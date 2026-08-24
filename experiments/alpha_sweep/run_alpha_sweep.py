"""Steering-strength sweep at the best layer.

In the smoke run α ∈ {0.5, 1.0, 1.5, 2.0} produced near-identical predictions,
which suggests α may simply be small next to the residual-stream norm. This
widens the grid to α ≤ 16 to map the whole response curve; large α is expected
to degrade the base task, so a peak followed by collapse is the shape to look for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, PersonaVectors, best_layer, compute_metric,
    load_model_and_tokenizer, persona_steered_generate, chat_kwargs_for,
    system_prompt_for, task_info,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--layer_idx", type=int, default=None)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/alpha_sweep")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metric = task_info(args.task)["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)
    layer_idx = (args.layer_idx if args.layer_idx is not None
                 else best_layer(args.model, args.task, fallback=16))

    print(f"=== α-sweep: {args.model} / {args.task} / layer {layer_idx} ===")
    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_samples,
                          data_dir=str(ROOT / "data"))
    extraction_questions = dataset.sample_train_inputs(k=1, seed=args.seed)

    # Vectors do not depend on α, so extract once and reuse across the grid.
    pv = PersonaVectors(model=model, tokenizer=tokenizer, layer_idx=layer_idx,
                        max_new_tokens=50, chat_template_kwargs=chat_kwargs)
    print("Extracting per-user vectors (shared across α values)...")
    user_vectors: list[torch.Tensor | None] = []
    t0 = time.time()
    for i, s in enumerate(dataset):
        try:
            v = pv.extract(
                positive_prompts=s["positive_system_prompts"],
                negative_prompts=s["negative_system_prompts"],
                extraction_questions=extraction_questions,
            )
        except RuntimeError:
            v = None
        user_vectors.append(v)
        if (i + 1) % 50 == 0:
            print(f"  extract [{i+1}/{len(dataset)}] {time.time()-t0:.0f}s")

    results = []
    for alpha in args.alphas:
        print(f"\n--- α={alpha} ---")
        preds, refs = [], []
        t0 = time.time()
        for i, s in enumerate(dataset):
            pred = persona_steered_generate(
                model, tokenizer,
                user_input=s["input_text"],
                persona_vector=user_vectors[i], layer_idx=layer_idx, alpha=alpha,
                max_new_tokens=dataset.max_new_tokens,
                chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            )
            preds.append(pred)
            refs.append(s["output_text"].strip())
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}] {time.time()-t0:.0f}s")
        m = compute_metric(metric, preds, refs)
        print(f"  α={alpha}: {m}")
        results.append({"alpha": alpha, "metric": metric, "value": m,
                        "wall_seconds": time.time() - t0,
                        "sample_preds": preds[:5], "sample_refs": refs[:5]})

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alpha_sweep_{args.model.split('/')[-1]}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "n_samples": args.n_samples,
            "layer_idx": layer_idx, "seed": args.seed,
            "alphas": args.alphas, "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
