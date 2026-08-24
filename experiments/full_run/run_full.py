"""Full-split evaluation on the two tasks where the smoke runs showed an effect.

Runs three passes over the same examples: a zero-shot control, steering at the
layer chosen by run_layer_search, and — when that layer differs from the
hand-picked default the smoke runs used — steering at the default too, so the
before/after comparison is on identical data.

    python run_full.py --model Qwen/Qwen3-8B --task LaMP-2
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

# Layers the smoke runs used before the layer search existed.
DEFAULT_LAYERS = {
    "Qwen3-8B": 18,
    "Qwen3-14B": 20,
    "Mistral-Small-24B-Instruct-2501": 22,
    "Llama-3.1-8B-Instruct": 16,
}


def default_layer(model_name: str) -> int:
    return DEFAULT_LAYERS.get(model_name.split("/")[-1], 16)


def _progress(label: str, i: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = elapsed / (i + 1)
    print(f"  {label}[{i+1}/{total}] {elapsed:.0f}s rate={rate:.1f}s/ex "
          f"eta={rate * (total - i - 1):.0f}s")


@torch.no_grad()
def run_eval_loop(
    model, tokenizer, dataset, *,
    persona_vectors_per_user: list[torch.Tensor | None] | None,
    layer_idx: int | None,
    alpha: float,
    chat_kwargs: dict,
    system_prompt: str,
) -> tuple[list[str], list[str], float]:
    preds, refs = [], []
    t0 = time.time()
    for i, sample in enumerate(dataset):
        v = persona_vectors_per_user[i] if persona_vectors_per_user else None
        pred = persona_steered_generate(
            model, tokenizer,
            user_input=sample["input_text"],
            persona_vector=v, layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=dataset.max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred)
        refs.append(sample["output_text"].strip())
        if (i + 1) % 50 == 0:
            _progress("", i, len(dataset), t0)
    return preds, refs, time.time() - t0


@torch.no_grad()
def extract_all_vectors(
    model, tokenizer, dataset, *,
    layer_idx: int, chat_kwargs: dict,
    extraction_questions: list[str],
) -> list[torch.Tensor | None]:
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    out: list[torch.Tensor | None] = []
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
        out.append(v)
        if (i + 1) % 50 == 0:
            _progress("extract ", i, len(dataset), t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_samples", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/full_run")
    ap.add_argument("--skip_default_layer", action="store_true",
                    help="Skip the default-layer comparison run (saves ~33%% time).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metric = task_info(args.task)["metric"]
    optimal = best_layer(args.model, args.task)
    default = default_layer(args.model)
    chosen_layer = optimal if optimal is not None else default
    print(f"=== Full run: {args.model} on {args.task}, n={args.n_samples}, α={args.alpha} ===")
    print(f"  optimal layer (from layer_search): {optimal}")
    print(f"  default layer (config fallback):   {default}")
    print(f"  chosen for primary run:            {chosen_layer}")

    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_samples,
                          data_dir=str(ROOT / "data"))
    extraction_questions = dataset.sample_train_inputs(k=1, seed=args.seed)

    print("\n--- Zero-shot ---")
    preds_zs, refs_zs, wall_zs = run_eval_loop(
        model, tokenizer, dataset,
        persona_vectors_per_user=None, layer_idx=None, alpha=0.0,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
    )
    m_zs = compute_metric(metric, preds_zs, refs_zs)
    print(f"  zero-shot {metric}: {m_zs}")

    print(f"\n--- Persona α={args.alpha} @ layer {chosen_layer} (optimal) ---")
    print("  extracting per-user vectors...")
    vecs_opt = extract_all_vectors(
        model, tokenizer, dataset,
        layer_idx=chosen_layer, chat_kwargs=chat_kwargs,
        extraction_questions=extraction_questions,
    )
    preds_opt, refs_opt, wall_opt = run_eval_loop(
        model, tokenizer, dataset,
        persona_vectors_per_user=vecs_opt, layer_idx=chosen_layer, alpha=args.alpha,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
    )
    m_opt = compute_metric(metric, preds_opt, refs_opt)
    print(f"  persona@{chosen_layer} {metric}: {m_opt}")

    m_def = wall_def = None
    if not args.skip_default_layer and default != chosen_layer:
        print(f"\n--- Persona α={args.alpha} @ layer {default} (default) ---")
        vecs_def = extract_all_vectors(
            model, tokenizer, dataset,
            layer_idx=default, chat_kwargs=chat_kwargs,
            extraction_questions=extraction_questions,
        )
        preds_def, refs_def, wall_def = run_eval_loop(
            model, tokenizer, dataset,
            persona_vectors_per_user=vecs_def, layer_idx=default, alpha=args.alpha,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        m_def = compute_metric(metric, preds_def, refs_def)
        print(f"  persona@{default} {metric}: {m_def}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full_{args.model.split('/')[-1]}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "n_samples": args.n_samples,
            "alpha": args.alpha, "seed": args.seed,
            "metric": metric,
            "optimal_layer": optimal, "default_layer": default,
            "chosen_layer": chosen_layer,
            "zero_shot": {"value": m_zs, "wall_seconds": wall_zs,
                          "num_eval": len(refs_zs),
                          "sample_preds": preds_zs[:5], "sample_refs": refs_zs[:5]},
            "persona_optimal": {"layer": chosen_layer, "value": m_opt,
                                "wall_seconds": wall_opt, "num_eval": len(refs_opt),
                                "sample_preds": preds_opt[:5],
                                "sample_refs": refs_opt[:5]},
            "persona_default": (None if m_def is None else {
                "layer": default, "value": m_def, "wall_seconds": wall_def,
            }),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
