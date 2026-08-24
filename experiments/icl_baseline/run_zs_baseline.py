"""Zero-shot baseline (no profile, no steering) for the same sample sizes
used in the ICL and steering experiments.  Needed for an apples-to-apples
comparison at n=100 (LaMP-2) and n=200 (LaMP-7), since the prior ZS numbers
were on the n=30 dedup'd subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, chat_kwargs_for, compute_metric,
    load_model_and_tokenizer, persona_steered_generate, system_prompt_for,
    task_info,
)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/icl_baseline_extended")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)

    model, tokenizer = load_model_and_tokenizer(args.model)

    preds, refs = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        pred = persona_steered_generate(
            model, tokenizer, user_input=s["input_text"],
            persona_vector=None, layer_idx=None, alpha=0.0,
            max_new_tokens=info["max_new_tokens"],
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred); refs.append(s["output_text"].strip())
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
    m = compute_metric(metric, preds, refs)
    print(f"  ZS {args.task} n={len(samples)}: {m}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_zs_n{len(samples)}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "n_users": len(samples),
            "metric": metric, "value": m,
            "preds": preds, "refs": refs,
            "wall_seconds": time.time() - t0,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
