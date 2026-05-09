"""Standalone template-vector extraction — no downstream inference.

Used to extend the sample size beyond the n=30 saved during the original
positive-control runs.  Saves to results/positive_control/vectors_*.npz
in the same shape (template field).  Idempotent: if the npz exists,
it loads it and only extracts new users beyond its length.

Usage:
    python scripts/extract_template_vectors.py \\
        --model Qwen/Qwen3-8B --task LaMP-7 --layer_idx 13 --n_users 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, PersonaVectors,
    chat_kwargs_for, load_model_and_tokenizer,
)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-7")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_npz",
                    default=None)
    args = ap.parse_args()

    if args.output_npz is None:
        short = args.model.split("/")[-1]
        args.output_npz = str(ROOT / "results/positive_control"
                               / f"vectors_{short}_{args.task}.npz")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    chat_kwargs = chat_kwargs_for(args.model)

    # 1. Load any existing vectors (idempotency)
    out_path = Path(args.output_npz)
    existing = {}
    if out_path.exists():
        d = np.load(out_path)
        existing = {k: d[k] for k in d.keys()}
        print(f"  loaded existing {len(existing.get('template', []))} template "
              f"and {len(existing.get('fact', []))} fact vectors from "
              f"{out_path.name}")

    n_existing = len(existing.get("template", []))
    if n_existing >= args.n_users:
        print(f"already have {n_existing} ≥ requested {args.n_users}; nothing to do.")
        return

    # 2. Load dataset (dedup'd) and slice the new users
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if len(samples) < args.n_users:
        print(f"  warning: only {len(samples)} unique users available "
              f"for {args.task}; capping.")
        args.n_users = len(samples)

    new_samples = samples[n_existing:args.n_users]
    print(f"  extracting {len(new_samples)} new template vectors "
          f"(users {n_existing}..{args.n_users-1})")

    # 3. Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # 4. Extract
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=args.layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    extraction_questions = dataset.sample_train_inputs(k=1, seed=args.seed)

    new_vectors = []
    t0 = time.time()
    for i, s in enumerate(new_samples):
        try:
            v = pv.extract(
                positive_prompts=s["positive_system_prompts"],
                negative_prompts=s["negative_system_prompts"],
                extraction_questions=extraction_questions,
            ).cpu().float().numpy()
        except RuntimeError as e:
            print(f"  user {n_existing + i}: extract failed ({e}); zero-filling")
            v = np.zeros(model.config.hidden_size, dtype=np.float32)
        new_vectors.append(v)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(new_samples) - (i + 1))
            print(f"  {n_existing + i + 1}/{args.n_users}  "
                  f"({elapsed:.0f}s, rate={rate:.1f}s/u, eta={eta:.0f}s)")
    new_vectors = np.stack(new_vectors, axis=0)

    # 5. Save merged
    if "template" in existing:
        merged_t = np.concatenate([existing["template"], new_vectors], axis=0)
    else:
        merged_t = new_vectors
    save = {"template": merged_t}
    if "fact" in existing:
        save["fact"] = existing["fact"]   # untouched
    np.savez_compressed(out_path, **save)
    print(f"\nsaved {out_path}  (template: {len(merged_t)}, "
          f"fact: {len(save.get('fact', []))})")


if __name__ == "__main__":
    main()
