"""Extract template-based persona vectors and nothing else.

The positive-control runs saved vectors for the first 30 users only; this
extends that set without re-running the downstream evaluation. It reads the
existing .npz, extracts the users beyond its current length, and writes the
merged array back, so an interrupted run can simply be restarted.

    python scripts/extract_template_vectors.py --task LaMP-7 --n_users 200
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-7")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_npz", default=None)
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    out_path = Path(args.output_npz or
                    ROOT / "results/positive_control" / f"vectors_{short}_{args.task}.npz")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    existing = {}
    if out_path.exists():
        d = np.load(out_path)
        existing = {k: d[k] for k in d}
        print(f"  loaded existing {len(existing.get('template', []))} template "
              f"and {len(existing.get('fact', []))} fact vectors from {out_path.name}")

    n_existing = len(existing.get("template", []))
    if n_existing >= args.n_users:
        print(f"already have {n_existing} ≥ requested {args.n_users}; nothing to do.")
        return

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

    model, tokenizer = load_model_and_tokenizer(args.model)
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=args.layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs_for(args.model),
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
            # Keep the row so user indices stay aligned with the fact vectors.
            print(f"  user {n_existing + i}: extract failed ({e}); zero-filling")
            v = np.zeros(model.config.hidden_size, dtype=np.float32)
        new_vectors.append(v)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            print(f"  {n_existing + i + 1}/{args.n_users}  "
                  f"({elapsed:.0f}s, rate={rate:.1f}s/u, "
                  f"eta={rate * (len(new_samples) - i - 1):.0f}s)")

    merged = np.stack(new_vectors, axis=0)
    if "template" in existing:
        merged = np.concatenate([existing["template"], merged], axis=0)
    save = {"template": merged}
    if "fact" in existing:
        save["fact"] = existing["fact"]
    np.savez_compressed(out_path, **save)
    print(f"\nsaved {out_path}  (template: {len(merged)}, "
          f"fact: {len(save.get('fact', []))})")


if __name__ == "__main__":
    main()
