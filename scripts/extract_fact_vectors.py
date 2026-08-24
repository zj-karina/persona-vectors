"""Extract fact-based persona vectors, mirroring extract_template_vectors.py.

Reuses the fact cache written by the positive-control run and only extracts
users the .npz does not already cover, so re-running is cheap.

    python scripts/extract_fact_vectors.py --task LaMP-2 --n_users 100
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
    FactExtractor, LaMPDataset, PersonaVectors,
    chat_kwargs_for, load_model_and_tokenizer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--n_users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_npz", default=None)
    ap.add_argument("--cache_path", default=None)
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    control_dir = ROOT / "results/positive_control"
    out_path = Path(args.output_npz or control_dir / f"vectors_{short}_{args.task}.npz")
    cache_path = Path(args.cache_path or control_dir / f"cache_facts_{args.task}.json")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    existing = {}
    if out_path.exists():
        d = np.load(out_path)
        existing = {k: d[k] for k in d}
        print(f"  loaded existing template={len(existing.get('template', []))} "
              f"fact={len(existing.get('fact', []))}")

    n_existing = len(existing.get("fact", []))
    if n_existing >= args.n_users:
        print(f"  already have {n_existing} ≥ {args.n_users} fact vectors; nothing to do.")
        return

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if len(samples) < args.n_users:
        print(f"  only {len(samples)} unique users available; capping.")
        args.n_users = len(samples)

    print(f"  extracting fact vectors for users {n_existing}..{args.n_users-1}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Facts are built for users 0..n so the indices line up with the template
    # vectors; the cache makes the already-done ones free.
    fe = FactExtractor(model=model, tokenizer=tokenizer, task=args.task)
    enriched = fe.build_artifacts_for_dataset(
        samples=samples, n_users=args.n_users, cache_path=str(cache_path),
    )

    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=args.layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs_for(args.model),
    )
    new_vectors = []
    t0 = time.time()
    for i in range(n_existing, args.n_users):
        s = enriched[i]
        try:
            v = pv.extract(
                positive_prompts=s["fact_positive_prompts"],
                negative_prompts=s["fact_negative_prompts"],
                extraction_questions=[s["input_text"]],
            ).cpu().float().numpy()
        except RuntimeError as e:
            print(f"  user {i}: extract failed ({e}); zero-fill")
            v = np.zeros(model.config.hidden_size, dtype=np.float32)
        new_vectors.append(v)
        done = i - n_existing + 1
        if done % 5 == 0 or i == args.n_users - 1:
            elapsed = time.time() - t0
            rate = elapsed / done
            print(f"  fact {i+1}/{args.n_users}  "
                  f"({elapsed:.0f}s, rate={rate:.1f}s/u, "
                  f"eta={rate * (args.n_users - i - 1):.0f}s)")

    merged = np.stack(new_vectors, axis=0)
    if "fact" in existing:
        merged = np.concatenate([existing["fact"], merged], axis=0)
    save = {"fact": merged}
    if "template" in existing:
        save["template"] = existing["template"]
    np.savez_compressed(out_path, **save)
    print(f"\nsaved {out_path}  "
          f"(template: {len(save.get('template', []))}, fact: {len(merged)})")


if __name__ == "__main__":
    main()
