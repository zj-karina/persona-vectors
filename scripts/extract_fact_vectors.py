"""Standalone fact-vector extraction.  Mirrors `extract_template_vectors.py`
but uses local-LLM-generated fact prompts (cached in
`results/positive_control/cache_facts_<task>.json`).

Idempotent: reuses existing template + fact vectors and the fact cache.
"""

from __future__ import annotations

import argparse
import json
import os
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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    if args.output_npz is None:
        args.output_npz = str(ROOT / "results/positive_control"
                               / f"vectors_{short}_{args.task}.npz")
    if args.cache_path is None:
        args.cache_path = str(ROOT / "results/positive_control"
                               / f"cache_facts_{args.task}.json")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    chat_kwargs = chat_kwargs_for(args.model)

    out_path = Path(args.output_npz)
    existing = {}
    if out_path.exists():
        d = np.load(out_path)
        existing = {k: d[k] for k in d.keys()}
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

    # Need ALL samples (0..n) so users align with template vectors
    print(f"  extracting fact vectors for users {n_existing}..{args.n_users-1}")

    model, tokenizer = load_model_and_tokenizer(args.model)

    # 1) Build / extend fact cache
    fe = FactExtractor(model=model, tokenizer=tokenizer, task=args.task)
    enriched = fe.build_artifacts_for_dataset(
        samples=samples, n_users=args.n_users, cache_path=args.cache_path,
    )

    # 2) Extract activation vectors using fact prompts
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=args.layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    new_vectors: list[np.ndarray] = []
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
            rate = elapsed / max(done, 1)
            eta = rate * (args.n_users - i - 1)
            print(f"  fact {i+1}/{args.n_users}  "
                  f"({elapsed:.0f}s, rate={rate:.1f}s/u, eta={eta:.0f}s)")

    new_vectors = np.stack(new_vectors, axis=0)

    if "fact" in existing:
        merged_f = np.concatenate([existing["fact"], new_vectors], axis=0)
    else:
        merged_f = new_vectors
    save = {"fact": merged_f}
    if "template" in existing:
        save["template"] = existing["template"]   # untouched
    np.savez_compressed(out_path, **save)
    print(f"\nsaved {out_path}  "
          f"(template: {len(save.get('template', []))}, fact: {len(merged_f)})")


if __name__ == "__main__":
    main()
