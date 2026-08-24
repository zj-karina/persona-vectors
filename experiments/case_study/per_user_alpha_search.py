"""The smallest α that flips each user's prediction to the gold answer.

The single-user case study shows a sharp phase boundary; this checks whether
that boundary sits in the same place for everyone. If the per-user α* is spread
out, a single global α is the wrong knob and the population average will hide
users that steering does help.

Downstream inference only — the vectors come from the positive-control .npz.

    python experiments/case_study/per_user_alpha_search.py --variant fact
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
    LaMPDataset, chat_kwargs_for, load_model_and_tokenizer,
    persona_steered_generate, system_prompt_for, task_info,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--variant", choices=["template", "fact"], default="fact")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--vectors_npz",
                    default="results/positive_control/vectors_Qwen3-8B_LaMP-2.npz")
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/case_study")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = task_info(args.task)
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    print(f"=== Per-user α search: {args.model} on {args.task} ===")
    print(f"  layer={args.layer_idx} variant={args.variant} αs={args.alphas}")

    npz_path = ROOT / args.vectors_npz
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} — run positive_control first")
    vectors = np.load(npz_path)[args.variant][:args.n_users]
    print(f"  loaded {len(vectors)} vectors of dim {vectors.shape[1]} from {npz_path.name}")

    # Same dedup as positive_control, or user i here is not user i there.
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if len(samples) != len(vectors):
        raise ValueError(f"mismatch: {len(samples)} samples vs {len(vectors)} vectors")

    model, tokenizer = load_model_and_tokenizer(args.model)

    rows = []
    t0 = time.time()
    for i, (s, v) in enumerate(zip(samples, vectors)):
        v_t = torch.from_numpy(v)
        gold = s["output_text"].strip().lower()
        per_alpha = []
        first_correct_alpha = None
        for alpha in args.alphas:
            pred = persona_steered_generate(
                model, tokenizer,
                user_input=s["input_text"],
                persona_vector=v_t if alpha != 0 else None,
                layer_idx=args.layer_idx, alpha=alpha,
                max_new_tokens=info["max_new_tokens"],
                chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            )
            match = pred.strip().lower() == gold
            per_alpha.append({"alpha": alpha, "pred": pred, "match": match})
            if match and first_correct_alpha is None:
                first_correct_alpha = alpha

        zs_correct = per_alpha[0]["match"]
        any_correct = any(p["match"] for p in per_alpha)
        rows.append({
            "user_idx": i,
            "gold": gold,
            "input_snippet": s["input_text"][:120],
            "vector_norm": float(np.linalg.norm(v)),
            "zs_correct": zs_correct,
            "any_correct": any_correct,
            "flip": not zs_correct and any_correct,
            "first_correct_alpha": first_correct_alpha,
            "per_alpha": per_alpha,
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(samples)} ({time.time()-t0:.0f}s)  "
                  f"flips so far: {sum(r['flip'] for r in rows)}")

    n = len(rows)
    n_zs_correct = sum(r["zs_correct"] for r in rows)
    n_flips = sum(r["flip"] for r in rows)
    n_unsteerable = sum(1 for r in rows if not r["any_correct"])
    flip_alphas = [r["first_correct_alpha"] for r in rows if r["flip"]]

    summary = {
        "n_users": n,
        "n_zs_correct": n_zs_correct,
        "n_flips_to_correct": n_flips,
        "n_unsteerable": n_unsteerable,
        "fraction_steerable_to_correct": n_flips / max(n - n_zs_correct, 1),
        "flip_alpha_distribution": {
            "values": flip_alphas,
            "min": min(flip_alphas) if flip_alphas else None,
            "max": max(flip_alphas) if flip_alphas else None,
            "mean": float(np.mean(flip_alphas)) if flip_alphas else None,
            "std":  float(np.std(flip_alphas)) if flip_alphas else None,
        },
    }
    print("\n=== Summary ===")
    print(f"  total users:         {n}")
    print(f"  zs already correct:  {n_zs_correct}")
    print(f"  flipped by steering: {n_flips}  "
          f"({summary['fraction_steerable_to_correct']:.1%} of zs-wrong users)")
    print(f"  never correct:       {n_unsteerable}")
    if flip_alphas:
        print(f"  first-correct α: min={min(flip_alphas)}, max={max(flip_alphas)}, "
              f"mean={np.mean(flip_alphas):.3f}, std={np.std(flip_alphas):.3f}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir /
                f"alpha_search_{args.model.split('/')[-1]}_{args.task}_{args.variant}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "layer_idx": args.layer_idx,
            "variant": args.variant, "alphas": args.alphas, "n_users": n,
            "summary": summary, "per_user": rows,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
