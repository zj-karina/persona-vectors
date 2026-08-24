"""Which users does steering help, and can it be predicted from the vector alone?

Steering is roughly neutral on average, but that average covers users it helps
and users it hurts. This correlates three properties computable from the vector
matrix alone against each user's measured steering delta:

  P1  cos(v_u, v̄) — alignment with the direction shared by the population.
  P2  ‖v_u‖ / mean‖v‖ — the effective perturbation at a constant α.
  P3  mean cosine to the k nearest user vectors — whether the user sits in a
      crowd or is an outlier with an idiosyncratic profile.

The delta itself is measured inline: zero-shot vs α=1 for every user, scored
with per-example ROUGE-L.
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
    LaMPDataset, chat_kwargs_for, compute_rouge,
    load_model_and_tokenizer, persona_steered_generate, system_prompt_for,
    task_info,
)

PROXIES = ("P1_proj_global_mean", "P2_relative_magnitude", "P3_top_k_coherence")


def per_user_static_stats(vectors: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(norms[:, None], 1e-8)

    v_mean = vectors.mean(axis=0)
    v_mean_unit = v_mean / max(np.linalg.norm(v_mean), 1e-8)

    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)

    return {
        "P1_proj_global_mean": unit @ v_mean_unit,
        "P2_relative_magnitude": norms / max(norms.mean(), 1e-8),
        "P3_top_k_coherence": np.sort(sim, axis=1)[:, -k:].mean(axis=1),
        "norms": norms,
    }


@torch.no_grad()
def per_user_steering_delta(
    model, tokenizer, samples, vectors: np.ndarray, *,
    layer_idx: int, alpha: float, max_new_tokens: int,
    chat_kwargs: dict, system_prompt: str,
) -> dict:
    n = len(samples)
    rouge_zs = np.zeros(n)
    rouge_st = np.zeros(n)
    preds_zs, preds_st, refs = [], [], []

    t0 = time.time()
    for i, (s, v) in enumerate(zip(samples, vectors)):
        gold = s["output_text"].strip()
        pred_zs = persona_steered_generate(
            model, tokenizer, user_input=s["input_text"],
            persona_vector=None, layer_idx=None, alpha=0.0,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        pred_st = persona_steered_generate(
            model, tokenizer, user_input=s["input_text"],
            persona_vector=torch.from_numpy(v), layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        rouge_zs[i] = compute_rouge([pred_zs], [gold])["ROUGE-L"]
        rouge_st[i] = compute_rouge([pred_st], [gold])["ROUGE-L"]
        preds_zs.append(pred_zs)
        preds_st.append(pred_st)
        refs.append(gold)
        if (i + 1) % 5 == 0:
            running = float(np.mean(rouge_st[:i+1] - rouge_zs[:i+1]))
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)  avgΔ={running:+.3f}")

    return {
        "rouge_zs": rouge_zs, "rouge_steered": rouge_st,
        "delta": rouge_st - rouge_zs,
        "preds_zs": preds_zs, "preds_st": preds_st, "refs": refs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-7")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--variant", choices=["template", "fact"], default="template")
    ap.add_argument("--vectors_npz",
                    default="results/positive_control/vectors_Qwen3-8B_LaMP-7.npz")
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/variance_analysis")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz_path = ROOT / args.vectors_npz
    vectors = np.load(npz_path)[args.variant][:args.n_users]
    print(f"=== Variance analysis: {args.model}/{args.task} ===")
    print(f"  loaded {len(vectors)} {args.variant} vectors from {npz_path.name}")

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if len(samples) != len(vectors):
        raise ValueError(f"sample/vector count mismatch: "
                         f"{len(samples)} vs {len(vectors)}")

    info = task_info(args.task)
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    stats = per_user_static_stats(vectors, k=args.top_k)

    print("\n[1/2] Loading model for per-user delta...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    print("[2/2] Per-user steering delta (zs vs α=1)...")
    delta_info = per_user_steering_delta(
        model, tokenizer, samples, vectors,
        layer_idx=args.layer_idx, alpha=args.alpha,
        max_new_tokens=info["max_new_tokens"],
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
    )

    delta = delta_info["delta"]
    correls = {}
    for k in PROXIES:
        if np.std(stats[k]) > 1e-8 and np.std(delta) > 1e-8:
            correls[k] = float(np.corrcoef(stats[k], delta)[0, 1])
        else:
            correls[k] = float("nan")
    print("\n=== Pearson r(stat, ΔROUGE-L) ===")
    for k, r in correls.items():
        print(f"  {k:>30s}: r={r:+.3f}")

    per_user = [
        {
            "user_idx": i,
            **{k: float(stats[k][i]) for k in PROXIES},
            "norm":          float(stats["norms"][i]),
            "rouge_zs":      float(delta_info["rouge_zs"][i]),
            "rouge_steered": float(delta_info["rouge_steered"][i]),
            "delta":         float(delta[i]),
            "pred_zs":       delta_info["preds_zs"][i],
            "pred_steered":  delta_info["preds_st"][i],
            "gold":          delta_info["refs"][i],
            "input":         samples[i]["input_text"][:200],
        }
        for i in range(len(samples))
    ]

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_per_user_stats_n{len(samples)}_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "layer_idx": args.layer_idx,
            "alpha": args.alpha, "variant": args.variant, "n_users": len(samples),
            "top_k_for_coherence": args.top_k,
            "summary": {
                "mean_delta": float(delta.mean()),
                "std_delta":  float(delta.std()),
                "n_positive_delta": int((delta > 0.01).sum()),
                "n_negative_delta": int((delta < -0.01).sum()),
                "correlations": correls,
            },
            "per_user": per_user,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
