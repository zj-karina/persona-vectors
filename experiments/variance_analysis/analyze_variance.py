"""Per-user signal-variance analysis for LaMP-7 persona vectors.

Hypothesis (paper §discussion): global single-layer additive steering at
constant α fails because users have heterogeneous dominant topics, so the
extracted persona vector averages out a coherent direction only for some
users.  User 3 (case study) works because their profile is topically
coherent.

This experiment correlates three per-user *static* signal-quality proxies
against per-user *downstream* steering delta:
    P1) projection of v_u onto the global mean v̄ (cos similarity).
        High P1 ⇒ user is well-aligned with the population-shared direction;
        steering moves all such users in the same way.
    P2) relative magnitude ‖v_u‖ / mean(‖v‖).
        Sets the *effective* perturbation α‖v_u‖ at a constant α.
    P3) topic coherence: mean cosine to top-k nearest-neighbour user vectors.
        High P3 ⇒ user has many neighbours in vector space (a "centroid"
        user); low P3 ⇒ outlier user with idiosyncratic profile.

Per-user steering delta is computed inline by running zero-shot vs.
α=1 steered inference for each user on their LaMP-7 paraphrase task and
taking ROUGE-L difference.

Output:
    results/variance_analysis/lamp7_per_user_stats.json
    figures/fig_variance_analysis_lamp7.pdf
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
    LaMPDataset, chat_kwargs_for, compute_rouge,
    load_model_and_tokenizer, persona_steered_generate, system_prompt_for,
    task_info,
)


def per_user_static_stats(vectors: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    """Compute P1, P2, P3 from the user-vector matrix only (no inference)."""
    n = len(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    u = vectors / np.maximum(norms[:, None], 1e-8)

    # P1 — projection on global mean direction
    v_mean = vectors.mean(axis=0)
    v_mean_unit = v_mean / max(np.linalg.norm(v_mean), 1e-8)
    p1 = u @ v_mean_unit

    # P2 — relative magnitude
    p2 = norms / max(norms.mean(), 1e-8)

    # P3 — topic coherence: mean cos to top-k nearest neighbours
    sim = u @ u.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.sort(sim, axis=1)[:, -k:]
    p3 = topk.mean(axis=1)

    return {
        "P1_proj_global_mean": p1,
        "P2_relative_magnitude": p2,
        "P3_top_k_coherence": p3,
        "norms": norms,
    }


@torch.no_grad()
def per_user_steering_delta(
    model, tokenizer, samples, vectors: np.ndarray, *,
    layer_idx: int, alpha: float, max_new_tokens: int,
    chat_kwargs: dict, system_prompt: str,
) -> dict[str, np.ndarray]:
    """For each user: ROUGE-L(steered) − ROUGE-L(zs)."""
    n = len(samples)
    rouge_zs = np.zeros(n)
    rouge_st = np.zeros(n)
    preds_zs, preds_st, refs = [], [], []

    t0 = time.time()
    for i, (s, v) in enumerate(zip(samples, vectors)):
        v_t = torch.from_numpy(v)
        gold = s["output_text"].strip()
        pred_zs = persona_steered_generate(
            model, tokenizer, user_input=s["input_text"],
            persona_vector=None, layer_idx=None, alpha=0.0,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        pred_st = persona_steered_generate(
            model, tokenizer, user_input=s["input_text"],
            persona_vector=v_t, layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        # Per-user single-pair ROUGE-L
        r_zs = compute_rouge([pred_zs], [gold])["ROUGE-L"]
        r_st = compute_rouge([pred_st], [gold])["ROUGE-L"]
        rouge_zs[i] = r_zs
        rouge_st[i] = r_st
        preds_zs.append(pred_zs); preds_st.append(pred_st); refs.append(gold)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n}  ({elapsed:.0f}s)  "
                  f"avgΔ={float(np.mean(rouge_st[:i+1]-rouge_zs[:i+1])):+.3f}")
    return {
        "rouge_zs": rouge_zs, "rouge_steered": rouge_st,
        "delta": rouge_st - rouge_zs,
        "preds_zs": preds_zs, "preds_st": preds_st, "refs": refs,
    }


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-7")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--variant", choices=["template", "fact"], default="template",
                    help="Which extracted vectors to analyse.")
    ap.add_argument("--vectors_npz",
                    default="results/positive_control/vectors_Qwen3-8B_LaMP-7.npz")
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/variance_analysis")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    npz_path = ROOT / args.vectors_npz
    arrs = np.load(npz_path)
    vectors = arrs[args.variant][:args.n_users]
    print(f"=== Variance analysis: {args.model}/{args.task} ===")
    print(f"  loaded {len(vectors)} {args.variant} vectors from {npz_path.name}")

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if len(samples) != len(vectors):
        raise ValueError(f"sample/vector count mismatch")

    info = task_info(args.task)
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    # Static stats — instant
    stats = per_user_static_stats(vectors, k=args.top_k)

    # Per-user delta — needs model
    print("\n[1/2] Loading model for per-user delta...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    print("[2/2] Per-user steering delta (zs vs α=1)...")
    delta_info = per_user_steering_delta(
        model, tokenizer, samples, vectors,
        layer_idx=args.layer_idx, alpha=args.alpha,
        max_new_tokens=info["max_new_tokens"],
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
    )

    # Correlations
    correls = {}
    for k in ("P1_proj_global_mean", "P2_relative_magnitude", "P3_top_k_coherence"):
        if np.std(stats[k]) > 1e-8 and np.std(delta_info["delta"]) > 1e-8:
            correls[k] = float(np.corrcoef(stats[k], delta_info["delta"])[0, 1])
        else:
            correls[k] = float("nan")
    print("\n=== Pearson r(stat, ΔROUGE-L) ===")
    for k, r in correls.items():
        print(f"  {k:>30s}: r={r:+.3f}")

    # Compose per-user table
    per_user = []
    for i in range(len(samples)):
        per_user.append({
            "user_idx": i,
            "P1_proj_global_mean":  float(stats["P1_proj_global_mean"][i]),
            "P2_relative_magnitude": float(stats["P2_relative_magnitude"][i]),
            "P3_top_k_coherence":   float(stats["P3_top_k_coherence"][i]),
            "norm":                 float(stats["norms"][i]),
            "rouge_zs":             float(delta_info["rouge_zs"][i]),
            "rouge_steered":        float(delta_info["rouge_steered"][i]),
            "delta":                float(delta_info["delta"][i]),
            "pred_zs":              delta_info["preds_zs"][i],
            "pred_steered":         delta_info["preds_st"][i],
            "gold":                 delta_info["refs"][i],
            "input":                samples[i]["input_text"][:200],
        })

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_per_user_stats_n{len(samples)}_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "layer_idx": args.layer_idx,
            "alpha": args.alpha, "variant": args.variant, "n_users": len(samples),
            "top_k_for_coherence": args.top_k,
            "summary": {
                "mean_delta": float(delta_info["delta"].mean()),
                "std_delta":  float(delta_info["delta"].std()),
                "n_positive_delta": int((delta_info["delta"] > 0.01).sum()),
                "n_negative_delta": int((delta_info["delta"] < -0.01).sum()),
                "correlations": correls,
            },
            "per_user": per_user,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
