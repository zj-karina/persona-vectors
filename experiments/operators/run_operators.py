"""Alternative steering operators for per-user persona vectors.

Baseline: single-layer additive steering at constant α.
This script tests modifications to the *vector preprocessing* before injection
and to the injection rule itself.  Pluggable via --operator.

Operators implemented:

* `additive` (baseline) -- h_l += α v_user (canonical persona-vector
  steering).
* `proj_nuisance` (Operator 1) -- compute a top-k PCA basis B of *all*
  user vectors, then steer with v_clean = v_user - B Bᵀ v_user.  The PCA
  directions capture variance shared across users (often a dominant
  ``act-as-author'' template direction); projecting them out leaves the
  user-specific component.

Future operators (e.g.\ multi-layer, per-token scaling) plug in here by
extending `make_steering_vector(...)` and/or the steering loop.

Usage:
    python experiments/operators/run_operators.py \\
        --task LaMP-7 --model Qwen/Qwen3-8B --layer_idx 13 \\
        --variant template --operator proj_nuisance --pca_k 5 \\
        --alphas 0.0 0.5 1.0 1.5 2.0 --n_users 500
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


# ---------------------------------------------------------------------------
# Operators — turn raw user vectors into "clean" per-user steering vectors.
# ---------------------------------------------------------------------------


def op_additive(vectors: np.ndarray, **_) -> tuple[np.ndarray, dict]:
    """Identity transform — canonical persona-vector additive steering."""
    return vectors, {"name": "additive", "params": {}}


def op_proj_nuisance(vectors: np.ndarray, *, pca_k: int) -> tuple[np.ndarray, dict]:
    """Project out the top-k PCA directions of the population from each user vector.

    The top-k PCs capture variance shared *across* users (template direction,
    backbone biases).  Removing them keeps only user-specific residual.
    """
    n, d = vectors.shape
    centroid = vectors.mean(axis=0, keepdims=True)
    centered = vectors - centroid
    # Truncated SVD: U S Vt = centered;  the right-singular-vectors V are the
    # PCA directions in feature space.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    B = Vt[:pca_k]                        # [k, d]
    # Project EACH original vector (not centered) to keep magnitude consistent
    # with the additive baseline; only directionally remove B-span.
    proj = vectors @ B.T @ B              # [n, d]
    cleaned = vectors - proj
    info = {
        "name": "proj_nuisance",
        "params": {
            "pca_k": pca_k,
            "removed_var_fraction": float(
                np.linalg.norm(proj) ** 2 / max(np.linalg.norm(vectors) ** 2, 1e-8)
            ),
            "preserved_var_fraction": float(
                np.linalg.norm(cleaned) ** 2 / max(np.linalg.norm(vectors) ** 2, 1e-8)
            ),
            "mean_norm_before": float(np.linalg.norm(vectors, axis=1).mean()),
            "mean_norm_after": float(np.linalg.norm(cleaned, axis=1).mean()),
        },
    }
    return cleaned, info


def op_norm_fixed(vectors: np.ndarray, *, target_norm: float | None = None,
                  ) -> tuple[np.ndarray, dict]:
    """Rescale each user vector to a fixed L2 norm to test whether the
    overshoot mechanism (longer prompts -> larger ||v||) explains
    fact-vs-template downstream gaps. If target_norm is None, use the
    population median of ||v_u||."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if target_norm is None:
        target_norm = float(np.median(norms))
    scale = target_norm / norms.clip(min=1e-8)
    rescaled = vectors * scale
    info = {
        "name": "norm_fixed",
        "params": {
            "target_norm": float(target_norm),
            "median_norm_before": float(np.median(norms)),
            "min_norm_before": float(norms.min()),
            "max_norm_before": float(norms.max()),
        },
    }
    return rescaled, info


OPERATORS = {
    "additive":       op_additive,
    "proj_nuisance":  op_proj_nuisance,
    "norm_fixed":     op_norm_fixed,
}


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_alpha(
    model, tokenizer, samples, vectors, *,
    layer_idx: int, alpha: float, max_new_tokens: int,
    chat_kwargs: dict, system_prompt: str,
) -> tuple[list[str], list[str], float]:
    preds, refs = [], []
    t0 = time.time()
    for i, (s, v) in enumerate(zip(samples, vectors)):
        v_t = torch.from_numpy(v) if alpha != 0 else None
        pred = persona_steered_generate(
            model, tokenizer, user_input=s["input_text"],
            persona_vector=v_t, layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred); refs.append(s["output_text"].strip())
        if (i + 1) % 50 == 0:
            print(f"    α={alpha} {i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
    return preds, refs, time.time() - t0


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-7")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--variant", choices=["template", "fact"], default="template")
    ap.add_argument("--operator", choices=list(OPERATORS), default="proj_nuisance")
    ap.add_argument("--pca_k", type=int, default=5,
                    help="For proj_nuisance: number of top PCs to project out.")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--n_users", type=int, default=500)
    ap.add_argument("--vectors_npz", default=None,
                    help="Defaults to results/positive_control/vectors_<short>_<task>.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/operators")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    short = args.model.split("/")[-1]
    if args.vectors_npz is None:
        args.vectors_npz = str(ROOT / "results/positive_control"
                                / f"vectors_{short}_{args.task}.npz")

    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    print(f"=== {args.operator} on {args.model} / {args.task} / {args.variant} ===")
    print(f"  layer={args.layer_idx}  α={args.alphas}  n_users={args.n_users}")

    # 1. Load vectors
    arrs = np.load(args.vectors_npz)
    if args.variant not in arrs:
        raise KeyError(f"{args.variant} not in {args.vectors_npz}: {list(arrs.keys())}")
    vectors = arrs[args.variant]
    if len(vectors) < args.n_users:
        print(f"  warning: only {len(vectors)} vectors available, capping n_users")
        args.n_users = len(vectors)
    vectors = vectors[: args.n_users]

    # 2. Apply operator transform
    if args.operator == "proj_nuisance":
        cleaned, op_info = OPERATORS[args.operator](vectors, pca_k=args.pca_k)
    else:
        cleaned, op_info = OPERATORS[args.operator](vectors)
    print(f"  operator info: {json.dumps(op_info['params'], indent=2)}")

    # 3. Load dataset
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if len(samples) != len(cleaned):
        raise ValueError(
            f"sample/vector mismatch: {len(samples)} vs {len(cleaned)}")

    model, tokenizer = load_model_and_tokenizer(args.model)

    # 4. Sweep α
    results_per_alpha = []
    for alpha in args.alphas:
        print(f"\n  --- α={alpha} ---")
        preds, refs, wall = eval_alpha(
            model, tokenizer, samples, cleaned,
            layer_idx=args.layer_idx, alpha=alpha,
            max_new_tokens=info["max_new_tokens"],
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        m = compute_metric(metric, preds, refs)
        print(f"  α={alpha}: {m}")
        results_per_alpha.append({
            "alpha": alpha, "metric": metric, "value": m,
            "wall_seconds": wall, "num_eval": len(refs),
        })

    # 5. Save
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.operator}"
    if args.operator == "proj_nuisance":
        tag = f"{tag}_k{args.pca_k}"
    out_path = out_dir / f"{args.task}_{short}_{args.variant}_{tag}_n{args.n_users}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "layer_idx": args.layer_idx,
            "variant": args.variant, "operator": op_info, "alphas": args.alphas,
            "n_users": args.n_users, "metric": metric,
            "results": results_per_alpha,
            "vectors_npz": args.vectors_npz,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
