"""Operator 2: rank-1 weight edit at the best steering layer (ROME-style).

Hypothesis: activation-space additive steering is overwritten by subsequent
layers. A direct weight edit persists through the rest of the forward pass.

Implementation (single, global rank-1 edit applied at MLP `down_proj`):

    W_down_new  =  W_down + α * (v_user_mean ⊗ k*) / (k* · k*)

where
  * `v_user_mean` is the mean persona vector across all users (in hidden dim H);
  * `k*` is the mean post-activation MLP intermediate at the last prompt token,
    averaged across the same users (in intermediate dim D_ff).

Per-user personalisation is achieved by scaling α by
`s_u = (v_u · v_user_mean) / ||v_user_mean||^2`, so users whose vector aligns
with the global mean get a larger edit, mis-aligned users get less / negative.

Each (α, user) pair edits a fresh copy of the weight tensor and reverts
afterwards (weights restored from a CPU snapshot to avoid drift).

Usage:
    python experiments/operators/run_rank1_edit.py \\
        --task LaMP-7 --variant template --layer_idx 13 \\
        --alphas 0.0 0.5 1.0 2.0 --n_users 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, build_chat_prompt, chat_kwargs_for, compute_metric,
    load_model_and_tokenizer, system_prompt_for, task_info,
)
from src.persona_vectors import get_decoder_layers


# ---------------------------------------------------------------------------
# Find the down_proj of an MLP block
# ---------------------------------------------------------------------------


def get_mlp_down_proj(layer):
    """Return the Linear layer that projects from intermediate -> hidden."""
    for name in ("down_proj", "fc2", "out_proj"):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, name):
            return getattr(layer.mlp, name)
    raise AttributeError("Cannot locate MLP output projection")


# ---------------------------------------------------------------------------
# Compute k* — mean intermediate activation at last prompt token
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_k_star(model, tokenizer, samples, *, layer_idx: int,
                   chat_kwargs: dict, system_prompt: str,
                   max_users: int = 200) -> np.ndarray:
    """Run forward over each user's input and capture the input to down_proj
    at the last prompt token (== output of gate*up == intermediate space)."""
    layers = get_decoder_layers(model)
    target = get_mlp_down_proj(layers[layer_idx])

    captured: list[torch.Tensor] = []

    def capture(module, inputs):
        # `inputs[0]` is the intermediate-dim tensor; shape [B, T, D_ff]
        captured.append(inputs[0][:, -1, :].detach().float().cpu())

    handle = target.register_forward_pre_hook(capture)
    try:
        for i, s in enumerate(samples[:max_users]):
            prompt = build_chat_prompt(tokenizer, s["input_text"], system_prompt,
                                       chat_kwargs)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=1024).to(next(model.parameters()).device)
            model(**enc, use_cache=False)
            if (i + 1) % 25 == 0:
                print(f"  k* capture {i+1}/{min(max_users, len(samples))}")
    finally:
        handle.remove()

    stack = torch.cat(captured, dim=0)  # [N, D_ff]
    return stack.mean(dim=0).numpy()    # [D_ff]


# ---------------------------------------------------------------------------
# Rank-1 edit context manager
# ---------------------------------------------------------------------------


@contextmanager
def rank1_edit(layer, v_h: np.ndarray, k_ff: np.ndarray, scale: float):
    """Temporarily add `scale * v_h ⊗ k_ff / (k_ff·k_ff)` to down_proj.weight.
    Reverts on exit.
    """
    target = get_mlp_down_proj(layer)
    W = target.weight  # shape [hidden_dim, intermediate_dim]
    device, dtype = W.device, W.dtype

    v = torch.from_numpy(v_h).to(device=device, dtype=dtype)        # [H]
    k = torch.from_numpy(k_ff).to(device=device, dtype=dtype)       # [D_ff]
    denom = (k.float() @ k.float()).clamp(min=1e-8).item()
    delta = (scale / denom) * torch.outer(v, k).to(dtype=dtype)     # [H, D_ff]

    with torch.no_grad():
        W.add_(delta)
    try:
        yield
    finally:
        with torch.no_grad():
            W.sub_(delta)


@torch.no_grad()
def eval_alpha_rank1(
    model, tokenizer, samples, vectors, v_mean, k_star, *,
    layer_idx: int, alpha: float,
    max_new_tokens: int, chat_kwargs: dict, system_prompt: str,
) -> tuple[list[str], list[str], float]:
    """For each user: compute personalised scale s_u, apply rank-1 edit, generate."""
    layers = get_decoder_layers(model)
    target_layer = layers[layer_idx]

    v_mean_norm_sq = float(np.dot(v_mean, v_mean))
    preds, refs = [], []
    t0 = time.time()
    for i, (s, v) in enumerate(zip(samples, vectors)):
        s_u = float(np.dot(v, v_mean) / max(v_mean_norm_sq, 1e-8))
        scale = alpha * s_u

        prompt = build_chat_prompt(tokenizer, s["input_text"], system_prompt, chat_kwargs)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024).to(next(model.parameters()).device)

        if alpha == 0:
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        else:
            with rank1_edit(target_layer, v_mean, k_star, scale):
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)

        new_tokens = out[0, enc["input_ids"].shape[1]:]
        preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        refs.append(s["output_text"].strip())
        if (i + 1) % 25 == 0:
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
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--n_users", type=int, default=200)
    ap.add_argument("--vectors_npz", default=None)
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

    print(f"=== rank-1 edit on {args.model}/{args.task}/{args.variant} ===")
    print(f"  layer={args.layer_idx} α={args.alphas} n={args.n_users}")

    arrs = np.load(args.vectors_npz)
    vectors = arrs[args.variant][: args.n_users]
    v_mean = vectors.mean(axis=0)

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)

    model, tokenizer = load_model_and_tokenizer(args.model)

    print("\n[1/2] Capturing k* (mean intermediate activation at last prompt token)...")
    k_star = compute_k_star(
        model, tokenizer, samples,
        layer_idx=args.layer_idx, chat_kwargs=chat_kwargs,
        system_prompt=system_prompt, max_users=min(args.n_users, 200),
    )
    print(f"  k* shape: {k_star.shape}, ‖k*‖={np.linalg.norm(k_star):.2f}")

    print("\n[2/2] Sweeping α...")
    grid_results = []
    for alpha in args.alphas:
        print(f"\n  --- α={alpha} ---")
        preds, refs, wall = eval_alpha_rank1(
            model, tokenizer, samples, vectors, v_mean, k_star,
            layer_idx=args.layer_idx, alpha=alpha,
            max_new_tokens=info["max_new_tokens"],
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        m = compute_metric(metric, preds, refs)
        print(f"  α={alpha}: {m}")
        grid_results.append({
            "alpha": alpha, "metric": metric, "value": m, "wall_seconds": wall,
        })

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank1_edit_{args.task}_{short}_{args.variant}_n{args.n_users}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "layer_idx": args.layer_idx,
            "variant": args.variant, "alphas": args.alphas,
            "n_users": args.n_users, "metric": metric,
            "k_star_norm": float(np.linalg.norm(k_star)),
            "v_mean_norm": float(np.linalg.norm(v_mean)),
            "results": grid_results,
            "vectors_npz": args.vectors_npz,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
