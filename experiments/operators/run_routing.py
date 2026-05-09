"""Operator 3: token-selective steering via cosine-gate routing.

Standard additive steering applies $h_l[t] \\mathrel{+}=\\alpha v_u$ at every
token position $t$ during decoding.  Hypothesis: most positions are not
"thinking about user identity" -- adding $v_u$ there is noise.  A simple
gate selects only positions whose pre-injection hidden state has high
projection on the persona direction:

    gate_t = (h_l[t] · v_u) / (||v_u|| ||h_l[t]||)
    h_l[t] += α v_u   if gate_t > τ else 0

Sweep $(\\tau, \\alpha)$ to find the operating point.

Usage:
    python experiments/operators/run_routing.py \\
        --task LaMP-7 --variant template --layer_idx 13 \\
        --taus 0.0 0.25 0.5 0.75 1.0 \\
        --alphas 0.0 0.5 1.0 2.0 \\
        --n_users 200
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
    LaMPDataset, chat_kwargs_for, compute_metric,
    load_model_and_tokenizer, persona_steered_generate, system_prompt_for,
    task_info,
)
from src.persona_vectors import (
    _layer_hidden, _replace_layer_hidden, get_decoder_layers,
)


# ---------------------------------------------------------------------------
# Routing hook
# ---------------------------------------------------------------------------


class RoutingSteering:
    """Per-token gated additive steering.

    Adds α v_u to position t only where  cos(h_l[t], v_u) > τ.
    """

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._layers = get_decoder_layers(model)

    @contextmanager
    def hook(self, vector: torch.Tensor, alpha: float, tau: float):
        v_cpu = vector.detach().float().cpu()
        cache: dict[tuple, torch.Tensor] = {}

        def vec_for(device, dtype):
            key = (device, dtype)
            if key not in cache:
                cache[key] = v_cpu.to(device=device, dtype=dtype)
            return cache[key]

        def fwd_hook(module, inputs, output):
            hidden = _layer_hidden(output)            # [B, T, H]
            v = vec_for(hidden.device, hidden.dtype)  # [H]
            v_norm = v.float().norm().clamp(min=1e-8)
            # cosine of each token with v
            h32 = hidden.float()
            h_norms = h32.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, T, 1]
            dots = (h32 @ v.float())                                   # [B, T]
            cos = dots / (h_norms.squeeze(-1) * v_norm)                # [B, T]
            mask = (cos > tau).to(hidden.dtype)                        # [B, T]
            hidden = hidden + alpha * mask.unsqueeze(-1) * v
            return _replace_layer_hidden(output, hidden)

        handle = self._layers[self.layer_idx].register_forward_hook(fwd_hook)
        try:
            yield self
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# Eval loop with routing
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_pair(
    model, tokenizer, samples, vectors, *,
    layer_idx: int, alpha: float, tau: float,
    max_new_tokens: int, chat_kwargs: dict, system_prompt: str,
) -> tuple[list[str], list[str], float]:
    preds, refs = [], []
    routing = RoutingSteering(model, layer_idx)
    t0 = time.time()
    for i, (s, v) in enumerate(zip(samples, vectors)):
        if alpha == 0:
            ctx = None
        else:
            ctx = routing.hook(torch.from_numpy(v), alpha=alpha, tau=tau)

        # Build prompt and generate
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": s["input_text"]})
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)
        else:
            prompt = s["input_text"]
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=1024).to(next(model.parameters()).device)

        if ctx is None:
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        else:
            with ctx:
                out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        refs.append(s["output_text"].strip())
        if (i + 1) % 50 == 0:
            print(f"    α={alpha} τ={tau}  {i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
    return preds, refs, time.time() - t0


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-7")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--variant", choices=["template", "fact"], default="template")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
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

    print(f"=== routing on {args.model}/{args.task}/{args.variant} ===")
    print(f"  layer={args.layer_idx} τ={args.taus} α={args.alphas} n={args.n_users}")

    arrs = np.load(args.vectors_npz)
    vectors = arrs[args.variant][: args.n_users]

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)

    model, tokenizer = load_model_and_tokenizer(args.model)

    grid_results = []
    for tau in args.taus:
        for alpha in args.alphas:
            print(f"\n  --- τ={tau}, α={alpha} ---")
            preds, refs, wall = eval_pair(
                model, tokenizer, samples, vectors,
                layer_idx=args.layer_idx, alpha=alpha, tau=tau,
                max_new_tokens=info["max_new_tokens"],
                chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            )
            m = compute_metric(metric, preds, refs)
            print(f"  τ={tau} α={alpha}: {m}")
            grid_results.append({
                "tau": tau, "alpha": alpha, "metric": metric, "value": m,
                "wall_seconds": wall,
            })

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"routing_{args.task}_{short}_{args.variant}_n{args.n_users}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "layer_idx": args.layer_idx,
            "variant": args.variant, "taus": args.taus, "alphas": args.alphas,
            "n_users": args.n_users, "metric": metric,
            "grid": grid_results,
            "vectors_npz": args.vectors_npz,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
