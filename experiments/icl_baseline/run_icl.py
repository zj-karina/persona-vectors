"""ICL (in-context-learning) baseline for personalization.

For each user we prepend K profile items as context to the test query and
generate.  This is the simplest "give the model the same user history that
the persona vector was extracted from, but in-context rather than as a
hidden-state perturbation" baseline.

NOTE on demonstrations: the LaMP `_titles_p6` profile we use only contains
*article-side* fields (titles or tweet text), not the user's chosen
category / paraphrase.  We therefore *cannot* form true input→output
demonstrations and instead frame the prompt as
"here are items the user has previously interacted with [...], now:".
This is the standard context-priming baseline for personalization
benchmarks where labels are not in the profile.

Usage:
    python experiments/icl_baseline/run_icl.py \\
        --model Qwen/Qwen3-8B --task LaMP-2 \\
        --K_grid 3 5 6 --n_users 30
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


_TASK_FRAMING: dict[str, str] = {
    "LaMP-1": "Articles this researcher has cited in past work:",
    "LaMP-2": "Articles this user has previously categorised:",
    "LaMP-3": "Reviews this user has previously written:",
    "LaMP-4": "Headlines this writer has previously produced:",
    "LaMP-5": "Paper titles this scholar has previously authored:",
    "LaMP-7": "Tweets this user has previously written:",
}


def build_icl_prompt(profile_items: list[str], test_input: str, K: int,
                     task: str) -> str:
    framing = _TASK_FRAMING.get(task, "User's history:")
    items = profile_items[:K]
    body = "\n".join(f"  {i+1}. {it}" for i, it in enumerate(items))
    return (
        f"{framing}\n\n{body}\n\n"
        f"Given this user's history, complete the following request as they "
        f"would.\n\n{test_input}"
    )


@torch.no_grad()
def run_one_K(model, tokenizer, samples, K: int, *,
              chat_kwargs: dict, system_prompt: str,
              max_new_tokens: int, task: str) -> dict:
    preds, refs = [], []
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = build_icl_prompt(s["behavior_profile_text"], s["input_text"],
                                  K=K, task=task)
        pred = persona_steered_generate(
            model, tokenizer, user_input=prompt,
            persona_vector=None, layer_idx=None, alpha=0.0,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred); refs.append(s["output_text"].strip())
        if (i + 1) % 10 == 0:
            print(f"  K={K}  {i+1}/{len(samples)}  ({time.time()-t0:.0f}s)")
    return {"preds": preds, "refs": refs, "wall_seconds": time.time() - t0}


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--K_grid", type=int, nargs="+", default=[3, 5, 6])
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/icl_baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    print(f"=== ICL baseline: {args.model}/{args.task} ===")
    print(f"  K grid={args.K_grid}  n_users={args.n_users}  metric={metric}")

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    print(f"  loaded {len(samples)} unique-user samples")

    model, tokenizer = load_model_and_tokenizer(args.model)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for K in args.K_grid:
        print(f"\n--- K={K} ---")
        run = run_one_K(
            model, tokenizer, samples, K=K,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            max_new_tokens=info["max_new_tokens"], task=args.task,
        )
        m = compute_metric(metric, run["preds"], run["refs"])
        print(f"  K={K}: {m}")
        all_results[f"K={K}"] = {
            "value": m, "wall_seconds": run["wall_seconds"],
            "sample_preds": run["preds"][:5],
            "sample_refs":  run["refs"][:5],
        }
        out_path = out_dir / f"{args.task}_icl_k{K}.json"
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model, "task": args.task, "K": K,
                "n_users": len(samples), "metric": metric,
                "value": m, "preds": run["preds"], "refs": run["refs"],
                "wall_seconds": run["wall_seconds"],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)
        print(f"  saved {out_path}")

    print("\n=== Summary ===")
    for k, v in all_results.items():
        print(f"  {k}: {v['value']}")


if __name__ == "__main__":
    main()
