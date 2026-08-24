"""In-context baseline: give the model the profile as text instead of a vector.

The `_titles_p6` profiles only contain the article side of each interaction —
titles or tweet text, never the category or paraphrase the user produced — so
real input→output demonstrations are not available. The prompt is therefore
context priming ("here is what this user has read, now answer as they would"),
which is the standard ICL baseline on personalization benchmarks whose profiles
carry no labels.

    python experiments/icl_baseline/run_icl.py --task LaMP-2 --K_grid 3 5 6
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
    LaMPDataset, chat_kwargs_for, compute_metric,
    load_model_and_tokenizer, persona_steered_generate, system_prompt_for,
    task_info,
)


# Deliberately worded for context priming, not for the fact-extraction prompt
# in src.fact_extractor — the committed results depend on this exact text.
TASK_FRAMING: dict[str, str] = {
    "LaMP-1": "Articles this researcher has cited in past work:",
    "LaMP-2": "Articles this user has previously categorised:",
    "LaMP-3": "Reviews this user has previously written:",
    "LaMP-4": "Headlines this writer has previously produced:",
    "LaMP-5": "Paper titles this scholar has previously authored:",
    "LaMP-7": "Tweets this user has previously written:",
}


def build_icl_prompt(profile_items: list[str], test_input: str, K: int,
                     task: str) -> str:
    framing = TASK_FRAMING.get(task, "User's history:")
    body = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(profile_items[:K]))
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
        preds.append(pred)
        refs.append(s["output_text"].strip())
        if (i + 1) % 10 == 0:
            print(f"  K={K}  {i+1}/{len(samples)}  ({time.time()-t0:.0f}s)")
    return {"preds": preds, "refs": refs, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--K_grid", type=int, nargs="+", default=[3, 5, 6])
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/icl_baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    summary = {}
    for K in args.K_grid:
        print(f"\n--- K={K} ---")
        run = run_one_K(
            model, tokenizer, samples, K=K,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            max_new_tokens=info["max_new_tokens"], task=args.task,
        )
        m = compute_metric(metric, run["preds"], run["refs"])
        print(f"  K={K}: {m}")
        summary[f"K={K}"] = m

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
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
