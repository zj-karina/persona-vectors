"""Render the single-user case study from run_case_study.py output.

Panel A stacks the top-token probabilities per α, B tracks the gold answer's
probability, C shows what the model actually predicted at each α.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to case_*.json")
    ap.add_argument("--out", default=None, help="PDF output path")
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)

    if args.out is None:
        repo_root = Path(args.input).resolve().parents[2]
        args.out = str(repo_root / "figures" / f"fig_{Path(args.input).stem}.pdf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    alphas = [r["alpha"] for r in d["rows"]]
    gold = d["gold"].lower()

    top_tokens: list[str] = []
    for r in d["rows"]:
        for t, _ in r["topk"][:5]:
            if t not in top_tokens:
                top_tokens.append(t)
    top_tokens = top_tokens[:8]

    probs = np.zeros((len(top_tokens), len(alphas)))
    for j, r in enumerate(d["rows"]):
        for t, p in r["topk"]:
            if t in top_tokens:
                probs[top_tokens.index(t), j] = p

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    bottoms = np.zeros(len(alphas))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(top_tokens):
        axes[0].bar(range(len(alphas)), probs[i], bottom=bottoms,
                    label=repr(t), color=cmap(i % 10), alpha=0.85)
        bottoms = bottoms + probs[i]
    axes[0].set_xticks(range(len(alphas)))
    axes[0].set_xticklabels([f"α={a}" for a in alphas])
    axes[0].set_ylabel("P(token | prompt)")
    axes[0].set_title("A) Top-token probabilities at answer position")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].set_ylim(0, 1.0)

    gold_probs = [next((p for t, p in r["topk"] if t.lower() == gold), 0.0)
                  for r in d["rows"]]
    axes[1].plot(alphas, gold_probs, marker="o", color="seagreen", lw=2)
    axes[1].axhline(gold_probs[0], color="gray", linestyle=":", alpha=0.6,
                    label="zero-shot")
    axes[1].set_xlabel("α (steering scale)")
    axes[1].set_ylabel(f"P(gold = {gold!r})")
    axes[1].set_title("B) Gold-answer probability vs α")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    colors = ["seagreen" if r["match_gold"] else "lightcoral" for r in d["rows"]]
    axes[2].bar(range(len(alphas)), [1] * len(alphas), color=colors, alpha=0.7)
    for j, r in enumerate(d["rows"]):
        axes[2].text(j, 0.5, r["prediction"][:20],
                     ha="center", va="center", fontsize=10,
                     rotation=90 if len(r["prediction"]) > 8 else 0)
    axes[2].set_xticks(range(len(alphas)))
    axes[2].set_xticklabels([f"α={a}" for a in alphas])
    axes[2].set_yticks([])
    axes[2].set_title(f"C) Greedy prediction at α (gold = {gold!r})")

    plt.suptitle(f"Case study: {Path(args.input).stem}\n"
                 f"layer {d['layer_idx']}, |v|={d['vector_norm']:.2f}, "
                 f"variant={d['variant']}", fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
