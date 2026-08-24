"""Three-panel summary of the per-user α search.

    python plot_alpha_distribution.py --input results/case_study/alpha_search_*.json
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
    ap.add_argument("--input", required=True, help="alpha_search_*.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)
    if args.out is None:
        repo_root = Path(args.input).resolve().parents[2]
        args.out = str(repo_root / "figures" / f"fig_{Path(args.input).stem}.pdf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    alphas = d["alphas"]
    rows = d["per_user"]
    n = len(rows)

    stayed = sum(1 for r in rows if r["zs_correct"] and all(p["match"] for p in r["per_alpha"]))
    broken = sum(1 for r in rows if r["zs_correct"] and not all(p["match"] for p in r["per_alpha"]))
    flipped = sum(r["flip"] for r in rows)
    never = sum(1 for r in rows if not r["any_correct"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    cats = [
        ("ZS correct\n(stays correct)", stayed, "seagreen"),
        ("ZS correct\n(broken by α>0)", broken, "tomato"),
        ("ZS wrong\n(flipped by α>0)", flipped, "steelblue"),
        ("Never correct", never, "lightgray"),
    ]
    axes[0].bar(range(len(cats)), [c[1] for c in cats],
                color=[c[2] for c in cats], alpha=0.85)
    for i, (_, v, _) in enumerate(cats):
        axes[0].text(i, v + 0.3, f"{v}/{n}", ha="center", fontsize=10)
    axes[0].set_xticks(range(len(cats)))
    axes[0].set_xticklabels([c[0] for c in cats], fontsize=9)
    axes[0].set_ylabel("# users")
    axes[0].set_ylim(0, n + 2)
    axes[0].set_title(f"A) Per-user effect of steering\n"
                      f"net Δ_users = +{flipped} − {broken} = {flipped - broken:+d}")
    axes[0].grid(axis="y", alpha=0.25)

    for r in rows:
        ys = [int(p["match"]) for p in r["per_alpha"]]
        if r["zs_correct"]:
            col = "lightgray" if all(ys) else "tomato"
        else:
            col = "steelblue" if r["flip"] else "k"
        axes[1].plot(alphas, ys, alpha=0.55, color=col, marker="o", linewidth=1)
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel("prediction == gold (1 / 0)")
    axes[1].set_yticks([0, 1])
    axes[1].set_title("B) Per-user correctness vs α\n(blue: flipped; red: broken)")
    axes[1].grid(alpha=0.25)

    flip_alphas = [r["first_correct_alpha"] for r in rows if r["flip"]]
    if flip_alphas:
        bins = np.arange(0, max(alphas) + 0.25 + 1e-9, 0.25) - 0.125
        axes[2].hist(flip_alphas, bins=bins, color="steelblue",
                     edgecolor="black", alpha=0.85)
        axes[2].axvline(1.0, color="red", linestyle="--",
                        label=r"global $\alpha{=}1$")
        axes[2].set_xticks(alphas)
        axes[2].legend(fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "no users flipped", ha="center", va="center")
    axes[2].set_xlabel(r"first $\alpha$ at which prediction = gold")
    axes[2].set_ylabel("# users")
    axes[2].set_title(f"C) Phase-boundary distribution\n"
                      f"({len(flip_alphas)} flipped users, "
                      f"global $\\alpha{{=}}1$ misses all of them)")
    axes[2].grid(axis="y", alpha=0.25)

    plt.suptitle(f"{Path(args.input).stem}: per-user α-search "
                 f"(n={n} unique users, {d['variant']} artifacts)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
