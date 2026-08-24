"""Render Experiment B figure: 3-panel scatter of static signal-quality
proxies (P1, P2, P3) vs per-user steering ΔROUGE-L."""

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
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)
    task_slug = d["task"].lower().replace("-", "")
    if args.out is None:
        repo_root = Path(args.input).resolve().parents[2]
        args.out = str(repo_root / "figures" / f"fig_variance_analysis_{task_slug}.pdf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows = d["per_user"]
    p1 = np.array([r["P1_proj_global_mean"] for r in rows])
    p2 = np.array([r["P2_relative_magnitude"] for r in rows])
    p3 = np.array([r["P3_top_k_coherence"] for r in rows])
    delta = np.array([r["delta"] for r in rows])
    correls = d["summary"]["correlations"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    panels = [
        (axes[0], p1, correls["P1_proj_global_mean"],
         "P1: proj on global $\\bar v$",
         "(alignment with population-shared direction)"),
        (axes[1], p2, correls["P2_relative_magnitude"],
         "P2: relative magnitude $\\|v_u\\|/\\overline{\\|v\\|}$",
         "(effective perturbation $\\alpha\\|v_u\\|$)"),
        (axes[2], p3, correls["P3_top_k_coherence"],
         "P3: top-5 NN cosine",
         "(topic coherence vs neighbours)"),
    ]
    for ax, x, r, xlabel, sub in panels:
        col = ["seagreen" if d > 0.01 else "tomato" if d < -0.01 else "gray"
               for d in delta]
        ax.scatter(x, delta, c=col, s=44, alpha=0.85, edgecolors="black", linewidths=0.4)
        ax.axhline(0, linestyle=":", color="black", alpha=0.4)
        # least-squares fit line
        if np.std(x) > 1e-8:
            slope, intercept = np.polyfit(x, delta, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, "--", color="steelblue", alpha=0.6)
        ax.set_xlabel(xlabel + "\n" + sub, fontsize=9)
        ax.set_ylabel(r"$\Delta$ROUGE-L (steered $-$ zero-shot)")
        ax.set_title(f"Pearson $r{{=}}{r:+.2f}$", fontsize=11)
        ax.grid(alpha=0.25)

    plt.suptitle(
        f"Per-user signal-quality proxies vs steering effect "
        f"({d['task']}, n={d['n_users']}, $\\alpha{{=}}{d['alpha']}$, "
        f"layer {d['layer_idx']}, {d['variant']})",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
