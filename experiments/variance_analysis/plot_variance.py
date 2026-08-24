"""Scatter the three signal-quality proxies against per-user ΔROUGE-L."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("P1_proj_global_mean", "P1: proj on global $\\bar v$",
     "(alignment with population-shared direction)"),
    ("P2_relative_magnitude", "P2: relative magnitude $\\|v_u\\|/\\overline{\\|v\\|}$",
     "(effective perturbation $\\alpha\\|v_u\\|$)"),
    ("P3_top_k_coherence", "P3: top-5 NN cosine",
     "(topic coherence vs neighbours)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)
    if args.out is None:
        task_slug = d["task"].lower().replace("-", "")
        repo_root = Path(args.input).resolve().parents[2]
        args.out = str(repo_root / "figures" / f"fig_variance_analysis_{task_slug}.pdf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows = d["per_user"]
    delta = np.array([r["delta"] for r in rows])
    correls = d["summary"]["correlations"]
    colors = ["seagreen" if v > 0.01 else "tomato" if v < -0.01 else "gray"
              for v in delta]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, xlabel, sub) in zip(axes, PANELS):
        x = np.array([r[key] for r in rows])
        ax.scatter(x, delta, c=colors, s=44, alpha=0.85,
                   edgecolors="black", linewidths=0.4)
        ax.axhline(0, linestyle=":", color="black", alpha=0.4)
        if np.std(x) > 1e-8:
            slope, intercept = np.polyfit(x, delta, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + intercept, "--", color="steelblue", alpha=0.6)
        ax.set_xlabel(f"{xlabel}\n{sub}", fontsize=9)
        ax.set_ylabel(r"$\Delta$ROUGE-L (steered $-$ zero-shot)")
        ax.set_title(f"Pearson $r{{=}}{correls[key]:+.2f}$", fontsize=11)
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
