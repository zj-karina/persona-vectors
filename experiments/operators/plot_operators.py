"""Combined comparison plot of all operator variants on LaMP-2 and LaMP-7.

Reads:
  results/operators/{LaMP-X}_Qwen3-8B_template_proj_nuisance_k{1,5,10}_n*.json
  results/operators/routing_LaMP-X_Qwen3-8B_template_n*.json
  results/operators/rank1_edit_LaMP-X_Qwen3-8B_template_n*.json
plus the additive-baseline numbers from prior variance runs.

Output: figures/fig_operators_comparison.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def primary(metric: str, value: dict) -> float:
    if metric == "accuracy":
        return value["accuracy"]
    if metric == "rouge":
        return value["ROUGE-L"]
    if metric == "regression":
        return value["mae"]
    return float("nan")


def collect_proj(task: str, n_users: int) -> dict[int, dict[float, float]]:
    """Return {k: {alpha: metric}}. Each row keyed by pca_k."""
    out = {}
    for k in [1, 5, 10]:
        p = ROOT / "results/operators" / \
            f"{task}_Qwen3-8B_template_proj_nuisance_k{k}_n{n_users}.json"
        d = load_json(p)
        if not d:
            continue
        metric = d["results"][0]["metric"]
        out[k] = {r["alpha"]: primary(metric, r["value"]) for r in d["results"]}
    return out


def collect_routing(task: str, n_users: int) -> dict[float, dict[float, float]]:
    p = ROOT / "results/operators" / f"routing_{task}_Qwen3-8B_template_n{n_users}.json"
    d = load_json(p)
    if not d:
        return {}
    metric = d["grid"][0]["metric"]
    out: dict[float, dict[float, float]] = {}
    for g in d["grid"]:
        out.setdefault(g["tau"], {})[g["alpha"]] = primary(metric, g["value"])
    return out


def collect_rank1(task: str, n_users: int) -> dict[float, float]:
    p = ROOT / "results/operators" / \
        f"rank1_edit_{task}_Qwen3-8B_template_n{n_users}.json"
    d = load_json(p)
    if not d:
        return {}
    metric = d["results"][0]["metric"]
    return {r["alpha"]: primary(metric, r["value"]) for r in d["results"]}


def plot_single_task(ax, task: str, n_users: int, metric_name: str,
                     additive_baseline: dict[float, float], zs: float):
    proj = collect_proj(task, n_users)
    routing = collect_routing(task, n_users)
    rank1 = collect_rank1(task, n_users)

    # Additive baseline
    if additive_baseline:
        xs = sorted(additive_baseline)
        ys = [additive_baseline[a] for a in xs]
        ax.plot(xs, ys, marker="x", linewidth=2, color="black",
                label="Additive (baseline)")

    # Op1: best k each (k=5 typically)
    colors = {1: "#ffa07a", 5: "#e74c3c", 10: "#7b241c"}
    for k, accs in sorted(proj.items()):
        xs = sorted(accs)
        ys = [accs[a] for a in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.6,
                color=colors.get(k, "gray"), label=f"Op1 proj_nuisance k={k}")

    # Op2: rank-1 edit
    if rank1:
        xs = sorted(rank1)
        ys = [rank1[a] for a in xs]
        ax.plot(xs, ys, marker="s", linewidth=1.6, color="#2980b9",
                label="Op2 rank-1 edit")

    # Op3: routing (best τ — pick the one with largest change from ZS)
    if routing:
        # Pick τ=0 (least restrictive) for visualisation
        if 0.0 in routing:
            xs = sorted(routing[0.0])
            ys = [routing[0.0][a] for a in xs]
            ax.plot(xs, ys, marker="d", linewidth=1.6, color="#27ae60",
                    label=r"Op3 routing $\tau{=}0$")

    ax.axhline(zs, linestyle=":", color="gray", alpha=0.7, label=f"Zero-shot ({zs:.3f})")
    ax.set_xlabel(r"$\alpha$  (steering scale)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{task} (n={n_users})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: LaMP-2 template (n=323)
    plot_single_task(
        axes[0, 0], "LaMP-2", n_users=323, metric_name="Accuracy",
        additive_baseline={0.0: 0.582, 1.0: 0.557},
        zs=0.582,
    )
    axes[0, 0].set_title("LaMP-2 template (n=323)")
    # Panel B: LaMP-2 fact (n=323)
    plot_lamp2_fact(axes[0, 1])
    # Panel C: LaMP-7 template (n=1497)
    plot_lamp7_template_n1497(axes[1, 0])
    # Panel D: LaMP-7 fact (n=1497)
    plot_lamp7_fact(axes[1, 1])

    plt.suptitle(
        "Alternative steering operators vs.\\ additive baseline "
        "(Qwen3-8B, layer 13, full unique-user samples)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    out = ROOT / "figures/fig_operators_comparison.pdf"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out}")


def plot_lamp2_fact(ax):
    """LaMP-2 fact n=323: Op1 (k=1,5,10), Op2, Op3 routing τ=0.25."""
    proj = collect_proj("LaMP-2", n_users=323)
    p2 = ROOT / "results/operators/rank1_edit_LaMP-2_Qwen3-8B_fact_n323.json"
    p3 = ROOT / "results/operators/routing_LaMP-2_Qwen3-8B_fact_n323.json"
    # Override path for fact variant
    proj_fact = {}
    for k in [1, 5, 10]:
        p = ROOT / f"results/operators/LaMP-2_Qwen3-8B_fact_proj_nuisance_k{k}_n323.json"
        d = load_json(p)
        if d:
            proj_fact[k] = {r["alpha"]: r["value"]["accuracy"] for r in d["results"]}
    colors = {1: "#ffa07a", 5: "#e74c3c", 10: "#7b241c"}
    for k, accs in sorted(proj_fact.items()):
        xs = sorted(accs); ys = [accs[a] for a in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.4,
                color=colors.get(k, "gray"), label=f"Op1 proj k={k}")
    d2 = load_json(p2)
    if d2:
        xs = [r["alpha"] for r in d2["results"]]
        ys = [r["value"]["accuracy"] for r in d2["results"]]
        ax.plot(xs, ys, marker="s", linewidth=1.6, color="#2980b9",
                label="Op2 rank-1")
    d3 = load_json(p3)
    if d3:
        # τ=0.25 row
        row = {c["alpha"]: c["value"]["accuracy"] for c in d3["grid"] if c["tau"] == 0.25}
        if row:
            xs = sorted(row); ys = [row[a] for a in xs]
            ax.plot(xs, ys, marker="d", linewidth=1.8, color="#27ae60",
                    label=r"Op3 routing $\tau{=}0.25$")
    ax.axhline(0.582, linestyle=":", color="gray", alpha=0.7, label="Zero-shot (0.582)")
    ax.set_xlabel(r"$\alpha$  (steering scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("LaMP-2 fact (n=323)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_lamp7_template_n1497(ax):
    """LaMP-7 template n=1497: Op1 k=5, Op2."""
    p1 = ROOT / "results/operators/LaMP-7_Qwen3-8B_template_proj_nuisance_k5_n1497.json"
    p2 = ROOT / "results/operators/rank1_edit_LaMP-7_Qwen3-8B_template_n1497.json"
    for path, marker, color, lbl in [
        (p1, "o", "#e74c3c", "Op1 proj k=5"),
        (p2, "s", "#2980b9", "Op2 rank-1"),
    ]:
        d = load_json(path)
        if not d: continue
        xs = [r["alpha"] for r in d["results"]]
        ys = [r["value"]["ROUGE-L"] for r in d["results"]]
        ax.plot(xs, ys, marker=marker, linewidth=1.6, color=color, label=lbl)
    ax.axhline(0.4262, linestyle=":", color="gray", alpha=0.7,
               label="Zero-shot (0.4262)")
    ax.set_xlabel(r"$\alpha$  (steering scale)")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("LaMP-7 template (n=1497)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_lamp7_fact(ax):
    """LaMP-7 fact n=1497: Op1 (k=1,5,10) + Op2."""
    colors = {1: "#ffa07a", 5: "#e74c3c", 10: "#7b241c"}
    for k in [1, 5, 10]:
        p = ROOT / f"results/operators/LaMP-7_Qwen3-8B_fact_proj_nuisance_k{k}_n1497.json"
        d = load_json(p)
        if not d: continue
        xs = [r["alpha"] for r in d["results"]]
        ys = [r["value"]["ROUGE-L"] for r in d["results"]]
        ax.plot(xs, ys, marker="o", linewidth=1.4,
                color=colors.get(k, "gray"), label=f"Op1 proj k={k}")
    p2 = ROOT / "results/operators/rank1_edit_LaMP-7_Qwen3-8B_fact_n1497.json"
    d2 = load_json(p2)
    if d2:
        xs = [r["alpha"] for r in d2["results"]]
        ys = [r["value"]["ROUGE-L"] for r in d2["results"]]
        ax.plot(xs, ys, marker="s", linewidth=1.6, color="#2980b9",
                label="Op2 rank-1")
    zs = 0.4262
    ax.axhline(zs, linestyle=":", color="gray", alpha=0.7, label=f"Zero-shot ({zs:.4f})")
    ax.set_xlabel(r"$\alpha$  (steering scale)")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("LaMP-7 fact (n=1497)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


if __name__ == "__main__":
    main()
