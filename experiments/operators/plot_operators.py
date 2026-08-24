"""All operator variants against the additive baseline, one panel per task/variant.

Reads whatever run_operators.py, run_rank1_edit.py and run_routing.py have
written for each panel and skips the runs that are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import FIGURES, primary_label, primary_value

OPERATORS_DIR = ROOT / "results/operators"
PROJ_COLORS = {1: "#ffa07a", 5: "#e74c3c", 10: "#7b241c"}

# Zero-shot references and the α=1 additive baseline come from the variance runs
# on the same unique-user samples; those runs write per-user rows, not a curve.
PANELS = [
    {"task": "LaMP-2", "variant": "template", "n_users": 323, "zs": 0.582,
     "additive": {0.0: 0.582, 1.0: 0.557}, "routing_tau": 0.0},
    {"task": "LaMP-2", "variant": "fact", "n_users": 323, "zs": 0.582,
     "additive": None, "routing_tau": 0.25},
    {"task": "LaMP-7", "variant": "template", "n_users": 1497, "zs": 0.4262,
     "additive": None, "routing_tau": None},
    {"task": "LaMP-7", "variant": "fact", "n_users": 1497, "zs": 0.4262,
     "additive": None, "routing_tau": None},
]


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def curve(results: list[dict], metric: str) -> dict[float, float]:
    return {r["alpha"]: primary_value(metric, r["value"]) for r in results}


def collect_proj(task: str, variant: str, n_users: int) -> dict[int, dict[float, float]]:
    out = {}
    for k in PROJ_COLORS:
        d = load_json(OPERATORS_DIR /
                      f"{task}_Qwen3-8B_{variant}_proj_nuisance_k{k}_n{n_users}.json")
        if d:
            out[k] = curve(d["results"], d["results"][0]["metric"])
    return out


def collect_rank1(task: str, variant: str, n_users: int) -> dict[float, float]:
    d = load_json(OPERATORS_DIR / f"rank1_edit_{task}_Qwen3-8B_{variant}_n{n_users}.json")
    return curve(d["results"], d["results"][0]["metric"]) if d else {}


def collect_routing(task: str, variant: str, n_users: int,
                    tau: float | None) -> dict[float, float]:
    if tau is None:
        return {}
    d = load_json(OPERATORS_DIR / f"routing_{task}_Qwen3-8B_{variant}_n{n_users}.json")
    if not d:
        return {}
    metric = d["grid"][0]["metric"]
    return {g["alpha"]: primary_value(metric, g["value"])
            for g in d["grid"] if g["tau"] == tau}


def line(ax, points: dict[float, float], **kwargs):
    if points:
        xs = sorted(points)
        ax.plot(xs, [points[x] for x in xs], **kwargs)


def plot_panel(ax, panel: dict, metric: str):
    task, variant, n_users = panel["task"], panel["variant"], panel["n_users"]

    line(ax, panel["additive"] or {}, marker="x", linewidth=2, color="black",
         label="Additive (baseline)")
    for k, points in sorted(collect_proj(task, variant, n_users).items()):
        line(ax, points, marker="o", linewidth=1.5, color=PROJ_COLORS[k],
             label=f"Op1 proj_nuisance k={k}")
    line(ax, collect_rank1(task, variant, n_users), marker="s", linewidth=1.6,
         color="#2980b9", label="Op2 rank-1 edit")
    line(ax, collect_routing(task, variant, n_users, panel["routing_tau"]),
         marker="d", linewidth=1.6, color="#27ae60",
         label=rf"Op3 routing $\tau{{=}}{panel['routing_tau']}$")

    ax.axhline(panel["zs"], linestyle=":", color="gray", alpha=0.7,
               label=f"Zero-shot ({panel['zs']:.3f})")
    ax.set_xlabel(r"$\alpha$  (steering scale)")
    ax.set_ylabel(primary_label(metric))
    ax.set_title(f"{task} {variant} (n={n_users})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, panel in zip(axes.ravel(), PANELS):
        metric = "accuracy" if panel["task"] == "LaMP-2" else "rouge"
        plot_panel(ax, panel, metric)

    plt.suptitle("Alternative steering operators vs. additive baseline "
                 "(Qwen3-8B, layer 13, full unique-user samples)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = FIGURES / "fig_operators_comparison.pdf"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
