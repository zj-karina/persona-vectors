"""Build the paper's figures and LaTeX tables from the JSON results.

Every step is skipped rather than failed when its results are missing, so this
can be run at any point during a session and will render whatever exists.

    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import primary_label, primary_value

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "paper" / "tables"

TASKS = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
BACKBONES = ["Qwen3-8B", "Qwen3-14B", "Mistral-Small-24B"]

# Flan-T5-XXL with a trained Q-Former, quoted from the BehavioralTwin runs the
# training-free setup is being compared against.
BEHAVIORAL_TWIN = {
    "LaMP-1": 0.567, "LaMP-2": 0.703, "LaMP-3": 0.251,
    "LaMP-4": 0.179, "LaMP-5": 0.437, "LaMP-7": 0.403,
}


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def short(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-Instruct-2501", "")


def plot_layer_search():
    files = sorted((RESULTS / "layer_search").glob("layer_search_*.json"))
    if not files:
        print("[fig1] no layer_search results — skipping")
        return

    by_task: dict[str, list[dict]] = {}
    for p in files:
        d = load_json(p)
        by_task.setdefault(d["task"], []).append(d)

    fig, axes = plt.subplots(1, len(by_task), figsize=(6 * len(by_task), 4), squeeze=False)
    for ax, (task, runs) in zip(axes[0], by_task.items()):
        for d in runs:
            metric = d["results"][0]["metric"] if d["results"] else "accuracy"
            ax.plot([r["layer_fraction"] for r in d["results"]],
                    [primary_value(metric, r["value"]) for r in d["results"]],
                    marker="o", linewidth=2, label=short(d["model"]))
            ax.axvline(d["best_layer"]["layer_fraction"], linestyle="--", alpha=0.4)
            ax.axhline(primary_value(metric, d["zero_shot"]["value"]),
                       linestyle=":", color="gray", alpha=0.5,
                       label=f"{short(d['model'])} ZS")
        ax.set_xlabel("Layer (fraction of total depth)")
        ax.set_ylabel(primary_label(runs[0]["results"][0]["metric"]))
        ax.set_title(task)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Persona-vector quality vs extraction layer", y=1.02, fontsize=13)
    plt.tight_layout()
    out = FIGURES / "fig1_layer_search.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig1] {out}")


def plot_magnitude():
    files = sorted((RESULTS / "geometry").glob("geometry_*.json"))
    if not files:
        print("[fig2] no geometry results — skipping")
        return

    means, labels = [], []
    for p in files:
        d = load_json(p)
        means.append(d["magnitude"]["mean_magnitude_ratio"])
        labels.append(f"{short(d['model'])}\n{d['task']} L{d['layer_idx']}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ypos = np.arange(len(labels))
    ax.barh(ypos, [m * 100 for m in means], color="steelblue", alpha=0.85)
    for i, m in enumerate(means):
        ax.text(m * 100 + 0.3, i, f"{m:.1%}", va="center", fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$\|v_{\mathrm{user}}\| \, / \, \|h_{\mathrm{residual}}\|$ (%)")
    ax.set_title("Persona vector magnitude relative to residual stream\n"
                 "(steering strength is proportional to this ratio)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "fig2_magnitude.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig2] {out}")


def load_smoke_table() -> dict[tuple[str, str], dict]:
    """{(backbone, task): {experiment: run}} from the n=200 smoke JSONs."""
    rows: dict[tuple[str, str], dict] = {}
    for p in (RESULTS / "main_table").glob("*.json"):
        d = load_json(p)
        if d:
            rows.setdefault((short(d["llm"]), d["task"]), {})[d.get("experiment", "?")] = d
    return rows


def plot_main_results():
    smoke = load_smoke_table()
    if not smoke:
        print("[fig4] no main_table results — skipping")
        return

    llms = sorted({llm for llm, _ in smoke})
    fig, axes = plt.subplots(1, len(TASKS), figsize=(3 * len(TASKS), 4), sharey=False)
    for ax, task in zip(axes, TASKS):
        labels, vals_zs, vals_ps = [], [], []
        for llm in llms:
            runs = smoke.get((llm, task))
            if not runs:
                continue
            labels.append(llm)
            for exp, target in (("zero_shot_control", vals_zs),
                                ("persona_steering", vals_ps)):
                run = runs.get(exp)
                target.append(primary_value(run["result"]["metric"], run["result"]["value"])
                              if run else float("nan"))
        x = np.arange(len(labels))
        ax.bar(x - 0.175, vals_zs, 0.35, label="Zero-shot", color="lightgray")
        ax.bar(x + 0.175, vals_ps, 0.35, label="Persona α=1", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("Mistral-Small-24B", "Mistral-24B") for l in labels],
                           rotation=30, fontsize=8)
        ax.set_title(task, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if task == TASKS[0]:
            ax.set_ylabel("Primary metric")
            ax.legend(fontsize=8)
    plt.suptitle("Zero-shot vs persona-steered, smoke n=200", y=1.02)
    plt.tight_layout()
    out = FIGURES / "fig4_main_results.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig4] {out}")


def _sweep_figure(subdir: str, glob: str, x_key: str, xlabel: str, title: str,
                  out_name: str, tag: str, marker: str, log_x: bool = False):
    files = sorted((RESULTS / subdir).glob(glob))
    if not files:
        print(f"[{tag}] no {subdir} results — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    metric = "accuracy"
    for p in files:
        d = load_json(p)
        metric = d["results"][0]["metric"]
        ax.plot([r[x_key] for r in d["results"]],
                [primary_value(metric, r["value"]) for r in d["results"]],
                marker=marker,
                label=f"{short(d['model'])} / {d['task']} L{d['layer_idx']}")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(primary_label(metric))
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both" if log_x else "major")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES / out_name
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[{tag}] {out}")


def make_table_main():
    smoke = load_smoke_table()
    body = [
        r"\multicolumn{7}{l}{\textit{Trained baseline (Flan-T5-XXL, Q-Former)}} \\",
        "BehavioralTwin & " + " & ".join(f"{BEHAVIORAL_TWIN[t]:.3f}" for t in TASKS) + r" \\",
        r"\midrule",
    ]

    for backbone in BACKBONES:
        body.append(rf"\multicolumn{{7}}{{l}}{{\textit{{Training-free: {backbone} (frozen)}}}} \\")
        for label, exp in (("Zero-shot", "zero_shot_control"),
                           (r"Persona ($\alpha{=}1$)", "persona_steering")):
            cells = []
            for task in TASKS:
                run = smoke.get((backbone, task), {}).get(exp)
                cells.append(f"{primary_value(run['result']['metric'], run['result']['value']):.3f}"
                             if run else "--")
            body.append(rf"{label} & {' & '.join(cells)} \\")

        deltas = []
        for task in TASKS:
            runs = smoke.get((backbone, task), {})
            zs, ps = runs.get("zero_shot_control"), runs.get("persona_steering")
            if zs and ps and zs["result"]["metric"] == ps["result"]["metric"]:
                metric = zs["result"]["metric"]
                delta = (primary_value(metric, ps["result"]["value"])
                         - primary_value(metric, zs["result"]["value"]))
                # MAE is the primary metric on LaMP-3, where lower is better.
                deltas.append(f"{(-delta if metric == 'regression' else delta):+.3f}")
            else:
                deltas.append("--")
        body.append(rf"$\Delta$ & {' & '.join(deltas)} \\")
        body.append(r"\midrule")

    latex = (
        "\\begin{table*}[t]\n"
        "\\centering\\small\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "\\textbf{Method} & "
        "\\textbf{LaMP-1} & \\textbf{LaMP-2} & \\textbf{LaMP-3} & "
        "\\textbf{LaMP-4} & \\textbf{LaMP-5} & \\textbf{LaMP-7} \\\\\n"
        " & Acc$\\uparrow$ & Acc$\\uparrow$ & MAE$\\downarrow$ & "
        "R-L$\\uparrow$ & R-L$\\uparrow$ & R-L$\\uparrow$ \\\\\n"
        "\\midrule\n"
        + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Main results. $\\Delta$ rows show effect of persona steering on the "
        "primary metric (sign-corrected: positive = improvement). Smoke evaluation "
        "with $n{=}200$.}\n"
        "\\label{tab:main}\n"
        "\\end{table*}\n"
    )
    out = TABLES / "table1_main.tex"
    out.write_text(latex)
    print(f"[tab1] {out}")


def make_table_layer_search():
    files = sorted((RESULTS / "layer_search").glob("layer_search_*.json"))
    if not files:
        print("[tab2] no layer_search results — skipping")
        return
    rows = []
    for p in files:
        d = load_json(p)
        best = d["best_layer"]
        metric = best["metric"]
        rows.append((short(d["model"]), d["task"], d["n_layers_total"],
                     best["layer_idx"], best["layer_fraction"],
                     primary_value(metric, best["value"]),
                     primary_value(metric, d["zero_shot"]["value"])))
    body = "\n".join(
        rf"{m} & {t} & {n} & {li} & {lf:.2f} & {best:.3f} & {zs:.3f} & {best - zs:+.3f} \\"
        for (m, t, n, li, lf, best, zs) in rows
    )
    latex = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\begin{tabular}{llcccccc}\n\\toprule\n"
        "Model & Task & $L$ & best & frac & metric@best & metric@ZS & $\\Delta$ \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Optimal extraction layer per (model, task), and effect over zero-shot.}\n"
        "\\label{tab:layer_search}\n\\end{table}\n"
    )
    out = TABLES / "table2_layer_search.tex"
    out.write_text(latex)
    print(f"[tab2] {out}")


def make_table_geometry():
    files = sorted((RESULTS / "geometry").glob("geometry_*.json"))
    if not files:
        print("[tab3] no geometry results — skipping")
        return
    rows = []
    for p in files:
        d = load_json(p)
        rows.append((short(d["model"]), d["task"], d["layer_idx"], d["n_users"],
                     d["cosine_similarity"]["mean_off_diagonal"],
                     d["magnitude"]["mean_magnitude_ratio"],
                     d["pca"]["total_explained_2pc"]))
    body = "\n".join(
        rf"{m} & {t} & {li} & {n} & {cos:.3f} & {mr:.3f} & {pca * 100:.1f}\% \\"
        for (m, t, li, n, cos, mr, pca) in rows
    )
    latex = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "Model & Task & Layer & $n$ & "
        r"$\overline{\cos}_{\mathrm{off}}$ & "
        r"$\overline{\|v\|/\|h\|}$ & "
        r"PCA 2-PC \\"
        "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Per-user persona-vector geometry. Low cosine $\\Rightarrow$ vectors "
        "are distinguishable. Low magnitude ratio $\\Rightarrow$ steering signal is small "
        "relative to residual stream norm.}\n"
        "\\label{tab:geometry}\n\\end{table}\n"
    )
    out = TABLES / "table3_geometry.tex"
    out.write_text(latex)
    print(f"[tab3] {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    print("Generating figures and tables...")
    plot_layer_search()
    plot_magnitude()
    plot_main_results()
    _sweep_figure("alpha_sweep", "alpha_sweep_*.json", "alpha",
                  r"$\alpha$ (steering scale)", "Persona steering response curve",
                  "fig5_alpha_curve.pdf", "fig5", marker="o")
    _sweep_figure("n_questions", "n_questions_*.json", "n_questions",
                  "Number of extraction questions",
                  "Persona-vector quality vs extraction noise",
                  "fig6_n_questions.pdf", "fig6", marker="s", log_x=True)
    print()
    make_table_main()
    make_table_layer_search()
    make_table_geometry()
    print(f"\nFigures: {FIGURES}")
    print(f"Tables:  {TABLES}")


if __name__ == "__main__":
    main()
