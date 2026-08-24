"""Paired significance tests for the claims in the body table.

LaMP-2 is a classification task, so the pairing is exact McNemar on the
discordant (zero-shot correct, method wrong) / (wrong, correct) counts. LaMP-7
is scored by ROUGE-L, so it gets a paired bootstrap over per-user differences.

Writes results/significance/significance_tests.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "significance"
SEED = 42
N_BOOT = 10_000


def mcnemar_exact(zs_only: int, method_only: int) -> float:
    n = zs_only + method_only
    if n == 0:
        return 1.0
    return binomtest(min(zs_only, method_only), n, p=0.5,
                     alternative="two-sided").pvalue


def paired_bootstrap_mean_diff(
    method: np.ndarray, zs: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = method - zs
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(p_two)


def _mcnemar_row(label: str, correct_zs: list[int], correct_method: list[int]) -> dict:
    n = len(correct_zs)
    zs_only = sum(1 for x, y in zip(correct_zs, correct_method) if x and not y)
    method_only = sum(1 for x, y in zip(correct_zs, correct_method) if y and not x)
    return {
        "label": label,
        "n": n,
        "acc_zs": sum(correct_zs) / n,
        "acc_method": sum(correct_method) / n,
        "delta": (sum(correct_method) - sum(correct_zs)) / n,
        "discordant_zs_only": zs_only,
        "discordant_method_only": method_only,
        "mcnemar_p_exact": mcnemar_exact(zs_only, method_only),
    }


def _bootstrap_row(label: str, rouge_method: np.ndarray, rouge_zs: np.ndarray) -> dict:
    obs, lo, hi, p = paired_bootstrap_mean_diff(rouge_method, rouge_zs)
    return {
        "label": label,
        "n": len(rouge_zs),
        "rouge_zs": float(rouge_zs.mean()),
        "rouge_method": float(rouge_method.mean()),
        "delta": obs,
        "boot_ci95": [lo, hi],
        "boot_p_two_sided": p,
    }


def lamp2_from_predictions(zs_path: Path, method_path: Path, label: str) -> dict:
    zs = json.loads(zs_path.read_text())
    method = json.loads(method_path.read_text())
    refs = zs["refs"]
    assert refs == method["refs"], f"{label}: refs mismatch"
    return _mcnemar_row(
        label,
        [int(p == r) for p, r in zip(zs["preds"], refs)],
        [int(p == r) for p, r in zip(method["preds"], refs)],
    )


def lamp2_from_variance(var_path: Path, label: str) -> dict:
    """The variance runs store both predictions per user, so pairing is direct."""
    rows = json.loads(var_path.read_text())["per_user"]
    return _mcnemar_row(
        label,
        [int(r["pred_zs"] == r["gold"]) for r in rows],
        [int(r["pred_steered"] == r["gold"]) for r in rows],
    )


def lamp7_from_variance(var_path: Path, label: str) -> dict:
    rows = json.loads(var_path.read_text())["per_user"]
    return _bootstrap_row(
        label,
        np.array([r["rouge_steered"] for r in rows], dtype=float),
        np.array([r["rouge_zs"] for r in rows], dtype=float),
    )


def lamp7_from_predictions(zs_path: Path, method_path: Path, label: str) -> dict:
    from rouge_score import rouge_scorer

    zs = json.loads(zs_path.read_text())
    method = json.loads(method_path.read_text())
    if zs["refs"] != method["refs"]:
        return {"label": label, "error": "refs mismatch"}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def per_example(d):
        return np.array([scorer.score(r, p)["rougeL"].fmeasure
                         for r, p in zip(d["refs"], d["preds"])])

    return _bootstrap_row(label, per_example(method), per_example(zs))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    icl = RESULTS / "icl_baseline_full"
    var = RESULTS / "variance_analysis"

    results = [
        lamp2_from_variance(var / "LaMP-2_per_user_stats_n323_template.json",
                            "LaMP-2: ZS vs steering(template) [variance]"),
        lamp2_from_variance(var / "LaMP-2_per_user_stats_n323_fact.json",
                            "LaMP-2: ZS vs steering(fact) [variance]"),
    ]
    results += [
        lamp2_from_predictions(icl / "LaMP-2_zs_n323.json",
                               icl / f"LaMP-2_icl_k{k}.json",
                               f"LaMP-2: ZS vs ICL(K={k})")
        for k in (3, 5, 6)
    ]
    results += [
        lamp7_from_variance(var / "LaMP-7_per_user_stats_n1497_template.json",
                            "LaMP-7: ZS vs steering(template) [variance]"),
        lamp7_from_variance(var / "LaMP-7_per_user_stats_n1497_fact.json",
                            "LaMP-7: ZS vs steering(fact) [variance]"),
        lamp7_from_predictions(icl / "LaMP-7_zs_n1497.json",
                               icl / "LaMP-7_icl_k3.json",
                               "LaMP-7: ZS vs ICL(K=3)"),
    ]

    out = OUT_DIR / "significance_tests.json"
    with open(out, "w") as f:
        json.dump({"seed": SEED, "n_boot": N_BOOT, "tests": results}, f, indent=2)
    print(f"wrote {out}")
    for r in results:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
