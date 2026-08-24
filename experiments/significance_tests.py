"""Paired significance tests for the body-table claims.

LaMP-2 (accuracy): McNemar's exact test on (correct_zs, correct_method) pairs.
LaMP-7 (ROUGE-L):  paired bootstrap (B=10000) on per-user mean ROUGE-L diff.

Outputs:
  results/significance/significance_tests.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "significance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
B = 10_000


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar two-sided p-value via binomial on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, p=0.5, alternative="two-sided").pvalue


def paired_bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = B, seed: int = SEED
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b  # method - zs
    obs = float(diffs.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_two = 2.0 * min((boot >= 0).mean(), (boot <= 0).mean())
    return obs, float(lo), float(hi), float(p_two)


def lamp2_paired(zs_path: Path, method_path: Path, label: str) -> dict:
    z = json.load(open(zs_path))
    m = json.load(open(method_path))
    refs = z["refs"]
    assert refs == m["refs"], f"{label}: refs mismatch"
    n = len(refs)
    cz = [int(p == r) for p, r in zip(z["preds"], refs)]
    cm = [int(p == r) for p, r in zip(m["preds"], refs)]
    # discordant cells of the 2x2 contingency table
    bb = sum(1 for x, y in zip(cz, cm) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(cz, cm) if x == 0 and y == 1)
    p = mcnemar_exact(bb, c)
    return {
        "label": label,
        "n": n,
        "acc_zs": sum(cz) / n,
        "acc_method": sum(cm) / n,
        "delta": (sum(cm) - sum(cz)) / n,
        "discordant_zs_only": bb,
        "discordant_method_only": c,
        "mcnemar_p_exact": p,
    }


def lamp2_paired_from_variance(var_path: Path, label: str) -> dict:
    """Use rouge_zs / rouge_steered (binary 1.0/0.0 for classification) from variance JSON."""
    d = json.load(open(var_path))
    rows = d["per_user"]
    cz = [1 if r["pred_zs"] == r["gold"] else 0 for r in rows]
    cm = [1 if r["pred_steered"] == r["gold"] else 0 for r in rows]
    n = len(rows)
    bb = sum(1 for x, y in zip(cz, cm) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(cz, cm) if x == 0 and y == 1)
    p = mcnemar_exact(bb, c)
    return {
        "label": label,
        "n": n,
        "acc_zs": sum(cz) / n,
        "acc_method": sum(cm) / n,
        "delta": (sum(cm) - sum(cz)) / n,
        "discordant_zs_only": bb,
        "discordant_method_only": c,
        "mcnemar_p_exact": p,
    }


def lamp7_paired_from_variance(var_path: Path, label: str) -> dict:
    d = json.load(open(var_path))
    rows = d["per_user"]
    rz = np.array([r["rouge_zs"] for r in rows], dtype=float)
    rs = np.array([r["rouge_steered"] for r in rows], dtype=float)
    obs, lo, hi, p = paired_bootstrap_mean_diff(rs, rz)
    return {
        "label": label,
        "n": len(rows),
        "rouge_zs": float(rz.mean()),
        "rouge_method": float(rs.mean()),
        "delta": obs,
        "boot_ci95": [lo, hi],
        "boot_p_two_sided": p,
    }


def lamp7_paired_icl(zs_path: Path, icl_path: Path, label: str) -> dict:
    """LaMP-7 ICL vs ZS: pair on the per-example index (sorted by user; same seed)."""
    z = json.load(open(zs_path))
    i = json.load(open(icl_path))
    if z["refs"] != i["refs"]:
        return {"label": label, "error": "refs mismatch"}
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rz = np.array(
        [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(z["refs"], z["preds"])]
    )
    rs = np.array(
        [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(i["refs"], i["preds"])]
    )
    obs, lo, hi, p = paired_bootstrap_mean_diff(rs, rz)
    return {
        "label": label,
        "n": len(rz),
        "rouge_zs": float(rz.mean()),
        "rouge_method": float(rs.mean()),
        "delta": obs,
        "boot_ci95": [lo, hi],
        "boot_p_two_sided": p,
    }


def lamp2_paired_icl(zs_path: Path, icl_path: Path, label: str) -> dict:
    return lamp2_paired(zs_path, icl_path, label)


def main():
    results = []
    icl = RESULTS / "icl_baseline_full"
    var = RESULTS / "variance_analysis"

    # LaMP-2 — McNemar
    results.append(
        lamp2_paired_from_variance(
            var / "LaMP-2_per_user_stats_n323_template.json",
            "LaMP-2: ZS vs steering(template) [variance]",
        )
    )
    results.append(
        lamp2_paired_from_variance(
            var / "LaMP-2_per_user_stats_n323_fact.json",
            "LaMP-2: ZS vs steering(fact) [variance]",
        )
    )
    for k in (3, 5, 6):
        results.append(
            lamp2_paired_icl(
                icl / "LaMP-2_zs_n323.json",
                icl / f"LaMP-2_icl_k{k}.json",
                f"LaMP-2: ZS vs ICL(K={k})",
            )
        )

    # LaMP-7 — paired bootstrap on per-user ROUGE-L
    results.append(
        lamp7_paired_from_variance(
            var / "LaMP-7_per_user_stats_n1497_template.json",
            "LaMP-7: ZS vs steering(template) [variance]",
        )
    )
    results.append(
        lamp7_paired_from_variance(
            var / "LaMP-7_per_user_stats_n1497_fact.json",
            "LaMP-7: ZS vs steering(fact) [variance]",
        )
    )
    results.append(
        lamp7_paired_icl(
            icl / "LaMP-7_zs_n1497.json",
            icl / "LaMP-7_icl_k3.json",
            "LaMP-7: ZS vs ICL(K=3)",
        )
    )

    out = OUT_DIR / "significance_tests.json"
    json.dump({"seed": SEED, "n_boot": B, "tests": results}, open(out, "w"), indent=2)
    print(f"wrote {out}")
    for r in results:
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
