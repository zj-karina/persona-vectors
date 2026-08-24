"""Locating the outputs of earlier runs.

The experiments are chained: `run_layer_search` picks the extraction layer that
the sweeps, full runs and operator experiments then reuse. Rather than passing
the layer around by hand, those scripts look up the previous run's JSON by
naming convention.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def best_layer(model_name: str, task: str, fallback: int | None = None) -> int | None:
    """Layer chosen by run_layer_search, or `fallback` if that search never ran."""
    short = model_name.split("/")[-1]
    path = RESULTS / "layer_search" / f"layer_search_{short}_{task}.json"
    if not path.exists():
        return fallback
    with open(path) as f:
        return int(json.load(f)["best_layer"]["layer_idx"])
