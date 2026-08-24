"""Finding the output of an earlier run.

The experiments are chained: layer_search picks the extraction layer that the
sweeps and full runs reuse, so several scripts need to read a previous run's
JSON by naming convention rather than by an explicit path.
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
    p = RESULTS / "layer_search" / f"layer_search_{short}_{task}.json"
    if not p.exists():
        return fallback
    with open(p) as f:
        return int(json.load(f)["best_layer"]["layer_idx"])
