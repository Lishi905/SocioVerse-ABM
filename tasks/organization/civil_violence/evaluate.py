"""
B / metrics — civil violence is characterised by *punctuated* outbursts: long quiet
spells broken by sudden rebellion peaks.

  - peak_active: largest number simultaneously rebelling (the size of the worst outburst).
  - mean_active / active_std: average unrest and its burstiness.
  - jailed_final: how many sit in jail at the end.

Decisions are discrete (rebel/quiet), so the consistency score uses strict equality.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def civil_violence_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    active = np.array([s["active"] for s in snapshots])
    return {
        "steps": len(snapshots) - 1,
        "n_citizens": snapshots[0]["n_citizens"],
        "peak_active": int(active.max()),
        "mean_active": float(active.mean()),
        "active_std": float(active.std()),
        "jailed_final": snapshots[-1]["jailed"],
    }
