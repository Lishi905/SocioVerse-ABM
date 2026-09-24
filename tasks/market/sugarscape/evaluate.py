"""
B / metrics — the Sugarscape outcome is the emergent wealth inequality.

  - gini: Gini coefficient of citizens' sugar (the headline inequality measure).
  - alive: surviving population (citizens starve when sugar hits 0; no respawn).
  - mean_sugar: average wealth of survivors.

Actions are discrete cell moves, so the consistency score uses strict
Action.matches (no custom equality needed).
"""
from __future__ import annotations

from typing import Sequence


def sugarscape_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    final = snapshots[-1]
    return {
        "steps": len(snapshots) - 1,
        "alive_start": snapshots[0]["alive"],
        "alive_final": final["alive"],
        "gini": final["gini"],
        "mean_sugar": final["mean_sugar"],
    }
