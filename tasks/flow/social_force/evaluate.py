"""
B / metrics — evacuation outcomes, plus the tolerance-based action equality used by
the consistency score (continuous vector actions, like Boids).

  - evac_fraction: share of pedestrians who reached the exit by the end.
  - evacuation_time: first step at which everyone is out (or total steps if not).
  - mean_dist_to_goal: how far the stragglers still are.
"""
from __future__ import annotations

from typing import Sequence

from socioverse_abm.eval import heading_equals  # noqa: F401  (re-exported for tasks/tests)


def social_force_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    final = snapshots[-1]
    evac_time = next((s["t"] for s in snapshots if s["evac_fraction"] >= 1.0),
                     len(snapshots) - 1)
    return {
        "n": final["n"],
        "steps": len(snapshots) - 1,
        "evac_fraction": final["evac_fraction"],
        "evacuation_time": evac_time,
        "mean_dist_to_goal": final["mean_dist_to_goal"],
    }
