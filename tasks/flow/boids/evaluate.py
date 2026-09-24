"""
B / metrics — flocking outcome measures, plus a tolerance-based action equality for
the consistency score (continuous vector actions can't use strict equality).

  - polarization (order parameter) Phi = |mean unit velocity| in [0, 1]; ~1 means a
    coherent flock, ~0 means disordered. The headline flocking outcome.
  - mean_speed: average speed.

`heading_equals(tol_deg)` returns an Action-equality that treats two velocity
actions as equal if their headings agree within a tolerance — this is the
`equals` passed to socioverse_abm.eval for the rule-vs-LLM consistency score.
"""
from __future__ import annotations

from typing import Sequence

from socioverse_abm.eval import heading_equals  # noqa: F401  (re-exported for tasks/tests)


def boids_metrics(snapshots: Sequence[dict], warmup_frac: float = 0.25) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    steps = len(snapshots) - 1
    start = 1 + int(warmup_frac * steps)
    window = snapshots[start:] or snapshots[-1:]
    return {
        "n": snapshots[0]["n"],
        "steps": steps,
        "polarization_final": snapshots[-1]["polarization"],
        "polarization_mean": sum(s["polarization"] for s in window) / len(window),
        "mean_speed": sum(s["mean_speed"] for s in window) / len(window),
    }
