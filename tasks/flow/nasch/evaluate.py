"""
B / metrics — NaSch fundamental-diagram measures.

  - mean_speed: average vehicle speed over the (post-warmup) run.
  - flow: average throughput q = sum(speeds)/L per step (the fundamental diagram's
    y-axis); for given density rho this is the headline traffic-flow outcome.
  - density: rho = N / L (constant; reported for the q-vs-rho diagram).

A warmup is skipped so the metrics reflect the steady state, not the random
initial condition.
"""
from __future__ import annotations

from typing import Sequence


def nasch_metrics(snapshots: Sequence[dict], warmup_frac: float = 0.2) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    steps = len(snapshots) - 1
    start = 1 + int(warmup_frac * steps)            # skip initial transient
    window = snapshots[start:] or snapshots[-1:]
    mean_speed = sum(s["mean_speed"] for s in window) / len(window)
    mean_flow = sum(s["flow"] for s in window) / len(window)
    return {
        "n": snapshots[0]["n"],
        "steps": steps,
        "density": snapshots[0]["density"],
        "mean_speed": mean_speed,
        "flow": mean_flow,
    }
