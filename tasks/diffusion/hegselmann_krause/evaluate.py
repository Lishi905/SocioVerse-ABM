"""
B / metrics — the HK outcome is the cluster structure: consensus (1 cluster),
polarization (2), or fragmentation (many), set by epsilon. Plus an opinion-tolerance
equality for the continuous-value consistency score.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from socioverse_abm.behavior_engine.action import Action


def cluster_count(opinions, tol: float = 1e-2) -> int:
    vals = np.sort(np.asarray(opinions, dtype=float))
    if len(vals) == 0:
        return 0
    return 1 + int((np.diff(vals) > tol).sum())


def hk_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    final = snapshots[-1]
    return {
        "n": final["n"],
        "steps": len(snapshots) - 1,
        "num_clusters": final["num_clusters"],
        "final_spread": final["spread"],
        "converged_from": snapshots[0]["num_clusters"],
    }


def opinion_equals(tol: float = 0.05) -> Callable[[Action, Action], bool]:
    def equals(a: Action, b: Action) -> bool:
        return abs(a.payload["opinion"] - b.payload["opinion"]) <= tol
    return equals
