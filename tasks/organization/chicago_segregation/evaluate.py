"""B — segregation-index summary of a trajectory.

Each snapshot is a per-step metrics row emitted by ``env.snapshot()`` (the vendored
``_compute_metrics`` output: Dissimilarity / Isolation / Exposure / R²_recovery, plus
mobility + direction indicators). This reduces the trajectory to start/final endpoints
per index and mobility stats — the quantities the Chicago case study reports.
"""
from __future__ import annotations

from typing import List, Mapping

_INDEX_KEYS = [
    "D_black_white", "D_hispanic_white", "D_asian_white",
    "Isolation_black", "Isolation_white", "Isolation_hispanic",
    "Exposure_black_white", "Exposure_white_black",
    "R2_recovery",
]


def chicago_metrics(snapshots: List[Mapping]) -> dict:
    if not snapshots:
        return {}
    first, last = snapshots[0], snapshots[-1]
    out: dict = {"steps": len(snapshots) - 1}

    for k in _INDEX_KEYS:
        if k in first or k in last:
            out[f"{k}_start"] = first.get(k)
            out[f"{k}_final"] = last.get(k)

    # Mobility over the move steps (snapshots[1:]).
    move_rows = [s for s in snapshots[1:]]
    if move_rows:
        movers = [s.get("n_movers", 0) or 0 for s in move_rows]
        pct_pop = [s.get("pct_pop_moved", 0.0) or 0.0 for s in move_rows]
        aligned = [s.get("mover_alignment_rate") for s in move_rows if s.get("mover_alignment_rate") is not None]
        out["avg_movers_per_step"] = sum(movers) / len(movers)
        out["avg_pct_pop_moved"] = sum(pct_pop) / len(pct_pop)
        if aligned:
            out["avg_mover_alignment_rate"] = sum(aligned) / len(aligned)

    return out
