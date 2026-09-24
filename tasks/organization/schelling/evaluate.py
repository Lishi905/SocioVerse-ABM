"""
B / metrics — Schelling's emergent segregation:

  - segregation: mean fraction of same-group neighbours (rises far above the
    minority_fraction baseline even when individual tolerance is modest — the model's
    famous result).
  - happy_fraction: share of residents content where they are.

Decisions are discrete (move/stay), so the consistency score uses strict equality.
"""
from __future__ import annotations

from typing import Sequence


def schelling_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    final = snapshots[-1]
    return {
        "steps": len(snapshots) - 1,
        "n": final["n"],
        "segregation_start": snapshots[0]["segregation"],
        "segregation_final": final["segregation"],
        "happy_fraction": final["happy_fraction"],
    }
