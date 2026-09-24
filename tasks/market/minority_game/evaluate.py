"""
B / metrics — the Minority Game's headline measure is the volatility sigma^2/N of the
attendance: lower means the population coordinates better (uses resources efficiently).
Mean win rate should sit near 0.5 (slightly below, since the minority is < half).

Actions are discrete sides, so the consistency score uses strict Action.matches.
"""
from __future__ import annotations

from typing import Sequence


def minority_game_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    final = snapshots[-1]
    return {
        "steps": len(snapshots) - 1,
        "n": final["n"],
        "volatility": final["volatility"],        # sigma^2 / N
        "attendance_final": final["attendance"],
    }
