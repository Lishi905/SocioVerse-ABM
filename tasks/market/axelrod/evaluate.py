"""
B / metrics — the cooperation outcome:

  - cooperation_rate: average share of C moves over the run (does cooperation survive?).
  - strategy_ranking: strategies ordered by mean accumulated payoff (Axelrod's result
    is that tit_for_tat-like cooperative strategies win round-robins).

Each agent's action is a dict of per-opponent moves; the consistency score uses
strict Action.matches (all per-opponent moves must agree).
"""
from __future__ import annotations

from typing import Sequence


def axelrod_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    rounds = snapshots[1:] or snapshots
    final = snapshots[-1]
    ranking = sorted(final["strategy_scores"].items(), key=lambda kv: kv[1], reverse=True)
    return {
        "rounds": len(snapshots) - 1,
        "n": final["n"],
        "cooperation_rate": sum(s["cooperation_rate"] for s in rounds) / len(rounds),
        "mean_score": final["mean_score"],
        "strategy_ranking": ranking,
    }
