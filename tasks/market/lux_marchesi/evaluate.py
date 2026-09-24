"""
B / metrics — the Lux-Marchesi outcome is the *stylized facts* of the price series:

  - volatility: std of log returns.
  - excess_kurtosis: fat tails (>0 means heavier-than-Gaussian, the hallmark stylized
    fact of real financial returns).
  - price_range: how far price wandered from the fundamental anchor.

Stances are discrete, so the consistency score uses strict Action.matches.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def lux_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    returns = np.array([s["log_return"] for s in snapshots[1:]])
    prices = np.array([s["price"] for s in snapshots])
    vol = float(returns.std()) if len(returns) else 0.0
    if len(returns) > 3 and returns.std() > 0:
        z = (returns - returns.mean()) / returns.std()
        excess_kurtosis = float((z ** 4).mean() - 3.0)
    else:
        excess_kurtosis = 0.0
    return {
        "steps": len(snapshots) - 1,
        "volatility": vol,
        "excess_kurtosis": excess_kurtosis,
        "price_min": float(prices.min()),
        "price_max": float(prices.max()),
        "final_opinion_index": snapshots[-1]["opinion_index"],
    }
