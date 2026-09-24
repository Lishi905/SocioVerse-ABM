"""
B / metrics — task-specific outcome measures fed to the kernel's evaluator.

The headline rumor-diffusion measures:
  - final_reach: cumulative number who ever left Ignorant (= N - final S). This is
    the standard "ever heard / forwarded" reach; the legacy script used the
    near-equivalent N - final_I.
  - peak_spreaders / time_to_peak: the height and timing of the spreading wave.
"""
from __future__ import annotations

from typing import List, Sequence


def sir_metrics(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        raise ValueError("no snapshots to evaluate")
    first = snapshots[0]
    n = first["S"] + first["I"] + first["R"]
    i_series: List[int] = [s["I"] for s in snapshots]
    peak = max(i_series)
    final = snapshots[-1]
    reach = n - final["S"]
    return {
        "n": n,
        "steps": len(snapshots) - 1,
        "final_reach": reach,
        "final_reach_frac": reach / n if n else 0.0,
        "peak_spreaders": peak,
        "time_to_peak": i_series.index(peak),
        "final": {k: final[k] for k in ("S", "I", "R")},
    }
