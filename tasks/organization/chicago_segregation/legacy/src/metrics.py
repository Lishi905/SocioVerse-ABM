"""
Segregation metrics for the Chicago ABM.

Implements key indices used in residential segregation research:
- Dissimilarity Index (D)
- Isolation Index (P*)
- Information Theory Index (H)
- Moran's I spatial autocorrelation
"""

import numpy as np
import pandas as pd


def dissimilarity_index(pop_minority: np.ndarray, pop_total: np.ndarray) -> float:
    """
    Compute Dissimilarity Index D between minority and rest of population.
    D = 0.5 * sum(|xi/X - yi/Y|)
    where xi = minority in tract i, X = total minority in city,
    yi = non-minority in tract i, Y = total non-minority in city.

    Returns value in [0, 1]. Target for Black-White in Chicago 2010: 0.825
    """
    pop_majority = pop_total - pop_minority
    X = pop_minority.sum()
    Y = pop_majority.sum()
    if X == 0 or Y == 0:
        return 0.0
    return 0.5 * np.sum(np.abs(pop_minority / X - pop_majority / Y))


def isolation_index(pop_group: np.ndarray, pop_total: np.ndarray) -> float:
    """
    Compute Isolation Index P*.
    P* = sum(xi/X * xi/ti)
    Measures the probability that a minority member shares a tract with another minority member.

    Target for Black in Chicago 2010: ~0.80
    """
    X = pop_group.sum()
    if X == 0:
        return 0.0
    mask = pop_total > 0
    return np.sum((pop_group[mask] / X) * (pop_group[mask] / pop_total[mask]))


def exposure_index(pop_group_a: np.ndarray, pop_group_b: np.ndarray,
                   pop_total: np.ndarray) -> float:
    """
    Compute Exposure Index: probability that a member of group A encounters group B.
    P*ab = sum(ai/A * bi/ti)
    """
    A = pop_group_a.sum()
    if A == 0:
        return 0.0
    mask = pop_total > 0
    return np.sum((pop_group_a[mask] / A) * (pop_group_b[mask] / pop_total[mask]))


def multi_group_dissimilarity(pop_groups: dict[str, np.ndarray],
                              pop_total: np.ndarray) -> dict[str, float]:
    """Compute pairwise Dissimilarity Indices for all group pairs."""
    results = {}
    groups = list(pop_groups.keys())
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            combined = pop_groups[g1] + pop_groups[g2]
            d = dissimilarity_index(pop_groups[g1], combined)
            results[f"{g1}_vs_{g2}"] = round(d, 4)
    return results


def recovery_r_squared(current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> float:
    """
    Compute R² between current per-tract racial percentages and the 2010 Census baseline.

    Flattens all (tract × race) percentage values into two vectors and computes
    the coefficient of determination. R²=1.0 means perfect recovery of the
    baseline distribution; R²≈0 means no correlation.

    Only tracts with non-zero population in the current state are included,
    so empty tracts (no agents) don't distort the metric.

    Both DataFrames must have columns: GEOID10, total_pop, pct_nh_white,
    pct_nh_black, pct_nh_asian, pct_hispanic.
    """
    pct_cols = ["pct_nh_white", "pct_nh_black", "pct_nh_asian", "pct_hispanic"]

    # Align on shared tracts
    merged = current_df.merge(baseline_df, on="GEOID10", suffixes=("_cur", "_base"))
    if merged.empty:
        return 0.0

    # Only compare tracts that have agents (non-zero current population)
    merged = merged[merged["total_pop_cur"] > 0]
    if merged.empty:
        return 0.0

    cur_vals = np.concatenate([merged[f"{c}_cur"].values for c in pct_cols])
    base_vals = np.concatenate([merged[f"{c}_base"].values for c in pct_cols])

    ss_res = np.sum((cur_vals - base_vals) ** 2)
    ss_tot = np.sum((base_vals - base_vals.mean()) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return round(float(1.0 - ss_res / ss_tot), 4)


def compute_all_metrics(race_df: pd.DataFrame) -> dict:
    """
    Compute all segregation metrics from a DataFrame with columns:
    nh_white, nh_black, nh_asian, hispanic, nh_other, total_pop

    Returns dict of metric name -> value.
    """
    total = race_df["total_pop"].values.astype(float)
    groups = {
        "nh_white": race_df["nh_white"].values.astype(float),
        "nh_black": race_df["nh_black"].values.astype(float),
        "nh_asian": race_df["nh_asian"].values.astype(float),
        "hispanic": race_df["hispanic"].values.astype(float),
    }

    metrics = {}

    # Dissimilarity Indices
    metrics["D_black_white"] = round(
        dissimilarity_index(groups["nh_black"], groups["nh_black"] + groups["nh_white"]), 4
    )
    metrics["D_hispanic_white"] = round(
        dissimilarity_index(groups["hispanic"], groups["hispanic"] + groups["nh_white"]), 4
    )
    metrics["D_asian_white"] = round(
        dissimilarity_index(groups["nh_asian"], groups["nh_asian"] + groups["nh_white"]), 4
    )

    # Isolation Indices
    metrics["Isolation_black"] = round(isolation_index(groups["nh_black"], total), 4)
    metrics["Isolation_white"] = round(isolation_index(groups["nh_white"], total), 4)
    metrics["Isolation_hispanic"] = round(isolation_index(groups["hispanic"], total), 4)

    # Key exposure indices
    metrics["Exposure_black_white"] = round(
        exposure_index(groups["nh_black"], groups["nh_white"], total), 4
    )
    metrics["Exposure_white_black"] = round(
        exposure_index(groups["nh_white"], groups["nh_black"], total), 4
    )

    return metrics
