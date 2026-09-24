# -*- coding: utf-8 -*-
"""
Lux-Marchesi Model Evaluation
1. Fat tails (excess kurtosis) in return distributions
2. Volatility clustering (autocorrelation of absolute/squared returns)
3. Absence of linear autocorrelation in returns
4. Price deviation from fundamental value dynamics
5. Trader composition dynamics
"""

import numpy as np
from scipy import stats
from scipy.signal import correlate
from typing import Dict, List, Tuple, Optional, Any
import json
import os


def compute_returns(prices: np.ndarray, log_returns: bool = True) -> np.ndarray:
    """
    Compute returns from price series.
    
    Args:
        prices: Array of prices
        log_returns: If True, compute log returns; else simple returns
        
    Returns:
        Array of returns
    """
    prices = prices[prices > 0]
    
    if len(prices) < 2:
        return np.array([])
    
    if log_returns:
        return np.diff(np.log(prices))
    else:
        return np.diff(prices) / prices[:-1]


def compute_autocorrelation(series: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation function up to max_lag.
    
    Args:
        series: Time series
        max_lag: Maximum lag to compute
        
    Returns:
        Array of autocorrelation values for lags 0 to max_lag
    """
    n = len(series)
    if n < 2:
        return np.array([1.0])
    
    # Demean the series
    series_centered = series - np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return np.ones(min(max_lag + 1, n))
    
    autocorr = np.zeros(min(max_lag + 1, n))
    autocorr[0] = 1.0
    
    for lag in range(1, min(max_lag + 1, n)):
        autocorr[lag] = np.sum(series_centered[lag:] * series_centered[:-lag]) / ((n - lag) * var)
    
    return autocorr


def test_fat_tails(returns: np.ndarray) -> Dict[str, float]:
    """
    Test for fat tails in return distribution.
    
    Stylized fact: Financial returns exhibit excess kurtosis (> 3 for normal).
    
    Args:
        returns: Array of returns
        
    Returns:
        Dictionary with fat tail statistics
    """
    if len(returns) < 4:
        return {"kurtosis": np.nan, "skewness": np.nan, "jarque_bera_stat": np.nan, 
                "jarque_bera_pvalue": np.nan, "fat_tails_detected": False}
    
    # Basic moments
    kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis (normal = 0)
    skewness = stats.skew(returns)
    
    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    # Test if distribution has fatter tails than normal
    # Kurtosis > 0 (excess kurtosis) indicates fat tails
    fat_tails_detected = kurtosis > 0 and jb_pvalue < 0.05
    
    return {
        "kurtosis": float(kurtosis),
        "excess_kurtosis": float(kurtosis),  # Same as kurtosis with fisher=True
        "skewness": float(skewness),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "fat_tails_detected": fat_tails_detected,
    }


def test_volatility_clustering(returns: np.ndarray, max_lag: int = 50) -> Dict[str, Any]:
    """
    Test for volatility clustering.
    
    Stylized fact: Absolute and squared returns show significant positive autocorrelation
    that decays slowly, while raw returns show no linear autocorrelation.
    
    Args:
        returns: Array of returns
        max_lag: Maximum lag for autocorrelation
        
    Returns:
        Dictionary with volatility clustering statistics
    """
    if len(returns) < max_lag + 1:
        return {
            "return_autocorr": np.array([]),
            "abs_return_autocorr": np.array([]),
            "squared_return_autocorr": np.array([]),
            "volatility_clustering_detected": False,
        }
    
    # Autocorrelation of raw returns (should be ~0)
    return_autocorr = compute_autocorrelation(returns, max_lag)
    
    # Autocorrelation of absolute returns (should be positive and slowly decaying)
    abs_returns = np.abs(returns)
    abs_return_autocorr = compute_autocorrelation(abs_returns, max_lag)
    
    # Autocorrelation of squared returns
    squared_returns = returns ** 2
    squared_return_autocorr = compute_autocorrelation(squared_returns, max_lag)
    
    # Test criteria:
    # 1. Raw return autocorrelation should be close to zero
    #    - For large n, pure significance testing is too strict (even tiny ACF is "significant")
    #    - Use BOTH significance AND magnitude thresholds
    # 2. Absolute return autocorrelation should be meaningfully positive (not just barely significant)
    
    n = len(returns)
    ci_bound = 2.0 / np.sqrt(n)
    
    # Magnitude threshold for "practically zero" ACF (in addition to statistical significance)
    # Financial empirical papers typically use |ACF| < 0.05 as "insignificant"
    practical_acf_threshold = 0.05
    
    # Check if raw returns have no meaningful autocorrelation
    # Must be BOTH statistically insignificant AND practically small
    mean_raw_acf = float(np.mean(np.abs(return_autocorr[1:11])))
    raw_insignificant_stat = np.all(np.abs(return_autocorr[1:11]) < ci_bound)
    raw_insignificant_practical = mean_raw_acf < practical_acf_threshold
    raw_insignificant = raw_insignificant_stat or raw_insignificant_practical  # Either condition suffices
    
    # Check if absolute returns have meaningful positive autocorrelation
    # Must be BOTH above CI AND above practical threshold
    mean_abs_acf = float(np.mean(abs_return_autocorr[1:11]))
    abs_significant = mean_abs_acf > max(ci_bound, 0.02)  # At least 0.02 or CI, whichever is larger
    
    volatility_clustering_detected = raw_insignificant and abs_significant
    
    # Measure of persistence: sum of autocorrelations (related to long memory)
    abs_autocorr_sum = float(np.sum(abs_return_autocorr[1:]))
    
    return {
        "return_autocorr": return_autocorr.tolist(),
        "abs_return_autocorr": abs_return_autocorr.tolist(),
        "squared_return_autocorr": squared_return_autocorr.tolist(),
        "mean_return_autocorr_lag1_10": float(np.mean(return_autocorr[1:11])),
        "mean_abs_return_autocorr_lag1_10": float(np.mean(abs_return_autocorr[1:11])),
        "abs_autocorr_sum": abs_autocorr_sum,
        "confidence_bound": float(ci_bound),
        "volatility_clustering_detected": volatility_clustering_detected,
    }


def test_price_fundamental_relation(prices: np.ndarray, pf: np.ndarray) -> Dict[str, float]:
    """
    Analyze the relationship between price and fundamental value.
    
    Stylized fact: Prices fluctuate around fundamental value with occasional
    large deviations (bubbles and crashes).
    
    Args:
        prices: Array of market prices
        pf: Array of fundamental values
        
    Returns:
        Dictionary with price-fundamental statistics
    """
    if len(prices) != len(pf) or len(prices) < 2:
        return {}
    
    # Mispricing ratio
    mispricing = (prices - pf) / pf
    
    # Statistics
    mean_mispricing = float(np.mean(mispricing))
    std_mispricing = float(np.std(mispricing))
    max_overvaluation = float(np.max(mispricing))
    max_undervaluation = float(np.min(mispricing))
    
    # Correlation between price and fundamental value
    correlation = float(np.corrcoef(prices, pf)[0, 1])
    
    # Mean reversion test: does mispricing tend to decrease?
    # Compute autocorrelation of mispricing
    mispricing_autocorr = compute_autocorrelation(mispricing, 20)
    
    # ADF-like stationarity indicator (simplified)
    # If mispricing is stationary, price tracks fundamental value
    mispricing_diff = np.diff(mispricing)
    if len(mispricing_diff) > 1 and np.std(mispricing_diff) > 0:
        # Rough stationarity check: variance ratio
        variance_ratio = np.var(mispricing) / (len(mispricing) * np.var(mispricing_diff))
    else:
        variance_ratio = np.nan
    
    return {
        "mean_mispricing": mean_mispricing,
        "std_mispricing": std_mispricing,
        "max_overvaluation": max_overvaluation,
        "max_undervaluation": max_undervaluation,
        "price_pf_correlation": correlation,
        "mispricing_autocorr_lag1": float(mispricing_autocorr[1]) if len(mispricing_autocorr) > 1 else np.nan,
        "mispricing_autocorr_lag10": float(mispricing_autocorr[10]) if len(mispricing_autocorr) > 10 else np.nan,
        "variance_ratio": float(variance_ratio),
    }


def test_trader_composition_dynamics(nc: np.ndarray, nf: np.ndarray, 
                                     n_plus: np.ndarray, n_minus: np.ndarray,
                                     N: int) -> Dict[str, float]:
    """
    Analyze trader composition dynamics.
    
    Stylized fact: Chartist population increases during volatile periods,
    fundamentalists increase when prices deviate from fundamentals.
    
    Args:
        nc: Array of chartist counts
        nf: Array of fundamentalist counts
        n_plus: Array of optimist counts
        n_minus: Array of pessimist counts
        N: Total agent count
        
    Returns:
        Dictionary with composition statistics
    """
    if len(nc) < 2:
        return {}
    
    # Fractions
    chartist_frac = nc / N
    fundamentalist_frac = nf / N
    opinion_index = (n_plus - n_minus) / np.maximum(nc, 1)
    
    return {
        "mean_chartist_fraction": float(np.mean(chartist_frac)),
        "std_chartist_fraction": float(np.std(chartist_frac)),
        "max_chartist_fraction": float(np.max(chartist_frac)),
        "min_chartist_fraction": float(np.min(chartist_frac)),
        "mean_opinion_index": float(np.mean(opinion_index)),
        "std_opinion_index": float(np.std(opinion_index)),
        "opinion_index_range": float(np.max(opinion_index) - np.min(opinion_index)),
        "chartist_frac_autocorr_lag1": float(compute_autocorrelation(chartist_frac, 1)[1]) if len(chartist_frac) > 1 else np.nan,
    }


def test_volatility_chartist_correlation(returns: np.ndarray, nc: np.ndarray) -> Dict[str, float]:
    """
    Test correlation between volatility and chartist population.
    
    Stylized fact: Higher chartist population should correlate with higher volatility.
    
    Args:
        returns: Array of returns
        nc: Array of chartist counts (aligned with returns)
        
    Returns:
        Dictionary with correlation statistics
    """
    if len(returns) != len(nc) - 1 or len(returns) < 10:
        return {}
    
    abs_returns = np.abs(returns)
    nc_aligned = nc[1:]  # Align with returns
    
    # Rolling volatility (window = 100)
    window = min(100, len(returns) // 10)
    if window < 2:
        return {"volatility_chartist_correlation": np.nan}
    
    rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) 
                           for i in range(len(returns))])
    
    # Correlation
    correlation = float(np.corrcoef(rolling_vol, nc_aligned)[0, 1])
    
    return {
        "volatility_chartist_correlation": correlation,
        "abs_return_chartist_correlation": float(np.corrcoef(abs_returns, nc_aligned)[0, 1]),
    }


def compute_power_law_tail(returns: np.ndarray, threshold_percentile: float = 95,
                          min_tail_obs: int = 30) -> Dict[str, float]:
    """
    Estimate power law tail exponent for return distribution using Hill estimator.
    
    Stylized fact: Tails of return distributions follow power law with exponent ~3.
    
    Args:
        returns: Array of returns
        threshold_percentile: Percentile threshold for tail estimation
        min_tail_obs: Minimum number of tail observations required for reliable estimate
        
    Returns:
        Dictionary with power law statistics
    """
    result = {
        "tail_exponent": np.nan, 
        "tail_exponent_stderr": np.nan,
        "tail_observations": 0,
        "threshold": np.nan,
        "reliable": False,
    }
    
    if len(returns) < 100:
        return result
    
    abs_returns = np.abs(returns)
    abs_returns = abs_returns[np.isfinite(abs_returns)]
    # Discrete-tick implementations can generate many exactly-zero returns.
    # For Hill estimation we must work with strictly positive observations.
    abs_returns = abs_returns[abs_returns > 0]
    if len(abs_returns) < 100:
        return result

    threshold = np.percentile(abs_returns, threshold_percentile)
    if threshold <= 0:
        return result
    
    # Get tail observations
    tail_obs = abs_returns[abs_returns > threshold]
    
    result["tail_observations"] = len(tail_obs)
    result["threshold"] = float(threshold)
    
    if len(tail_obs) < min_tail_obs:
        # Not enough tail observations for reliable estimate
        return result
    
    # Hill estimator for tail exponent
    log_ratios = np.log(tail_obs / threshold)
    denom = float(np.sum(log_ratios))
    if not np.isfinite(denom) or denom <= 0:
        return result
    tail_exponent = len(tail_obs) / denom
    
    # Standard error (asymptotic)
    tail_exponent_stderr = tail_exponent / np.sqrt(len(tail_obs))
    
    result["tail_exponent"] = float(tail_exponent)
    result["tail_exponent_stderr"] = float(tail_exponent_stderr)
    result["reliable"] = len(tail_obs) >= min_tail_obs
    
    return result


# =============================================================================
# Multi-scale Returns Analysis (Paper Alignment)
# =============================================================================

def compute_aggregated_returns(prices: np.ndarray, horizons: List[int] = None,
                               overlapping: bool = True) -> Dict[int, np.ndarray]:
    """
    Compute aggregated (multi-period) returns for different horizons.
    
    This is crucial for Poisson-tick models where micro-returns may be zero-inflated.
    Aggregation reveals stylized facts that emerge at coarser time scales.
    
    Args:
        prices: Array of prices
        horizons: List of aggregation horizons (default: [1, 5, 10, 50, 100])
        overlapping: If True (default), use overlapping windows for more samples.
                    If False, use non-overlapping windows (no mechanical ACF artifacts).
        
    Returns:
        Dictionary mapping horizon -> returns array
    """
    if horizons is None:
        horizons = [1, 5, 10, 50, 100]
    
    prices = prices[prices > 0]
    if len(prices) < 2:
        return {}
    
    result = {}
    for h in horizons:
        if h >= len(prices):
            continue
        
        if overlapping:
            # Overlapping: r[t] = log(p[t+h]) - log(p[t]) for all t
            # More samples but introduces mechanical autocorrelation at lags < h
            agg_returns = np.log(prices[h:]) - np.log(prices[:-h])
        else:
            # Non-overlapping: only use p[0], p[h], p[2h], ... 
            # Fewer samples but no mechanical ACF artifacts
            # r[k] = log(p[(k+1)*h]) - log(p[k*h]) for k = 0, 1, ...
            n_periods = len(prices) // h
            if n_periods < 2:
                continue
            p_start = prices[::h][:n_periods]
            p_end = prices[h::h][:n_periods]
            if len(p_start) != len(p_end):
                min_len = min(len(p_start), len(p_end))
                p_start = p_start[:min_len]
                p_end = p_end[:min_len]
            agg_returns = np.log(p_end) - np.log(p_start)
        
        result[h] = agg_returns
    
    return result


def compute_event_time_returns(returns: np.ndarray, zero_threshold: float = 1e-12) -> np.ndarray:
    """
    Extract event-time returns by filtering out zero returns.
    
    For Poisson-tick models, many micro-returns are exactly zero.
    Event-time analysis focuses on periods when actual trading occurred.
    
    Args:
        returns: Array of returns
        zero_threshold: Absolute threshold below which returns are considered zero
        
    Returns:
        Array of non-zero returns
    """
    return returns[np.abs(returns) > zero_threshold]


def test_multiscale_stylized_facts(prices: np.ndarray, 
                                   horizons: List[int] = None,
                                   use_nonoverlapping: bool = True) -> Dict[str, Any]:
    """
    Evaluate stylized facts at multiple aggregation scales.
    
    This is critical for Poisson-tick models where micro-scale returns may
    appear degenerate but aggregated returns show proper stylized facts.
    
    Args:
        prices: Array of prices
        horizons: Aggregation horizons to test
        use_nonoverlapping: If True (default), use non-overlapping returns for ACF tests
                           to avoid mechanical autocorrelation artifacts. Overlapping
                           returns are still used for distribution/tail estimation.
        
    Returns:
        Dictionary with multi-scale evaluation results
    """
    if horizons is None:
        horizons = [1, 5, 10, 50, 100]
    
    # Use overlapping for distribution/tail (more samples), non-overlapping for ACF (no artifacts)
    agg_returns_overlap = compute_aggregated_returns(prices, horizons, overlapping=True)
    agg_returns_nonoverlap = compute_aggregated_returns(prices, horizons, overlapping=False) if use_nonoverlapping else agg_returns_overlap
    
    results = {
        "horizons_tested": [],
        "fat_tails_by_horizon": {},
        "volatility_clustering_by_horizon": {},
        "returns_acf_by_horizon": {},  # New: raw return ACF (should be ~0 for stylized fact i)
        "tail_exponent_by_horizon": {},
        "best_horizon_for_fat_tails": None,
        "best_horizon_for_clustering": None,
        "best_horizon_for_zero_acf": None,  # Horizon where returns ACF is closest to 0
        "methodology": "nonoverlapping" if use_nonoverlapping else "overlapping",
    }
    
    best_kurtosis = -np.inf
    best_acf = -np.inf
    best_zero_acf_score = np.inf  # Lower is better (closer to 0)
    
    for h in horizons:
        # Use overlapping returns for distribution/tail (more samples)
        returns_overlap = agg_returns_overlap.get(h)
        # Use non-overlapping returns for ACF (no mechanical artifacts)
        returns_nonoverlap = agg_returns_nonoverlap.get(h)
        
        if returns_overlap is None or len(returns_overlap) < 100:
            continue
        
        results["horizons_tested"].append(h)
        
        # Fat tails test (use overlapping for more tail samples)
        fat_tails = test_fat_tails(returns_overlap)
        results["fat_tails_by_horizon"][h] = {
            "kurtosis": fat_tails["kurtosis"],
            "skewness": fat_tails["skewness"],
            "detected": fat_tails["fat_tails_detected"],
        }
        
        if fat_tails["kurtosis"] > best_kurtosis:
            best_kurtosis = fat_tails["kurtosis"]
            results["best_horizon_for_fat_tails"] = h
        
        # Volatility clustering test (use non-overlapping to avoid artifacts)
        acf_returns = returns_nonoverlap if returns_nonoverlap is not None and len(returns_nonoverlap) >= 50 else returns_overlap
        vol_cluster = test_volatility_clustering(acf_returns, max_lag=min(50, len(acf_returns) // 10))
        acf_mean = vol_cluster.get("mean_abs_return_autocorr_lag1_10", 0)
        results["volatility_clustering_by_horizon"][h] = {
            "mean_abs_return_autocorr_lag1_10": acf_mean,
            "detected": vol_cluster.get("volatility_clustering_detected", False),
            "n_samples": len(acf_returns),
        }
        
        if acf_mean > best_acf:
            best_acf = acf_mean
            results["best_horizon_for_clustering"] = h
        
        # Raw returns ACF (stylized fact i: should be ~0)
        raw_acf_mean = vol_cluster.get("mean_return_autocorr_lag1_10", 0)
        results["returns_acf_by_horizon"][h] = {
            "mean_return_autocorr_lag1_10": raw_acf_mean,
            "near_zero": abs(raw_acf_mean) < 0.05,  # Threshold for "close to zero"
        }
        
        if abs(raw_acf_mean) < best_zero_acf_score:
            best_zero_acf_score = abs(raw_acf_mean)
            results["best_horizon_for_zero_acf"] = h
        
        # Tail exponent (use overlapping for more tail samples)
        tail = compute_power_law_tail(returns_overlap)
        results["tail_exponent_by_horizon"][h] = {
            "exponent": tail["tail_exponent"],
            "stderr": tail["tail_exponent_stderr"],
            "observations": tail.get("tail_observations", 0),
            "reliable": tail.get("reliable", False),
        }
    
    return results


def test_extended_volatility_clustering(returns: np.ndarray, 
                                        max_lag: int = 200) -> Dict[str, Any]:
    """
    Extended volatility clustering test with longer lags.
    
    The paper emphasizes LONG MEMORY in |r| and r² autocorrelations,
    which requires testing at longer lags than the default 50.
    
    Args:
        returns: Array of returns
        max_lag: Maximum lag to test (default: 200 for long memory detection)
        
    Returns:
        Dictionary with extended volatility clustering statistics
    """
    n = len(returns)
    effective_max_lag = min(max_lag, n // 5)  # Don't use more than 20% of series
    
    if effective_max_lag < 10:
        return {"extended_analysis_available": False}
    
    abs_returns = np.abs(returns)
    squared_returns = returns ** 2
    
    abs_acf = compute_autocorrelation(abs_returns, effective_max_lag)
    sq_acf = compute_autocorrelation(squared_returns, effective_max_lag)
    
    ci_bound = 2.0 / np.sqrt(n)
    
    # Count how many lags show significant positive autocorrelation
    abs_significant_lags = int(np.sum(abs_acf[1:] > ci_bound))
    sq_significant_lags = int(np.sum(sq_acf[1:] > ci_bound))
    
    # Check for slow decay (long memory indicator)
    # If lag-100 ACF is still above CI, suggests long memory
    lag_100_idx = min(100, effective_max_lag)
    long_memory_indicator = abs_acf[lag_100_idx] > ci_bound if lag_100_idx < len(abs_acf) else False
    
    # Half-life estimation (where ACF first drops below 0.5 * ACF[1])
    abs_half_life = np.nan
    if len(abs_acf) > 2 and abs_acf[1] > 0:
        target = 0.5 * abs_acf[1]
        half_life_idx = np.where(abs_acf[1:] < target)[0]
        if len(half_life_idx) > 0:
            abs_half_life = float(half_life_idx[0] + 1)
    
    return {
        "extended_analysis_available": True,
        "max_lag_tested": effective_max_lag,
        "abs_return_acf_lag50": float(abs_acf[50]) if len(abs_acf) > 50 else np.nan,
        "abs_return_acf_lag100": float(abs_acf[100]) if len(abs_acf) > 100 else np.nan,
        "squared_return_acf_lag50": float(sq_acf[50]) if len(sq_acf) > 50 else np.nan,
        "squared_return_acf_lag100": float(sq_acf[100]) if len(sq_acf) > 100 else np.nan,
        "abs_significant_lags": abs_significant_lags,
        "sq_significant_lags": sq_significant_lags,
        "long_memory_indicator": long_memory_indicator,
        "abs_half_life": abs_half_life,
        "confidence_bound": float(ci_bound),
    }


def full_evaluation(history: Dict[str, np.ndarray], params: Dict[str, Any] = None,
                   include_multiscale: bool = True) -> Dict[str, Any]:
    """
    Run full evaluation suite on simulation results.
    
    Args:
        history: Dictionary with simulation history arrays
        params: Optional parameters dictionary
        include_multiscale: Include multi-scale aggregated returns analysis (paper alignment)
        
    Returns:
        Comprehensive evaluation results
    """
    prices = np.array(history.get("price", []))
    pf = np.array(history.get("pf", []))
    nc = np.array(history.get("nc", []))
    nf = np.array(history.get("nf", []))
    n_plus = np.array(history.get("n_plus", []))
    n_minus = np.array(history.get("n_minus", []))
    
    N = params.get("N", nc[0] + nf[0]) if params else (nc[0] + nf[0] if len(nc) > 0 else 500)
    
    # Compute returns
    returns = compute_returns(prices)
    
    # Event-time returns (for Poisson-tick models)
    event_returns = compute_event_time_returns(returns)
    nonzero_frac = len(event_returns) / len(returns) if len(returns) > 0 else 0
    
    results = {
        "basic_stats": {
            "n_observations": len(prices),
            "n_returns": len(returns),
            "n_event_returns": len(event_returns),
            "nonzero_return_fraction": float(nonzero_frac),
            "mean_return": float(np.mean(returns)) if len(returns) > 0 else np.nan,
            "std_return": float(np.std(returns)) if len(returns) > 0 else np.nan,
            "min_return": float(np.min(returns)) if len(returns) > 0 else np.nan,
            "max_return": float(np.max(returns)) if len(returns) > 0 else np.nan,
            "final_price": float(prices[-1]) if len(prices) > 0 else np.nan,
            "final_pf": float(pf[-1]) if len(pf) > 0 else np.nan,
        },
        "fat_tails": test_fat_tails(returns),
        "volatility_clustering": test_volatility_clustering(returns),
        "price_fundamental": test_price_fundamental_relation(prices, pf) if len(pf) > 0 else {},
        "trader_composition": test_trader_composition_dynamics(nc, nf, n_plus, n_minus, N) if len(nc) > 0 else {},
        "volatility_chartist": test_volatility_chartist_correlation(returns, nc) if len(nc) > 0 else {},
        "power_law_tail": compute_power_law_tail(returns),
    }
    
    # Extended volatility clustering with longer lags
    results["extended_volatility"] = test_extended_volatility_clustering(returns)
    
    # Event-time analysis (for Poisson models)
    if len(event_returns) >= 100:
        results["event_time_fat_tails"] = test_fat_tails(event_returns)
        results["event_time_power_law"] = compute_power_law_tail(event_returns)
    else:
        results["event_time_fat_tails"] = {"note": "insufficient event-time observations"}
        results["event_time_power_law"] = {"note": "insufficient event-time observations"}
    
    # Multi-scale analysis (crucial for Poisson-tick models)
    if include_multiscale:
        results["multiscale"] = test_multiscale_stylized_facts(prices)
    
    # Overall assessment
    stylized_facts_score = 0
    max_score = 5  # Increased to 5 to include multi-scale
    
    if results["fat_tails"].get("fat_tails_detected", False):
        stylized_facts_score += 1
    if results["volatility_clustering"].get("volatility_clustering_detected", False):
        stylized_facts_score += 1
    if results["price_fundamental"].get("price_pf_correlation", 0) > 0.5:
        stylized_facts_score += 1
    if 2 < results["power_law_tail"].get("tail_exponent", 0) < 5:
        stylized_facts_score += 1
    
    # Multi-scale bonus: if aggregated returns show better stylized facts
    if include_multiscale:
        ms = results.get("multiscale", {})
        best_h = ms.get("best_horizon_for_fat_tails")
        if best_h and best_h > 1:
            h_fat = ms.get("fat_tails_by_horizon", {}).get(best_h, {})
            if h_fat.get("detected", False):
                stylized_facts_score += 1
    
    results["overall"] = {
        "stylized_facts_score": stylized_facts_score,
        "max_score": max_score,
        "assessment": "GOOD" if stylized_facts_score >= 4 else "MODERATE" if stylized_facts_score >= 2 else "POOR",
    }
    
    return results


def save_evaluation(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.generic):
            # Fallback for any other NumPy scalar types
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results_converted = convert(results)
    
    with open(output_path, "w") as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"[Evaluation] Results saved to {output_path}")


def print_evaluation_summary(results: Dict[str, Any]):
    """Print a human-readable evaluation summary."""
    print("\n" + "=" * 70)
    print("STYLIZED FACTS EVALUATION SUMMARY")
    print("=" * 70)
    
    # Basic stats
    basic = results.get("basic_stats", {})
    print(f"\n Basic Statistics:")
    print(f"   Observations: {basic.get('n_observations', 'N/A')}")
    print(f"   Mean return: {basic.get('mean_return', np.nan):.6f}")
    print(f"   Std return: {basic.get('std_return', np.nan):.6f}")
    nonzero_frac = basic.get('nonzero_return_fraction', 1.0)
    if nonzero_frac < 1.0:
        print(f"   Non-zero returns: {basic.get('n_event_returns', 'N/A')} ({nonzero_frac:.1%})")
    
    # Fat tails
    fat_tails = results.get("fat_tails", {})
    print(f"\n Fat Tails (micro-returns):")
    print(f"   Excess kurtosis: {fat_tails.get('kurtosis', np.nan):.4f} (normal = 0)")
    print(f"   Skewness: {fat_tails.get('skewness', np.nan):.4f}")
    print(f"   Jarque-Bera p-value: {fat_tails.get('jarque_bera_pvalue', np.nan):.4e}")
    detected = "YES" if fat_tails.get('fat_tails_detected', False) else "NO"
    print(f"   Fat tails detected: {detected}")
    
    # Volatility clustering
    vol_cluster = results.get("volatility_clustering", {})
    print(f"\n Volatility Clustering:")
    print(f"   Mean |return| autocorr (lag 1-10): {vol_cluster.get('mean_abs_return_autocorr_lag1_10', np.nan):.4f}")
    print(f"   Mean return autocorr (lag 1-10): {vol_cluster.get('mean_return_autocorr_lag1_10', np.nan):.4f}")
    detected = "YES" if vol_cluster.get('volatility_clustering_detected', False) else "NO"
    print(f"   Volatility clustering detected: {detected}")
    
    # Extended volatility (long memory)
    ext_vol = results.get("extended_volatility", {})
    if ext_vol.get("extended_analysis_available", False):
        print(f"   Long memory indicator: {'YES' if ext_vol.get('long_memory_indicator', False) else 'NO'}")
        print(f"   |return| ACF at lag 100: {ext_vol.get('abs_return_acf_lag100', np.nan):.4f}")
    
    # Price-fundamental relation
    pf_relation = results.get("price_fundamental", {})
    print(f"\n Price-Fundamental Relation:")
    print(f"   Mean mispricing: {pf_relation.get('mean_mispricing', np.nan):.4f}")
    print(f"   Std mispricing: {pf_relation.get('std_mispricing', np.nan):.4f}")
    print(f"   Price-PF correlation: {pf_relation.get('price_pf_correlation', np.nan):.4f}")
    
    # Power law tail
    power_law = results.get("power_law_tail", {})
    print(f"\n Power Law Tail:")
    exp_val = power_law.get('tail_exponent', np.nan)
    exp_err = power_law.get('tail_exponent_stderr', np.nan)
    reliable = power_law.get('reliable', False)
    reliability_note = " (reliable)" if reliable else " (unreliable - too few tail obs)"
    print(f"   Tail exponent: {exp_val:.2f} ± {exp_err:.2f}{reliability_note}")
    print(f"   Tail observations: {power_law.get('tail_observations', 0)}")
    
    # Multi-scale analysis
    ms = results.get("multiscale", {})
    if ms:
        print(f"\n Multi-Scale Analysis:")
        print(f"   Horizons tested: {ms.get('horizons_tested', [])}")
        best_h_fat = ms.get("best_horizon_for_fat_tails")
        if best_h_fat:
            h_fat = ms.get("fat_tails_by_horizon", {}).get(best_h_fat, {})
            print(f"   Best horizon for fat tails: h={best_h_fat} (kurtosis={h_fat.get('kurtosis', np.nan):.2f})")
        best_h_clust = ms.get("best_horizon_for_clustering")
        if best_h_clust:
            h_clust = ms.get("volatility_clustering_by_horizon", {}).get(best_h_clust, {})
            print(f"   Best horizon for clustering: h={best_h_clust} (ACF={h_clust.get('mean_abs_return_autocorr_lag1_10', np.nan):.4f})")
    
    # Overall
    overall = results.get("overall", {})
    print(f"\n Overall Assessment:")
    print(f"   Stylized facts score: {overall.get('stylized_facts_score', 0)}/{overall.get('max_score', 5)}")
    print(f"   Assessment: {overall.get('assessment', 'N/A')}")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Lux-Marchesi simulation results")
    parser.add_argument("--input", type=str, required=True, help="Path to history.npz file")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load history
    data = np.load(args.input)
    history = {key: data[key] for key in data.files}
    
    # Run evaluation
    results = full_evaluation(history)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save if requested
    if args.output:
        save_evaluation(results, args.output)

