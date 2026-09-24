# -*- coding: utf-8 -*-
"""
Lux-Marchesi Model Visualization

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from typing import Dict, Optional, Tuple, List
import os

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')


def set_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.0,
    })


def plot_price_series(prices: np.ndarray, pf: np.ndarray, 
                      ax: Optional[plt.Axes] = None,
                      title: str = "Price vs Fundamental Value") -> plt.Axes:
    """
    Plot price and fundamental value time series.
    
    Args:
        prices: Array of market prices
        pf: Array of fundamental values
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    t = np.arange(len(prices))
    
    ax.plot(t, prices, 'b-', alpha=0.8, label='Market Price', linewidth=0.8)
    # Use a solid line for the fundamental value to avoid the visual impression of
    # "discontinuities" caused by dashed linestyles.
    ax.plot(t, pf, 'r-', alpha=0.8, label='Fundamental Value', linewidth=1.0)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_returns(returns: np.ndarray, ax: Optional[plt.Axes] = None,
                 title: str = "Log Returns") -> plt.Axes:
    """
    Plot return time series.
    
    Args:
        returns: Array of returns
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    
    t = np.arange(len(returns))
    
    ax.plot(t, returns, 'k-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Return')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_return_distribution(returns: np.ndarray, ax: Optional[plt.Axes] = None,
                            title: str = "Return Distribution") -> plt.Axes:
    """
    Plot return distribution with normal comparison.
    
    Args:
        returns: Array of returns
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histogram
    n_bins = min(100, len(returns) // 20)
    counts, bins, patches = ax.hist(returns, bins=n_bins, density=True, 
                                    alpha=0.7, color='steelblue', 
                                    edgecolor='white', label='Empirical')
    
    # Normal fit
    mu, std = np.mean(returns), np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 200)
    normal_pdf = stats.norm.pdf(x, mu, std)
    ax.plot(x, normal_pdf, 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={std:.4f})')
    
    # Stats annotation
    kurtosis = stats.kurtosis(returns)
    skewness = stats.skew(returns)
    stats_text = f'Kurtosis: {kurtosis:.2f}\nSkewness: {skewness:.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_qq(returns: np.ndarray, ax: Optional[plt.Axes] = None,
            title: str = "Q-Q Plot (Normal)") -> plt.Axes:
    """
    Plot Q-Q plot against normal distribution.
    
    Args:
        returns: Array of returns
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_autocorrelation(series: np.ndarray, max_lag: int = 50,
                         ax: Optional[plt.Axes] = None,
                         title: str = "Autocorrelation") -> plt.Axes:
    """
    Plot autocorrelation function.
    
    Args:
        series: Time series
        max_lag: Maximum lag to plot
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    n = len(series)
    series_centered = series - np.mean(series)
    var = np.var(series)
    
    if var == 0:
        autocorr = np.ones(min(max_lag + 1, n))
    else:
        autocorr = np.zeros(min(max_lag + 1, n))
        autocorr[0] = 1.0
        for lag in range(1, len(autocorr)):
            autocorr[lag] = np.sum(series_centered[lag:] * series_centered[:-lag]) / ((n - lag) * var)
    
    lags = np.arange(len(autocorr))
    
    ax.bar(lags, autocorr, color='steelblue', alpha=0.7, width=0.8)
    
    # Confidence bounds
    ci = 2 / np.sqrt(n)
    ax.axhline(y=ci, color='r', linestyle='--', alpha=0.5, label=f'95% CI (±{ci:.3f})')
    ax.axhline(y=-ci, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_clustering(returns: np.ndarray, max_lag: int = 50,
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot autocorrelation of absolute and squared returns.
    
    Args:
        returns: Array of returns
        max_lag: Maximum lag to plot
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    n = len(returns)
    
    # Compute autocorrelations
    def compute_acf(series, max_lag):
        series_centered = series - np.mean(series)
        var = np.var(series)
        if var == 0:
            return np.ones(min(max_lag + 1, len(series)))
        acf = np.zeros(min(max_lag + 1, len(series)))
        acf[0] = 1.0
        for lag in range(1, len(acf)):
            acf[lag] = np.sum(series_centered[lag:] * series_centered[:-lag]) / ((len(series) - lag) * var)
        return acf
    
    acf_returns = compute_acf(returns, max_lag)
    acf_abs = compute_acf(np.abs(returns), max_lag)
    acf_squared = compute_acf(returns ** 2, max_lag)
    
    lags = np.arange(len(acf_returns))
    
    ax.plot(lags, acf_returns, 'b-', label='Returns', marker='o', markersize=3)
    ax.plot(lags, acf_abs, 'r-', label='|Returns|', marker='s', markersize=3)
    ax.plot(lags, acf_squared, 'g-', label='Returns²', marker='^', markersize=3)
    
    # Confidence bounds
    ci = 2 / np.sqrt(n)
    ax.axhline(y=ci, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-ci, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Volatility Clustering: ACF of Returns, |Returns|, and Returns²')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_trader_composition(nc: np.ndarray, nf: np.ndarray,
                           n_plus: np.ndarray, n_minus: np.ndarray,
                           N: int, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot trader composition over time.
    
    Args:
        nc: Array of chartist counts
        nf: Array of fundamentalist counts
        n_plus: Array of optimist counts
        n_minus: Array of pessimist counts
        N: Total agent count
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    t = np.arange(len(nc))
    
    # Stacked area plot
    ax.fill_between(t, 0, nf / N * 100, alpha=0.5, label='Fundamentalists', color='blue')
    ax.fill_between(t, nf / N * 100, (nf + n_minus) / N * 100, alpha=0.5, 
                   label='Pessimistic Chartists', color='red')
    ax.fill_between(t, (nf + n_minus) / N * 100, 100, alpha=0.5, 
                   label='Optimistic Chartists', color='green')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Population Share (%)')
    ax.set_title('Trader Composition Over Time')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_opinion_index(n_plus: np.ndarray, n_minus: np.ndarray, nc: np.ndarray,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot opinion index (sentiment) over time.
    
    Args:
        n_plus: Array of optimist counts
        n_minus: Array of pessimist counts
        nc: Array of chartist counts
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    
    opinion_index = (n_plus - n_minus) / np.maximum(nc, 1)
    t = np.arange(len(opinion_index))
    
    ax.plot(t, opinion_index, 'purple', alpha=0.7, linewidth=0.8)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.fill_between(t, opinion_index, 0, where=(opinion_index >= 0), 
                   alpha=0.3, color='green', label='Bullish')
    ax.fill_between(t, opinion_index, 0, where=(opinion_index < 0), 
                   alpha=0.3, color='red', label='Bearish')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Opinion Index')
    ax.set_title('Market Sentiment (Opinion Index = (n+ - n-) / nc)')
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_volatility_vs_chartists(returns: np.ndarray, nc: np.ndarray,
                                  window: int = 100,
                                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot rolling volatility vs chartist population.
    
    Args:
        returns: Array of returns
        nc: Array of chartist counts (aligned with prices, so len = len(returns) + 1)
        window: Rolling window size
        ax: Optional axes to plot on
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Compute rolling volatility
    rolling_vol = np.array([np.std(returns[max(0, i-window):i+1]) 
                           for i in range(len(returns))])
    
    # Align nc with returns
    nc_aligned = nc[1:len(returns)+1] if len(nc) > len(returns) else nc[:len(returns)]
    
    # Scatter plot
    ax.scatter(nc_aligned, rolling_vol, alpha=0.3, s=5, c='steelblue')
    
    # Add trend line
    z = np.polyfit(nc_aligned, rolling_vol, 1)
    p = np.poly1d(z)
    x_line = np.linspace(nc_aligned.min(), nc_aligned.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.2e})')
    
    # Correlation
    corr = np.corrcoef(nc_aligned, rolling_vol)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Number of Chartists')
    ax.set_ylabel(f'Rolling Volatility (window={window})')
    ax.set_title('Volatility vs Chartist Population')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_comprehensive_figure(history: Dict[str, np.ndarray], 
                                 params: Dict = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive multi-panel figure with all key plots.
    
    Args:
        history: Dictionary with simulation history
        params: Optional parameters dictionary
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    prices = np.array(history.get("price", []))
    pf = np.array(history.get("pf", []))
    nc = np.array(history.get("nc", []))
    nf = np.array(history.get("nf", []))
    n_plus = np.array(history.get("n_plus", []))
    n_minus = np.array(history.get("n_minus", []))
    
    N = params.get("N", 500) if params else 500
    
    # Compute returns
    returns = np.diff(np.log(prices[prices > 0]))
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Row 1: Price series (full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_price_series(prices, pf, ax=ax1)
    
    # Row 2: Returns (full width)
    ax2 = fig.add_subplot(gs[1, :])
    plot_returns(returns, ax=ax2)
    
    # Row 3: Return distribution and Q-Q plot
    ax3a = fig.add_subplot(gs[2, 0])
    plot_return_distribution(returns, ax=ax3a)
    
    ax3b = fig.add_subplot(gs[2, 1])
    plot_qq(returns, ax=ax3b)
    
    # Row 4: Volatility clustering
    ax4 = fig.add_subplot(gs[3, :])
    plot_volatility_clustering(returns, ax=ax4)
    
    # Row 5: Trader composition and Opinion index
    ax5a = fig.add_subplot(gs[4, 0])
    plot_trader_composition(nc, nf, n_plus, n_minus, N, ax=ax5a)
    
    ax5b = fig.add_subplot(gs[4, 1])
    plot_opinion_index(n_plus, n_minus, nc, ax=ax5b)
    
    # Row 6: Volatility vs Chartists
    ax6 = fig.add_subplot(gs[5, :])
    if len(returns) > 100:
        plot_volatility_vs_chartists(returns, nc, ax=ax6)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for this plot', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Volatility vs Chartist Population')
    
    # Main title
    # Note: mode is not in LuxMarchesiParams, need to pass separately or detect from data
    mode = "ABM"
    if params:
        mode = params.get("mode", "ABM")
        # If mode not explicitly passed, try to detect from data patterns
        if mode == "ABM" and params.get("llm_model") is not None:
            mode = "LLM"
    fig.suptitle(f'Lux-Marchesi Model Analysis ({mode.upper()} Mode)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Figure saved to {save_path}")
    
    return fig


def create_comparison_figure(history_abm: Dict[str, np.ndarray],
                              history_llm: Dict[str, np.ndarray],
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison figure between ABM and LLM modes.
    
    Args:
        history_abm: ABM simulation history
        history_llm: LLM simulation history
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    for col, (history, mode) in enumerate([(history_abm, 'ABM'), (history_llm, 'LLM')]):
        prices = np.array(history.get("price", []))
        pf = np.array(history.get("pf", []))
        returns = np.diff(np.log(prices[prices > 0])) if len(prices) > 1 else np.array([])
        
        # Price series
        if len(prices) > 0:
            axes[0, col].plot(prices, 'b-', alpha=0.8, linewidth=0.5, label='Price')
            axes[0, col].plot(pf, 'r--', alpha=0.8, linewidth=0.8, label='Fundamental')
            axes[0, col].set_title(f'{mode}: Price vs Fundamental')
            axes[0, col].legend()
            axes[0, col].grid(True, alpha=0.3)
        
        # Return distribution
        if len(returns) > 10:
            axes[1, col].hist(returns, bins=50, density=True, alpha=0.7, color='steelblue')
            mu, std = np.mean(returns), np.std(returns)
            x = np.linspace(returns.min(), returns.max(), 100)
            axes[1, col].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
            kurtosis = stats.kurtosis(returns)
            axes[1, col].set_title(f'{mode}: Returns (Kurt={kurtosis:.2f})')
            axes[1, col].grid(True, alpha=0.3)
        
        # Autocorrelation
        if len(returns) > 50:
            plot_autocorrelation(np.abs(returns), max_lag=30, ax=axes[2, col], 
                               title=f'{mode}: ACF of |Returns|')
    
    fig.suptitle('ABM vs LLM Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Comparison figure saved to {save_path}")
    
    return fig


def _as_float_array(x) -> np.ndarray:
    """Convert input to 1D float numpy array (safe for missing keys)."""
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _compute_log_change(series: np.ndarray, lag_steps: int) -> np.ndarray:
    """
    Compute log-change over a fixed lag:
        log(x[t+lag]) - log(x[t])
    Invalid (non-positive) points are dropped.
    """
    s = _as_float_array(series)
    if s.size == 0:
        return s
    lag = int(lag_steps)
    if lag <= 0:
        lag = 1
    if s.size <= lag:
        return np.array([], dtype=float)

    a = s[:-lag]
    b = s[lag:]
    valid = (a > 0) & (b > 0)
    if not np.any(valid):
        return np.array([], dtype=float)
    return np.log(b[valid]) - np.log(a[valid])


def plot_fig16_style(returns: np.ndarray, nc: np.ndarray, N: int,
                     ax_returns: plt.Axes = None, ax_chartist: plt.Axes = None,
                     title: str = "Fig.16-style: Returns & Chartist Fraction") -> Tuple[plt.Axes, plt.Axes]:
    """
    Create a Fig.16-style plot from the paper: returns (top) + chartist fraction (bottom).
    
    This shows the co-movement of volatility bursts and chartist population rises.
    
    Args:
        returns: Array of returns
        nc: Array of chartist counts (len = len(returns) + 1)
        N: Total agent count
        ax_returns: Optional axes for returns plot
        ax_chartist: Optional axes for chartist fraction plot
        title: Plot title
        
    Returns:
        Tuple of (ax_returns, ax_chartist)
    """
    if ax_returns is None or ax_chartist is None:
        fig, (ax_returns, ax_chartist) = plt.subplots(2, 1, figsize=(14, 6), 
                                                       sharex=True, height_ratios=[1, 1])
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    t = np.arange(len(returns))
    
    # Top: Returns
    ax_returns.plot(t, returns, 'k-', linewidth=0.5, alpha=0.8)
    ax_returns.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax_returns.set_ylabel('Log Returns')
    ax_returns.grid(True, alpha=0.3)
    
    # Bottom: Chartist fraction
    chartist_frac = nc[1:len(returns)+1] / N if len(nc) > len(returns) else nc[:len(returns)] / N
    ax_chartist.plot(t, chartist_frac * 100, 'b-', linewidth=0.8, alpha=0.8)
    ax_chartist.fill_between(t, 0, chartist_frac * 100, alpha=0.3, color='blue')
    ax_chartist.set_xlabel('Time Step')
    ax_chartist.set_ylabel('Chartist Fraction (%)')
    ax_chartist.set_ylim(0, 100)
    ax_chartist.grid(True, alpha=0.3)
    
    return ax_returns, ax_chartist


def plot_log_log_tail_ccdf(returns: np.ndarray, ax: plt.Axes = None,
                           title: str = "Log-Log CCDF of |Returns|") -> plt.Axes:
    """
    Plot complementary CDF of |returns| on log-log scale for tail analysis.
    
    A power-law tail appears as a straight line on this plot.
    
    Args:
        returns: Array of returns
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    abs_returns = np.abs(returns)
    abs_returns = abs_returns[abs_returns > 0]  # Filter zeros for log scale
    
    if len(abs_returns) < 100:
        ax.text(0.5, 0.5, 'Insufficient non-zero returns', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    # Sort and compute empirical CCDF
    sorted_returns = np.sort(abs_returns)[::-1]  # Descending
    n = len(sorted_returns)
    ccdf = np.arange(1, n + 1) / n
    
    ax.loglog(sorted_returns, ccdf, 'b.', alpha=0.5, markersize=2, label='Empirical')
    
    # Fit power-law to tail (top 5%)
    tail_idx = int(0.05 * n)
    if tail_idx > 10:
        tail_x = sorted_returns[:tail_idx]
        tail_y = ccdf[:tail_idx]
        
        # Linear regression in log-log space
        log_x = np.log(tail_x)
        log_y = np.log(tail_y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        
        # Plot fitted line
        fit_x = np.logspace(np.log10(tail_x.min()), np.log10(tail_x.max()), 50)
        fit_y = np.exp(intercept) * fit_x ** slope
        ax.loglog(fit_x, fit_y, 'r-', linewidth=2, 
                  label=f'Power-law fit (α ≈ {-slope:.2f})')
    
    ax.set_xlabel('|Return|')
    ax.set_ylabel('CCDF P(X > x)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    return ax


def plot_extended_acf(returns: np.ndarray, max_lag: int = 200,
                      ax: plt.Axes = None,
                      title: str = "Extended ACF: |Returns| and Returns²") -> plt.Axes:
    """
    Plot autocorrelation of |returns| and returns² at extended lags.
    
    This shows the long-memory property emphasized in the paper.
    
    Args:
        returns: Array of returns
        max_lag: Maximum lag to plot
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    n = len(returns)
    effective_max_lag = min(max_lag, n // 5)
    
    if effective_max_lag < 10:
        ax.text(0.5, 0.5, 'Insufficient data for ACF', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    def compute_acf(series, max_lag):
        series_centered = series - np.mean(series)
        var = np.var(series)
        if var == 0:
            return np.ones(min(max_lag + 1, len(series)))
        acf = np.zeros(min(max_lag + 1, len(series)))
        acf[0] = 1.0
        for lag in range(1, len(acf)):
            acf[lag] = np.sum(series_centered[lag:] * series_centered[:-lag]) / ((len(series) - lag) * var)
        return acf
    
    abs_acf = compute_acf(np.abs(returns), effective_max_lag)
    sq_acf = compute_acf(returns ** 2, effective_max_lag)
    
    lags = np.arange(len(abs_acf))
    
    ax.plot(lags, abs_acf, 'r-', label='|Returns|', linewidth=1.5, alpha=0.8)
    ax.plot(lags, sq_acf, 'b--', label='Returns²', linewidth=1.5, alpha=0.8)
    
    # Confidence bounds
    ci = 2 / np.sqrt(n)
    ax.axhline(y=ci, color='gray', linestyle=':', alpha=0.5, label=f'95% CI (±{ci:.3f})')
    ax.axhline(y=-ci, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_ed_decomposition(history: Dict[str, np.ndarray], params: Dict = None,
                          ax: plt.Axes = None,
                          title: str = "ED Decomposition: |ED_c| vs |ED_f|") -> plt.Axes:
    """
    Plot the decomposition of excess demand into chartist and fundamentalist components.
    
    This diagnostic shows which mechanism is driving price dynamics:
    - When |ED_c| dominates: chartist speculation drives volatility (Fig.16-like)
    - When |ED_f| dominates: fundamentalist mean-reversion dominates
    
    Args:
        history: Simulation history dictionary
        params: Parameters dictionary (needs tc, gamma)
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    
    prices = np.array(history.get("price", []))
    pf = np.array(history.get("pf", []))
    n_plus = np.array(history.get("n_plus", []))
    n_minus = np.array(history.get("n_minus", []))
    nf = np.array(history.get("nf", []))
    
    if len(prices) < 2 or len(pf) < 2:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Get parameters
    tc = params.get("tc", 0.001) if params else 0.001
    gamma = params.get("gamma", 0.01) if params else 0.01
    
    # Compute ED components
    ed_c = (n_plus - n_minus) * tc
    ed_f = nf * gamma * (pf - prices) / prices
    
    # Use rolling average for smoother visualization
    window = min(100, len(prices) // 50)
    if window > 1:
        abs_ed_c = np.convolve(np.abs(ed_c), np.ones(window)/window, mode='valid')
        abs_ed_f = np.convolve(np.abs(ed_f), np.ones(window)/window, mode='valid')
        t = np.arange(len(abs_ed_c))
    else:
        abs_ed_c = np.abs(ed_c)
        abs_ed_f = np.abs(ed_f)
        t = np.arange(len(abs_ed_c))
    
    ax.plot(t, abs_ed_c, 'b-', alpha=0.7, linewidth=0.8, label='|ED_c| (chartist)')
    ax.plot(t, abs_ed_f, 'r-', alpha=0.7, linewidth=0.8, label='|ED_f| (fundamentalist)')
    
    # Add ratio annotation
    mean_ratio = np.mean(abs_ed_c) / np.mean(abs_ed_f) if np.mean(abs_ed_f) > 0 else np.inf
    frac_c_dom = np.mean(abs_ed_c > abs_ed_f)
    ax.text(0.02, 0.95, f'Mean |ED_c|/|ED_f| = {mean_ratio:.3f}\n|ED_c| > |ED_f|: {frac_c_dom:.1%}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('|Excess Demand|')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_burst_substeps(history: Dict[str, np.ndarray], ax: plt.Axes = None,
                        title: str = "Burst Sub-stepping Triggers") -> plt.Axes:
    """
    Visualize when burst sub-stepping was triggered.
    
    Args:
        history: Simulation history dictionary (needs "burst_substeps" key)
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 2))
    
    burst_substeps = np.array(history.get("burst_substeps", []))
    
    if len(burst_substeps) == 0 or np.max(burst_substeps) == 0:
        ax.text(0.5, 0.5, 'No burst sub-stepping triggered', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    t = np.arange(len(burst_substeps))
    triggered = burst_substeps > 0
    
    # Plot as vertical bars where triggered
    ax.fill_between(t, 0, triggered.astype(float), alpha=0.6, color='orange', 
                   label=f'Triggered ({triggered.sum()} steps)')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Burst Active')
    ax.set_title(title)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_multiscale_returns_comparison(prices: np.ndarray, 
                                       horizons: List[int] = None,
                                       ax: plt.Axes = None,
                                       title: str = "Return Distribution by Aggregation Horizon") -> plt.Axes:
    """
    Compare return distributions at different aggregation horizons.
    
    Args:
        prices: Array of prices
        horizons: Aggregation horizons (default: [1, 10, 50])
        ax: Optional axes to plot on
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if horizons is None:
        horizons = [1, 10, 50]
    
    prices = prices[prices > 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(horizons)))
    
    for h, color in zip(horizons, colors):
        if h >= len(prices):
            continue
        
        agg_returns = np.log(prices[h:]) - np.log(prices[:-h])
        
        if len(agg_returns) < 50:
            continue
        
        # Standardize for comparison
        std_returns = (agg_returns - np.mean(agg_returns)) / np.std(agg_returns)
        
        # Histogram
        n_bins = min(80, len(std_returns) // 20)
        counts, bins, _ = ax.hist(std_returns, bins=n_bins, density=True, 
                                   alpha=0.4, color=color, label=f'h={h}')
    
    # Normal reference
    x = np.linspace(-5, 5, 200)
    ax.plot(x, stats.norm.pdf(x), 'k--', linewidth=2, label='Normal')
    
    ax.set_xlabel('Standardized Return')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.grid(True, alpha=0.3)
    
    return ax


def create_paper_aligned_figure(history: Dict[str, np.ndarray],
                                params: Dict = None,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive figure with paper-aligned visualizations.
    
    Includes:
    - Fig.16-style returns + chartist fraction panel
    - ED decomposition (|ED_c| vs |ED_f|) to show driving mechanism
    - Log-log CCDF tail plot
    - Extended ACF plot
    - Multi-scale return distributions
    - Burst sub-stepping trigger visualization (if enabled)
    
    Args:
        history: Dictionary with simulation history
        params: Optional parameters dictionary
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    prices = np.array(history.get("price", []))
    nc = np.array(history.get("nc", []))
    nf = np.array(history.get("nf", []))
    burst_substeps = np.array(history.get("burst_substeps", []))
    
    N = params.get("N", 500) if params else 500
    if len(nc) > 0 and len(nf) > 0:
        N = int(nc[0] + nf[0])
    
    # Compute returns
    returns = np.diff(np.log(prices[prices > 0]))
    
    # Check if burst sub-stepping was used
    has_burst_data = len(burst_substeps) > 0 and np.max(burst_substeps) > 0
    
    # Create figure with adaptive layout
    n_rows = 6 if has_burst_data else 5
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    row_idx = 0
    
    # Row 1: Fig.16-style panel (returns)
    ax1a = fig.add_subplot(gs[row_idx, :])
    if len(returns) > 0 and len(nc) > 0:
        t = np.arange(len(returns))
        ax1a.plot(t, returns, 'k-', linewidth=0.5, alpha=0.8)
        ax1a.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1a.set_ylabel('Log Returns')
    ax1a.set_title('Returns (Fig.16-style)', fontsize=11)
    ax1a.grid(True, alpha=0.3)
    row_idx += 1
    
    # Row 2: Chartist fraction
    ax1b = fig.add_subplot(gs[row_idx, :], sharex=ax1a)
    if len(nc) > 0:
        chartist_frac = nc / N
        t = np.arange(len(chartist_frac))
        ax1b.plot(t, chartist_frac * 100, 'b-', linewidth=0.8, alpha=0.8)
        ax1b.fill_between(t, 0, chartist_frac * 100, alpha=0.3, color='blue')
    ax1b.set_xlabel('Time Step')
    ax1b.set_ylabel('Chartist Fraction (%)')
    ax1b.set_ylim(0, 100)
    ax1b.grid(True, alpha=0.3)
    row_idx += 1
    
    # Row 3: ED decomposition (key diagnostic)
    ax_ed = fig.add_subplot(gs[row_idx, :])
    plot_ed_decomposition(history, params, ax=ax_ed)
    row_idx += 1
    
    # Row 4: Burst sub-stepping (if used)
    if has_burst_data:
        ax_burst = fig.add_subplot(gs[row_idx, :])
        plot_burst_substeps(history, ax=ax_burst)
        row_idx += 1
    
    # Row 5: Log-log CCDF and Extended ACF
    ax2a = fig.add_subplot(gs[row_idx, 0])
    plot_log_log_tail_ccdf(returns, ax=ax2a)
    
    ax2b = fig.add_subplot(gs[row_idx, 1])
    plot_extended_acf(returns, max_lag=200, ax=ax2b)
    row_idx += 1
    
    # Row 6: Multi-scale returns and Q-Q
    ax3a = fig.add_subplot(gs[row_idx, 0])
    plot_multiscale_returns_comparison(prices, horizons=[1, 10, 50, 100], ax=ax3a)
    
    ax3b = fig.add_subplot(gs[row_idx, 1])
    if len(returns) > 0:
        plot_qq(returns, ax=ax3b, title='Q-Q Plot (Normal)')
    
    # Main title
    mode = params.get("mode", "ABM") if params else "ABM"
    fig.suptitle(f'Paper-Aligned Analysis: Lux-Marchesi ({mode.upper()} Mode)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Paper-aligned figure saved to {save_path}")
    
    return fig


def save_legacy_abm_style_figures(
    history: Dict[str, np.ndarray],
    params: Optional[Dict] = None,
    save_dir: Optional[str] = None,
    dpi: int = 300,
) -> Dict[str, str]:
    """
    Save a set of figures consistent with the legacy `Lux-Marchesi-ABM` repo.

    This reproduces the following PNG names / contents (as in `Lux-Marchesi-ABM/main.py`):
    - agents.png
    - pi_values.png
    - prices.png
    - relative_changes_of_market_price.png
    - relative_changes_of_fundamental_value.png
    - relative_changes_of_fundamental_value_and_price.png

    Notes on the "relative changes" plots:
    - The legacy implementation uses a 1-time-unit lag: `lag_steps = int(1 / delta_t)`
    - And computes: `log(x[t+lag]) - log(x[t])`

    Args:
        history: Simulation history dict (Lux_Marchesi format).
        params: Optional params dict; should contain `dt` for correct lag conversion.
        save_dir: Directory to save figures into (created if missing). If None, uses cwd.
        dpi: Output dpi (legacy uses 300).

    Returns:
        Mapping from figure key to absolute saved file path.
    """
    out_dir = os.path.abspath(save_dir) if save_dir else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # Pull series (Lux_Marchesi names already match legacy labels closely)
    nf = _as_float_array(history.get("nf"))
    n_plus = _as_float_array(history.get("n_plus"))
    n_minus = _as_float_array(history.get("n_minus"))
    price = _as_float_array(history.get("price"))
    pf = _as_float_array(history.get("pf"))

    pi_minus_plus = _as_float_array(history.get("pi_minus_plus"))
    pi_plus_minus = _as_float_array(history.get("pi_plus_minus"))
    pi_plus_f = _as_float_array(history.get("pi_plus_f"))
    pi_f_plus = _as_float_array(history.get("pi_f_plus"))
    pi_minus_f = _as_float_array(history.get("pi_minus_f"))
    pi_f_minus = _as_float_array(history.get("pi_f_minus"))

    # 1-time-unit lag (legacy uses int(1 / delta_t))
    dt = None
    if params:
        dt = params.get("dt", None)
    try:
        dt = float(dt) if dt is not None else None
    except Exception:
        dt = None
    if dt is None or dt <= 0:
        # fall back to the legacy default delta_t used in many configs
        dt = 0.002
    # Match legacy implementation: `one = int(1 / delta_t)`
    lag_steps = max(1, int(1.0 / dt))

    changes_price = _compute_log_change(price, lag_steps)
    changes_pf = _compute_log_change(pf, lag_steps)

    saved: Dict[str, str] = {}

    # Use rc_context with defaults so it stays consistent even if `set_style()` was called before.
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        # agents.png
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        ax.plot(nf, label="fundamentalists")
        ax.plot(n_plus, label="noise optimists")
        ax.plot(n_minus, label="noise pessimists")
        ax.legend()
        fig_path = os.path.join(out_dir, "agents.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["agents"] = fig_path

        # pi_values.png
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        ax.plot(pi_minus_plus, label="pi_minus_plus")
        ax.plot(pi_plus_minus, label="pi_plus_minus")
        ax.plot(pi_plus_f, label="pi_plus_f")
        ax.plot(pi_f_plus, label="pi_f_plus")
        ax.plot(pi_minus_f, label="pi_minus_f")
        ax.plot(pi_f_minus, label="pi_f_minus")
        ax.legend()
        fig_path = os.path.join(out_dir, "pi_values.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["pi_values"] = fig_path

        # prices.png
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        ax.plot(price, label="market price")
        ax.plot(pf, label="real value")
        ax.legend()
        fig_path = os.path.join(out_dir, "prices.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["prices"] = fig_path

        # relative_changes_of_market_price.png
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        ax.plot(changes_price, label="market price")
        ax.legend()
        fig_path = os.path.join(out_dir, "relative_changes_of_market_price.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["relative_changes_of_market_price"] = fig_path

        # relative_changes_of_fundamental_value.png
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        ax.plot(changes_pf, label="fundamental value")
        ax.legend()
        fig_path = os.path.join(out_dir, "relative_changes_of_fundamental_value.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["relative_changes_of_fundamental_value"] = fig_path

        # relative_changes_of_fundamental_value_and_price.png
        fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)
        ax.plot(changes_pf, label="fundamental value")
        ax.plot(changes_price, label="market price")
        ax.legend()
        fig_path = os.path.join(out_dir, "relative_changes_of_fundamental_value_and_price.png")
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["relative_changes_of_fundamental_value_and_price"] = fig_path

    return saved


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Lux-Marchesi simulation results")
    parser.add_argument("--input", type=str, required=True, help="Path to history.npz file")
    parser.add_argument("--output", type=str, default="lux_marchesi_analysis.png", 
                       help="Output figure path")
    
    args = parser.parse_args()
    
    # Load history
    data = np.load(args.input)
    history = {key: data[key] for key in data.files}
    
    # Create comprehensive figure
    fig = create_comprehensive_figure(history, save_path=args.output)
    
    print(f"[Done] Figure saved to {args.output}")

