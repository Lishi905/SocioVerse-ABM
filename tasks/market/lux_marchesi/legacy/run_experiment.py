#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lux-Marchesi Model Experiment Runner
====================================
Convenience script to run ABM/LLM experiments with full analysis pipeline.

Usage:
    python run_experiment.py --mode abm --T 50000
    python run_experiment.py --mode llm --T 10000 --model gpt-4o-2024-08-06
    python run_experiment.py --compare  # Run both modes and compare
"""

import argparse
import os
import sys
import json
import numpy as np
from datetime import datetime
from dataclasses import replace

# Ensure local modules resolve when running with `python -m ...`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from lux_marchesi_model import LuxMarchesiModel, LuxMarchesiParams


def run_single_experiment(mode: str, params: LuxMarchesiParams, 
                          llm_model: str = "gpt-4o-2024-08-06",
                          output_dir: str = "results/Lux_Marchesi",
                          llm_decision_interval: int = 50,
                          llm_cache_duration: int = 100,
                          llm_max_agents: int = 20,
                          llm_rate_limit: float = 0.0,
                          llm_sync_mode: bool = True,
                          llm_sync_timeout: float = 120.0,
                          llm_concurrent_batch_size: int = 10,
                          llm_batch_delay: float = 0.5,
                          llm_debug: bool = False):
    """Run a single experiment with full analysis.
    
    Args:
        mode: "abm" or "llm"
        params: Model parameters
        llm_model: LLM model name for LLM mode
        output_dir: Output directory for results
        llm_decision_interval: Steps between LLM decision rounds
        llm_cache_duration: Steps to cache LLM decisions
        llm_max_agents: Max agents to process per LLM round
        llm_rate_limit: Minimum seconds between LLM requests (0 = no limit)
        llm_sync_mode: Wait for LLM responses before proceeding
        llm_sync_timeout: Timeout for sync mode waiting
        llm_concurrent_batch_size: Max concurrent API calls within a batch
        llm_batch_delay: Seconds to wait between sub-batches
        llm_debug: Enable LLM debug output
    """
    
    # Create output directory
    if mode == "llm":
        exp_dir = os.path.join(output_dir, llm_model, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        exp_dir = os.path.join(output_dir, mode, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Lux-Marchesi Experiment ({mode.upper()} Mode)")
    print("=" * 70)
    print(f"Output directory: {exp_dir}")
    print(f"Parameters: N={params.N}, T={params.T}, dt={params.dt}, seed={params.seed}")
    print(f"Price Update Mode: {params.price_update_mode}")
    if mode == "llm":
        print(f"LLM Model: {llm_model}")
        print(f"LLM Decision Interval: {llm_decision_interval} steps")
        print(f"LLM Cache Duration: {llm_cache_duration} steps")
        print(f"LLM Max Agents/Tick: {llm_max_agents}")
        print(f"LLM Rate Limit: {llm_rate_limit}s between requests")
        print(f"LLM Concurrent Batch Size: {llm_concurrent_batch_size} (max parallel API calls)")
        print(f"LLM Batch Delay: {llm_batch_delay}s between sub-batches")
        print(f"LLM Sync Mode: {'ON (wait for responses)' if llm_sync_mode else 'OFF (async)'}")
    print("=" * 70)
    
    # Run simulation
    print("\n[1/4] Running simulation...")
    model = LuxMarchesiModel(params, mode=mode, llm_model=llm_model)
    
    # Configure LLM parameters if in LLM mode (MUST be before run())
    if mode == "llm":
        model.llm_decision_tick_interval = llm_decision_interval
        model.llm_cache_duration = llm_cache_duration
        model.max_llm_agents_per_tick = llm_max_agents
        model.min_seconds_between_requests = llm_rate_limit
        model.llm_sync_mode = llm_sync_mode  # Sync mode: wait for responses
        model.llm_sync_timeout = llm_sync_timeout
        model.llm_concurrent_batch_size = llm_concurrent_batch_size
        model.llm_batch_delay = llm_batch_delay
        model.llm_debug = llm_debug
        print(f"[LLM Config] interval={llm_decision_interval}, cache={llm_cache_duration}, "
              f"max_agents={llm_max_agents}, concurrent_batch={llm_concurrent_batch_size}, "
              f"batch_delay={llm_batch_delay}s, sync={llm_sync_mode}")
    
    history = model.run(progress_bar=True)
    
    # Save results
    print("\n[2/4] Saving results...")
    model.save_results(exp_dir)
    
    # Run evaluation
    print("\n[3/4] Running evaluation...")
    eval_results = None
    try:
        from dataclasses import asdict
        from evaluation import full_evaluation, print_evaluation_summary, save_evaluation
        eval_results = full_evaluation(history, asdict(params))
        print_evaluation_summary(eval_results)
        save_evaluation(eval_results, os.path.join(exp_dir, "evaluation.json"))
    except Exception as e:
        print(f"[Evaluation skipped] {e}")
    
    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    try:
        from dataclasses import asdict
        from visualization import (create_comprehensive_figure, save_legacy_abm_style_figures,
                                   create_paper_aligned_figure)
        fig_path = os.path.join(exp_dir, "analysis.png")
        # Add mode and llm_model to params dict for visualization title
        params_dict = asdict(params)
        params_dict["mode"] = mode
        params_dict["llm_model"] = llm_model if mode == "llm" else None
        create_comprehensive_figure(history, params_dict, save_path=fig_path)
        # Legacy figures compatible with `Lux-Marchesi-ABM` repo outputs
        save_legacy_abm_style_figures(history, params_dict, save_dir=exp_dir)
        # Paper-aligned figure (Fig.16-style + tail/ACF plots)
        paper_fig_path = os.path.join(exp_dir, "analysis_paper.png")
        create_paper_aligned_figure(history, params_dict, save_path=paper_fig_path)
    except Exception as e:
        print(f"[Visualization skipped] {e}")
    
    # Print LLM statistics if in LLM mode
    if mode == "llm":
        print("\n" + "-" * 70)
        print("LLM Statistics Summary")
        print("-" * 70)
        stats = model.llm_control_stats
        print(f"  Requests sent: {stats['requests_sent']}")
        print(f"  Responses received: {stats['responses_received']}")
        print(f"  LLM decisions applied: {stats['llm_decisions']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Errors: {stats['errors']}")
        
        # Calculate effective LLM coverage
        total_decisions = stats['llm_decisions'] + stats['cache_hits']
        if total_decisions > 0:
            llm_ratio = stats['llm_decisions'] / total_decisions
            print(f"  LLM decision ratio: {llm_ratio:.1%} (vs cache)")
        
        # Token usage statistics (actual from API)
        usage = model.llm_usage_total
        print(f"\n  Token Usage (actual):")
        print(f"    Prompt tokens: {usage['prompt_tokens']:,}")
        print(f"    Completion tokens: {usage['completion_tokens']:,}")
        print(f"    Total tokens: {usage['total_tokens']:,}")
        
        if usage['num_successful'] > 0:
            avg_tokens = usage['total_tokens'] / usage['num_successful']
            print(f"    Avg tokens/request: {avg_tokens:.1f}")
        
        # Estimate API cost based on actual token usage
        # GPT-4o pricing: $2.50/1M input, $10.00/1M output (as of 2024)
        input_cost = usage['prompt_tokens'] / 1_000_000 * 2.50
        output_cost = usage['completion_tokens'] / 1_000_000 * 10.00
        total_cost = input_cost + output_cost
        print(f"\n  Estimated API Cost (GPT-4o pricing):")
        print(f"    Input cost: ${input_cost:.4f}")
        print(f"    Output cost: ${output_cost:.4f}")
        print(f"    Total cost: ${total_cost:.4f}")
        
        print(f"\n  Usage files saved: llm_usage.jsonl, llm_usage_total.json")
    
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print(f"Results saved to: {exp_dir}")
    print("=" * 70)
    
    return model, history, eval_results, exp_dir


def run_comparison_experiment(params: LuxMarchesiParams,
                               llm_model_name: str = "gpt-4o-2024-08-06",
                               output_dir: str = "results/Lux_Marchesi",
                               llm_decision_interval: int = 50,
                               llm_cache_duration: int = 100,
                               llm_max_agents: int = 20,
                               llm_rate_limit: float = 0.0,
                               llm_concurrent_batch_size: int = 10,
                               llm_batch_delay: float = 0.5,
                               llm_T_ratio: float = 1.0):
    """Run ABM and LLM experiments and compare results.
    
    Args:
        params: Model parameters
        llm_model_name: LLM model name for LLM mode
        output_dir: Output directory for results
        llm_decision_interval: Steps between LLM decision rounds
        llm_cache_duration: Steps to cache LLM decisions
        llm_max_agents: Max agents to process per LLM round
        llm_rate_limit: Minimum seconds between LLM API requests
        llm_concurrent_batch_size: Max concurrent API calls within a batch
        llm_batch_delay: Seconds to wait between sub-batches
        llm_T_ratio: Ratio of LLM simulation length to ABM (default 1.0 = same length)
    """
    
    # Create comparison output directory
    comp_dir = os.path.join(output_dir, "comparison", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(comp_dir, exist_ok=True)
    
    print("=" * 70)
    print("Lux-Marchesi ABM vs LLM Comparison Experiment")
    print("=" * 70)
    print(f"ABM Time steps: {params.T}")
    llm_T = int(params.T * llm_T_ratio)
    print(f"LLM Time steps: {llm_T} (ratio: {llm_T_ratio})")
    print(f"LLM Model: {llm_model_name}")
    print(f"LLM Decision Interval: {llm_decision_interval} steps")
    print(f"LLM Cache Duration: {llm_cache_duration} steps")
    print(f"LLM Max Agents/Tick: {llm_max_agents}")
    print("=" * 70)
    
    # Run ABM
    print("\n>>> Running ABM simulation...")
    abm_model = LuxMarchesiModel(params, mode="abm")
    history_abm = abm_model.run(progress_bar=True)
    
    # Save ABM results
    abm_dir = os.path.join(comp_dir, "abm")
    os.makedirs(abm_dir, exist_ok=True)
    abm_model.save_results(abm_dir)
    
    # Run LLM (with configurable T for cost efficiency)
    print("\n>>> Running LLM simulation...")
    llm_params = replace(params, T=llm_T)
    
    llm_model = LuxMarchesiModel(llm_params, mode="llm", llm_model=llm_model_name)
    # Configure LLM-specific parameters
    llm_model.llm_decision_tick_interval = llm_decision_interval
    llm_model.llm_cache_duration = llm_cache_duration
    llm_model.max_llm_agents_per_tick = llm_max_agents
    llm_model.min_seconds_between_requests = llm_rate_limit
    llm_model.llm_concurrent_batch_size = llm_concurrent_batch_size
    llm_model.llm_batch_delay = llm_batch_delay
    print(f"[LLM Config] Parameters applied: interval={llm_decision_interval}, cache={llm_cache_duration}, "
          f"max_agents={llm_max_agents}, concurrent_batch={llm_concurrent_batch_size}, "
          f"batch_delay={llm_batch_delay}s, rate_limit={llm_rate_limit}s")
    
    history_llm = llm_model.run(progress_bar=True)
    
    # Save LLM results
    llm_dir = os.path.join(comp_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    llm_model.save_results(llm_dir)
    
    # Run evaluation for both
    print("\n>>> Evaluating results...")
    eval_abm = eval_llm = None
    try:
        from dataclasses import asdict
        from evaluation import full_evaluation, print_evaluation_summary, save_evaluation
        
        eval_abm = full_evaluation(history_abm, asdict(params))
        eval_llm = full_evaluation(history_llm, asdict(llm_params))
        
        print("\n" + "-" * 35 + " ABM " + "-" * 35)
        print_evaluation_summary(eval_abm)
        
        print("\n" + "-" * 35 + " LLM " + "-" * 35)
        print_evaluation_summary(eval_llm)
        
        # Save evaluations
        save_evaluation(eval_abm, os.path.join(abm_dir, "evaluation.json"))
        save_evaluation(eval_llm, os.path.join(llm_dir, "evaluation.json"))
    except Exception as e:
        print(f"[Evaluation skipped] {e}")
    
    # Create comparison visualization
    print("\n>>> Creating comparison visualization...")
    try:
        from visualization import create_comparison_figure, create_comprehensive_figure, save_legacy_abm_style_figures
        from dataclasses import asdict
        comp_fig_path = os.path.join(comp_dir, "comparison.png")
        create_comparison_figure(history_abm, history_llm, save_path=comp_fig_path)
        
        # Create individual analysis figures with mode information
        abm_params_dict = asdict(params)
        abm_params_dict["mode"] = "abm"
        abm_fig_path = os.path.join(abm_dir, "analysis.png")
        create_comprehensive_figure(history_abm, abm_params_dict, save_path=abm_fig_path)
        save_legacy_abm_style_figures(history_abm, abm_params_dict, save_dir=abm_dir)
        
        llm_params_dict = asdict(llm_params)
        llm_params_dict["mode"] = "llm"
        llm_params_dict["llm_model"] = llm_model_name
        llm_fig_path = os.path.join(llm_dir, "analysis.png")
        create_comprehensive_figure(history_llm, llm_params_dict, save_path=llm_fig_path)
        save_legacy_abm_style_figures(history_llm, llm_params_dict, save_dir=llm_dir)
    except Exception as e:
        print(f"[Visualization skipped] {e}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # LLM statistics
    llm_stats = llm_model.llm_control_stats
    comparison = {
        "ABM": {
            "kurtosis": eval_abm["fat_tails"]["kurtosis"] if eval_abm else None,
            "vol_cluster": eval_abm["volatility_clustering"]["mean_abs_return_autocorr_lag1_10"] if eval_abm else None,
            "score": eval_abm["overall"]["stylized_facts_score"] if eval_abm else None,
        },
        "LLM": {
            "kurtosis": eval_llm["fat_tails"]["kurtosis"] if eval_llm else None,
            "vol_cluster": eval_llm["volatility_clustering"]["mean_abs_return_autocorr_lag1_10"] if eval_llm else None,
            "score": eval_llm["overall"]["stylized_facts_score"] if eval_llm else None,
            "llm_requests": llm_stats["requests_sent"],
            "llm_decisions": llm_stats["llm_decisions"],
            "cache_hits": llm_stats["cache_hits"],
            "errors": llm_stats["errors"],
        }
    }
    
    if eval_abm and eval_llm:
        print(f"{'Metric':<30} {'ABM':>15} {'LLM':>15}")
        print("-" * 60)
        print(f"{'Excess Kurtosis':<30} {comparison['ABM']['kurtosis']:>15.4f} {comparison['LLM']['kurtosis']:>15.4f}")
        print(f"{'Vol Clustering (ACF)':<30} {comparison['ABM']['vol_cluster']:>15.4f} {comparison['LLM']['vol_cluster']:>15.4f}")
        print(f"{'Stylized Facts Score':<30} {comparison['ABM']['score']:>15}/4 {comparison['LLM']['score']:>15}/4")
    else:
        print("Evaluation unavailable; comparison summary contains null metrics.")
    
    # LLM performance summary
    print("\n" + "-" * 60)
    print("LLM Mode Statistics:")
    print(f"  API requests: {llm_stats['requests_sent']}")
    print(f"  Decisions applied: {llm_stats['llm_decisions']}")
    print(f"  Cache hits: {llm_stats['cache_hits']}")
    print(f"  Errors: {llm_stats['errors']}")
    
    # Token usage statistics
    usage = llm_model.llm_usage_total
    print(f"\n  Token Usage:")
    print(f"    Total tokens: {usage['total_tokens']:,}")
    if usage['num_successful'] > 0:
        avg_tokens = usage['total_tokens'] / usage['num_successful']
        print(f"    Avg tokens/request: {avg_tokens:.1f}")
    
    # Estimate cost
    input_cost = usage['prompt_tokens'] / 1_000_000 * 2.50
    output_cost = usage['completion_tokens'] / 1_000_000 * 10.00
    print(f"  Estimated cost: ${input_cost + output_cost:.4f}")
    
    # Save comparison summary
    with open(os.path.join(comp_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Comparison results saved to: {comp_dir}")
    print("=" * 70)
    
    return {
        "abm": {"model": abm_model, "history": history_abm, "eval": eval_abm},
        "llm": {"model": llm_model, "history": history_llm, "eval": eval_llm},
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Lux-Marchesi Model Experiment Runner")
    
    # Legacy config (Lux-Marchesi-ABM) convenience
    parser.add_argument("--config", type=str, default=None,
                       help="Path to Lux-Marchesi-ABM style configuration.json (optional)")

    # Mode
    parser.add_argument("--mode", type=str, default="abm", choices=["abm", "llm"],
                       help="Simulation mode")
    parser.add_argument("--compare", action="store_true",
                       help="Run both ABM and LLM and compare")
    
    # Model parameters
    parser.add_argument("--N", type=int, default=500, help="Number of agents")
    parser.add_argument("--T", type=int, default=50000, help="Time steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("--time_of_simulation", type=float, default=None,
                       help="Legacy total simulation time (overrides --T via T=int(time_of_simulation/dt))")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Legacy explicit initial populations (optional; if set, overrides --N)
    parser.add_argument("--num_fundamentalists", type=int, default=None, help="Legacy initial number of fundamentalists")
    parser.add_argument("--num_noise_optimist", type=int, default=None, help="Legacy initial number of noise optimists")
    parser.add_argument("--num_noise_pessimist", type=int, default=None, help="Legacy initial number of noise pessimists")
    
    # Key parameters (matching original Lux-Marchesi-ABM defaults)
    parser.add_argument("--alpha1", type=float, default=0.6, help="Herding weight")
    parser.add_argument("--alpha2", type=float, default=1.5, help="Trend weight")
    parser.add_argument("--alpha3", type=float, default=1.0, help="Profit sensitivity")
    parser.add_argument("--v1", type=float, default=2.0, help="Opinion re-evaluation frequency")
    parser.add_argument("--v2", type=float, default=0.6, help="Strategy re-evaluation frequency")
    parser.add_argument("--beta", type=float, default=4.0, help="Price adjustment speed")
    parser.add_argument("--gamma", type=float, default=0.01, help="Fundamentalist sensitivity")
    parser.add_argument("--tc", type=float, default=0.001, help="Chartist trade volume")
    parser.add_argument("--t_c", type=float, default=None, help="Legacy alias for --tc")
    parser.add_argument("--price_tick", type=float, default=0.001, help="Price tick size (legacy: delta_p)")
    parser.add_argument("--delta_p", type=float, default=None, help="Legacy alias for --price_tick")
    parser.add_argument("--price_update_mode", type=str, default="conditional_tick",
                       choices=["walrasian", "poisson", "conditional_tick", "bernoulli"],
                       help="Price update scheme (see LuxMarchesiParams.price_update_mode)")
    parser.add_argument("--trend_lookback_steps", type=int, default=10, help="dp/dt lookback in steps (paper: 10)")

    # Returns / discount (legacy uses R and s; r is kept for API completeness)
    parser.add_argument("--r", type=float, default=0.0004, help="Dividend yield (unused in legacy ABM formulas)")
    parser.add_argument("--R", type=float, default=0.0004, help="Alternative return")
    parser.add_argument("--s", type=float, default=0.75, help="Fundamental return discount")

    # Initial conditions
    parser.add_argument("--initial_price", type=float, default=100.0, help="Initial price")
    parser.add_argument("--initial_pf", type=float, default=100.0, help="Initial fundamental value")

    # Noise
    parser.add_argument("--price_noise_sigma", type=float, default=0.05, help="Price noise std")
    parser.add_argument("--value_noise_sigma", type=float, default=0.005, help="Fundamental value noise std")
    parser.add_argument("--price_change_mu", type=float, default=0.0, help="Price noise mean")
    parser.add_argument("--value_change_mu", type=float, default=0.0, help="Fundamental value noise mean")

    # Population bounds
    parser.add_argument("--min_population_frac", type=float, default=0.01, help="Minimum fraction for each agent type")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/Lux_Marchesi",
                       help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06",
                       help="LLM model for llm mode")
    
    # LLM per-agent scheduling parameters
    parser.add_argument("--llm_decision_interval", type=int, default=50,
                       help="LLM decision tick interval (steps between LLM calls)")
    parser.add_argument("--llm_cache_duration", type=int, default=100,
                       help="How many steps to cache LLM decisions")
    parser.add_argument("--llm_max_agents", type=int, default=20,
                       help="Maximum agents to process per LLM decision round")
    parser.add_argument("--llm_rate_limit", type=float, default=0.0,
                       help="Minimum seconds between LLM API requests (0 = no limit)")
    parser.add_argument("--llm_concurrent_batch_size", type=int, default=10,
                       help="Max concurrent API calls within a batch (prevents 429/timeout errors)")
    parser.add_argument("--llm_batch_delay", type=float, default=0.5,
                       help="Seconds to wait between sub-batches (prevents rate limiting)")
    parser.add_argument("--llm_sync", action="store_true", default=True,
                       help="Enable sync mode: wait for LLM responses before proceeding (default: True)")
    parser.add_argument("--llm_async", action="store_true",
                       help="Enable async mode: don't wait for LLM responses (faster but less accurate)")
    parser.add_argument("--llm_sync_timeout", type=float, default=120.0,
                       help="Timeout in seconds for waiting LLM responses in sync mode")
    parser.add_argument("--llm_debug", action="store_true",
                       help="Enable LLM debug output")
    parser.add_argument("--llm_T_ratio", type=float, default=1.0,
                       help="Ratio of LLM simulation length to ABM in comparison mode (default: 1.0)")
    
    # Paper-alignment preset and dynamics
    parser.add_argument("--preset", type=str, default=None,
                       choices=["default", "paper_fig16", "paper_stylized"],
                       help="Apply a preset configuration for paper alignment")
    parser.add_argument("--abm_update_scheme", type=str, default="synchronous",
                       choices=["synchronous", "sequential"],
                       help="ABM update scheme: synchronous (legacy) or sequential (asynchronous approximation)")
    parser.add_argument("--burst_substep_enabled", action="store_true",
                       help="Enable burst sub-stepping for high-activity periods")
    parser.add_argument("--burst_substep_threshold", type=float, default=0.1,
                       help="Activity threshold for triggering burst sub-stepping")
    parser.add_argument("--burst_substep_factor", type=int, default=5,
                       help="Number of substeps when burst sub-stepping is triggered")
    
    args = parser.parse_args()

    def _flag_provided(flag: str) -> bool:
        argv = sys.argv[1:]
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    # =========================================================================
    # Apply preset defaults (can be overridden by CLI flags)
    # =========================================================================
    # Presets set sensible defaults for reproducing paper-like behavior.
    # CLI flags always override preset values.
    
    PRESETS = {
        "paper_fig16": {
            # Fig.16-like intermittency: returns + chartist fraction co-move.
            # Key insight: After fixing the strategy switching rates (source-group scaling),
            # chartists should no longer collapse. We tune parameters to ensure:
            # - ED_c and ED_f are comparable in magnitude (avoid fundamentalist dominance)
            # - Burst sub-stepping triggers during high-activity periods
            # - Price starts at fundamental value to avoid initial mispricing bias
            "price_update_mode": "poisson",
            "abm_update_scheme": "sequential",
            "burst_substep_enabled": True,
            "burst_substep_factor": 5,
            "burst_substep_threshold": 0.05,  # Lower threshold to catch more bursts
            "initial_price": 100.0,  # Start at fundamental value
            "initial_pf": 100.0,
            "T": 100000,
            "N": 500,
            # Noise tuned to balance ED_c vs ED_f dynamics:
            # - Moderate price noise allows chartist-driven volatility to emerge
            # - Lower fundamental noise prevents extreme mispricing from dominating
            "price_noise_sigma": 0.04,
            "value_noise_sigma": 0.002,  # Reduced to limit pf drift
            # Increase t_c slightly to boost chartist contribution to ED
            "tc": 0.002,
        },
        "paper_stylized": {
            # Stylized facts: fat tails + volatility clustering + power-law tails.
            # Walrasian mode provides continuous price updates (better for tail estimation).
            # Sequential update scheme prevents artificial synchronicity.
            "price_update_mode": "walrasian",
            "abm_update_scheme": "sequential",
            "burst_substep_enabled": False,  # Not needed for walrasian
            "initial_price": 100.0,
            "initial_pf": 100.0,
            "T": 200000,  # Long run for stable statistics
            "N": 500,
            "price_noise_sigma": 0.05,
            "value_noise_sigma": 0.003,  # Moderate drift
            "tc": 0.002,  # Boost chartist ED contribution
        },
    }
    
    if args.preset and args.preset in PRESETS:
        preset = PRESETS[args.preset]
        print(f"[Preset] Applying '{args.preset}' defaults...")
        for key, value in preset.items():
            cli_flag = f"--{key}"
            if not _flag_provided(cli_flag):
                if hasattr(args, key):
                    setattr(args, key, value)
                    print(f"  {key} = {value}")
        print()

    # Aliases
    tc = args.t_c if args.t_c is not None else args.tc
    price_tick = args.delta_p if args.delta_p is not None else args.price_tick

    # Determine T from time_of_simulation if provided
    T = int(args.time_of_simulation / args.dt) if args.time_of_simulation is not None else args.T

    # Create parameters (optionally from legacy config)
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)

        cfg_dt = float(cfg.get("delta_t", args.dt))
        cfg_time = float(cfg.get("time_of_simulation", 0.0))

        dt = args.dt if _flag_provided("--dt") else cfg_dt
        if args.time_of_simulation is not None:
            cfg_time = float(args.time_of_simulation)
        T = args.T if _flag_provided("--T") else int(cfg_time / dt) if cfg_time > 0 else args.T

        tc = tc if (_flag_provided("--tc") or _flag_provided("--t_c")) else float(cfg.get("t_c", cfg.get("tc", 0.001)))
        price_tick = price_tick if (_flag_provided("--price_tick") or _flag_provided("--delta_p")) else float(cfg.get("delta_p", 0.001))

        params = LuxMarchesiParams(
            # explicit populations (legacy)
            num_fundamentalists=int(cfg.get("num_fundamentalists", 0)),
            num_noise_optimist=int(cfg.get("num_noise_optimist", 0)),
            num_noise_pessimist=int(cfg.get("num_noise_pessimist", 0)),

            # time
            T=T,
            dt=dt,
            trend_lookback_steps=args.trend_lookback_steps,

            # core parameters
            v1=float(cfg.get("v1", args.v1)) if not _flag_provided("--v1") else args.v1,
            alpha1=float(cfg.get("alpha1", args.alpha1)) if not _flag_provided("--alpha1") else args.alpha1,
            alpha2=float(cfg.get("alpha2", args.alpha2)) if not _flag_provided("--alpha2") else args.alpha2,
            v2=float(cfg.get("v2", args.v2)) if not _flag_provided("--v2") else args.v2,
            alpha3=float(cfg.get("alpha3", args.alpha3)) if not _flag_provided("--alpha3") else args.alpha3,
            R=float(cfg.get("R", args.R)) if not _flag_provided("--R") else args.R,
            s=float(cfg.get("s", args.s)) if not _flag_provided("--s") else args.s,
            beta=float(cfg.get("beta", args.beta)) if not _flag_provided("--beta") else args.beta,
            tc=tc,
            gamma=float(cfg.get("gamma", args.gamma)) if not _flag_provided("--gamma") else args.gamma,
            price_tick=price_tick,
            price_update_mode=str(cfg.get("price_update_mode", args.price_update_mode))
            if not _flag_provided("--price_update_mode") else args.price_update_mode,

            # noise
            price_change_mu=float(cfg.get("price_change_mu", args.price_change_mu)) if not _flag_provided("--price_change_mu") else args.price_change_mu,
            price_change_sigma=float(cfg.get("price_change_sigma", args.price_noise_sigma)) if not _flag_provided("--price_noise_sigma") else args.price_noise_sigma,
            value_change_mu=float(cfg.get("value_change_mu", args.value_change_mu)) if not _flag_provided("--value_change_mu") else args.value_change_mu,
            value_change_sigma=float(cfg.get("value_change_sigma", args.value_noise_sigma)) if not _flag_provided("--value_noise_sigma") else args.value_noise_sigma,

            # population bounds
            min_population_frac=float(cfg.get("minimal_agent_population_frac", args.min_population_frac)) if not _flag_provided("--min_population_frac") else args.min_population_frac,

            # initial conditions
            initial_price=float(cfg.get("market_price", args.initial_price)) if not _flag_provided("--initial_price") else args.initial_price,
            initial_pf=float(cfg.get("market_value", args.initial_pf)) if not _flag_provided("--initial_pf") else args.initial_pf,

            # seed
            seed=args.seed,

            # keep r for API completeness
            r=args.r,

            # Paper-alignment dynamics (CLI/preset overrides config)
            abm_update_scheme=args.abm_update_scheme if _flag_provided("--abm_update_scheme") else str(cfg.get("abm_update_scheme", "synchronous")),
            burst_substep_enabled=args.burst_substep_enabled if _flag_provided("--burst_substep_enabled") else bool(cfg.get("burst_substep_enabled", False)),
            burst_substep_threshold=args.burst_substep_threshold if _flag_provided("--burst_substep_threshold") else float(cfg.get("burst_substep_threshold", 0.1)),
            burst_substep_factor=args.burst_substep_factor if _flag_provided("--burst_substep_factor") else int(cfg.get("burst_substep_factor", 5)),
        )
    else:
        params = LuxMarchesiParams(
            N=args.N,
            num_fundamentalists=args.num_fundamentalists,
            num_noise_optimist=args.num_noise_optimist,
            num_noise_pessimist=args.num_noise_pessimist,
            T=T,
            dt=args.dt,
            trend_lookback_steps=args.trend_lookback_steps,
            v1=args.v1,
            alpha1=args.alpha1,
            alpha2=args.alpha2,
            alpha3=args.alpha3,
            v2=args.v2,
            r=args.r,
            R=args.R,
            s=args.s,
            beta=args.beta,
            tc=tc,
            gamma=args.gamma,
            price_tick=price_tick,
            price_update_mode=args.price_update_mode,
            price_change_mu=args.price_change_mu,
            price_change_sigma=args.price_noise_sigma,
            value_change_mu=args.value_change_mu,
            value_change_sigma=args.value_noise_sigma,
            min_population_frac=args.min_population_frac,
            initial_price=args.initial_price,
            initial_pf=args.initial_pf,
            seed=args.seed,
            # Paper-alignment dynamics
            abm_update_scheme=args.abm_update_scheme,
            burst_substep_enabled=args.burst_substep_enabled,
            burst_substep_threshold=args.burst_substep_threshold,
            burst_substep_factor=args.burst_substep_factor,
        )
    
    if args.compare:
        run_comparison_experiment(
            params, 
            llm_model_name=args.model, 
            output_dir=args.output_dir,
            llm_decision_interval=args.llm_decision_interval,
            llm_cache_duration=args.llm_cache_duration,
            llm_max_agents=args.llm_max_agents,
            llm_rate_limit=args.llm_rate_limit,
            llm_concurrent_batch_size=args.llm_concurrent_batch_size,
            llm_batch_delay=args.llm_batch_delay,
            llm_T_ratio=args.llm_T_ratio,
        )
    else:
        # Determine sync mode: --llm_async overrides default sync mode
        sync_mode = not args.llm_async
        
        run_single_experiment(
            args.mode, 
            params, 
            llm_model=args.model, 
            output_dir=args.output_dir,
            llm_decision_interval=args.llm_decision_interval,
            llm_cache_duration=args.llm_cache_duration,
            llm_max_agents=args.llm_max_agents,
            llm_rate_limit=args.llm_rate_limit,
            llm_sync_mode=sync_mode,
            llm_sync_timeout=args.llm_sync_timeout,
            llm_concurrent_batch_size=args.llm_concurrent_batch_size,
            llm_batch_delay=args.llm_batch_delay,
            llm_debug=args.llm_debug,
        )


if __name__ == "__main__":
    main()

