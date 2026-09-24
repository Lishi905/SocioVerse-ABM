#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minority Game Simulation - Main Script

A multi-agent system where N players choose between two options (A=0, B=1) each round.
The minority side wins. Players use strategies based on the history of past M winners.

This implementation uses the Mesa agent-based modeling framework.

Usage:
    python minority_game.py --N 1001 --M 6 --S 5 --steps 20000 --seed 42 --plot
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

from minority_game_model import MinorityGameModel


def parse_M_mix(m_mix_str):
    """
    Parse M-mix string like "1:100,2:100,3:100"
    
    Args:
        m_mix_str: String in format "M1:count1,M2:count2,..."
        
    Returns:
        Dictionary {M_value: count} or None if no mix specified
    """
    if not m_mix_str:
        return None
    
    M_mix = {}
    pairs = m_mix_str.split(',')
    for pair in pairs:
        M_val, count = pair.split(':')
        M_mix[int(M_val)] = int(count)
    
    return M_mix


def save_agent_stats(agent_stats, filename='agent_stats.csv'):
    """
    Save agent statistics to CSV file.
    
    Args:
        agent_stats: List of dictionaries containing agent statistics
        filename: Output CSV filename
    """
    if not agent_stats:
        print("No agent statistics to save.")
        return
    
    fieldnames = ['AgentID', 'M', 'wins', 'moves', 'win_rate', 'switches', 'switch_freq']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stats in agent_stats:
            writer.writerow(stats)
    
    print(f"Agent statistics saved to {filename}")


def plot_time_series(step_data, M, N, S, steps, seed, filename=None):
    """
    Plot A_count vs step with statistics.
    
    Args:
        step_data: DataFrame with step data
        M: Memory length (or "mixed" for heterogeneous)
        N, S, steps, seed: Model parameters for title
        filename: If provided, save to file instead of showing
    """
    steps_idx = step_data.index.values
    A_counts = step_data['A_count'].values
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps_idx, A_counts, alpha=0.7, linewidth=0.5, color='steelblue')
    plt.axhline(y=N/2, color='red', linestyle='--', alpha=0.5, label=f'N/2 = {N/2}')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Number choosing A', fontsize=12)
    plt.title(f'Minority Game: A Count vs Step (M={M}, N={N}, S={S})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    variance = np.var(A_counts)
    std = np.std(A_counts)
    mean = np.mean(A_counts)
    
    stats_text = f'Mean: {mean:.2f}\nStd: {std:.2f}\nVar: {variance:.2f}'
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()
    
    plt.close()


def plot_win_rate_by_M(agent_stats, filename='win_rate_by_M.png'):
    """
    Plot average win rate vs memory length M.
    
    Args:
        agent_stats: List of agent statistics dictionaries
        filename: Output plot filename
    """
    # Group by M
    M_to_rates = defaultdict(list)
    for stats in agent_stats:
        M_to_rates[stats['M']].append(stats['win_rate'])
    
    # Calculate statistics for each M
    M_values = sorted(M_to_rates.keys())
    means = [np.mean(M_to_rates[M]) for M in M_values]
    stds = [np.std(M_to_rates[M]) for M in M_values]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(M_values, means, yerr=stds, marker='o', markersize=8,
                capsize=5, capthick=2, linewidth=2, color='darkgreen',
                label='Mean win rate ± std')
    
    plt.xlabel('Memory Length (M)', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.title('Average Win Rate vs Memory Length', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(M_values)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Win rate plot saved to {filename}")
    plt.close()


def main():
    """Main entry point for the Minority Game simulation."""
    parser = argparse.ArgumentParser(
        description='Minority Game Simulation using Mesa Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--N', type=int, required=True,
                       help='Number of agents (should be odd to avoid ties)')
    parser.add_argument('--M', type=int, default=6,
                       help='Memory length (default if M-mix not specified)')
    parser.add_argument('--S', type=int, required=True,
                       help='Number of strategies per agent')
    parser.add_argument('--steps', type=int, required=True,
                       help='Number of simulation steps to run')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--M-mix', type=str, default=None,
                       help='Mixed M configuration, e.g., "1:100,2:100,3:100"')
    parser.add_argument('--burn-in', type=int, default=0,
                       help='Burn-in period (steps to discard from statistics)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate time series plot')
    parser.add_argument('--dump-agent-stats', action='store_true',
                       help='Export agent statistics to CSV file')
    
    args = parser.parse_args()
    
    # Parse M-mix if provided
    M_mix = parse_M_mix(args.M_mix) if args.M_mix else None
    
    # Validate parameters
    if args.N % 2 == 0:
        print(f"Warning: N={args.N} is even. Ties may occur and will be broken randomly.")
    
    if M_mix:
        total_agents = sum(M_mix.values())
        if total_agents != args.N:
            print(f"Warning: M-mix specifies {total_agents} agents but N={args.N}")
            print(f"Using M-mix total of {total_agents} agents.")
    
    # Print configuration
    print("=" * 80)
    print("MINORITY GAME SIMULATION")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  N (agents): {args.N}")
    print(f"  M (memory): {args.M}")
    print(f"  S (strategies): {args.S}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed}")
    print(f"  Burn-in: {args.burn_in}")
    if M_mix:
        print(f"  M-mix: {M_mix}")
    print()
    
    # Create and run model
    print("Initializing model...")
    model = MinorityGameModel(N=args.N, M=args.M, S=args.S, seed=args.seed, M_mix=M_mix)
    
    print("Running simulation...")
    for i in range(args.steps):
        model.step()
        if (i + 1) % 1000 == 0:
            print(f"  Step {i + 1}/{args.steps}")
    
    print("Simulation complete!")
    print()
    
    # Get and display results
    step_data = model.get_step_data(burn_in=args.burn_in)
    
    print("=" * 80)
    print(f"RESULTS (after burn-in of {args.burn_in} steps)")
    print("=" * 80)
    
    A_counts = step_data['A_count'].values
    print(f"A Count Statistics:")
    print(f"  Mean: {np.mean(A_counts):.2f}")
    print(f"  Std:  {np.std(A_counts):.2f}")
    print(f"  Var:  {np.var(A_counts):.2f}")
    print(f"  Min:  {np.min(A_counts)}")
    print(f"  Max:  {np.max(A_counts)}")
    print()
    
    # Save agent stats if requested
    if args.dump_agent_stats:
        print("Exporting agent statistics...")
        agent_stats = model.get_agent_stats(burn_in=args.burn_in)
        save_agent_stats(agent_stats)
        
        # Print M-wise summary if heterogeneous
        if M_mix:
            print()
            print("Win Rate by Memory Length (M):")
            print("-" * 40)
            M_to_rates = defaultdict(list)
            for stats in agent_stats:
                M_to_rates[stats['M']].append(stats['win_rate'])
            
            for M_val in sorted(M_to_rates.keys()):
                rates = M_to_rates[M_val]
                print(f"  M={M_val:2d}: mean={np.mean(rates):.4f}, std={np.std(rates):.4f}, n={len(rates)}")
            
            # Also create win rate plot
            plot_win_rate_by_M(agent_stats)
    
    # Generate plot if requested
    if args.plot:
        print()
        print("Generating plot...")
        M_label = "mixed" if M_mix else args.M
        filename = f"minority_game_M{M_label}_N{args.N}_S{args.S}_steps{args.steps}_seed{args.seed}.png"
        plot_time_series(step_data, M_label, args.N, args.S, args.steps, args.seed, filename)
    
    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()

