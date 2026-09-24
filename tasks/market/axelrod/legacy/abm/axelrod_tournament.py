#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Axelrod Tournament - Main Script

A comprehensive round-robin tournament for all 200+ Iterated Prisoner's Dilemma 
strategies from the Axelrod library.

Usage:
    python axelrod_tournament.py [--turns N] [--repetitions N] [--processes N] [--strategy_set SET]
    
Strategy Sets:
    all       - All strategies that obey Axelrod's rules (~204 strategies)
    short     - Only short run time strategies (~180 strategies) 
    demo      - Demo strategies (5 strategies)
    first     - Axelrod's first tournament (15 strategies)
    basic     - Basic deterministic strategies
"""

import argparse
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axelrod as axl
from axelrod_agent import (
    get_all_strategies,
    get_short_run_strategies, 
    get_demo_strategies,
    get_axelrod_first_strategies,
    get_basic_strategies
)


def run_tournament(strategies, turns=200, repetitions=1, processes=None, seed=None):
    """
    Run a round-robin tournament with the given strategies.
    
    Args:
        strategies: List of strategy classes
        turns: Number of turns per match
        repetitions: Number of times to repeat each match
        processes: Number of parallel processes (None = auto)
        seed: Random seed for reproducibility
        
    Returns:
        axl.Tournament: Tournament object with results
    """
    print(f"\n{'='*70}")
    print(f"AXELROD ITERATED PRISONER'S DILEMMA TOURNAMENT")
    print(f"{'='*70}")
    print(f"Strategies: {len(strategies)}")
    print(f"Turns per match: {turns}")
    print(f"Repetitions: {repetitions}")
    print(f"Total matches: {len(strategies) * (len(strategies) - 1) // 2}")
    print(f"Processes: {processes if processes else 'auto'}")
    print(f"Random seed: {seed if seed else 'random'}")
    print(f"{'='*70}\n")
    
    # Create player instances
    players = [strategy() for strategy in strategies]
    
    # Create tournament
    # Note: Axelrod library uses 'processes' for parallel processing in play() method
    tournament = axl.Tournament(
        players=players,
        turns=turns,
        repetitions=repetitions,
        seed=seed
    )
    
    # Run tournament
    print("Running tournament...")
    start_time = time.time()
    results = tournament.play(progress_bar=True, processes=processes)
    elapsed_time = time.time() - start_time
    
    print(f"\nTournament completed in {elapsed_time:.2f} seconds")
    print(f"  ({elapsed_time/60:.2f} minutes)")
    
    return results


def print_rankings(results, top_n=None):
    """
    Print tournament rankings.
    
    Args:
        results: Tournament results object
        top_n: Number of top strategies to show (None = all)
    """
    # Get rankings
    ranked_names = results.ranked_names
    scores = results.scores
    
    # Calculate normalized scores (per turn)
    normalised_scores = results.normalised_scores
    
    print(f"\n{'='*70}")
    print(f"TOURNAMENT RANKINGS")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Strategy':<40} {'Score':<12} {'Avg/Turn':<10}")
    print(f"{'-'*70}")
    
    display_count = top_n if top_n else len(ranked_names)
    
    for rank, name in enumerate(ranked_names[:display_count], 1):
        idx = results.players.index(name)
        total_score = np.mean(scores[idx])
        avg_score = np.mean(normalised_scores[idx])
        print(f"{rank:<6} {name:<40} {total_score:<12.2f} {avg_score:<10.4f}")
    
    if top_n and len(ranked_names) > top_n:
        print(f"... and {len(ranked_names) - top_n} more strategies")
    
    print(f"{'='*70}\n")


def save_results_csv(results, filename='tournament_results.csv'):
    """
    Save tournament results to CSV file.
    
    Args:
        results: Tournament results object
        filename: Output CSV filename
    """
    ranked_names = results.ranked_names
    scores = results.scores
    normalised_scores = results.normalised_scores
    
    # Create dataframe
    data = []
    for rank, name in enumerate(ranked_names, 1):
        idx = results.players.index(name)
        data.append({
            'Rank': rank,
            'Strategy': name,
            'Total_Score': np.mean(scores[idx]),
            'Avg_Score_Per_Turn': np.mean(normalised_scores[idx]),
            'Std_Score': np.std(scores[idx]),
            'Median_Score': np.median(scores[idx]),
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")


def plot_rankings(results, filename='tournament_rankings.png', top_n=50):
    """
    Plot tournament rankings as a bar chart.
    
    Args:
        results: Tournament results object
        filename: Output filename
        top_n: Number of top strategies to show
    """
    ranked_names = results.ranked_names
    normalised_scores = results.normalised_scores
    
    # Get top N strategies
    top_names = ranked_names[:top_n]
    top_scores = [np.mean(normalised_scores[results.players.index(name)]) 
                  for name in top_names]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_scores, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()  # Top rank at top
    ax.set_xlabel('Average Score per Turn', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strategy', fontsize=14, fontweight='bold')
    ax.set_title(f'Axelrod Tournament Rankings (Top {top_n})', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}',
                ha='left', va='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Add rank numbers
    for i, name in enumerate(top_names):
        ax.text(-0.05, i, f'#{i+1}',
                ha='right', va='center', fontsize=9, fontweight='bold',
                transform=ax.get_yaxis_transform())
    
    # Add payoff matrix info as text box
    payoff_text = (
        "Payoff Matrix:\n"
        "R=3 (Mutual Coop)\n"
        "T=5 (Temptation)\n"
        "S=0 (Sucker)\n"
        "P=1 (Mutual Defect)"
    )
    ax.text(0.98, 0.02, payoff_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Rankings plot saved to: {filename}")
    plt.close()


def plot_payoff_matrix(results, filename='payoff_matrix.png', top_n=30):
    """
    Plot payoff matrix heatmap showing head-to-head scores.
    
    Args:
        results: Tournament results object
        filename: Output filename
        top_n: Number of top strategies to include
    """
    # Get top N strategies
    ranked_names = results.ranked_names[:top_n]
    
    # Get score matrix for these strategies
    score_matrix = np.zeros((top_n, top_n))
    for i, name1 in enumerate(ranked_names):
        idx1 = results.players.index(name1)
        for j, name2 in enumerate(ranked_names):
            idx2 = results.players.index(name2)
            score_matrix[i, j] = np.mean(results.payoff_matrix[idx1][idx2])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(score_matrix, cmap='RdYlGn', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(top_n))
    ax.set_yticks(np.arange(top_n))
    ax.set_xticklabels(ranked_names, rotation=90, fontsize=8)
    ax.set_yticklabels(ranked_names, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Score', rotation=270, labelpad=20, fontsize=12)
    
    # Labels
    ax.set_xlabel('Opponent Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Strategy', fontsize=14, fontweight='bold')
    ax.set_title(f'Head-to-Head Payoff Matrix (Top {top_n} Strategies)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Payoff matrix plot saved to: {filename}")
    plt.close()


def plot_wins_distribution(results, filename='wins_distribution.png', top_n=30):
    """
    Plot distribution of wins for top strategies.
    
    Args:
        results: Tournament results object
        filename: Output filename
        top_n: Number of top strategies to show
    """
    ranked_names = results.ranked_names[:top_n]
    
    # Get wins for each ranked strategy
    wins = []
    for name in ranked_names:
        idx = results.players.index(name)
        wins.append(results.wins[idx])
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(ranked_names))
    bars = ax.bar(x_pos, wins, color='forestgreen', alpha=0.8, edgecolor='black')
    
    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ranked_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Wins', fontsize=14, fontweight='bold')
    ax.set_title(f'Win Distribution (Top {top_n} Strategies)', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Wins distribution plot saved to: {filename}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Axelrod Tournament - All 200+ IPD Strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--turns', type=int, default=200,
                       help='Number of turns per match (default: 200)')
    parser.add_argument('--repetitions', type=int, default=1,
                       help='Number of times to repeat tournament (default: 1)')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
    parser.add_argument('--strategy_set', type=str, default='short',
                       choices=['all', 'short', 'demo', 'first', 'basic'],
                       help='Set of strategies to use (default: short)')
    parser.add_argument('--top_n', type=int, default=50,
                       help='Number of top strategies to show in plots (default: 50)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results (default: current dir)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select strategy set
    if args.strategy_set == 'all':
        strategies = get_all_strategies()
        set_name = 'All Strategies'
    elif args.strategy_set == 'short':
        strategies = get_short_run_strategies()
        set_name = 'Short Run Time Strategies'
    elif args.strategy_set == 'demo':
        strategies = get_demo_strategies()
        set_name = 'Demo Strategies'
    elif args.strategy_set == 'first':
        strategies = get_axelrod_first_strategies()
        set_name = 'Axelrod First Tournament Strategies'
    else:  # basic
        strategies = get_basic_strategies()
        set_name = 'Basic Strategies'
    
    print(f"\nStrategy Set: {set_name} ({len(strategies)} strategies)")
    
    # Run tournament
    results = run_tournament(
        strategies=strategies,
        turns=args.turns,
        repetitions=args.repetitions,
        processes=args.processes,
        seed=args.seed
    )
    
    # Print rankings
    print_rankings(results, top_n=args.top_n)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(args.output_dir, 
                                f'tournament_results_{args.strategy_set}_{timestamp}.csv')
    save_results_csv(results, csv_filename)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        
        rankings_file = os.path.join(args.output_dir, 
                                     f'rankings_{args.strategy_set}_{timestamp}.png')
        plot_rankings(results, rankings_file, top_n=args.top_n)
        
        matrix_file = os.path.join(args.output_dir, 
                                   f'payoff_matrix_{args.strategy_set}_{timestamp}.png')
        plot_payoff_matrix(results, matrix_file, top_n=min(30, len(strategies)))
        
        wins_file = os.path.join(args.output_dir, 
                                f'wins_{args.strategy_set}_{timestamp}.png')
        plot_wins_distribution(results, wins_file, top_n=min(30, len(strategies)))
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TOURNAMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Winner: {results.ranked_names[0]}")
    print(f"Total strategies: {len(strategies)}")
    print(f"Turns per match: {args.turns}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Results saved to: {csv_filename}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
