#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the Axelrod ABM tournament with unified parameters and export the top-50 strategies.

Outputs (all timestamped):
* CSV with complete rankings
* JSON list of the top 50 strategy names (for LLM experiments)
* Horizontal bar chart of average score per turn for the top 50 strategies

Usage:
    python run_experiments.py [--turns 10] [--seed 123] [--processes N]
"""

import argparse
import json
import os
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import axelrod as axl
from axelrod_agent import (
    get_all_strategies,
    get_selected_strategies,
    get_short_run_strategies,
)

STRATEGY_LOADERS: dict[str, Callable[[], list[type]]] = {
    "selected": get_selected_strategies,
    "short": get_short_run_strategies,
    "all": get_all_strategies,
}


def run_tournament(
    strategy_set: str,
    turns: int,
    repetitions: int,
    processes: int | None,
    seed: int | None,
    include_self_play: bool,
):
    """Execute the round-robin tournament for the requested strategy set."""
    try:
        loader = STRATEGY_LOADERS[strategy_set]
    except KeyError:
        raise ValueError(
            f"Unknown strategy set '{strategy_set}'. Choose from: {', '.join(STRATEGY_LOADERS)}"
        )

    strategies = loader()
    players = [strategy() for strategy in strategies]

    print("=" * 80)
    print("AXELROD ABM TOURNAMENT")
    print("=" * 80)
    print(f"Total strategies     : {len(players)}")
    print(f"Strategy set         : {strategy_set}")
    print(f"Turns per match      : {turns}")
    print(f"Repetitions          : {repetitions}")
    print(f"Parallel processes   : {processes if processes else 'auto'}")
    print(f"Random seed          : {seed if seed is not None else 'random'}")
    print("=" * 80)

    edges = None
    if include_self_play:
        print("Self matches        : enabled")
        total_matches = len(players) * (len(players) - 1) // 2 + len(players)
        print(f"Total matches       : {total_matches}")
    else:
        edges = [(i, j) for i, j in combinations(range(len(players)), 2)]
        print("Self matches        : disabled")
        print(f"Total matches       : {len(edges)}")

    tournament = axl.Tournament(
        players=players,
        turns=turns,
        repetitions=repetitions,
        seed=seed,
        edges=edges,
    )

    start_time = time.time()
    results = tournament.play(progress_bar=True, processes=processes)
    elapsed = time.time() - start_time
    print(f"\nTournament finished in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    return results, players


def build_rankings(results, players) -> pd.DataFrame:
    """Create a ranking DataFrame sorted by average score per turn."""
    names = [player.name for player in players]
    modules = [player.__class__.__module__ for player in players]
    class_names = [player.__class__.__name__ for player in players]
    docstrings = [(player.__class__.__doc__ or "").strip() for player in players]

    normalised_scores = np.asarray(results.normalised_scores, dtype=float)
    avg_scores = normalised_scores.mean(axis=1)

    total_scores = np.asarray(results.scores, dtype=float).mean(axis=1)
    ranking = pd.DataFrame({
        "strategy": names,
        "avg_score_per_turn": avg_scores,
        "mean_total_score": total_scores,
        "class_module": modules,
        "class_name": class_names,
        "docstring": docstrings
    })

    ranking.sort_values(by="avg_score_per_turn", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking


def plot_top_average_scores(df: pd.DataFrame, output_path: str, top_n: int = 50):
    """Plot horizontal bar chart for top-N strategies by average score per turn."""
    limit = top_n if top_n and top_n > 0 else len(df)
    top_df = df.head(limit).iloc[::-1]  # reverse for horizontal bar

    plt.figure(figsize=(12, max(8, top_n * 0.25)))
    bars = plt.barh(top_df["strategy"], top_df["avg_score_per_turn"], color="steelblue", alpha=0.85)
    plt.xlabel("Average Score per Turn", fontsize=12, fontweight="bold")
    plt.title(f"Top {top_n} Axelrod Strategies (ABM) – Average Score per Turn", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, value in zip(bars, top_df["avg_score_per_turn"]):
        plt.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Average score plot saved to: {output_path}")


def save_top_strategies(df: pd.DataFrame, output_path: str, top_n: int = 50, turns: int = 10):
    """Save top-N strategy metadata to JSON (for LLM experiments)."""
    limit = top_n if top_n and top_n > 0 else len(df)
    top_records = []
    for _, row in df.head(limit).iterrows():
        top_records.append({
            "rank": int(row["rank"]),
            "strategy": row["strategy"],
            "avg_score_per_turn": float(row["avg_score_per_turn"]),
            "mean_total_score": float(row["mean_total_score"]),
            "class_module": row["class_module"],
            "class_name": row["class_name"],
            "docstring": row["docstring"]
        })
    payload = {"turns": turns, "strategies": top_records}

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Top {top_n} strategy metadata saved to: {output_path}")

    latest_path = output_path.with_name("abm_top_strategies_latest.json")
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Latest top strategy list refreshed at: {latest_path}")

    root_latest = Path(__file__).resolve().parent / "abm_top_strategies_latest.json"
    if root_latest.resolve() != latest_path.resolve():
        with root_latest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Root-level latest file updated at: {root_latest}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ABM Axelrod tournament and export top-50 strategies.")
    parser.add_argument("--turns", type=int, default=10, help="Turns per match (default: 10)")
    parser.add_argument("--repetitions", type=int, default=1, help="Tournament repetitions (default: 1)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--processes", type=int, default=None, help="Parallel processes for Axelrod (default: auto)")
    parser.add_argument("--top-n", type=int, default=0, help="Number of top strategies to export (default: all)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for outputs (default: current)")
    parser.add_argument(
        "--strategy-set",
        type=str,
        default="all",
        choices=tuple(STRATEGY_LOADERS.keys()),
        help="Strategy set to load: selected (50 curated), short (~200), all (~240). Default: all.",
    )
    parser.add_argument(
        "--exclude-self-play",
        action="store_true",
        help="Exclude self-play matches (default: include self-play).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results, players = run_tournament(
        strategy_set=args.strategy_set,
        turns=args.turns,
        repetitions=args.repetitions,
        processes=args.processes,
        seed=args.seed,
        include_self_play=not args.exclude_self_play,
    )

    ranking_df = build_rankings(results, players)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(args.output_dir, f"abm_rankings_{timestamp}.csv")
    ranking_df.to_csv(csv_path, index=False)
    print(f"Full ranking CSV saved to: {csv_path}")

    json_path = os.path.join(args.output_dir, f"abm_top_{args.top_n}_strategies.json")
    save_top_strategies(ranking_df, json_path, args.top_n, turns=args.turns)

    plot_path = os.path.join(args.output_dir, f"abm_top_{args.top_n}_avg_scores_{timestamp}.png")
    plot_top_average_scores(ranking_df, plot_path, args.top_n)

    print("\nSummary:")
    print(f"  Winner: {ranking_df.iloc[0]['strategy']} (avg per turn {ranking_df.iloc[0]['avg_score_per_turn']:.3f})")
    print(f"  Outputs located in: {os.path.abspath(args.output_dir)}")
