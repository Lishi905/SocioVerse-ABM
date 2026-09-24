#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABM tournament restricted to the ABM Top-N strategy subset.

This script mirrors the LLM experiment configuration so both pipelines evaluate
the exact same roster (self-play enabled, 10-turn matches).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import axelrod as axl

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from top_strategy_loader import load_abm_top_strategies  # noqa: E402
from run_experiments import build_rankings, plot_top_average_scores  # noqa: E402

DEFAULT_RANKING_PATH = PROJECT_ROOT / "abm" / "abm_top_strategies_latest.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "outputs_top_subset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ABM Axelrod tournament on the ABM Top-N ranking subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ranking-path",
        type=str,
        default=str(DEFAULT_RANKING_PATH),
        help="Path to the ABM ranking artifact (JSON or CSV).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top strategies to include.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Turns per match (keep aligned with the LLM setup).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Tournament repetitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Parallel processes for the Axelrod tournament (None = auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV/JSON/plot artifacts.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render a ranking bar chart for the final results.",
    )
    return parser.parse_args()


def instantiate_players(selection: list[dict]):
    """Instantiate Axelrod strategy classes from the metadata selection."""
    players = []
    for entry in selection:
        module = importlib.import_module(entry["class_module"])
        cls = getattr(module, entry["class_name"])
        players.append(cls())
    return players


def export_summary(df, output_path: Path, turns: int):
    """Persist the subset metadata + fresh ranking to JSON."""
    payload = {
        "turns": turns,
        "strategies": [],
    }
    for _, row in df.iterrows():
        payload["strategies"].append({
            "rank": int(row["rank"]),
            "strategy": row["strategy"],
            "avg_score_per_turn": float(row["avg_score_per_turn"]),
            "mean_total_score": float(row["mean_total_score"]),
            "class_module": row["class_module"],
            "class_name": row["class_name"],
            "docstring": row["docstring"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = Path(args.ranking_path).expanduser().resolve()
    selection = load_abm_top_strategies(ranking_path, args.top_n)
    if not selection:
        raise RuntimeError("No strategies loaded from ranking artifact.")

    players = instantiate_players(selection)

    print("=" * 80)
    print("ABM TOP-N AXELROD TOURNAMENT")
    print("=" * 80)
    print(f"Ranking source     : {ranking_path}")
    print(f"Strategies         : {len(players)} (Top {args.top_n})")
    print(f"Turns per match    : {args.turns}")
    print(f"Repetitions        : {args.repetitions}")
    print(f"Self matches       : enabled")
    print(f"Processes          : {args.processes if args.processes else 'auto'}")
    print(f"Random seed        : {args.seed if args.seed is not None else 'random'}")
    print("=" * 80)

    tournament = axl.Tournament(
        players=players,
        turns=args.turns,
        repetitions=args.repetitions,
        seed=args.seed,
    )

    start = time.time()
    results = tournament.play(progress_bar=True, processes=args.processes)
    elapsed = time.time() - start
    print(f"\nTournament finished in {elapsed:.1f}s ({elapsed/60:.2f} min)")

    ranking_df = build_rankings(results, players)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"abm_top{len(players)}"

    csv_path = output_dir / f"{prefix}_rankings_{timestamp}.csv"
    ranking_df.to_csv(csv_path, index=False)

    summary_path = output_dir / f"{prefix}_summary_{timestamp}.json"
    export_summary(ranking_df, summary_path, args.turns)

    plot_path = None
    if args.plot:
        plot_path = output_dir / f"{prefix}_bar_{timestamp}.png"
        plot_top_average_scores(ranking_df, str(plot_path), top_n=len(players))

    print("\nTop 5 strategies:")
    for _, row in ranking_df.head(5).iterrows():
        print(f"  #{int(row['rank']):<2} {row['strategy']:<35} {row['avg_score_per_turn']:.4f}/turn")

    print("\nArtifacts:")
    print(f"  - Rankings CSV : {csv_path}")
    print(f"  - Summary JSON : {summary_path}")
    if plot_path:
        print(f"  - Rankings plot: {plot_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
