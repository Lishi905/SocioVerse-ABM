#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mini Axelrod tournament restricted to five canonical Iterated Prisoner's Dilemma strategies.

Strategies (self-play included):
    1. Always Cooperate
    2. Always Defect
    3. Tit-for-Tat
    4. Grim Trigger (Grudger)
    5. Random

Outputs: rankings CSV + bar plot with progress bar during execution.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import axelrod as axl

from run_experiments import build_rankings, plot_top_average_scores

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "outputs_five_strategy_showcase"

STRATEGY_SPEC = [
    ("Always Cooperate", axl.Cooperator),
    ("Always Defect", axl.Defector),
    ("Tit-for-Tat", axl.TitForTat),
    ("Grim Trigger", axl.Grudger),
    ("Random", axl.Random),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a five-strategy ABM Axelrod tournament (self-play enabled).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--turns", type=int, default=100, help="Rounds per match.")
    parser.add_argument("--repetitions", type=int, default=1, help="Tournament repetitions.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (None uses Axelrod default).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV/plot outputs.",
    )
    return parser.parse_args()


def instantiate_players():
    players = []
    for alias, strategy_cls in STRATEGY_SPEC:
        player = strategy_cls()
        player.classifier["name"] = alias
        players.append(player)
    return players


def main():
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    players = instantiate_players()

    print("=" * 80)
    print("FIVE-STRATEGY AXELROD ABM TOURNAMENT")
    print("=" * 80)
    print("Roster              :", [alias for alias, _ in STRATEGY_SPEC])
    print(f"Turns per match     : {args.turns}")
    print(f"Repetitions         : {args.repetitions}")
    print("Self matches        : enabled")
    print(f"Random seed         : {args.seed if args.seed is not None else 'random'}")
    print(f"Output directory    : {output_dir}")
    print("=" * 80)

    tournament = axl.Tournament(
        players=players,
        turns=args.turns,
        repetitions=args.repetitions,
        seed=args.seed,
    )

    start = time.time()
    results = tournament.play(progress_bar=True)
    elapsed = time.time() - start
    print(f"\nTournament finished in {elapsed:.2f}s ({elapsed/60:.2f} min)")

    ranking_df = build_rankings(results, players)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"abm_five_strategy_{args.turns}t"

    csv_path = output_dir / f"{prefix}_rankings_{timestamp}.csv"
    ranking_df.to_csv(csv_path, index=False)

    plot_path = output_dir / f"{prefix}_bar_{timestamp}.png"
    plot_top_average_scores(ranking_df, str(plot_path), top_n=len(players))

    print("\nFinal ranking (avg score/turn):")
    for _, row in ranking_df.iterrows():
        print(f"  #{int(row['rank']):<2} {row['strategy']:<15} {row['avg_score_per_turn']:.3f}")

    print("\nArtifacts:")
    print(f"  - Rankings CSV : {csv_path}")
    print(f"  - Ranking plot : {plot_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
