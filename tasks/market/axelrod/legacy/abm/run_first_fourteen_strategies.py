#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Axelrod first-tournament showcase (14 named strategies + self-play).

Strategies included (order preserved):
    1. Tit For Tat (Rapoport)
    2. Tideman and Chieruzzi
    3. Nydegger
    4. Grofman
    5. Shubik
    6. Stein and Rapoport
    7. Friedman (a.k.a. Grudger)
    8. Davis
    9. Graaskamp
    10. Downing
    11. Feld
    12. Joss
    13. Tullock
    14. Random

Each head-to-head match runs for 100 turns by default (self-play enabled),
and the script exports both a rankings CSV and a bar chart identical to the
main ABM workflow. Use this when you only need the canonical Axelrod 1980
entrants rather than the entire 200+ strategy roster.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import axelrod as axl

from run_experiments import build_rankings, plot_top_average_scores

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "outputs_first_fourteen"

# (display_label, axelrod_name)
TARGET_STRATEGIES: Sequence[Tuple[str, str]] = [
    ("Tit For Tat (Rapoport)", "Tit For Tat"),
    ("Tideman and Chieruzzi", "First by Tideman and Chieruzzi"),
    ("Nydegger", "First by Nydegger"),
    ("Grofman", "First by Grofman"),
    ("Shubik", "First by Shubik"),
    ("Stein and Rapoport", "First by Stein and Rapoport"),
    ("Friedman (Grudger)", "Grudger"),
    ("Davis", "First by Davis"),
    ("Graaskamp", "First by Graaskamp"),
    ("Downing", "First by Downing"),
    ("Feld", "First by Feld"),
    ("Joss", "First by Joss"),
    ("Tullock", "First by Tullock"),
    ("Random", "Random"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 14 classic Axelrod strategies with self-play (100 turns).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=100,
        help="Rounds per match.",
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
        help="Random seed (set None to use Axelrod's default RNG).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV and plot artifacts.",
    )
    parser.add_argument(
        "--include-anonymous",
        action="store_true",
        help="Also include 'First by Anonymous' (Axelrod's additional submission).",
    )
    return parser.parse_args()


def resolve_strategy_classes(include_anonymous: bool) -> List[Tuple[str, type]]:
    """Map display labels to concrete Axelrod Player classes."""
    name_to_class = {cls().name: cls for cls in axl.axelrod_first_strategies}

    missing = []
    resolved: List[Tuple[str, type]] = []

    for display, library_name in TARGET_STRATEGIES:
        strategy_class = name_to_class.get(library_name)
        if strategy_class is None:
            missing.append(library_name)
            continue
        resolved.append((display, strategy_class))

    if include_anonymous:
        anon_class = name_to_class.get("First by Anonymous")
        if anon_class is None:
            missing.append("First by Anonymous")
        else:
            resolved.append(("Anonymous", anon_class))

    if missing:
        raise ValueError(
            "Unable to resolve the following Axelrod strategies: "
            + ", ".join(missing)
        )

    return resolved


def instantiate_players(include_anonymous: bool) -> List[axl.Player]:
    players = []
    for display, strategy_class in resolve_strategy_classes(include_anonymous):
        player = strategy_class()
        player.name = display
        player.classifier["name"] = display
        players.append(player)
    return players


def main():
    args = parse_args()

    players = instantiate_players(include_anonymous=args.include_anonymous)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AXELROD FIRST-TOURNAMENT (14 STRATEGIES)")
    print("=" * 80)
    print("Roster              :", [player.name for player in players])
    print(f"Turns per match     : {args.turns}")
    print(f"Repetitions         : {args.repetitions}")
    print("Self matches        : enabled")
    print(f"Random seed         : {args.seed if args.seed is not None else 'random'}")
    print(f"Output directory    : {output_dir}")
    if args.include_anonymous:
        print("Extra entry         : First by Anonymous")
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
    suffix = "15_with_anonymous" if args.include_anonymous else "14_classic"
    prefix = f"axelrod_first_{suffix}_{args.turns}t"

    csv_path = output_dir / f"{prefix}_rankings_{timestamp}.csv"
    ranking_df.to_csv(csv_path, index=False)

    plot_path = output_dir / f"{prefix}_bar_{timestamp}.png"
    plot_top_average_scores(ranking_df, str(plot_path), top_n=len(players))

    print("\nFinal ranking (avg score per turn):")
    for _, row in ranking_df.iterrows():
        print(f"  #{int(row['rank']):<2} {row['strategy']:<20} {row['avg_score_per_turn']:.3f}")

    print("\nArtifacts:")
    print(f"  - Rankings CSV : {csv_path}")
    print(f"  - Ranking plot : {plot_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
