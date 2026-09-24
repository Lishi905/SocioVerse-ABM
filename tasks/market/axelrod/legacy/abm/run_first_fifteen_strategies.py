#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Axelrod first tournament (15 strategies, 200-turn matches, self-play enabled).

Roster (Axelrod 1980 entrants + Anonymous):
    1. Tit For Tat (Rapoport)
    2. Tideman and Chieruzzi
    3. Nydegger
    4. Grofman
    5. Shubik
    6. Stein and Rapoport
    7. Friedman (Grudger)
    8. Davis
    9. Graaskamp
    10. Downing
    11. Feld
    12. Joss
    13. Tullock
    14. Anonymous (Name withheld)
    15. Random

Matches run for 200 rounds by default. The script emits a CSV ranking table and a
bar chart consistent with the broader ABM workflow, complete with Axelrod's progress
bar during tournament execution.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import axelrod as axl
import matplotlib.pyplot as plt
import numpy as np

from run_experiments import build_rankings, plot_top_average_scores

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "outputs_first_fifteen"


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
    ("Anonymous (Name withheld)", "First by Anonymous"),
    ("Random", "Random"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 15 Axelrod (1980) strategies including Anonymous; 200 turns per matchup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=200,
        help="Rounds per match (default: 200).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Tournament repetitions (Axelrod ran the full round robin 5 times).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (set None for randomised runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV / PNG artifacts are stored.",
    )
    return parser.parse_args()


def resolve_strategy_classes() -> List[Tuple[str, type]]:
    """Resolve display labels to concrete Axelrod strategy classes."""
    name_to_class = {cls().name: cls for cls in axl.axelrod_first_strategies}

    missing = []
    resolved: List[Tuple[str, type]] = []

    for display, lib_name in TARGET_STRATEGIES:
        strategy_class = name_to_class.get(lib_name)
        if strategy_class is None:
            missing.append(lib_name)
            continue
        resolved.append((display, strategy_class))

    if missing:
        raise ValueError(
            "Unable to resolve the following Axelrod strategies: "
            + ", ".join(missing)
        )

    return resolved


def instantiate_players() -> List[axl.Player]:
    roster = []
    for display, strategy_class in resolve_strategy_classes():
        player = strategy_class()
        player.name = display
        player.classifier["name"] = display
        roster.append(player)
    return roster


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "pair"


def compute_defection_rates(results, turns: int, repetitions: int) -> np.ndarray:
    """Return a matrix of player-1 defection rates (%) for each pairing."""
    total_rounds = turns * repetitions
    cooperation = np.asarray(results.cooperation, dtype=float)
    defections = total_rounds - cooperation
    return (defections / total_rounds) * 100.0


def compute_cooperation_rates(results, turns: int, repetitions: int) -> np.ndarray:
    """Return a matrix of player-1 cooperation rates (%) for each pairing."""
    total_rounds = turns * repetitions
    cooperation = np.asarray(results.cooperation, dtype=float)
    return (cooperation / total_rounds) * 100.0


def compute_score_percent(results) -> np.ndarray:
    """
    Return a matrix of player-1 average score per turn scaled to 0-100.

    Scale by the mutual-cooperation reward (R=3) so that all-cooperate pairings
    map to 100; values above 100 are clipped for readability.
    """
    payoff_matrix = np.asarray(results.payoff_matrix, dtype=float)  # per-turn payoff
    coop_reward = 3.0
    scaled = (payoff_matrix / coop_reward) * 100.0
    return np.clip(scaled, 0.0, 100.0)


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, cbar_label: str, output_path: Path, cmap: str = "viridis"):
    """Render a labeled heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Player 2")
    ax.set_ylabel("Player 1")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text_color = "white" if matrix[i, j] > 50 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_action_timeline(pair_actions: list[tuple], player_labels: tuple[str, str], turns: int, output_path: Path):
    """Plot per-round actions (C/D) for a single matchup."""
    rounds = np.arange(1, turns + 1)
    actions1 = [1 if a1 == axl.Action.C else 0 for a1, _ in pair_actions]
    actions2 = [1 if a2 == axl.Action.C else 0 for _, a2 in pair_actions]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.step(rounds, actions1, where="mid", label=player_labels[0], color="steelblue", marker="o")
    ax.step(rounds, actions2, where="mid", label=player_labels[1], color="darkorange", marker="o")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Defect", "Cooperate"])
    ax.set_xlabel("Round")
    ax.set_title(f"Action timeline: {player_labels[0]} vs {player_labels[1]}")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()

    players = instantiate_players()
    name_to_class = {name: cls for name, cls in resolve_strategy_classes()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AXELROD FIRST-TOURNAMENT (15 STRATEGIES, 200 TURNS)")
    print("=" * 80)
    print("Roster              :", [player.name for player in players])
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
    matches_per_player = len(players)  # includes self-play
    ranking_df["paper_total_score_per_run"] = (
        ranking_df["avg_score_per_turn"] * args.turns * matches_per_player
    )
    ranking_df["paper_total_score_all_runs"] = (
        ranking_df["paper_total_score_per_run"] * args.repetitions
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"axelrod_first_15_{args.turns}t"

    csv_path = output_dir / f"{prefix}_rankings_{timestamp}.csv"
    ranking_df.to_csv(csv_path, index=False)

    plot_path = output_dir / f"{prefix}_bar_{timestamp}.png"
    plot_top_average_scores(ranking_df, str(plot_path), top_n=len(players))

    # New visualizations modeled on the paper figure
    defection_matrix = compute_defection_rates(results, args.turns, args.repetitions)
    cooperation_matrix = compute_cooperation_rates(results, args.turns, args.repetitions)
    score_matrix = compute_score_percent(results)
    heatmap_labels = [player.name for player in players]

    defection_plot = plot_heatmap(
        defection_matrix,
        heatmap_labels,
        "Player 1 defection rate (%)",
        "Defection rate (%)",
        output_dir / f"{prefix}_defection_heatmap_{timestamp}.png",
        cmap="RdYlGn_r",  # low defection = green
    )
    cooperation_plot = plot_heatmap(
        cooperation_matrix,
        heatmap_labels,
        "Player 1 cooperation rate (%)",
        "Cooperation rate (%)",
        output_dir / f"{prefix}_cooperation_heatmap_{timestamp}.png",
        cmap="Greens",  # high cooperation = green
    )
    score_plot = plot_heatmap(
        score_matrix,
        heatmap_labels,
        "Player 1 accrued scores (scaled 0-100)",
        "Scaled avg score (per turn)",
        output_dir / f"{prefix}_score_heatmap_{timestamp}.png",
    )

    # Action timeline for the top two ranked strategies
    top_two = ranking_df.head(2)["strategy"].tolist()
    if len(top_two) == 2 and all(name in name_to_class for name in top_two):
        p1 = name_to_class[top_two[0]]()
        p2 = name_to_class[top_two[1]]()
        p1.name = top_two[0]
        p2.name = top_two[1]
        match = axl.Match((p1, p2), turns=args.turns, seed=args.seed)
        actions = match.play()
        timeline_path = output_dir / f"{prefix}_timeline_{_slugify(top_two[0])}_vs_{_slugify(top_two[1])}_{timestamp}.png"
        timeline_plot = plot_action_timeline(actions, (top_two[0], top_two[1]), args.turns, timeline_path)
    else:
        timeline_plot = None

    print("\nFinal ranking (avg score / turn, total score across runs):")
    for _, row in ranking_df.iterrows():
        print(
            f"  #{int(row['rank']):<2} {row['strategy']:<25} "
            f"{row['avg_score_per_turn']:.3f} "
            f"(total≈{row['paper_total_score_all_runs']:.1f})"
        )

    print("\nArtifacts:")
    print(f"  - Rankings CSV : {csv_path}")
    print(f"  - Ranking plot : {plot_path}")
    print(f"  - Defection heatmap : {defection_plot}")
    print(f"  - Cooperation heatmap: {cooperation_plot}")
    print(f"  - Score heatmap     : {score_plot}")
    if timeline_plot:
        print(f"  - Action timeline   : {timeline_plot}")
    print("\nDone!")


if __name__ == "__main__":
    main()
