#!/usr/bin/env python3
"""
Headless evaluator for Boids (traditional ABM).
Computes: polarisation (alignment_order), mean_dist_to_centroid, nn_dist (mean_nnd), mean_speed.
Exports to bench_exports/boids_traditional.csv and prints one-line metrics.
"""

import os
import math
import csv
import numpy as np
from boids_core import BoidsEngine


def compute_additional_metrics(engine: BoidsEngine):
    if not engine.boids:
        return 0.0, 0.0
    # centroid
    xs = np.array([b.x for b in engine.boids], dtype=float)
    ys = np.array([b.y for b in engine.boids], dtype=float)
    cx = float(xs.mean())
    cy = float(ys.mean())
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mean_dist_to_centroid = float(dists.mean())
    # mean speed
    speeds = np.array([math.hypot(b.vx, b.vy) for b in engine.boids], dtype=float)
    mean_speed = float(speeds.mean()) if speeds.size else 0.0
    return mean_dist_to_centroid, mean_speed


def compute_alignment_order(engine: BoidsEngine) -> float:
    if not engine.boids:
        return 0.0
    velocities = np.array([[b.vx, b.vy] for b in engine.boids], dtype=float)
    norms = np.linalg.norm(velocities, axis=1)
    norms[norms == 0] = 1.0
    directions = velocities / norms[:, None]
    avg_direction = directions.mean(axis=0)
    return float(np.linalg.norm(avg_direction))


def compute_mean_nnd(engine: BoidsEngine) -> float:
    positions = np.array([[b.x, b.y] for b in engine.boids], dtype=float)
    if len(positions) < 2:
        return 0.0
    min_dists = []
    for i, pos in enumerate(positions):
        diff = positions - pos
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        min_dists.append(dist.min())
    return float(np.mean(min_dists))


def main():
    engine = BoidsEngine(width=800, height=600, num_boids=50, mode="traditional", scenario="open_field")
    steps = 100
    export_every = 1
    export_path = os.path.join("bench_exports", "boids_traditional.csv")
    os.makedirs("bench_exports", exist_ok=True)

    alignment_hist: list[float] = []
    nnd_hist: list[float] = []
    centroid_hist: list[float] = []
    speed_hist: list[float] = []

    with open(export_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "alignment_order", "mean_nnd", "mean_dist_to_centroid", "mean_speed"])

        def after_step(i: int) -> None:
            if (i + 1) % export_every == 0:
                alignment_order = compute_alignment_order(engine)
                mean_nnd = compute_mean_nnd(engine)
                mean_dist_to_centroid, mean_speed = compute_additional_metrics(engine)
                alignment_hist.append(alignment_order)
                nnd_hist.append(mean_nnd)
                centroid_hist.append(mean_dist_to_centroid)
                speed_hist.append(mean_speed)
                writer.writerow([
                    i + 1,
                    f"{alignment_order:.3f}",
                    f"{mean_nnd:.3f}",
                    f"{mean_dist_to_centroid:.3f}",
                    f"{mean_speed:.3f}",
                ])

        engine.run_steps(
            steps,
            show_progress=True,
            desc="Traditional Boids Simulation",
            after_step=after_step,
        )

        mean_alignment = float(np.mean(alignment_hist)) if alignment_hist else 0.0
        mean_nnd = float(np.mean(nnd_hist)) if nnd_hist else 0.0
        mean_dist = float(np.mean(centroid_hist)) if centroid_hist else 0.0
        mean_speed = float(np.mean(speed_hist)) if speed_hist else 0.0
        writer.writerow([
            "mean",
            f"{mean_alignment:.3f}",
            f"{mean_nnd:.3f}",
            f"{mean_dist:.3f}",
            f"{mean_speed:.3f}",
        ])

    # Gather metrics for one-line summary
    alignment_order = compute_alignment_order(engine)
    mean_nnd = compute_mean_nnd(engine)
    mean_dist_to_centroid, mean_speed = compute_additional_metrics(engine)
    mean_alignment = float(np.mean(alignment_hist)) if alignment_hist else 0.0
    mean_nnd_hist = float(np.mean(nnd_hist)) if nnd_hist else 0.0
    mean_dist_hist = float(np.mean(centroid_hist)) if centroid_hist else 0.0
    mean_speed_hist = float(np.mean(speed_hist)) if speed_hist else 0.0

    print(
        f"boids: polarisation={alignment_order:.3f}, mean_dist_to_centroid={mean_dist_to_centroid:.3f}, "
        f"nn_dist={mean_nnd:.3f}, mean_speed={mean_speed:.3f}"
    )
    print(
        f"averages over {steps} steps -> polarisation={mean_alignment:.3f}, "
        f"mean_dist_to_centroid={mean_dist_hist:.3f}, nn_dist={mean_nnd_hist:.3f}, mean_speed={mean_speed_hist:.3f}"
    )


if __name__ == "__main__":
    main()


