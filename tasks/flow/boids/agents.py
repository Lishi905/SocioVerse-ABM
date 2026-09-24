"""
P — the Boids population: N agents with a continuous 2-D position and velocity.

Positions/velocities are kept as NumPy arrays (N, 2) so the env can do vectorized
radius neighbour queries via the kernel's ContinuousField2D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    positions: np.ndarray   # (N, 2)
    velocities: np.ndarray  # (N, 2)

    def __len__(self) -> int:
        return len(self.positions)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    p = cfg["population"]
    n = p["num_boids"]

    positions = rng.uniform([0.0, 0.0], [p["width"], p["height"]], size=(n, 2))
    # random initial headings at half max speed
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    speed0 = 0.5 * cfg["dynamics"]["max_speed"]
    velocities = np.column_stack([np.cos(angles), np.sin(angles)]) * speed0
    return Population(positions=positions, velocities=velocities)
