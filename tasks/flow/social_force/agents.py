"""
P — the pedestrian crowd: N agents with continuous position, velocity and a
heterogeneous desired walking speed (Helbing's Gaussian mu=1.34, sigma=0.26 m/s).
All share one goal (the exit). NumPy arrays for vectorized neighbour queries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    positions: np.ndarray    # (N, 2)
    velocities: np.ndarray   # (N, 2)
    desired_speed: np.ndarray  # (N,)
    goal: np.ndarray         # (2,)

    def __len__(self) -> int:
        return len(self.positions)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    p = cfg["population"]
    d = cfg["dynamics"]
    n = p["num_pedestrians"]

    xs = rng.uniform(p["spawn"]["x"][0], p["spawn"]["x"][1], size=n)
    ys = rng.uniform(p["spawn"]["y"][0], p["spawn"]["y"][1], size=n)
    positions = np.column_stack([xs, ys])
    velocities = np.zeros((n, 2))
    desired = np.clip(rng.normal(d["desired_speed_mu"], d["desired_speed_sigma"], size=n),
                      0.5, d["v_max"])
    return Population(positions=positions, velocities=velocities,
                      desired_speed=desired, goal=np.asarray(p["goal"], float))
