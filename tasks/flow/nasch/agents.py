"""
P — the NaSch population: N vehicles on a circular single-lane road of L cells.

Each vehicle occupies one cell and carries an integer speed in [0, vmax]. Unlike
SIR's graph nodes, the interaction structure here is *spatial and ordered*: a
vehicle only ever interacts with the one immediately ahead (its gap), so the
population is kept sorted by position.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class Vehicle:
    agent_id: int
    position: int
    speed: int


@dataclass
class Population:
    lane_length: int
    vehicles: List[Vehicle]

    def __len__(self) -> int:
        return len(self.vehicles)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    L = cfg["population"]["lane_length"]
    n = round(cfg["population"]["density"] * L)
    n = max(0, min(n, L))
    vmax = cfg["dynamics"]["vmax"]

    cells = sorted(rng.sample(range(L), n))          # distinct occupied cells
    vehicles = [
        Vehicle(agent_id=i, position=pos, speed=rng.randint(0, vmax))
        for i, pos in enumerate(cells)
    ]
    return Population(lane_length=L, vehicles=vehicles)
