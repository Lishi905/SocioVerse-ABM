"""
P — the Sugarscape population: citizens scattered on the grid, each with a vision,
a metabolism and a sugar stock (the wealth that the model's inequality is measured
on). Heterogeneous traits are the source of the emergent wealth distribution.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Citizen:
    agent_id: int
    pos: Tuple[int, int]
    vision: int
    metabolism: int
    sugar: int
    alive: bool = True


@dataclass
class Population:
    citizens: List[Citizen]

    def __len__(self) -> int:
        return len(self.citizens)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    p, d = cfg["population"], cfg["dynamics"]
    w, h, n = p["width"], p["height"], p["n_citizens"]

    cells = rng.sample([(x, y) for x in range(w) for y in range(h)], n)
    citizens = [
        Citizen(
            agent_id=i,
            pos=cells[i],
            vision=rng.randint(*d["vision_range"]),
            metabolism=rng.randint(*d["metabolism_range"]),
            sugar=rng.randint(*d["initial_sugar_range"]),
        )
        for i in range(n)
    ]
    return Population(citizens=citizens)
