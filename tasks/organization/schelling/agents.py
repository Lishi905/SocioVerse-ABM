"""
P — the resident population: two groups placed on a grid, leaving some cells empty
so unhappy residents have somewhere to move. Group membership is the only attribute
the homophily rule cares about.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Resident:
    agent_id: int
    group: int          # 1 or 2
    pos: Tuple[int, int]


@dataclass
class Population:
    residents: List[Resident]
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.residents)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    p = cfg["population"]
    w, h = p["width"], p["height"]
    n = int(p["density"] * w * h)

    cells = rng.sample([(x, y) for x in range(w) for y in range(h)], n)
    residents = [
        Resident(agent_id=i, group=(1 if rng.random() < p["minority_fraction"] else 2), pos=cells[i])
        for i in range(n)
    ]
    return Population(residents=residents, width=w, height=h)
