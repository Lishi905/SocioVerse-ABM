"""
P — the population of Epstein's civil-violence model: citizens, each with a private
hardship and risk aversion (so grievance and fear differ across people), plus a small
force of cops. Citizens may be quiet or actively rebelling; arrested citizens sit in
jail for a term.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Citizen:
    agent_id: int
    pos: Tuple[int, int]
    hardship: float
    risk_aversion: float
    active: bool = False
    jail_remaining: int = 0


@dataclass
class Cop:
    agent_id: int
    pos: Tuple[int, int]


@dataclass
class Population:
    citizens: List[Citizen]
    cops: List[Cop]
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.citizens) + len(self.cops)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = random.Random(seed)
    p = cfg["population"]
    w, h = p["width"], p["height"]
    n_cells = w * h
    n_cit = int(p["citizen_density"] * n_cells)
    n_cop = int(p["cop_density"] * n_cells)

    cells = rng.sample([(x, y) for x in range(w) for y in range(h)], n_cit + n_cop)
    citizens = [
        Citizen(agent_id=i, pos=cells[i], hardship=rng.random(), risk_aversion=rng.random())
        for i in range(n_cit)
    ]
    cops = [Cop(agent_id=n_cit + j, pos=cells[n_cit + j]) for j in range(n_cop)]
    return Population(citizens=citizens, cops=cops, width=w, height=h)
