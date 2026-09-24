"""
P — the Minority Game population. Each agent owns S strategies; a strategy is a
lookup table of size 2^M mapping the recent winning-side history to an action (0/1).
Agents track a virtual score per strategy (how well it would have done) and pick the
best one each round.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MGAgent:
    agent_id: int
    strategies: List[np.ndarray]   # S tables, each shape (2^M,) of 0/1
    virtual_scores: np.ndarray     # (S,)
    real_score: int = 0


@dataclass
class Population:
    agents: List[MGAgent]
    memory: int                    # M

    def __len__(self) -> int:
        return len(self.agents)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    p = cfg["population"]
    m, s, n = p["memory"], p["n_strategies"], p["n_agents"]
    table_size = 2 ** m
    agents = [
        MGAgent(
            agent_id=i,
            strategies=[rng.integers(0, 2, size=table_size) for _ in range(s)],
            virtual_scores=np.zeros(s),
        )
        for i in range(n)
    ]
    return Population(agents=agents, memory=m)
