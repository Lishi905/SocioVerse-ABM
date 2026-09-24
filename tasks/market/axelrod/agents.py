"""
P — the tournament population. Each agent plays a fixed strategy; the population is
a mix of several strategies (multiple copies each) so the round-robin reveals which
strategies accumulate the most payoff (Axelrod's question: does cooperation pay?).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    agent_id: int
    strategy: str


@dataclass
class Population:
    agents: List[Agent]

    def __len__(self) -> int:
        return len(self.agents)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    p = cfg["population"]
    agents: List[Agent] = []
    aid = 0
    for strategy in p["strategies"]:
        for _ in range(p["copies_per_strategy"]):
            agents.append(Agent(agent_id=aid, strategy=strategy))
            aid += 1
    return Population(agents=agents)
