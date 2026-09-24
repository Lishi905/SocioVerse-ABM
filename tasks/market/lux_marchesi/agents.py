"""
P — the trader population. Each trader holds one of three stances:
  - fundamentalist: trades on the gap between price and fundamental value
  - optimist (noise trader): buys, expecting the trend to continue
  - pessimist (noise trader): sells
The emergent dynamics come from noise traders herding between optimist/pessimist.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

FUNDAMENTALIST, OPTIMIST, PESSIMIST = "fundamentalist", "optimist", "pessimist"
STANCES = (FUNDAMENTALIST, OPTIMIST, PESSIMIST)


@dataclass
class Trader:
    agent_id: int
    stance: str


@dataclass
class Population:
    traders: List[Trader]

    def __len__(self) -> int:
        return len(self.traders)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    p = cfg["population"]
    stances = (
        [FUNDAMENTALIST] * p["num_fundamentalists"]
        + [OPTIMIST] * p["num_optimists"]
        + [PESSIMIST] * p["num_pessimists"]
    )
    traders = [Trader(agent_id=i, stance=s) for i, s in enumerate(stances)]
    return Population(traders=traders)
