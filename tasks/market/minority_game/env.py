"""
E — the repeated Minority Game. Holds the shared winning-side history; each round the
minority side wins. After actions are in, the env awards real points, updates every
strategy's virtual score by counterfactual (did it predict the winning side?), and
appends the winning side to the history.
"""
from __future__ import annotations

import random
from typing import List

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.market.minority_game.agents import Population


class MinorityGameEnv:
    def __init__(self, population: Population, cfg: dict):
        self.M = population.memory
        self.agents = population.agents
        self.N = len(self.agents)
        self.rng = random.Random(cfg["seed"])
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        for a in self.agents:
            a.virtual_scores[:] = 0.0
            a.real_score = 0
        self.history = [self.rng.randint(0, 1) for _ in range(self.M)]
        self.attendance = []          # count choosing side 1, per round
        return self.snapshot()

    def history_index(self) -> int:
        idx = 0
        for i, bit in enumerate(self.history[-self.M:]):
            idx += int(bit) * (2 ** (self.M - 1 - i))
        return idx

    def observe_batch(self) -> List[Observation]:
        idx = self.history_index()
        recent = list(self.history[-self.M:])
        obs = []
        for a in self.agents:
            ctx = {"history_index": idx, "strategies": a.strategies,
                   "virtual_scores": a.virtual_scores, "history": recent}
            obs.append(Observation(agent_id=a.agent_id, state={"real_score": a.real_score},
                                   context=ctx, rendered=self._render(recent)))
        return obs

    def _render(self, recent) -> str:
        return (f"In the minority game, the recent winning (minority) sides were "
                f"{recent}. Which side (0 or 1) will be in the minority this round?")

    def apply(self, actions: List[Action]) -> None:
        idx = self.history_index()
        side = {a.agent_id: int(a.payload["side"]) for a in actions}
        count_b = sum(1 for s in side.values() if s == 1)
        count_a = self.N - count_b
        winning_side = 0 if count_a < count_b else 1        # minority wins
        for a in self.agents:
            if side[a.agent_id] == winning_side:
                a.real_score += 1
            for strat, k in zip(a.strategies, range(len(a.strategies))):
                if int(strat[idx]) == winning_side:
                    a.virtual_scores[k] += 1.0
        self.history.append(winning_side)
        self.attendance.append(count_b)

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        var = float(np.var(self.attendance)) if self.attendance else 0.0
        return {
            "t": self.t,
            "n": self.N,
            "attendance": self.attendance[-1] if self.attendance else None,
            "volatility": var / self.N,        # sigma^2 / N, the MG efficiency measure
        }
