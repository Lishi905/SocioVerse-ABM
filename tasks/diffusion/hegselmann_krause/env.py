"""
E — the opinion field for Hegselmann-Krause. The "environment" each agent perceives
is the *set of other opinions* (not a spatial/network neighbourhood): at each step an
agent is shown all opinions and listens only to those within its confidence bound.

Synchronous update: everyone computes a new opinion from the current frame, then all
update together (so the behavior function returns each agent's new opinion).
"""
from __future__ import annotations

from typing import List

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.diffusion.hegselmann_krause.agents import Population


class HKOpinionEnv:
    def __init__(self, population: Population, epsilon: float):
        self.epsilon = epsilon
        self._pop = population
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.opinions = self._pop.opinions.astype(float).copy()
        return self.snapshot()

    def observe_batch(self) -> List[Observation]:
        all_ops = self.opinions
        obs = []
        for i in range(len(all_ops)):
            x = float(all_ops[i])
            within = all_ops[np.abs(all_ops - x) <= self.epsilon]
            ctx = {"opinion": x, "peers": all_ops.copy(), "epsilon": self.epsilon}
            obs.append(Observation(agent_id=i, state={"opinion": x}, context=ctx,
                                   rendered=self._render(x, within)))
        return obs

    def _render(self, x: float, within: np.ndarray) -> str:
        vals = ", ".join(f"{v:.2f}" for v in np.sort(within)[:20])
        return (f"Your opinion is {x:.2f} on a 0-1 scale. The people whose views are "
                f"close enough to yours to consider hold: [{vals}]. Your updated opinion:")

    def apply(self, actions: List[Action]) -> None:
        for a in actions:
            self.opinions[a.agent_id] = float(np.clip(a.payload["opinion"], 0.0, 1.0))

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        from tasks.diffusion.hegselmann_krause.evaluate import cluster_count
        return {
            "t": self.t,
            "n": len(self.opinions),
            "num_clusters": cluster_count(self.opinions),
            "spread": float(self.opinions.max() - self.opinions.min()),
            "variance": float(self.opinions.var()),
        }
