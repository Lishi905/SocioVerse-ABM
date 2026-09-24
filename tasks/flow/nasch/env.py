"""
E — the circular single-lane road for NaSch.

This is a different *kind* of environment from SIR's network: a 1-D cellular
lattice with motion. It still speaks the same kernel contract
(observe_batch / apply / snapshot), which is the point of using NaSch as
a layout template — to show the standard layout survives a non-network E.

NaSch is a synchronous (parallel) update: every vehicle computes its new speed
from the *current* gaps, then all vehicles move. So one kernel step is:
  observe_batch (gaps now) -> f returns speeds -> apply (set speeds + move all).
"""
from __future__ import annotations

from typing import List

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.flow.nasch.agents import Population


class NaSchLaneEnv:
    def __init__(self, population: Population, vmax: int, randomization_prob: float):
        self.pop = population
        self.L = population.lane_length
        self.vmax = vmax
        self.p = randomization_prob
        self.t = 0
        self.vehicles = sorted(population.vehicles, key=lambda v: v.position)

    def reset(self) -> dict:
        self.t = 0
        self.vehicles = sorted(self.pop.vehicles, key=lambda v: v.position)
        return self.snapshot()

    def _gap_ahead(self, idx: int) -> int:
        """Empty cells between vehicle idx and the next vehicle ahead (ring)."""
        n = len(self.vehicles)
        if n <= 1:
            return self.L - 1
        here = self.vehicles[idx].position
        ahead = self.vehicles[(idx + 1) % n].position
        return (ahead - here - 1) % self.L

    def _render(self, v: int, gap: int) -> str:
        return (f"You are driving at speed {v} on a single-lane road; the next car "
                f"is {gap} cells ahead. vmax={self.vmax}, p={self.p}. What is your new speed?")

    def observe_batch(self) -> List[Observation]:
        obs = []
        for idx, veh in enumerate(self.vehicles):
            gap = self._gap_ahead(idx)
            ctx = {
                "current_speed": veh.speed,
                "gap_ahead": gap,
                "max_speed": self.vmax,
                "randomization_prob": self.p,
            }
            obs.append(Observation(agent_id=veh.agent_id, state={"speed": veh.speed},
                                   context=ctx, rendered=self._render(veh.speed, gap)))
        return obs

    def apply(self, actions: List[Action]) -> None:
        by_id = {a.agent_id: int(a.payload["speed"]) for a in actions}
        for veh in self.vehicles:
            new_speed = by_id[veh.agent_id]
            if not 0 <= new_speed <= self.vmax:
                raise ValueError(f"vehicle {veh.agent_id}: speed {new_speed} out of [0,{self.vmax}]")
            veh.speed = new_speed
            veh.position = (veh.position + new_speed) % self.L  # NaSch rule 4: move
        self.vehicles.sort(key=lambda v: v.position)

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        n = len(self.vehicles)
        total_speed = sum(v.speed for v in self.vehicles)
        return {
            "t": self.t,
            "n": n,
            "density": n / self.L,
            "mean_speed": total_speed / n if n else 0.0,
            "flow": total_speed / self.L,           # vehicles passing per cell per step
        }
