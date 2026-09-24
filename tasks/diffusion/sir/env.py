"""
E — the network environment for SIR rumor spreading.

Turns the monolithic ndlib trajectory (legacy/SIR_simple.py) into the kernel's
per-step f(P, E) contract: each step the env produces one Observation per agent,
the behavior function returns one Action (the agent's next state), and the env
applies them synchronously. This is what lets rule-f and llm-f be compared
agent-by-agent (the consistency score).

Maps to SocioVerse2's EnvironmentProvider: reset / observe_batch / apply / snapshot.
"""
from __future__ import annotations

from collections import Counter
from typing import List

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.diffusion.sir.agents import HEALTH_LABEL, I, R, S, Population


class SIRNetworkEnv:
    def __init__(self, population: Population, beta_eff: float, gamma: float):
        self.pop = population
        self.graph = population.graph
        self.beta_eff = beta_eff
        self.gamma = gamma
        self.t = 0
        self.health = {}
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.health = {p.agent_id: p.health for p in self.pop.personas}
        return self.snapshot()

    def spreader_neighbors(self, agent_id: int) -> int:
        return sum(1 for j in self.graph.neighbors(agent_id) if self.health[j] == I)

    def _render(self, state: str, k: int, deg: int) -> str:
        if state == S:
            return (f"You have not heard a particular rumor. {k} of your {deg} "
                    f"contacts are actively spreading it. Do you start spreading it too?")
        if state == I:
            return ("You are currently spreading a rumor. Do you keep spreading it, "
                    "or lose interest and stop?")
        return "You have already stopped caring about this rumor."

    def observe_batch(self) -> List[Observation]:
        obs = []
        for i in sorted(self.health):
            state = self.health[i]
            deg = self.graph.degree(i)
            k = self.spreader_neighbors(i)
            ctx = {
                "state": state,
                "spreader_neighbors": k,
                "degree": deg,
                "beta_eff": self.beta_eff,
                "gamma": self.gamma,
            }
            obs.append(Observation(agent_id=i, state={"health": state},
                                   context=ctx, rendered=self._render(state, k, deg)))
        return obs

    def apply(self, actions: List[Action]) -> None:
        for a in actions:
            if a.kind not in (S, I, R):
                raise ValueError(f"agent {a.agent_id}: illegal next state '{a.kind}'")
            self.health[a.agent_id] = a.kind

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        c = Counter(self.health.values())
        return {"t": self.t, "S": c.get(S, 0), "I": c.get(I, 0), "R": c.get(R, 0)}
