"""
E — the Schelling grid (Moore neighbourhood, toroidal). Holds who is where and the
empty cells; unhappy residents relocate to a random empty cell.

The `context` variant (ocm | lcm) changes only how the environment is *described* to
the LLM: ocm gives the raw neighbour counts, lcm renders the same situation as a short
narrative. The numeric context is identical either way, so the rule path is unaffected.
"""
from __future__ import annotations

import random
from typing import List

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.organization.schelling.agents import Population

_MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class SchellingGrid:
    def __init__(self, population: Population, cfg: dict):
        self.w, self.h = population.width, population.height
        self.threshold = cfg["dynamics"]["homophily"]
        self.context = cfg["behavior"].get("context", "ocm")
        self._pop = population
        self.rng = random.Random(cfg["seed"])
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.residents = {r.agent_id: r for r in self._pop.residents}
        self.occupied = {r.pos: r.agent_id for r in self.residents.values()}
        self.empties = {(x, y) for x in range(self.w) for y in range(self.h)} - set(self.occupied)
        return self.snapshot()

    def _neighbors(self, pos):
        x, y = pos
        for dx, dy in _MOORE:
            np = ((x + dx) % self.w, (y + dy) % self.h)
            aid = self.occupied.get(np)
            if aid is not None:
                yield self.residents[aid]

    def _counts(self, r):
        n_same = n_diff = 0
        for nb in self._neighbors(r.pos):
            if nb.group == r.group:
                n_same += 1
            else:
                n_diff += 1
        return n_same, n_diff

    def same_fraction(self, r):
        n_same, n_diff = self._counts(r)
        total = n_same + n_diff
        return (n_same / total) if total else 1.0      # isolated residents count as content

    def observe_batch(self) -> List[Observation]:
        obs = []
        for r in self.residents.values():
            n_same, n_diff = self._counts(r)
            total = n_same + n_diff
            frac = (n_same / total) if total else 1.0
            ctx = {"group": r.group, "n_same": n_same, "n_diff": n_diff,
                   "total_neighbors": total, "same_fraction": frac, "threshold": self.threshold}
            obs.append(Observation(agent_id=r.agent_id, state={"pos": r.pos}, context=ctx,
                                   rendered=self._render(ctx)))
        return obs

    def _render(self, c) -> str:
        if self.context == "lcm":
            if c["total_neighbors"] == 0:
                feel = "You live alone with no neighbours nearby."
            else:
                share = c["same_fraction"]
                vibe = ("surrounded by people like you" if share >= 0.66
                        else "in a fairly mixed area" if share >= 0.34
                        else "mostly among people unlike you")
                feel = (f"Your neighbourhood feels {vibe}: of {c['total_neighbors']} "
                        f"neighbours, {c['n_same']} share your background.")
            return feel
        # ocm: raw numeric context
        return (f"You are group {c['group']}. Among your {c['total_neighbors']} neighbours, "
                f"{c['n_same']} are your group and {c['n_diff']} are the other group.")

    def apply(self, actions: List[Action]) -> None:
        movers = [a.agent_id for a in actions if a.kind == "move"]
        self.rng.shuffle(movers)
        for aid in movers:
            if not self.empties:
                break
            r = self.residents[aid]
            dest = self.rng.choice(tuple(self.empties))
            del self.occupied[r.pos]
            self.empties.add(r.pos)
            self.empties.discard(dest)
            r.pos = dest
            self.occupied[dest] = aid

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        fracs = [self.same_fraction(r) for r in self.residents.values()]
        happy = sum(1 for r in self.residents.values()
                    if self.same_fraction(r) >= self.threshold)
        n = len(self.residents)
        return {
            "t": self.t,
            "n": n,
            "segregation": sum(fracs) / n if n else 0.0,
            "happy_fraction": happy / n if n else 0.0,
        }
