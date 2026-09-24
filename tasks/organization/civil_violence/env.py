"""
E — the Epstein civil-violence grid (toroidal, Chebyshev vision). The env presents
each citizen with what it can see (cops and active rebels nearby) and, as fixed
enforcement mechanics, lets cops arrest active citizens and jail them.

Only citizens make a behavioral decision (rebel or stay quiet) — that is the f the
LLM can replace. Cops, arrests, jail terms and movement are environment dynamics.

The `context` variant (ocm | lcm) changes only the natural-language description.
"""
from __future__ import annotations

import math
import random
from typing import List

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.organization.civil_violence.agents import Population


class CivilViolenceGrid:
    def __init__(self, population: Population, cfg: dict):
        d = cfg["dynamics"]
        self.w, self.h = population.width, population.height
        self.legitimacy = d["legitimacy"]
        self.vision = d["vision"]
        self.k = d["arrest_k"]
        self.threshold = d["active_threshold"]
        self.max_jail = d["max_jail_term"]
        self.context = cfg["behavior"].get("context", "ocm")
        self._pop = population
        self.rng = random.Random(cfg["seed"])
        v = self.vision
        self._offsets = [(dx, dy) for dx in range(-v, v + 1) for dy in range(-v, v + 1)
                         if not (dx == 0 and dy == 0)]
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.citizens = {c.agent_id: c for c in self._pop.citizens}
        self.cops = list(self._pop.cops)
        for c in self.citizens.values():
            c.active = False
            c.jail_remaining = 0
        self.occupied = {}
        for c in self.citizens.values():
            self.occupied[c.pos] = ("C", c.agent_id)
        for p in self.cops:
            self.occupied[p.pos] = ("P", p.agent_id)
        self.empties = {(x, y) for x in range(self.w) for y in range(self.h)} - set(self.occupied)
        return self.snapshot()

    def _vision(self, pos):
        x, y = pos
        for dx, dy in self._offsets:
            yield ((x + dx) % self.w, (y + dy) % self.h)

    def _scan(self, pos):
        cops = actives = 0
        for cell in self._vision(pos):
            occ = self.occupied.get(cell)
            if occ is None:
                continue
            if occ[0] == "P":
                cops += 1
            elif self.citizens[occ[1]].active:
                actives += 1
        return cops, actives

    def observe_batch(self) -> List[Observation]:
        obs = []
        for c in self.citizens.values():
            if c.jail_remaining > 0:
                continue
            cops, actives = self._scan(c.pos)
            a = actives + 1                                   # count self as a potential active
            arrest_prob = 1.0 - math.exp(-self.k * math.floor(cops / a))
            grievance = c.hardship * (1.0 - self.legitimacy)
            ctx = {"hardship": c.hardship, "risk_aversion": c.risk_aversion,
                   "grievance": grievance, "legitimacy": self.legitimacy,
                   "arrest_prob": arrest_prob, "cops_in_vision": cops,
                   "actives_in_vision": a, "threshold": self.threshold}
            obs.append(Observation(agent_id=c.agent_id, state={"active": c.active},
                                   context=ctx, rendered=self._render(ctx)))
        return obs

    def _render(self, c) -> str:
        if self.context == "lcm":
            mood = ("furious at your hardship" if c["grievance"] > 0.1 else "fairly content")
            danger = ("police are close and could arrest you" if c["arrest_prob"] > 0.5
                      else "few police are around")
            return (f"You are {mood}; {danger}. {c['cops_in_vision']} officers and "
                    f"{c['actives_in_vision'] - 1} other rebels are within sight.")
        return (f"grievance={c['grievance']:.2f}, risk_aversion={c['risk_aversion']:.2f}, "
                f"arrest_prob={c['arrest_prob']:.2f} (cops={c['cops_in_vision']}, "
                f"actives={c['actives_in_vision']}).")

    def _move_agent(self, kind, aid, pos):
        empties_in_view = [cell for cell in self._vision(pos) if cell in self.empties]
        if not empties_in_view:
            return pos
        dest = self.rng.choice(empties_in_view)
        del self.occupied[pos]
        self.empties.add(pos)
        self.empties.discard(dest)
        self.occupied[dest] = (kind, aid)
        return dest

    def apply(self, actions: List[Action]) -> None:
        # 1) citizens set their stance
        for act in actions:
            self.citizens[act.agent_id].active = bool(act.payload["active"])
        # 2) release finished jail terms (decrement those jailed before this step)
        for c in self.citizens.values():
            if c.jail_remaining > 0:
                c.jail_remaining -= 1
                if c.jail_remaining == 0 and self.empties:
                    c.pos = self.rng.choice(tuple(self.empties))
                    self.empties.discard(c.pos)
                    self.occupied[c.pos] = ("C", c.agent_id)
        # 3) cops arrest active citizens in view
        arrested = set()
        for cop in self.rng.sample(self.cops, len(self.cops)):
            targets = [occ[1] for cell in self._vision(cop.pos)
                       if (occ := self.occupied.get(cell)) and occ[0] == "C"
                       and self.citizens[occ[1]].active and occ[1] not in arrested]
            if targets:
                pick = self.rng.choice(targets)
                c = self.citizens[pick]
                c.active = False
                c.jail_remaining = self.rng.randint(1, self.max_jail)
                del self.occupied[c.pos]
                self.empties.add(c.pos)
                arrested.add(pick)
        # 4) movement of on-grid agents
        for c in self.citizens.values():
            if c.jail_remaining == 0:
                c.pos = self._move_agent("C", c.agent_id, c.pos)
        for cop in self.cops:
            cop.pos = self._move_agent("P", cop.agent_id, cop.pos)

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        on_grid = [c for c in self.citizens.values() if c.jail_remaining == 0]
        n_active = sum(1 for c in on_grid if c.active)
        n_jailed = sum(1 for c in self.citizens.values() if c.jail_remaining > 0)
        return {
            "t": self.t,
            "n_citizens": len(self.citizens),
            "active": n_active,
            "jailed": n_jailed,
            "quiet": len(on_grid) - n_active,
        }
