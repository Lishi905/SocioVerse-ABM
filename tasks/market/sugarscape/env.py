"""
E — the toroidal sugar grid. A native reimplementation of the legacy Mesa model
(legacy/sugarscape_simple.py): two layered sugar hills that regrow each step, with
citizens that see along the four cardinal directions.

Per kernel step: observe (each citizen's visible empty cells + their sugar) ->
behavior picks a destination -> apply moves citizens (sequentially, to avoid two
landing on one cell), harvests, metabolizes, and removes the dead; sugar regrows in
advance(). Movement is the behavior; harvest/metabolism/regrowth are env mechanics.
"""
from __future__ import annotations

import math
import random
from typing import List

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.market.sugarscape.agents import Population

_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _build_hills(w: int, h: int, cap_max: int) -> np.ndarray:
    """Two diagonal sugar hills with concentric capacity terraces (classic G1)."""
    cap = np.zeros((w, h), dtype=int)
    c1 = (int(w * 0.28), int(h * 0.28))
    c2 = (int(w * 0.72), int(h * 0.72))
    r0, r1, r2, r3 = (k * min(w, h) for k in (0.06, 0.18, 0.27, 0.45))
    for x in range(w):
        for y in range(h):
            d = min(math.hypot(x - c1[0], y - c1[1]), math.hypot(x - c2[0], y - c2[1]))
            if d <= r0:
                level = cap_max
            elif d <= r1:
                level = max(1, cap_max - 1)
            elif d <= r2:
                level = max(1, cap_max - 2)
            elif d <= r3:
                level = 1
            else:
                level = 0
            cap[x, y] = level
    return cap


class SugarscapeGrid:
    def __init__(self, population: Population, cfg: dict):
        p, d = cfg["population"], cfg["dynamics"]
        self.w, self.h = p["width"], p["height"]
        self.capacity = _build_hills(self.w, self.h, d["patch_capacity_max"])
        self.regrowth_rate = d["regrowth_rate"]
        self._pop = population
        self.rng = random.Random(cfg["seed"])
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.sugar = self.capacity.copy()
        self.citizens = self._pop.citizens
        for c in self.citizens:
            c.alive = True
        self.occupied = {c.pos: c.agent_id for c in self.citizens}
        return self.snapshot()

    def _alive(self) -> List:
        return [c for c in self.citizens if c.alive]

    def _free(self, pos, self_id: int) -> bool:
        return pos not in self.occupied or self.occupied[pos] == self_id

    def _visible_candidates(self, c) -> list:
        x, y = c.pos
        cands = [{"to": c.pos, "sugar": int(self.sugar[x, y]), "dist": 0}]
        for dx, dy in _DIRS:
            for r in range(1, c.vision + 1):
                nx, ny = (x + dx * r) % self.w, (y + dy * r) % self.h
                if not self._free((nx, ny), c.agent_id):
                    break                       # blocked by another citizen
                cands.append({"to": (nx, ny), "sugar": int(self.sugar[nx, ny]), "dist": r})
        return cands

    def observe_batch(self) -> List[Observation]:
        obs = []
        for c in self._alive():
            cands = self._visible_candidates(c)
            ctx = {"pos": c.pos, "sugar": c.sugar, "vision": c.vision,
                   "metabolism": c.metabolism, "candidates": cands}
            obs.append(Observation(agent_id=c.agent_id, state={"sugar": c.sugar},
                                   context=ctx, rendered=self._render(c, cands)))
        return obs

    def _render(self, c, cands) -> str:
        top = sorted(cands, key=lambda d: (-d["sugar"], d["dist"]))[:4]
        opts = "; ".join(f"{d['to']}=sugar{d['sugar']}@dist{d['dist']}" for d in top)
        return (f"You have {c.sugar} sugar. Reachable cells: [{opts}]. "
                f"Move to which cell to gather the most sugar?")

    def apply(self, actions: List[Action]) -> None:
        by_id = {a.agent_id: tuple(a.payload["to"]) for a in actions}
        citizen = {c.agent_id: c for c in self._alive()}
        for cid in self.rng.sample(list(by_id), len(by_id)):   # sequential, random order
            c = citizen.get(cid)
            if c is None or not c.alive:
                continue
            dest = by_id[cid]
            if self._free(dest, cid):                          # move if still free
                del self.occupied[c.pos]
                c.pos = dest
                self.occupied[dest] = cid
            x, y = c.pos
            c.sugar += int(self.sugar[x, y])                   # harvest
            self.sugar[x, y] = 0
            c.sugar -= c.metabolism                            # metabolize
            if c.sugar <= 0:                                   # starve
                c.alive = False
                self.occupied.pop(c.pos, None)

    def advance(self) -> None:
        self.t += 1
        self.sugar = np.minimum(self.capacity, self.sugar + self.regrowth_rate)

    def snapshot(self) -> dict:
        wealth = [c.sugar for c in self._alive()]
        return {
            "t": self.t,
            "alive": len(wealth),
            "mean_sugar": float(np.mean(wealth)) if wealth else 0.0,
            "gini": _gini(wealth),
        }


def _gini(values) -> float:
    vals = sorted(max(0, int(v)) for v in values)
    n = len(vals)
    if n == 0 or vals[-1] == 0:
        return 0.0
    weighted = sum(i * v for i, v in enumerate(vals, start=1))
    total = sum(vals)
    mean = total / n
    return max(0.0, min(1.0, (2 * weighted / n - (n + 1) * mean) / (n * mean)))
