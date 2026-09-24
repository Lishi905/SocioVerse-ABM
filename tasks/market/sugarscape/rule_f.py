"""
f (rule) — the Sugarscape movement rule (Epstein-Axtell): move to the visible empty
cell with the most sugar, breaking ties by nearest distance then randomly. The
parity baseline. Harvest / metabolism / death are env mechanics, not the decision.
"""
from __future__ import annotations

import random
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def sugarscape_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        cands = o.context["candidates"]
        max_sugar = max(d["sugar"] for d in cands)
        best = [d for d in cands if d["sugar"] == max_sugar]
        min_dist = min(d["dist"] for d in best)
        nearest = [d for d in best if d["dist"] == min_dist]
        choice = _RNG.choice(nearest)
        actions.append(Action(agent_id=o.agent_id, kind="move",
                              payload={"to": tuple(choice["to"])}))
    return actions
