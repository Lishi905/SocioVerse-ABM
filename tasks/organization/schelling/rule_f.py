"""
f (rule) — the classic Schelling rule, the parity baseline: a resident is unhappy
(and wants to move) when the fraction of same-group neighbours is below the homophily
threshold. Deterministic; unaffected by the ocm/lcm and tbf/lbf variants (those only
shape the LLM path).
"""
from __future__ import annotations

from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation


def schelling_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        move = c["same_fraction"] < c["threshold"]
        actions.append(Action(agent_id=o.agent_id, kind="move" if move else "stay",
                              payload={"move": move}))
    return actions
