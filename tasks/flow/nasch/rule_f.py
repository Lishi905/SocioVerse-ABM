"""
f (rule) — the classic Nagel-Schreckenberg speed update, the parity baseline.
Faithful port of legacy/behavior_engine/ABM_based.py (rules 1-3; rule 4 = move is
done by the env). Per vehicle, given (current_speed, gap_ahead, vmax, p):

  1. accelerate:        v = min(v + 1, vmax)
  2. brake to gap:      v = min(v, gap_ahead)
  3. random slowdown:   with prob p, v = max(v - 1, 0)
"""
from __future__ import annotations

import random
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def nasch_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        v = min(c["current_speed"] + 1, c["max_speed"])   # 1. accelerate
        v = min(v, c["gap_ahead"])                         # 2. brake to gap
        if _RNG.random() < c["randomization_prob"] and v > 0:
            v = max(v - 1, 0)                              # 3. random slowdown
        actions.append(Action(agent_id=o.agent_id, kind="set_speed", payload={"speed": v}))
    return actions
