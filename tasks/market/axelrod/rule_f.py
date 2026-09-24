"""
f (rule) — each agent plays its fixed strategy against every opponent, the parity
baseline. Supported strategies:

  always_cooperate / always_defect : unconditional
  tit_for_tat                      : copy the opponent's last move (cooperate first)
  grudger                          : cooperate until the opponent ever defects, then defect forever
  random                           : 50/50
"""
from __future__ import annotations

import random
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def _move(strategy: str, last, ever_defected: bool) -> str:
    if strategy == "always_cooperate":
        return "C"
    if strategy == "always_defect":
        return "D"
    if strategy == "tit_for_tat":
        return last if last is not None else "C"
    if strategy == "grudger":
        return "D" if ever_defected else "C"
    if strategy == "random":
        return _RNG.choice(["C", "D"])
    return "C"


def axelrod_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        strategy = o.context["strategy"]
        moves = {
            j: _move(strategy, opp["last"], opp["ever_defected"])
            for j, opp in o.context["opponents"].items()
        }
        actions.append(Action(agent_id=o.agent_id, kind="play", payload={"moves": moves}))
    return actions
