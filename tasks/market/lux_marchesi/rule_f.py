"""
f (rule) — the Lux-Marchesi herding rule, the parity baseline. Fundamentalists keep
their strategy (the price anchor); each noise trader becomes optimist or pessimist
with a probability driven by the crowd's mood and the recent price trend:

    U = a1 * opinion_index + a2 * tanh(50 * trend)
    P(optimist) = sigmoid(U)

The positive feedback (more optimists -> higher U -> more optimists) produces the
bubbles/crashes and volatility clustering the model is known for.
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation
from tasks.market.lux_marchesi.agents import FUNDAMENTALIST, OPTIMIST, PESSIMIST

_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def lux_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        if c["stance"] == FUNDAMENTALIST:
            new = FUNDAMENTALIST
        else:
            u = c["a1"] * c["opinion_index"] + c["a2"] * math.tanh(50.0 * c["trend"])
            prob_opt = 1.0 / (1.0 + math.exp(-u))
            new = OPTIMIST if _RNG.random() < prob_opt else PESSIMIST
        actions.append(Action(agent_id=o.agent_id, kind=new, payload={"from": c["stance"]}))
    return actions
