"""
f (rule) — the Hegselmann-Krause bounded-confidence update, the parity baseline.
Each agent's new opinion is the mean of all opinions within its confidence bound
epsilon (including its own). Deterministic; native reimplementation of the legacy
ndlib HKModel so it runs without ndlib.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation


def hk_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        x = o.context["opinion"]
        peers = np.asarray(o.context["peers"], dtype=float)
        within = peers[np.abs(peers - x) <= o.context["epsilon"]]
        new_opinion = float(within.mean()) if len(within) else x
        actions.append(Action(agent_id=o.agent_id, kind="set_opinion",
                              payload={"opinion": new_opinion}))
    return actions
