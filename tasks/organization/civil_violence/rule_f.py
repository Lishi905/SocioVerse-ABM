"""
f (rule) — Epstein's citizen decision rule, the parity baseline: a citizen rebels
(goes active) when grievance exceeds the net risk of doing so by more than a threshold:

    grievance  = hardship * (1 - legitimacy)
    net_risk   = risk_aversion * arrest_probability
    active     <=>  grievance - net_risk > threshold

Deterministic; unaffected by the ocm/lcm and tbf/lbf variants (those shape only the
LLM path).
"""
from __future__ import annotations

from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation


def civil_violence_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        net_risk = c["risk_aversion"] * c["arrest_prob"]
        active = (c["grievance"] - net_risk) > c["threshold"]
        actions.append(Action(agent_id=o.agent_id, kind="active" if active else "quiet",
                              payload={"active": active}))
    return actions
