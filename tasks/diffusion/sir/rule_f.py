"""
f (rule) — the rule-based SIR behavior function, the parity baseline that the
LLM behavior function must reproduce (consistency score).

This is a native per-agent reimplementation of the network-SIR contact process
(the legacy ndlib version is kept verbatim in legacy/SIR_simple.py). Per step:

  - Ignorant (S): becomes Spreader with prob 1 - (1 - beta_eff)^k, where k is the
    number of spreading neighbours (i.e. "at least one of k contacts convinces me").
  - Spreader (I): becomes Stifler (R) with prob gamma; else keeps spreading.
  - Stifler (R): absorbing, stays R.

Reproducibility: a module RNG is seeded once per run via `seed()`. The behavior
function satisfies the bare kernel contract `f(observations) -> actions`, reading
beta_eff / gamma from each Observation's context.
"""
from __future__ import annotations

import random
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.diffusion.sir.agents import I, R, S

_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def sir_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        state = o.context["state"]
        if state == S:
            k = o.context["spreader_neighbors"]
            p_spread = 1.0 - (1.0 - o.context["beta_eff"]) ** k
            nxt = I if _RNG.random() < p_spread else S
        elif state == I:
            nxt = R if _RNG.random() < o.context["gamma"] else I
        else:
            nxt = R
        actions.append(Action(agent_id=o.agent_id, kind=nxt, payload={"from": state}))
    return actions
