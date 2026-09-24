"""
f (LLM) — same contract as rule_f: each trader picks its next stance
(optimist / pessimist / fundamentalist) from the market view. openai is lazy.
"""
from __future__ import annotations

import os
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation
from tasks.market.lux_marchesi.agents import FUNDAMENTALIST, OPTIMIST, PESSIMIST

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You are a trader deciding your stance for the next round. Reply with ONLY one "
    "word: optimist, pessimist, or fundamentalist."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _parse_stance(text: str, default: str) -> str:
    t = str(text).lower()
    if "fundamental" in t:
        return FUNDAMENTALIST
    if "pessimist" in t or "sell" in t or "bear" in t:
        return PESSIMIST
    if "optimist" in t or "buy" in t or "bull" in t:
        return OPTIMIST
    return default


def lux_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        raw = llm.generate(_MODEL, o.rendered, sys_prompt=_SYS_PROMPT,
                           max_tokens=4, temperature=0.2)
        new = _parse_stance(raw, default=o.context["stance"])
        actions.append(Action(agent_id=o.agent_id, kind=new,
                              payload={"from": o.context["stance"]}, raw=raw))
    return actions
