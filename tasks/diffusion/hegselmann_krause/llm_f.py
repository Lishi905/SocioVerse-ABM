"""
f (LLM) — same contract as rule_f: output a new opinion in [0, 1]. The agent is
shown the peers within its confidence bound and asked for its updated opinion.
openai is imported lazily.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You update your opinion by averaging the views of people close enough to yours "
    "to take seriously. Respond with ONLY a number between 0 and 1. No other text."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _parse_opinion(text: str, default: float) -> float:
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    if not m:
        return default
    return min(1.0, max(0.0, float(m.group())))


def hk_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        raw = llm.generate(_MODEL, o.rendered, sys_prompt=_SYS_PROMPT,
                           max_tokens=8, temperature=0.2)
        new_opinion = _parse_opinion(raw, default=o.context["opinion"])
        actions.append(Action(agent_id=o.agent_id, kind="set_opinion",
                              payload={"opinion": new_opinion}, raw=raw))
    return actions
