"""
f (LLM) — same contract as rule_f: pick a side (0 or 1). The LLM is shown the recent
winning-side history and asked which side it expects to be the minority. openai lazy.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You play the minority game: you win by being on the side fewer people choose. "
    "Respond with ONLY a single digit, 0 or 1."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _parse_side(text: str, default: int = 0) -> int:
    m = re.search(r"[01]", str(text))
    return int(m.group()) if m else default


def minority_game_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        raw = llm.generate(_MODEL, o.rendered, sys_prompt=_SYS_PROMPT, max_tokens=4, temperature=0.2)
        actions.append(Action(agent_id=o.agent_id, kind="choose",
                              payload={"side": _parse_side(raw)}, raw=raw))
    return actions
