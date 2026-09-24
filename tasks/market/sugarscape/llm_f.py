"""
f (LLM) — same contract as rule_f: choose one of the visible cells to move to. The
candidates are presented as a numbered menu; the LLM returns an index. openai is
imported lazily.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You are a forager who wants to gather as much sugar as possible. From the "
    "numbered cells you can reach, pick the best one. Respond with ONLY the index "
    "number. No other text."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _menu(cands) -> str:
    return "\n".join(
        f"{i}: cell {d['to']} has {d['sugar']} sugar at distance {d['dist']}"
        for i, d in enumerate(cands)
    )


def _parse_index(text: str, n: int, default: int = 0) -> int:
    m = re.search(r"\d+", str(text))
    if not m:
        return default
    return min(n - 1, max(0, int(m.group())))


def sugarscape_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        cands = o.context["candidates"]
        prompt = f"You have {o.context['sugar']} sugar. Reachable cells:\n{_menu(cands)}\nMove to index:"
        raw = llm.generate(_MODEL, prompt, sys_prompt=_SYS_PROMPT, max_tokens=8, temperature=0.2)
        idx = _parse_index(raw, len(cands))
        actions.append(Action(agent_id=o.agent_id, kind="move",
                              payload={"to": tuple(cands[idx]["to"])}, raw=raw))
    return actions
