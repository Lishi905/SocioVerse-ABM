"""
f (LLM) — same contract as rule_f: decide one integer speed per vehicle. Faithful
port of legacy/behavior_engine/LLM_based.py prompt + integer-parsing, but routed
through the kernel's LLM helper (so all tasks share one provider config) and with
the openai import done lazily (rule-only runs and fake-LLM parity tests need no key).
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = """You are a driver on a single-lane circular road.

To take a valid action, you MUST compute your speed according to these rules:
1. Acceleration: v_temp = min(v_current + 1, vmax)
2. Safety braking: v_safe = min(v_temp, gap_ahead)
3. Random hesitation: with probability p, reduce by 1: max(v_safe - 1, 0)

Any output outside the integer range [0, vmax] is an invalid action.
You MUST respond with ONLY a single integer. No explanation, no other text."""


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _build_prompt(c: dict) -> str:
    return (f"Given v_current={c['current_speed']}, gap_ahead={c['gap_ahead']}, "
            f"vmax={c['max_speed']}, p={c['randomization_prob']:.6f},\nmy speed is:")


def _parse_speed(text: str, vmax: int, default: int = 0) -> int:
    """Take the last integer in [0, vmax] from the response (legacy strategy)."""
    ints = re.findall(r"\b(\d+)\b", str(text).strip())
    valid = [int(x) for x in ints if 0 <= int(x) <= vmax]
    if valid:
        return valid[-1]
    if ints:
        return max(0, min(int(ints[-1]), vmax))
    return default


def nasch_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        c = o.context
        raw = llm.generate(_MODEL, _build_prompt(c), sys_prompt=_SYS_PROMPT,
                           max_tokens=16, temperature=0.1)
        speed = _parse_speed(raw, c["max_speed"])
        actions.append(Action(agent_id=o.agent_id, kind="set_speed",
                              payload={"speed": speed}, raw=raw))
    return actions
