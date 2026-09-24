"""
f (LLM) — the LLM-based SIR behavior function. Same contract and same per-agent
granularity as rule_f, so the two can be compared by the consistency score.

Each non-stifler agent is shown the natural-language render of its situation
(Observation.rendered) and asked to decide. The kernel's LLM helpers are imported
lazily so that importing this task does not require `openai` to be installed
(rule-only runs, `sv-abm list`, and fake-LLM parity tests all work without it).
"""
from __future__ import annotations

import os
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.diffusion.sir.agents import I, R, S

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_PROMPT = """{situation}

Reply with strict JSON: {{"decision": "<choice>"}} where <choice> is one of:
  - "spread"  : you start (or keep) spreading the rumor
  - "ignore"  : you do not spread it
"""


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _decision_to_state(decision: str, current: str) -> str:
    """Map the LLM's choice onto the next health state, given current state."""
    spreading = str(decision).strip().lower() == "spread"
    if current == S:
        return I if spreading else S
    if current == I:
        return I if spreading else R     # stop spreading -> stifler
    return R


def sir_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    # Lazy import: only needed when actually running the LLM path.
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # needs openai
    actions: List[Action] = []
    for o in observations:
        state = o.context["state"]
        if state == R:
            actions.append(Action(agent_id=o.agent_id, kind=R, payload={"from": R}))
            continue
        raw = llm.generate_and_parser(_MODEL, _PROMPT.format(situation=o.rendered))
        decision = raw.get("decision", "ignore") if isinstance(raw, dict) else "ignore"
        nxt = _decision_to_state(decision, state)
        actions.append(Action(agent_id=o.agent_id, kind=nxt,
                              payload={"from": state}, raw=raw))
    return actions
