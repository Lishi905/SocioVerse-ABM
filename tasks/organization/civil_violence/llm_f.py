"""
f (LLM) — same contract as rule_f (rebel / stay quiet). Same four-variant structure
as Schelling:

  context (ocm | lcm)      : carried in Observation.rendered by the env.
  llm_behavior (tbf | lbf) : set here — tbf states Epstein's rule and asks the LLM to
                             apply it; lbf asks the citizen to decide by free reasoning.

openai is imported lazily.
"""
from __future__ import annotations

import os
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")
_BEHAVIOR = "tbf"

_SYS_PROMPT = "You are a citizen deciding whether to rebel. Reply with ONLY one word: rebel or quiet."


def configure(model: str, llm_behavior: str = "tbf") -> None:
    global _MODEL, _BEHAVIOR
    _MODEL, _BEHAVIOR = model, llm_behavior


def _prompt(o: Observation) -> str:
    c = o.context
    if _BEHAVIOR == "tbf":      # traditional behavior function: apply Epstein's rule
        return (f"{o.rendered} You rebel only if your grievance exceeds your fear of "
                f"arrest (risk_aversion x arrest_prob) by more than {c['threshold']}. "
                f"Applying this rule, do you rebel or stay quiet?")
    # lbf: free reasoning
    return f"{o.rendered} Do you join the uprising or stay quiet?"


def _parse(text: str, default: str = "quiet") -> str:
    t = str(text).strip().lower()
    if "rebel" in t or "active" in t or "uprising" in t:
        return "active"
    if "quiet" in t or "calm" in t or "stay" in t:
        return "quiet"
    return default


def civil_violence_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        raw = llm.generate(_MODEL, _prompt(o), sys_prompt=_SYS_PROMPT, max_tokens=4, temperature=0.2)
        decision = _parse(raw)
        actions.append(Action(agent_id=o.agent_id, kind=decision,
                              payload={"active": decision == "active"}, raw=raw))
    return actions
