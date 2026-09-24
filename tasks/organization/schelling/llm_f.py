"""
f (LLM) — same contract as rule_f (move / stay). This is where the four variants act:

  context (ocm | lcm)        : carried in Observation.rendered by the env — ocm gives raw
                               neighbour counts, lcm a narrative of the same situation.
  llm_behavior (tbf | lbf)   : set here — tbf states the Schelling rule and asks the LLM
                               to apply it; lbf asks the LLM to decide by free reasoning.

The 2x2 of these is the LCM/LBF/OCM/TBF matrix from the companion ABM study, now a parameter pair.
openai is imported lazily.
"""
from __future__ import annotations

import os
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")
_BEHAVIOR = "tbf"

_SYS_PROMPT = "You decide whether to move home. Reply with ONLY one word: move or stay."


def configure(model: str, llm_behavior: str = "tbf") -> None:
    global _MODEL, _BEHAVIOR
    _MODEL, _BEHAVIOR = model, llm_behavior


def _prompt(o: Observation) -> str:
    c = o.context
    if _BEHAVIOR == "tbf":      # traditional behavior function: apply the rule
        return (f"{o.rendered} You are content only if at least {c['threshold']:.0%} of "
                f"your neighbours share your group. Applying this rule, do you move or stay?")
    # lbf: language behavior function — free reasoning
    return f"{o.rendered} Would you rather move to a different neighbourhood or stay? Move or stay?"


def _parse(text: str, default: str = "stay") -> str:
    t = str(text).strip().lower()
    if "move" in t:
        return "move"
    if "stay" in t:
        return "stay"
    return default


def schelling_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        raw = llm.generate(_MODEL, _prompt(o), sys_prompt=_SYS_PROMPT, max_tokens=4, temperature=0.2)
        decision = _parse(raw)
        actions.append(Action(agent_id=o.agent_id, kind=decision,
                              payload={"move": decision == "move"}, raw=raw))
    return actions
