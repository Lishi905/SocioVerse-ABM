"""
f (LLM) — same contract as rule_f: each agent chooses C/D against every opponent.
The LLM is told its disposition and, per opponent, the opponent's last move and
whether they have defected before. openai is imported lazily.

Note: one call per (agent, opponent) per round, so the LLM path is costly for large
populations — rule mode and the fake-LLM parity test do not call a real model.
"""
from __future__ import annotations

import os
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You are playing iterated Prisoner's Dilemma. Reply with ONLY one letter: "
    "C to cooperate or D to defect."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _parse_move(text: str, default: str = "C") -> str:
    t = str(text).strip().upper()
    if "D" in t and "C" not in t:
        return "D"
    if "C" in t and "D" not in t:
        return "C"
    return default


def _opp_prompt(strategy: str, last, ever_defected: bool) -> str:
    hist = "this is your first encounter" if last is None else f"their last move was {last}"
    grudge = " They have defected against you before." if ever_defected else ""
    return (f"Your disposition is '{strategy}'. Against this opponent, {hist}.{grudge} "
            f"Do you Cooperate or Defect?")


def axelrod_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        strategy = o.context["strategy"]
        moves = {}
        for j, opp in o.context["opponents"].items():
            raw = llm.generate(_MODEL, _opp_prompt(strategy, opp["last"], opp["ever_defected"]),
                               sys_prompt=_SYS_PROMPT, max_tokens=2, temperature=0.2)
            moves[j] = _parse_move(raw)
        actions.append(Action(agent_id=o.agent_id, kind="play", payload={"moves": moves}))
    return actions
