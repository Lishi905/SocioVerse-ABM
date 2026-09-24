"""
Behavior engine — the f and B side of B = f(P, E).

    action.py  typed Observation / Action / Decision (the f contract)      [dep-free]
    hybrid.py  rule+LLM routing for `--mode hybrid`                         [dep-free]
    llm_f.py   LLM helpers (generate / parse / embedding)                  [needs openai]
    prompt.py  shared prompt templates

Rule-based behavior functions live with each task (`tasks/<family>/<task>/rule_f.py`).
Only the dep-free pieces are re-exported; import llm_f explicitly so a bare
`import socioverse_abm.behavior_engine` does not require openai.
"""
from socioverse_abm.behavior_engine.action import Action, Decision, Observation
from socioverse_abm.behavior_engine.hybrid import (
    always_llm,
    always_rule,
    hybrid_decide,
)

__all__ = [
    "Action",
    "Decision",
    "Observation",
    "hybrid_decide",
    "always_rule",
    "always_llm",
]
