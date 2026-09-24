"""f (LLM) — the validated two-phase behaviour function.

Delegates to the vendored model's OWN batched-LLM phases
(``_assess_archetype_satisfaction`` + ``_evaluate_move_candidates``), which write
``group_would_move`` / ``move_targets`` per (archetype, tract). We then project each
household's resulting intent into a typed move/stay Action. No validated dynamic is
re-implemented; the actual relocation happens later in ``env.apply`` (budget gate +
shuffle_do), exactly as in legacy ``step()``.

Mirror of SocioVerse2's ``adapter/decision.py``.
"""
from __future__ import annotations

from typing import List, Optional

from socioverse_abm.behavior_engine.action import Action, Observation

from socioverse_abm.scenario_engine.registry import TaskSetupError

from ._engine import ChicagoEngine, get_active


def chicago_llm_f(observations: List[Observation], engine: Optional[ChicagoEngine] = None) -> List[Action]:
    eng = engine or get_active()
    if not eng.has_llm:
        # Built for rule mode, or for a hybrid run without a key: the vendored client
        # would fall back silently on every call, so stop with the setup hint instead.
        raise TaskSetupError(
            "chicago_segregation: the LLM behaviour function was called on an engine "
            "built without an LLM (no SV_LLM_API_KEY / OPENAI_API_KEY found). Set the key "
            "in your environment or in SocioVerse-ABM/.env, or pass "
            "llm_client=DeterministicLLMClient() for a no-token run."
        )
    m = eng.model
    # Phase 1 + 2 of legacy step() — batched LLM calls that set the per-group intent.
    m._assess_archetype_satisfaction()
    m._evaluate_move_candidates()

    actions: List[Action] = []
    for ob in observations:
        agent = eng.agent(ob.agent_id)
        key = (agent.archetype_key, agent.tract_id)
        wants = m.group_would_move.get(key, agent.archetype.would_move)
        targets = m.move_targets.get(key, [])
        move = bool(wants and targets)
        actions.append(Action(
            agent_id=ob.agent_id,
            kind="move" if move else "stay",
            payload={"move": move},           # minimal — consistency compares move/stay
            raw={"ranked_targets": targets, "source": "llm"},
        ))
    return actions
