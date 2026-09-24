"""f (rule) — the classical Schelling similarity-threshold baseline on Chicago geography.

This is the rule-based reference baseline: a single similarity threshold τ on the
Queen-contiguity tract graph, run with the SAME per-step move budget as the LLM
population. A household is content when the own-race share of its 1-hop tract
neighbourhood is ≥ τ; discontent households target adjacent tracts with a higher
own-race share.

To guarantee "same move budget / same execution" as llm mode, rule_f writes the exact
same model state the LLM phases write (``group_would_move`` / ``move_targets`` /
``agent.satisfaction``); the relocation itself is then performed by the identical
validated machinery in ``env.apply`` (``_apply_move_budget`` + ``shuffle_do("step")``).
The ONLY thing that differs between rule and llm is how that intent is decided.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from socioverse_abm.behavior_engine.action import Action, Observation

from ._engine import ChicagoEngine, get_active

_THRESHOLD = 0.5      # τ — desired own-race neighbourhood share
_HOPS = 1             # how far a discontent household searches for a better tract
_TOP_K = 3            # ranked candidates handed to the shared executor


def configure(threshold: Optional[float] = None, search_radius_hops: Optional[int] = None) -> None:
    global _THRESHOLD, _HOPS
    if threshold is not None:
        _THRESHOLD = float(threshold)
    if search_radius_hops is not None:
        _HOPS = int(search_radius_hops)


def chicago_rule_f(observations: List[Observation], engine: Optional[ChicagoEngine] = None) -> List[Action]:
    eng = engine or get_active()
    m = eng.model
    env = m.environment
    tau, hops = _THRESHOLD, _HOPS

    # Group households by (archetype, tract) — the same granularity the LLM phases use.
    groups: dict[tuple, list] = defaultdict(list)
    for agent in eng.agent_order:
        groups[(agent.archetype_key, agent.tract_id)].append(agent)

    m.move_targets.clear()                       # refreshed every step, like _evaluate_move_candidates
    own_share_cache: dict[tuple[str, str], float] = {}

    def own_race_share(tract_id: str, race: str) -> float:
        ck = (tract_id, race)
        if ck not in own_share_cache:
            surrounding = env.get_surrounding_area_summary(tract_id)
            own_share_cache[ck] = surrounding.get(f"pct_{race}", 0.0) / 100.0
        return own_share_cache[ck]

    for (key, tract_id), agents_in_group in groups.items():
        race = agents_in_group[0].race
        own = own_race_share(tract_id, race)
        would_move = own < tau
        # Monotone satisfaction on 0–10 so the shared budget gate ranks the most
        # discontent first (10 - satisfaction), exactly as in llm mode.
        sat = max(0.0, min(10.0, 10.0 * own))
        for a in agents_in_group:
            a.satisfaction = sat
        m.group_would_move[(key, tract_id)] = would_move

        if not would_move:
            continue

        # Rank adjacent tracts that IMPROVE own-race share and can physically absorb us.
        max_pop = max((a.pop_count for a in agents_in_group), default=1)
        ranked = []
        for cid in env.get_neighbors(tract_id, hops):
            if cid == tract_id or cid not in env.dynamic_pop:
                continue
            cand = own_race_share(cid, race)
            if cand <= own:
                continue
            if not env.has_capacity(cid, max_pop):
                continue
            ranked.append({"tract_id": cid, "score": round(cand * 10.0, 3)})
        ranked.sort(key=lambda d: d["score"], reverse=True)
        if ranked:
            m.move_targets[(key, tract_id)] = ranked[:_TOP_K]

    actions: List[Action] = []
    for ob in observations:
        agent = eng.agent(ob.agent_id)
        k = (agent.archetype_key, agent.tract_id)
        targets = m.move_targets.get(k, [])
        move = bool(m.group_would_move.get(k, False) and targets)
        actions.append(Action(
            agent_id=ob.agent_id,
            kind="move" if move else "stay",
            payload={"move": move},
            raw={"ranked_targets": targets, "source": "rule"},
        ))
    return actions
