"""Stronger rule baselines for the Chicago task (supplementary ablation E0).

Both variants are *calibrated* from the same literature-sourced archetype parameters
the LLM condition uses (``legacy/config/archetypes.csv``, derived from
``persona_parameters.json``: Farley/Krysan preference points, satisfaction weights),
which makes them the strongest fair non-LLM references: same E, same executor
(``env.apply`` = budget gate + capacity + inflow caps), same per-step move budget;
only the decision rule differs. NOTE: ``min/ideal_own_group_pct`` are FRACTIONS 0-1.

- ``calibrated_multigroup``: asymmetric per-archetype thresholds. A group wants to
  move when the own-race share of its 1-hop neighbourhood falls below
  ``min_own_group_pct``; candidates (searched within the archetype's calibrated
  ``search_radius_hops``) are ranked by closeness to ``ideal_own_group_pct``.
- ``logit``: a discrete-choice baseline. Tract utility = sum_k w_k * f_k(tract) with
  w_k the archetype's satisfaction weights and f_k normalized tract features (racial
  fit, safety, schools, transit, amenities, housing-cost match, social proximity).
  A group wants to move when the best candidate beats the current tract's utility by
  more than ``MOVE_COST`` (deterministic argmax = the beta->inf limit of logit choice;
  stochasticity still enters through the shared budget gate and shuffled execution).

Select via config.yaml:
``behavior.rule_variant: classical | classical_random | calibrated_multigroup | logit``
(``classical_random``: the classical threshold with a random feasible destination; see below).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from socioverse_abm.behavior_engine.action import Action, Observation

from ._engine import ChicagoEngine, get_active

MOVE_COST = 0.05      # logit: minimum utility gain (0-1 scale) that justifies moving
_TOP_K = 3

# Rough 2010 Chicago per-capita income anchors per archetype income bracket, used
# only by the logit housing-cost-match feature (documented approximation).
_BRACKET_INCOME = {"low": 12_000, "lower_middle": 22_000, "upper_middle": 35_000, "high": 60_000}

_WEIGHT_FEATURES = (
    "racial_composition", "school_quality", "safety_crime", "housing_cost_match",
    "commute_transit", "amenities", "social_network_proximity",
)


def get(name: str):
    fns = {"calibrated_multigroup": calibrated_multigroup_rule_f, "logit": logit_choice_rule_f,
           "classical_random": classical_random_rule_f}
    if name not in fns:
        raise ValueError(
            "unknown rule_variant '%s' (classical|%s)" % (name, "|".join(sorted(fns))))
    return fns[name]


# ── shared helpers ───────────────────────────────────────────────────────────

def _own_share(env, cache, tract_id: str, race: str) -> float:
    """Own-race share (fraction 0-1) of the 1-hop neighbourhood around a tract."""
    key = (tract_id, race)
    if key not in cache:
        s = env.get_surrounding_area_summary(tract_id)
        cache[key] = s.get("pct_%s" % race, 0.0) / 100.0
    return cache[key]


def _emit_actions(eng, m, observations) -> List[Action]:
    actions = []
    for ob in observations:
        agent = eng.agent(ob.agent_id)
        k = (agent.archetype_key, agent.tract_id)
        targets = m.move_targets.get(k, [])
        move = bool(m.group_would_move.get(k, False) and targets)
        actions.append(Action(agent_id=ob.agent_id, kind="move" if move else "stay",
                              payload={"move": move}, raw={"ranked_targets": targets}))
    return actions


def _groups(eng):
    groups = defaultdict(list)
    for agent in eng.agent_order:
        groups[(agent.archetype_key, agent.tract_id)].append(agent)
    return groups


# ── E0a': classical with RANDOM feasible destination (Schelling's original
# "relocate to a random vacant spot" reading; contrast with rule_f.py's
# best-improvement ranking). Same tau gate, same budget/executor; only the
# destination choice differs. Uses the model's seeded RNG -> reproducible. ──

def classical_random_rule_f(observations: List[Observation],
                            engine: Optional[ChicagoEngine] = None) -> List[Action]:
    from . import rule_f as _rf                       # reuse the configured tau / hops
    eng = engine or get_active()
    m, env = eng.model, eng.model.environment
    tau, hops = _rf._THRESHOLD, _rf._HOPS
    m.move_targets.clear()
    cache: dict = {}
    rng = m.random

    for (key, tract_id), members in _groups(eng).items():
        race = members[0].race
        own = _own_share(env, cache, tract_id, race)
        would_move = own < tau
        sat = max(0.0, min(10.0, 10.0 * own))
        for a in members:
            a.satisfaction = sat
        m.group_would_move[(key, tract_id)] = would_move
        if not would_move:
            continue
        max_pop = max((a.pop_count for a in members), default=1)
        feasible = [cid for cid in env.get_neighbors(tract_id, hops)
                    if cid != tract_id and cid in env.dynamic_pop
                    and env.has_capacity(cid, max_pop)]
        if not feasible:
            continue
        rng.shuffle(feasible)
        m.move_targets[(key, tract_id)] = [
            {"tract_id": cid, "score": 1.0} for cid in feasible[:_TOP_K]]

    return _emit_actions(eng, m, observations)


# ── E0b: calibrated multi-group thresholds ──────────────────────────────────

def calibrated_multigroup_rule_f(observations: List[Observation],
                                 engine: Optional[ChicagoEngine] = None) -> List[Action]:
    eng = engine or get_active()
    m, env = eng.model, eng.model.environment
    m.move_targets.clear()
    cache: dict = {}

    for (key, tract_id), members in _groups(eng).items():
        arch = m.active_archetypes[key]
        race = members[0].race
        own = _own_share(env, cache, tract_id, race)
        min_frac = float(arch.min_own_group_pct)      # fractions 0-1 (see module docstring)
        ideal = float(arch.ideal_own_group_pct)

        would_move = own < min_frac
        sat = 10.0 * min(1.0, own / ideal) if ideal > 0 else 10.0
        for a in members:
            a.satisfaction = sat                       # ranks the shared budget gate
        m.group_would_move[(key, tract_id)] = would_move
        if not would_move:
            continue

        max_pop = max((a.pop_count for a in members), default=1)
        cur_gap = abs(own - ideal)
        ranked = []
        for cid in env.get_neighbors(tract_id, int(arch.search_radius_hops)):
            if cid == tract_id or cid not in env.dynamic_pop:
                continue
            if not env.has_capacity(cid, max_pop):
                continue
            gap = abs(_own_share(env, cache, cid, race) - ideal)
            if gap < cur_gap:                          # only genuinely-closer-to-ideal moves
                ranked.append({"tract_id": cid, "score": round(10.0 * (1.0 - gap), 3)})
        ranked.sort(key=lambda d: d["score"], reverse=True)
        if ranked:
            m.move_targets[(key, tract_id)] = ranked[:_TOP_K]

    return _emit_actions(eng, m, observations)


# ── E0c: logit / discrete-choice utility ────────────────────────────────────

def _tract_utility(env, caches, tract_id: str, arch) -> float:
    race = arch.race
    share = _own_share(env, caches["share"], tract_id, race)
    if tract_id not in caches["feat"]:
        info = env.get_tract_info(tract_id)
        caches["feat"][tract_id] = {
            "safety_crime": env._compute_safety_score(info["crime_count_2010"])[0] / 10.0,
            "school_quality": env._compute_school_score(info["schools"])[0] / 10.0,
            "commute_transit": env._compute_transit_score(
                info["cta_stations"], info["cta_bus_stops"])[0] / 10.0,
            "amenities": min(1.0, (info["parks"] + info["religious_places"]) / 6.0),
            "pci": float(info["per_capita_income"] or 0.0),
        }
    f = caches["feat"][tract_id]
    ideal = float(arch.ideal_own_group_pct)
    target = _BRACKET_INCOME.get(arch.income_bracket, 30_000)
    feats = {
        "racial_composition": 1.0 - min(1.0, abs(share - ideal)),
        "school_quality": f["school_quality"],
        "safety_crime": f["safety_crime"],
        "housing_cost_match": 1.0 - min(1.0, abs(f["pci"] - target) / max(target, 1.0)),
        "commute_transit": f["commute_transit"],
        "amenities": f["amenities"],
        "social_network_proximity": share,
    }
    w = arch.satisfaction_weights
    return sum(float(w.get(k, 0.0)) * feats[k] for k in _WEIGHT_FEATURES)


def logit_choice_rule_f(observations: List[Observation],
                        engine: Optional[ChicagoEngine] = None) -> List[Action]:
    eng = engine or get_active()
    m, env = eng.model, eng.model.environment
    m.move_targets.clear()
    caches = {"share": {}, "feat": {}}

    for (key, tract_id), members in _groups(eng).items():
        arch = m.active_archetypes[key]
        u_cur = _tract_utility(env, caches, tract_id, arch)
        sat = 10.0 * max(0.0, min(1.0, u_cur))
        for a in members:
            a.satisfaction = sat

        max_pop = max((a.pop_count for a in members), default=1)
        scored = []
        for cid in env.get_neighbors(tract_id, int(arch.search_radius_hops)):
            if cid == tract_id or cid not in env.dynamic_pop:
                continue
            if not env.has_capacity(cid, max_pop):
                continue
            u = _tract_utility(env, caches, cid, arch)
            if u > u_cur + MOVE_COST:                  # inertia premium
                scored.append({"tract_id": cid, "score": round(10.0 * u, 3)})
        scored.sort(key=lambda d: d["score"], reverse=True)

        would_move = bool(scored)
        m.group_would_move[(key, tract_id)] = would_move
        if scored:
            m.move_targets[(key, tract_id)] = scored[:_TOP_K]

    return _emit_actions(eng, m, observations)
