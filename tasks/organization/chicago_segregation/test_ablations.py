"""Tests for the supplementary ablation switches (E0-E4).

Covers: race-blind scrubbing (E2), attribute shuffle invariants (E4), constraints-off
config passthrough (E3), and the two calibrated rule baselines (E0). All offline
(DeterministicLLMClient / rule mode); auto-skip when the geo stack or data is absent.
"""
from __future__ import annotations

import pathlib

import pytest

_LEGACY = pathlib.Path(__file__).with_name("legacy")
_DATA_OK = (_LEGACY / "processed_data" / "chicago_tracts.geojson").exists()

needs_chicago = pytest.mark.skipif(
    not _DATA_OK, reason="chicago extra / vendored data not available"
)

_SEED = 42


def _fresh_engine(**kw):
    pytest.importorskip("geopandas")
    pytest.importorskip("mesa")
    from ._engine import ChicagoEngine
    from ._fake_llm import DeterministicLLMClient

    return ChicagoEngine(scale="small", init_mode="census", seed=_SEED,
                         llm_client=DeterministicLLMClient(), **kw)


@needs_chicago
def test_race_blind_scrubs_all_llm_visible_text():
    from . import ablations

    eng = _fresh_engine()
    eng.ensure_built()
    try:
        ablations.apply(eng, {"race_blind": True})
        env = eng.model.environment

        tid = env.tract_ids[0]
        desc = env.describe_tract_with_context_for_llm(tid)
        assert "Racial composition" not in desc
        assert "Non-Hispanic" not in desc and "Hispanic" not in desc

        arch = next(iter(eng.model.active_archetypes.values()))
        persona = arch.describe_for_llm()
        assert "Regarding neighborhood demographics" not in persona
        assert "Non-Hispanic" not in persona and "Hispanic/Latino" not in persona
        assert "demographic familiarity" not in persona

        import src.llm_client as _lc
        for prompt in (_lc.SATISFACTION_SYSTEM_PROMPT, _lc.MOVE_EVAL_SYSTEM_PROMPT):
            low = prompt.lower()
            assert "race" not in low and "demographic" not in low
            assert "valid JSON" in prompt          # output contract preserved
    finally:
        ablations.restore_prompts()                # don't leak the patch to other tests


@needs_chicago
def test_shuffle_permutes_profiles_but_preserves_demographics():
    from . import ablations

    eng = _fresh_engine()
    eng.ensure_built()
    gdf = eng.model.environment.gdf
    race_before = gdf["nh_black"].tolist()
    income_before = gdf["per_capita_income"].tolist()
    crime_before = gdf["crime_count_2010"].tolist()
    pop_before = {k: dict(v) for k, v in eng.model.environment.dynamic_pop.items()}

    ablations.apply(eng, {"shuffle_tract_attributes": 7})

    # multiset preserved, order changed, one permutation for all columns
    assert sorted(gdf["per_capita_income"].tolist()) == sorted(income_before)
    assert gdf["per_capita_income"].tolist() != income_before
    # income and crime moved with the SAME permutation (profiles move as blocks)
    pairs_before = sorted(zip(income_before, crime_before))
    pairs_after = sorted(zip(gdf["per_capita_income"].tolist(), gdf["crime_count_2010"].tolist()))
    assert pairs_before == pairs_after
    # demographics untouched
    assert gdf["nh_black"].tolist() == race_before
    assert {k: dict(v) for k, v in eng.model.environment.dynamic_pop.items()} == pop_before


@needs_chicago
def test_constraints_off_config_runs():
    pytest.importorskip("geopandas")
    pytest.importorskip("mesa")
    from . import model, rule_f
    cfg = {
        "seed": _SEED,
        "population": {"scale": "small", "init_mode": "perturbed_n",
                       "perturb_pct": 0.15, "perturb_radius_hops": 7},
        "dynamics": {"steps": 1, "move_budget_pct": 0.05},
        "behavior": {"mode": "rule", "homophily": 0.5, "search_radius_hops": 1},
        "model": {"own_race_ceiling_ratio": 0, "race_size_budget_scaling": False,
                  "per_step_tract_inflow_cap_ratio": 0, "enclave_candidate_bias": 0},
    }
    env, _ = model.build(cfg)
    actions = list(rule_f.chicago_rule_f(env.observe_batch()))
    env.apply(actions)
    env.advance()
    snap = env.snapshot()
    assert "D_black_white" in snap


@needs_chicago
@pytest.mark.parametrize("variant", ["calibrated_multigroup", "logit"])
def test_calibrated_rule_variants_run(variant):
    from . import rule_variants
    from ._engine import set_active
    from .env import ChicagoTractEnv

    eng = _fresh_engine()
    eng.ensure_built()
    set_active(eng)
    env = ChicagoTractEnv(eng, {"behavior": {"homophily": 0.5}})
    fn = rule_variants.get(variant)
    snaps = [env.snapshot()]
    for _ in range(2):
        actions = list(fn(env.observe_batch()))
        assert len(actions) == len(eng.agent_order)
        env.apply(actions)
        env.advance()
        snaps.append(env.snapshot())
    assert "D_black_white" in snaps[-1]
    # satisfaction was written for the budget gate (0-10 scale)
    sats = [a.satisfaction for a in eng.agent_order[:50]]
    assert all(0.0 <= s <= 10.0 for s in sats)
