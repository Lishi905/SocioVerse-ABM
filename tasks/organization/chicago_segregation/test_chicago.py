"""Tests for the chicago_segregation task.

- registration is import-safe (no heavy geo deps needed just to discover the task);
- PARITY: the env-driven wrapper reproduces legacy ``SegregationModel.step()`` bit-for-bit
  under the DeterministicLLMClient (the proof that wrapping preserves the validated effect);
- rule mode runs offline (no API key) and moves the segregation indices.

The build/parity/rule tests require the `chicago` extra (geopandas, libpysal, mesa) and the
vendored data; they auto-skip if either is missing (mirrors SocioVerse2's Chicago test skip).
"""
from __future__ import annotations

import pathlib

import pytest

_LEGACY = pathlib.Path(__file__).with_name("legacy")
_DATA_OK = (_LEGACY / "processed_data" / "chicago_tracts.geojson").exists()


def test_registration():
    """Discoverable + correctly typed without importing any heavy dependency."""
    import tasks  # noqa: F401  triggers discovery
    from socioverse_abm.scenario_engine import registry

    spec = registry.get("chicago_segregation")
    assert spec.family == "organization"
    assert spec.rule_f is not None and spec.llm_f is not None
    assert spec.socioverse2_study == "chicago_schelling"


needs_chicago = pytest.mark.skipif(
    not _DATA_OK, reason="chicago extra / vendored data not available"
)

_STEPS = 4
_SEED = 42
_KW = {"move_budget_pct": 0.05}


@needs_chicago
def test_parity_wrapper_matches_legacy_step():
    """Env-driven wrapper == monolithic model.step() on identical metric trajectories."""
    pytest.importorskip("geopandas")
    pytest.importorskip("libpysal")
    pytest.importorskip("mesa")
    import numpy as np

    from ._engine import ChicagoEngine, set_active
    from ._fake_llm import DeterministicLLMClient
    from .env import ChicagoTractEnv
    from .llm_f import chicago_llm_f
    from . import rule_f

    # --- reference: drive legacy step() directly ---
    np.random.seed(_SEED)
    ref = ChicagoEngine(scale="small", init_mode="census", seed=_SEED,
                        llm_client=DeterministicLLMClient(), model_kwargs=_KW)
    m_ref = ref.ensure_built()
    for _ in range(_STEPS):
        m_ref.step()

    # --- wrapper: drive the kernel observe→decide→apply→snapshot loop ---
    np.random.seed(_SEED)
    wrp = ChicagoEngine(scale="small", init_mode="census", seed=_SEED,
                        llm_client=DeterministicLLMClient(), model_kwargs=_KW)
    wrp.ensure_built()
    set_active(wrp)
    rule_f.configure(0.5, 1)
    env = ChicagoTractEnv(wrp, {"behavior": {"homophily": 0.5}})
    snaps = [env.snapshot()]
    for _ in range(_STEPS):
        actions = list(chicago_llm_f(env.observe_batch()))
        env.apply(actions)
        env.advance()
        snaps.append(env.snapshot())

    assert len(snaps) == _STEPS + 1 == len(m_ref.metrics_history)
    keys = ["D_black_white", "D_hispanic_white", "D_asian_white",
            "Isolation_black", "Isolation_white", "R2_recovery"]
    for i in range(_STEPS + 1):
        ref_row = m_ref.metrics_history[i]
        for k in keys:
            assert abs(ref_row[k] - snaps[i][k]) < 1e-9, (
                f"parity broke at step {i}, metric {k}: "
                f"legacy={ref_row[k]} wrapper={snaps[i][k]}"
            )


@needs_chicago
def test_rule_mode_runs_offline_and_moves_indices():
    """Classical Schelling baseline runs with no API key and changes segregation."""
    pytest.importorskip("geopandas")
    pytest.importorskip("mesa")

    from . import model

    cfg = {
        "seed": _SEED,
        "population": {"scale": "small", "init_mode": "perturbed_n",
                       "perturb_pct": 0.15, "perturb_radius_hops": 7},
        "dynamics": {"steps": 3, "move_budget_pct": 0.05},
        "behavior": {"mode": "rule", "homophily": 0.5, "search_radius_hops": 1},
    }
    env, _ = model.build(cfg)
    from . import rule_f
    snaps = [env.snapshot()]
    for _ in range(cfg["dynamics"]["steps"]):
        actions = list(rule_f.chicago_rule_f(env.observe_batch()))
        env.apply(actions)
        env.advance()
        snaps.append(env.snapshot())

    assert "D_black_white" in snaps[0]
    assert len(snaps) == cfg["dynamics"]["steps"] + 1
    # At least one step produced movers (perturbed start is not at equilibrium).
    assert any(s.get("n_movers", 0) > 0 for s in snaps[1:])


def test_legacy_root_without_data_raises_clear_error(monkeypatch, tmp_path):
    """A regular (non-editable) install has the code but not the data: say what to do."""
    from . import _engine

    monkeypatch.delenv("SV_CHICAGO_LEGACY", raising=False)
    monkeypatch.delenv("SV_ABM_ROOT", raising=False)
    monkeypatch.setattr(_engine, "_LEGACY_ROOT", tmp_path / "legacy")
    with pytest.raises(RuntimeError, match="SV_CHICAGO_LEGACY"):
        _engine.legacy_root()

    # an explicit SV_CHICAGO_LEGACY is used as given
    monkeypatch.setenv("SV_CHICAGO_LEGACY", str(tmp_path / "elsewhere"))
    assert _engine.legacy_root() == tmp_path / "elsewhere"


@needs_chicago
def test_legacy_root_in_checkout_is_vendored_dir(monkeypatch):
    from . import _engine

    monkeypatch.delenv("SV_CHICAGO_LEGACY", raising=False)
    monkeypatch.delenv("SV_ABM_ROOT", raising=False)
    assert _engine.legacy_root() == _LEGACY.resolve()


@needs_chicago
def test_hybrid_without_key_runs_and_llm_f_refuses_without_llm(monkeypatch, tmp_path):
    """hybrid needs no key under the default routing (every agent follows the rule); the
    LLM behaviour function refuses an engine that has no LLM instead of falling back."""
    pytest.importorskip("geopandas")
    pytest.importorskip("mesa")
    import yaml

    from socioverse_abm.scenario_engine.registry import TaskSetupError

    from . import _engine, model
    from .llm_f import chicago_llm_f

    for var in ("SV_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_engine, "_load_dotenv", lambda *a, **k: None)   # ignore a local .env
    cfg = model.load_config()
    cfg["dynamics"]["steps"] = 2
    path = tmp_path / "chicago_small.yaml"
    path.write_text(yaml.safe_dump(cfg))

    out = model.run(mode="hybrid", config=str(path), seed=_SEED)
    assert len(out["snapshots"]) >= 2

    eng = _engine.get_active()
    assert not eng.has_llm
    with pytest.raises(TaskSetupError, match="SV_LLM_API_KEY"):
        chicago_llm_f([])

