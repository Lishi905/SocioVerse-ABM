"""Tests for the `sv-abm run` error paths: a missing optional dependency, a missing
LLM key and an incomplete local setup each print a hint and exit with status 2
(no traceback). The tasks used here are throwaway registrations, so no heavy
dependency, network access or API key is needed."""
from __future__ import annotations

import pytest

from socioverse_abm import runner
from socioverse_abm.scenario_engine import registry

_KEY_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "SV_LLM_API_KEY", "SV_LLM_BASE_URL")


@pytest.fixture
def fake_task(monkeypatch):
    """Register a throwaway task whose run() is the given callable."""
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)

    def _make(run, name="_runner_test_task", **meta):
        spec = registry.TaskSpec(name=name, family="organization", run=run, meta=dict(meta))
        monkeypatch.setitem(registry._REGISTRY, name, spec)
        return name

    return _make


def _must_not_run(**_kwargs):
    raise AssertionError("run() must not be called")


def test_missing_chicago_extra_prints_install_hint(fake_task, capsys):
    def run(**_kwargs):
        raise ModuleNotFoundError("No module named 'mesa'", name="mesa")

    name = fake_task(run)
    assert runner.main(["run", name, "--mode", "rule"]) == 2
    err = capsys.readouterr().err
    assert "mesa" in err
    assert 'pip install -e ".[chicago]"' in err
    assert "Traceback" not in err


def test_missing_kernel_dependency_prints_install_hint(fake_task, capsys):
    def run(**_kwargs):
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    name = fake_task(run)
    assert runner.main(["run", name, "--mode", "rule"]) == 2
    err = capsys.readouterr().err
    assert "pip install -e ." in err
    assert ".[chicago]" not in err


def test_socks_proxy_import_error_prints_hint(fake_task, capsys):
    def run(**_kwargs):
        raise ImportError("Using SOCKS proxy, but the 'socksio' package is not installed.")

    name = fake_task(run)
    assert runner.main(["run", name]) == 2
    assert 'httpx[socks]' in capsys.readouterr().err


def test_llm_mode_without_key_prints_how_to_set_it(fake_task, capsys):
    name = fake_task(_must_not_run)
    assert runner.main(["run", name, "--mode", "llm"]) == 2
    err = capsys.readouterr().err
    assert "export OPENAI_API_KEY=" in err
    assert "OPENAI_BASE_URL" in err
    assert "--mode rule" in err


def test_hybrid_mode_without_key_runs_with_a_note(fake_task, capsys):
    """The default hybrid routing (always_rule) never calls the LLM, so no key is needed."""
    calls = []
    name = fake_task(lambda **kw: calls.append(kw))
    assert runner.main(["run", name, "--mode", "hybrid"]) == 0
    assert calls == [{"mode": "hybrid", "config": None}]
    err = capsys.readouterr().err
    assert "no LLM API key set" in err
    assert "always_rule" in err


def test_hybrid_mode_with_key_prints_no_note(fake_task, monkeypatch, capsys):
    name = fake_task(lambda **kw: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert runner.main(["run", name, "--mode", "hybrid"]) == 0
    assert capsys.readouterr().err == ""


class OpenAIError(Exception):
    """Stand-in for openai.OpenAIError (the CI job does not install openai)."""


OpenAIError.__module__ = "openai"

# The client's missing-credentials messages (openai 1.x, then 3.x).
_NO_KEY_MESSAGES = (
    "The api_key client option must be set either by passing api_key to the client "
    "or by setting the OPENAI_API_KEY environment variable",
    "Missing credentials. Please pass an `api_key`, `workload_identity`, `admin_api_key`, "
    "or set the `OPENAI_API_KEY` or `OPENAI_ADMIN_KEY` environment variable.",
)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("message", _NO_KEY_MESSAGES)
def test_hybrid_route_reaching_the_llm_without_key_prints_how_to_set_it(
    fake_task, capsys, wrapped, message
):
    """A custom route_to_llm policy that calls the LLM without a key: hint, status 2."""
    def run(**_kwargs):
        if not wrapped:
            raise OpenAIError(message)
        try:
            raise OpenAIError(message)
        except OpenAIError as exc:  # llm_f.generate_and_parser re-raises as Exception(e)
            raise Exception(exc)

    name = fake_task(run)
    assert runner.main(["run", name, "--mode", "hybrid"]) == 2
    err = capsys.readouterr().err
    assert "export OPENAI_API_KEY=" in err
    assert "Traceback" not in err


def test_hybrid_mode_other_errors_still_raise(fake_task):
    def run(**_kwargs):
        raise ValueError("a real bug")

    name = fake_task(run)
    with pytest.raises(ValueError):
        runner.main(["run", name, "--mode", "hybrid"])


def test_hybrid_run_of_a_real_task_needs_no_key_and_matches_rule(fake_task, monkeypatch, tmp_path, capsys):
    """sv-abm run <task> --mode hybrid, no key: no LLM client is created and the
    output equals the rule run (default routing sends every agent to the rule)."""
    import yaml

    import tasks  # noqa: F401  triggers discovery
    from socioverse_abm.behavior_engine import llm_f as kernel_llm
    from tasks.diffusion.hegselmann_krause import model as hk_model

    def _no_client():
        raise AssertionError("hybrid with the default routing must not create an LLM client")

    monkeypatch.setattr(kernel_llm, "get_client", _no_client)
    cfg = hk_model.load_config()
    cfg["population"]["n"] = 12
    cfg["dynamics"]["steps"] = 5
    path = tmp_path / "hk_small.yaml"
    path.write_text(yaml.safe_dump(cfg))

    assert runner.main(["run", "hegselmann_krause", "--mode", "rule", "--config", str(path)]) == 0
    rule_out = capsys.readouterr().out
    assert runner.main(["run", "hegselmann_krause", "--mode", "hybrid", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert "no LLM API key set" in captured.err
    assert captured.out.replace("/hybrid]", "/rule]") == rule_out


def test_llm_mode_with_key_runs(fake_task, monkeypatch):
    calls = []
    name = fake_task(lambda **kw: calls.append(kw))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert runner.main(["run", name, "--mode", "llm"]) == 0
    assert calls == [{"mode": "llm", "config": None}]


def test_task_specific_key_names_and_env_loader(fake_task, monkeypatch, capsys):
    """A task may accept other key names (Chicago: SV_LLM_API_KEY) and hydrate a .env."""
    calls = []
    keys = ("SV_LLM_API_KEY", "OPENAI_API_KEY")
    name = fake_task(
        lambda **kw: calls.append(kw),
        llm_key_envs=keys,
        llm_base_url_envs=("SV_LLM_BASE_URL", "OPENAI_BASE_URL"),
        load_env=lambda: None,
    )
    assert runner.main(["run", name, "--mode", "llm"]) == 2
    err = capsys.readouterr().err
    assert "SV_LLM_API_KEY or OPENAI_API_KEY" in err
    assert "export SV_LLM_BASE_URL=" in err

    monkeypatch.setenv("SV_LLM_API_KEY", "sk-test")
    assert runner.main(["run", name, "--mode", "llm"]) == 0
    assert len(calls) == 1

    # the loader runs before the check, so a key it provides counts
    monkeypatch.delenv("SV_LLM_API_KEY")
    name2 = fake_task(
        lambda **kw: calls.append(kw),
        name="_runner_test_task_dotenv",
        llm_key_envs=keys,
        load_env=lambda: monkeypatch.setenv("SV_LLM_API_KEY", "sk-from-dotenv"),
    )
    assert runner.main(["run", name2, "--mode", "hybrid"]) == 0
    assert len(calls) == 2


def test_chicago_declares_sv_llm_key():
    import tasks  # noqa: F401  triggers discovery

    meta = registry.get("chicago_segregation").meta
    assert meta["llm_key_envs"] == ("SV_LLM_API_KEY", "OPENAI_API_KEY")
    assert callable(meta["load_env"])


def test_task_setup_error_is_reported_without_traceback(fake_task, capsys):
    def run(**_kwargs):
        raise registry.TaskSetupError("Chicago data not found: set SV_CHICAGO_LEGACY")

    name = fake_task(run)
    assert runner.main(["run", name]) == 2
    err = capsys.readouterr().err
    assert "SV_CHICAGO_LEGACY" in err
    assert "Traceback" not in err


def test_unexpected_errors_still_raise(fake_task):
    def run(**_kwargs):
        raise ValueError("a real bug")

    name = fake_task(run)
    with pytest.raises(ValueError):
        runner.main(["run", name])


def test_hybrid_route_to_llm_without_key_real_client(fake_task, monkeypatch, tmp_path, capsys):
    """End to end with the real OpenAI client: a task routed with always_llm, no key."""
    pytest.importorskip("openai")
    import yaml

    import tasks  # noqa: F401  triggers discovery
    from socioverse_abm.behavior_engine import llm_f as kernel_llm
    from socioverse_abm.behavior_engine.hybrid import always_llm, hybrid_decide
    from tasks.diffusion.hegselmann_krause import llm_f as hk_llm, model as hk_model, rule_f as hk_rule

    monkeypatch.setattr(kernel_llm, "_client", None)
    monkeypatch.setattr(
        hk_model, "_behavior_fn",
        lambda mode: (lambda obs: hybrid_decide(obs, hk_rule.hk_rule_f, hk_llm.hk_llm_f,
                                                route_to_llm=always_llm).actions),
    )
    cfg = hk_model.load_config()
    cfg["population"]["n"] = 3
    cfg["dynamics"]["steps"] = 1
    path = tmp_path / "hk_tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))

    assert runner.main(["run", "hegselmann_krause", "--mode", "hybrid", "--config", str(path)]) == 2
    err = capsys.readouterr().err
    assert "export OPENAI_API_KEY=" in err
    assert kernel_llm._client is None
