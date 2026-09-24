"""
Minimal smoke test for the SIR task — the per-task testing template.
Rule mode needs no LLM, so it runs in CI without keys.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.diffusion.sir import llm_f, model, rule_f


class _FakeLLM:
    """Deterministic stand-in so the LLM contract is exercised without openai."""

    def generate_and_parser(self, _model, _prompt):
        return {"decision": "spread"}


def test_registered():
    assert "sir" in registry.all_tasks()
    assert registry.get("sir").family == "diffusion"


def test_rule_run_and_determinism():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["n"] == 500
    assert 0 < m["final_reach"] <= m["n"]
    assert a["metrics"] == b["metrics"]  # same seed -> identical


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    rule_f.seed(cfg["seed"])
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.sir_rule_f(obs)
    llm_acts = llm_f.sir_llm_f(obs, llm=_FakeLLM())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts)
    assert 0.0 <= score <= 1.0
