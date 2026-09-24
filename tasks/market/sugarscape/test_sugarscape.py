"""
Smoke test for Sugarscape. Rule mode needs no LLM; verifies the simulation runs,
inequality is well-defined, and the discrete-move consistency plumbing works.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.market.sugarscape import llm_f, model, rule_f


class _FakeForager:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=8, temperature=0.2):
        return "0"  # always stay put


def test_registered():
    assert "sugarscape" in registry.all_tasks()
    assert registry.get("sugarscape").family == "market"


def test_rule_run_and_determinism():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["alive_start"] == 200
    assert 0.0 <= m["gini"] <= 1.0
    assert a["metrics"] == b["metrics"]


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    rule_f.seed(cfg["seed"])
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.sugarscape_rule_f(obs)
    llm_acts = llm_f.sugarscape_llm_f(obs, llm=_FakeForager())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts)  # discrete: strict match
    assert 0.0 <= score <= 1.0
