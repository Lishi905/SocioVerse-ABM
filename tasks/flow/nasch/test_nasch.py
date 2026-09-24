"""
Smoke test for the NaSch task (per-task testing template). Rule mode needs
no LLM; the LLM contract is exercised with a deterministic fake driver.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.flow.nasch import llm_f, model, rule_f


class _FakeDriver:
    """Stand-in kernel LLM: always answers the max plausible speed as bare text."""

    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=16, temperature=0.1):
        return "3"


def test_registered():
    assert "nasch" in registry.all_tasks()
    assert registry.get("nasch").family == "flow"


def test_rule_run_and_determinism():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["n"] > 0
    assert 0.0 <= m["mean_speed"] <= model.load_config()["dynamics"]["vmax"]
    assert m["flow"] >= 0.0
    assert a["metrics"] == b["metrics"]  # same seed -> identical


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    rule_f.seed(cfg["seed"])
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.nasch_rule_f(obs)
    llm_acts = llm_f.nasch_llm_f(obs, llm=_FakeDriver())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    # every action is a legal speed
    for a in llm_acts:
        assert 0 <= a.payload["speed"] <= cfg["dynamics"]["vmax"]
    score = sv_eval.step_consistency(rule_acts, llm_acts)
    assert 0.0 <= score <= 1.0
