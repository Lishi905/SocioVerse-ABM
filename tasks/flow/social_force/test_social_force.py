"""
Smoke test for Social Force evacuation. Rule mode needs no LLM; verifies that
pedestrians actually evacuate (fraction rises) and the vector-action consistency
plumbing works with a tolerance-based equality.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.flow.social_force import evaluate, llm_f, model, rule_f


class _FakeWalker:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=8, temperature=0.2):
        return "0"  # always head toward +x (the exit side)


def test_registered():
    assert "social_force" in registry.all_tasks()
    assert registry.get("social_force").family == "flow"


def test_rule_run_and_evacuation():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["n"] == 60
    assert m["evac_fraction"] > 0.5         # most pedestrians reach the exit
    assert a["metrics"] == b["metrics"]      # deterministic


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.social_force_rule_f(obs)
    llm_acts = llm_f.social_force_llm_f(obs, llm=_FakeWalker())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts, equals=evaluate.heading_equals(30))
    assert 0.0 <= score <= 1.0
