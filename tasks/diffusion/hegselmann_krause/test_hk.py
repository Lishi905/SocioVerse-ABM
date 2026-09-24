"""
Smoke test for Hegselmann-Krause. Rule mode needs no LLM; verifies that opinions
coalesce (fewer clusters than the random start) and the float-opinion consistency
plumbing works with a tolerance equality.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.diffusion.hegselmann_krause import evaluate, llm_f, model, rule_f


class _FakeOpinion:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=8, temperature=0.2):
        return "0.5"


def test_registered():
    assert "hegselmann_krause" in registry.all_tasks()
    assert registry.get("hegselmann_krause").socioverse2_study == "abm_hegselmann_krause"


def test_rule_run_and_coalescence():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["n"] == 100
    # bounded-confidence dynamics reduce the number of opinion clusters
    assert m["num_clusters"] < m["converged_from"]
    assert a["metrics"] == b["metrics"]


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.hk_rule_f(obs)
    llm_acts = llm_f.hk_llm_f(obs, llm=_FakeOpinion())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts, equals=evaluate.opinion_equals(0.05))
    assert 0.0 <= score <= 1.0
