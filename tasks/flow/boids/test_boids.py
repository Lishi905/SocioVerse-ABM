"""
Smoke test for the Boids task. Rule mode needs no LLM. Verifies that flocking
actually emerges (polarization rises) and that the continuous-vector consistency
plumbing works with a tolerance-based equality.
"""
import numpy as np

from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.flow.boids import evaluate, llm_f, model, rule_f


class _FakeBird:
    """Deterministic kernel-LLM stand-in: always answers heading '90'."""

    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=8, temperature=0.2):
        return "90"


def test_registered():
    assert "boids" in registry.all_tasks()
    assert registry.get("boids").family == "flow"


def test_rule_run_and_flocking():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["n"] == 40
    # flocking emerges: final polarization clearly higher than the random start
    assert m["polarization_final"] > a["snapshots"][0]["polarization"] + 0.1
    assert a["metrics"] == b["metrics"]  # deterministic


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.boids_rule_f(obs)
    llm_acts = llm_f.boids_llm_f(obs, llm=_FakeBird())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts, equals=evaluate.heading_equals(30))
    assert 0.0 <= score <= 1.0
