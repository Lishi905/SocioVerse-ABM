"""
Smoke test for Schelling. Rule mode needs no LLM and must produce segregation; the
test also checks all four context x llm_behavior variants build a prompt and run via
a fake LLM, and that the move/stay consistency plumbing works.
"""
import copy

from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.organization.schelling import llm_f, model, rule_f


class _FakeMover:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=4, temperature=0.2):
        return "stay"


def test_registered():
    assert "schelling" in registry.all_tasks()
    assert registry.get("schelling").family == "organization"


def test_rule_run_and_segregation():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    # segregation rises well above the 50/50 baseline
    assert m["segregation_final"] > m["segregation_start"]
    assert m["segregation_final"] > 0.6
    assert a["metrics"] == b["metrics"]


def test_four_variants_build_and_run():
    base = model.load_config()
    for context in ("ocm", "lcm"):
        for behavior in ("tbf", "lbf"):
            cfg = copy.deepcopy(base)
            cfg["behavior"]["context"] = context
            cfg["behavior"]["llm_behavior"] = behavior
            llm_f.configure(cfg["behavior"]["llm_model"], behavior)
            env, _ = model.build(cfg)
            obs = env.observe_batch()
            acts = llm_f.schelling_llm_f(obs, llm=_FakeMover())
            assert len(acts) == len(obs)
            assert all(a.kind in ("move", "stay") for a in acts)


def test_consistency_plumbing():
    cfg = model.load_config()
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.schelling_rule_f(obs)
    llm_acts = llm_f.schelling_llm_f(obs, llm=_FakeMover())
    score = sv_eval.step_consistency(rule_acts, llm_acts)
    assert 0.0 <= score <= 1.0
