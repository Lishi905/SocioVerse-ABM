"""
Smoke test for Civil Violence. Rule mode needs no LLM and must produce some rebellion;
the test also checks all four context x llm_behavior variants build a prompt and run
via a fake LLM, and that the rebel/quiet consistency plumbing works.
"""
import copy

from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.organization.civil_violence import llm_f, model, rule_f


class _FakeCitizen:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=4, temperature=0.2):
        return "quiet"


def test_registered():
    assert "civil_violence" in registry.all_tasks()
    assert registry.get("civil_violence").family == "organization"


def test_rule_run_and_unrest():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["peak_active"] > 0            # rebellion does break out
    assert m["active_std"] >= 0.0
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
            acts = llm_f.civil_violence_llm_f(obs, llm=_FakeCitizen())
            assert len(acts) == len(obs)
            assert all(a.kind in ("active", "quiet") for a in acts)


def test_consistency_plumbing():
    cfg = model.load_config()
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.civil_violence_rule_f(obs)
    llm_acts = llm_f.civil_violence_llm_f(obs, llm=_FakeCitizen())
    score = sv_eval.step_consistency(rule_acts, llm_acts)
    assert 0.0 <= score <= 1.0
