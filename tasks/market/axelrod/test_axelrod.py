"""
Smoke test for the Axelrod tournament. Rule mode needs no LLM; verifies the
tournament runs, produces a strategy ranking, and the per-opponent-move consistency
plumbing works. always_defect should never be beaten head-to-head, but cooperative
strategies should accumulate competitive total scores.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.market.axelrod import llm_f, model, rule_f


class _FakeOpponent:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=2, temperature=0.2):
        return "C"


def test_registered():
    assert "axelrod" in registry.all_tasks()
    assert registry.get("axelrod").family == "market"


def test_rule_run_and_determinism():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert 0.0 <= m["cooperation_rate"] <= 1.0
    assert len(m["strategy_ranking"]) == 5
    assert a["metrics"] == b["metrics"]


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    rule_f.seed(cfg["seed"])
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.axelrod_rule_f(obs)
    llm_acts = llm_f.axelrod_llm_f(obs, llm=_FakeOpponent())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts)  # strict per-agent moves dict
    assert 0.0 <= score <= 1.0
