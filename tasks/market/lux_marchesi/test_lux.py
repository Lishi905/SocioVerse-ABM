"""
Smoke test for Lux-Marchesi. Rule mode needs no LLM; verifies the market runs and
produces price movement, and the discrete-stance consistency plumbing works.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.market.lux_marchesi import llm_f, model, rule_f


class _FakeTrader:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=4, temperature=0.2):
        return "optimist"


def test_registered():
    assert "lux_marchesi" in registry.all_tasks()
    assert registry.get("lux_marchesi").family == "market"


def test_rule_run_and_determinism():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["price_max"] > m["price_min"]      # price actually moves
    assert m["volatility"] >= 0.0
    assert a["metrics"] == b["metrics"]


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    rule_f.seed(cfg["seed"])
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.lux_rule_f(obs)
    llm_acts = llm_f.lux_llm_f(obs, llm=_FakeTrader())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    score = sv_eval.step_consistency(rule_acts, llm_acts)  # discrete stance: strict
    assert 0.0 <= score <= 1.0
