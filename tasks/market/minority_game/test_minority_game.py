"""
Smoke test for the Minority Game. Rule mode needs no LLM; verifies the game runs,
volatility is well-defined, and the discrete-side consistency plumbing works.
"""
from socioverse_abm import eval as sv_eval
from socioverse_abm.scenario_engine import registry

from tasks.market.minority_game import llm_f, model, rule_f


class _FakePlayer:
    def generate(self, _model, _prompt, sys_prompt=None, max_tokens=4, temperature=0.2):
        return "1"


def test_registered():
    assert "minority_game" in registry.all_tasks()
    assert registry.get("minority_game").family == "market"


def test_rule_run_and_determinism():
    a = model.run("rule")
    b = model.run("rule")
    m = a["metrics"]
    assert m["n"] == 101
    assert m["volatility"] >= 0.0
    assert a["metrics"] == b["metrics"]


def test_llm_contract_and_consistency():
    cfg = model.load_config()
    rule_f.seed(cfg["seed"])
    env, _ = model.build(cfg)
    obs = env.observe_batch()
    rule_acts = rule_f.minority_game_rule_f(obs)
    llm_acts = llm_f.minority_game_llm_f(obs, llm=_FakePlayer())
    assert len(rule_acts) == len(llm_acts) == len(obs)
    for a in llm_acts:
        assert a.payload["side"] in (0, 1)
    score = sv_eval.step_consistency(rule_acts, llm_acts)  # discrete: strict
    assert 0.0 <= score <= 1.0
