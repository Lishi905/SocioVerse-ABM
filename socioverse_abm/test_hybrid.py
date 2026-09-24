"""Tests for the hybrid routing kernel (socioverse_abm.behavior_engine.hybrid)."""
from __future__ import annotations

from socioverse_abm.behavior_engine.action import Action, Observation
from socioverse_abm.behavior_engine.hybrid import always_llm, hybrid_decide


def _obs(n):
    return [Observation(agent_id=i) for i in range(n)]


def _f(tag, calls):
    def f(batch):
        calls.append((tag, [o.agent_id for o in batch]))
        return [Action(agent_id=o.agent_id, kind=tag) for o in batch]
    return f


def test_default_routing_sends_every_agent_to_the_rule():
    calls = []
    d = hybrid_decide(_obs(4), _f("rule", calls), _f("llm", calls))
    assert calls == [("rule", [0, 1, 2, 3])]      # llm_f is never called
    assert d.source == ["rule"] * 4
    assert [a.kind for a in d.actions] == ["rule"] * 4


def test_always_llm_sends_every_agent_to_the_llm():
    calls = []
    d = hybrid_decide(_obs(3), _f("rule", calls), _f("llm", calls), route_to_llm=always_llm)
    assert calls == [("llm", [0, 1, 2])]
    assert d.source == ["llm"] * 3


def test_custom_policy_splits_and_keeps_agent_order():
    calls = []
    d = hybrid_decide(_obs(5), _f("rule", calls), _f("llm", calls),
                      route_to_llm=lambda o: o.agent_id % 2 == 1)
    assert calls == [("rule", [0, 2, 4]), ("llm", [1, 3])]   # one batch per f
    assert [a.agent_id for a in d.actions] == [0, 1, 2, 3, 4]
    assert d.source == ["rule", "llm", "rule", "llm", "rule"]
