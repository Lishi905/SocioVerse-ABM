"""
Hybrid behavior dispatch: route each agent's decision to the rule-based f or the
LLM-based f according to a policy. This is the companion ABM study's "hybrid"
design knob and the third value of `--mode {rule, llm, hybrid}`.

The kernel only provides the *routing*; the actual rule_f / llm_f are supplied by
each task (registered in `scenario_engine.registry`). Dependency-free.

Default routing: `hybrid_decide` uses `always_rule` unless a caller passes
`route_to_llm`, and every shipped task calls it without one. So `sv-abm run <task>
--mode hybrid` sends every agent to the rule, makes no LLM call, needs no API key
and reproduces the `--mode rule` run. To mix the two, pass a policy where the
task's `model.py` (`_behavior_fn`) calls `hybrid_decide`, for example
`route_to_llm=always_llm` or any predicate `Observation -> bool` (say, on
`obs.agent_id` or a field of `obs.context`). A policy that routes agents to the LLM
needs the task's API key, as `--mode llm` does.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from socioverse_abm.behavior_engine.action import Action, Decision, Observation

# A behavior function maps a batch of observations to a batch of actions.
BehaviorFn = Callable[[Sequence[Observation]], Sequence[Action]]
# A routing policy decides, per observation, whether to use the LLM path.
RoutePolicy = Callable[[Observation], bool]


def always_rule(_obs: Observation) -> bool:
    return False


def always_llm(_obs: Observation) -> bool:
    return True


def hybrid_decide(
    obs: Sequence[Observation],
    rule_f: BehaviorFn,
    llm_f: BehaviorFn,
    route_to_llm: RoutePolicy = always_rule,
) -> Decision:
    """Split `obs` by `route_to_llm`, call each f once on its slice, then stitch
    the results back into original order. Calling each f on a *batch* (not per
    agent) preserves the batched-LLM efficiency the tasks rely on.
    """
    llm_idx = [i for i, o in enumerate(obs) if route_to_llm(o)]
    rule_idx = [i for i, o in enumerate(obs) if i not in set(llm_idx)]

    out: list[Action | None] = [None] * len(obs)
    src: list[str] = [""] * len(obs)

    for idx, f, tag in ((rule_idx, rule_f, "rule"), (llm_idx, llm_f, "llm")):
        if not idx:
            continue
        results = list(f([obs[i] for i in idx]))
        if len(results) != len(idx):
            raise ValueError(f"{tag}_f returned {len(results)} actions for {len(idx)} obs")
        for slot, action in zip(idx, results):
            out[slot] = action
            src[slot] = tag

    return Decision(actions=[a for a in out if a is not None], source=src)
