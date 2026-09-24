"""
Consistency-score evaluator — the kernel-level metric that asks the project's
core question: *does the LLM behavior function reproduce the rule-based one?*

The companion ABM study reports an agent-level agreement (~0.9) between LLM-f and rule-f
behavior. This module makes that a single reusable function so every task scores
the same way instead of re-implementing it. Pure-Python; numpy is optional.

Definitions
-----------
- Per-step consistency: fraction of agents whose LLM action == rule action at
  that step (Action.matches: same kind + payload).
- Trajectory consistency: mean of per-step consistency over all steps.

Calibration note: the exact reported figure (90.5%) depends on each task's action
equivalence; tasks may pass a custom `equals` for fuzzy matching (e.g. position
within tolerance). The default uses strict Action.matches.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from socioverse_abm.behavior_engine.action import Action

ActionEq = Callable[[Action, Action], bool]


def _default_eq(a: Action, b: Action) -> bool:
    return a.matches(b)


def step_consistency(
    rule_actions: Sequence[Action],
    llm_actions: Sequence[Action],
    equals: ActionEq = _default_eq,
) -> float:
    """Agreement rate at one step. Actions are paired by position; both
    sequences must describe the same agents in the same order."""
    if len(rule_actions) != len(llm_actions):
        raise ValueError(
            f"step length mismatch: rule={len(rule_actions)} llm={len(llm_actions)}"
        )
    if not rule_actions:
        return 1.0
    agree = sum(1 for r, l in zip(rule_actions, llm_actions) if equals(r, l))
    return agree / len(rule_actions)


def trajectory_consistency(
    rule_traj: Sequence[Sequence[Action]],
    llm_traj: Sequence[Sequence[Action]],
    equals: ActionEq = _default_eq,
) -> dict:
    """Mean agreement over a full run. Returns {'mean': float, 'per_step': [...]}.

    This is the headline agent-level agreement number (the ABM study reports ~0.9).
    """
    if len(rule_traj) != len(llm_traj):
        raise ValueError(
            f"trajectory length mismatch: rule={len(rule_traj)} llm={len(llm_traj)}"
        )
    per_step = [step_consistency(r, l, equals) for r, l in zip(rule_traj, llm_traj)]
    mean = sum(per_step) / len(per_step) if per_step else 1.0
    return {"mean": mean, "per_step": per_step}


def heading_equals(tol_deg: float = 30.0,
                   vx: str = "vx", vy: str = "vy") -> Callable[[Action, Action], bool]:
    """Tolerance equality for continuous *vector* actions (Boids, Social Force): two
    actions agree if their heading (atan2 of the payload's vx/vy) is within tol_deg.
    Shared here so every continuous-action task scores consistency the same way."""
    tol = math.radians(tol_deg)

    def equals(a: Action, b: Action) -> bool:
        ha = math.atan2(a.payload[vy], a.payload[vx])
        hb = math.atan2(b.payload[vy], b.payload[vx])
        diff = abs(ha - hb) % (2 * math.pi)
        return min(diff, 2 * math.pi - diff) <= tol

    return equals
