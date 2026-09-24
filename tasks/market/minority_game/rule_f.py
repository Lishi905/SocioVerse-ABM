"""
f (rule) — the Minority Game decision: use the strategy with the highest virtual
score (random tie-break) and read off its action for the current history. The parity
baseline. Counterfactual score updates are env mechanics, not the decision.
"""
from __future__ import annotations

import random
from typing import List, Sequence

from socioverse_abm.behavior_engine.action import Action, Observation

_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def minority_game_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        vs = c["virtual_scores"]
        max_score = vs.max()
        best = [i for i in range(len(vs)) if vs[i] == max_score]
        chosen = _RNG.choice(best)
        side = int(c["strategies"][chosen][c["history_index"]])
        actions.append(Action(agent_id=o.agent_id, kind="choose", payload={"side": side}))
    return actions
