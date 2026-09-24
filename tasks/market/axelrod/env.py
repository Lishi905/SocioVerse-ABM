"""
E — the round-robin arena. Every pair of agents plays an iterated Prisoner's Dilemma;
all matches advance one round per kernel step in lockstep (independent histories, so
this equals a standard Axelrod round-robin). The env tracks, for each ordered pair,
the last move played, and which opponents have ever defected against you.

The behavior is each agent's move toward every opponent this round (C/D); the env
scores the pairwise outcomes with the payoff matrix.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.market.axelrod.agents import Population


class AxelrodArena:
    def __init__(self, population: Population, cfg: dict):
        self.agents = population.agents
        self.ids = [a.agent_id for a in self.agents]
        self.strategy = {a.agent_id: a.strategy for a in self.agents}
        self.payoff = cfg["dynamics"]["payoff"]
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.last_move: Dict[tuple, str] = {}      # (i, j) -> i's last move toward j
        self.betrayed: Dict[int, set] = {i: set() for i in self.ids}
        self.score: Dict[int, float] = {i: 0.0 for i in self.ids}
        self._round_coop = 0
        self._round_total = 0
        return self.snapshot()

    def observe_batch(self) -> List[Observation]:
        obs = []
        for i in self.ids:
            opponents = {
                j: {"last": self.last_move.get((j, i)), "ever_defected": j in self.betrayed[i]}
                for j in self.ids if j != i
            }
            ctx = {"strategy": self.strategy[i], "opponents": opponents}
            obs.append(Observation(agent_id=i, state={"strategy": self.strategy[i]},
                                   context=ctx, rendered=self._render(self.strategy[i], len(opponents))))
        return obs

    def _render(self, strategy: str, n_opp: int) -> str:
        return (f"You are playing iterated Prisoner's Dilemma against {n_opp} opponents "
                f"(your disposition: {strategy}). For each, choose Cooperate or Defect.")

    def _pay(self, a: str, b: str):
        p = self.payoff
        if a == "C" and b == "C":
            return p["R"], p["R"]
        if a == "C" and b == "D":
            return p["S"], p["T"]
        if a == "D" and b == "C":
            return p["T"], p["S"]
        return p["P"], p["P"]

    def apply(self, actions: List[Action]) -> None:
        moves = {act.agent_id: act.payload["moves"] for act in actions}
        # record moves + grudges
        self._round_coop = self._round_total = 0
        for i in self.ids:
            for j, m in moves[i].items():
                self.last_move[(i, j)] = m
                if m == "D":
                    self.betrayed[j].add(i)
                self._round_total += 1
                self._round_coop += (m == "C")
        # score every unordered pair
        for i, j in combinations(self.ids, 2):
            pi, pj = self._pay(moves[i][j], moves[j][i])
            self.score[i] += pi
            self.score[j] += pj

    def advance(self) -> None:
        self.t += 1

    def _strategy_scores(self) -> Dict[str, float]:
        agg: Dict[str, list] = {}
        for i in self.ids:
            agg.setdefault(self.strategy[i], []).append(self.score[i])
        return {s: sum(v) / len(v) for s, v in agg.items()}

    def snapshot(self) -> dict:
        coop_rate = self._round_coop / self._round_total if self._round_total else 0.0
        return {
            "t": self.t,
            "n": len(self.ids),
            "cooperation_rate": coop_rate,
            "mean_score": sum(self.score.values()) / len(self.ids),
            "strategy_scores": self._strategy_scores(),
        }
