"""
E — the market. Aggregates the traders' stances into excess demand and updates the
price; the fundamental value follows a random walk. The "environment" each trader
perceives is the market state (price, value, crowd mood, recent trend).

Synchronous step: observe (current market) -> behavior picks each trader's next
stance -> apply (update stances, then clear the market into a new price).
"""
from __future__ import annotations

import random
from typing import List

from socioverse_abm.behavior_engine.action import Action, Observation

from tasks.market.lux_marchesi.agents import OPTIMIST, PESSIMIST, FUNDAMENTALIST, Population


class LuxMarket:
    def __init__(self, population: Population, cfg: dict):
        d = cfg["dynamics"]
        self._pop = population
        self.tc, self.gamma, self.beta = d["tc"], d["gamma"], d["beta"]
        self.a1, self.a2, self.dt = d["a1"], d["a2"], d["dt"]
        self.value_sigma = d["value_sigma"]
        self.p0, self.v0 = d["market_price"], d["market_value"]
        self.rng = random.Random(cfg["seed"])
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.price = self.p0
        self.value = self.v0
        self.prev_price = self.p0
        self.traders = self._pop.traders
        for tr, s in zip(self.traders, self._pop.traders):
            tr.stance = s.stance
        return self.snapshot()

    def _counts(self):
        n_opt = sum(1 for t in self.traders if t.stance == OPTIMIST)
        n_pess = sum(1 for t in self.traders if t.stance == PESSIMIST)
        n_fund = len(self.traders) - n_opt - n_pess
        return n_opt, n_pess, n_fund

    def opinion_index(self) -> float:
        n_opt, n_pess, _ = self._counts()
        nc = n_opt + n_pess
        return (n_opt - n_pess) / nc if nc else 0.0

    def trend(self) -> float:
        return (self.price - self.prev_price) / self.prev_price if self.prev_price else 0.0

    def observe_batch(self) -> List[Observation]:
        x = self.opinion_index()
        tr = self.trend()
        ctx_common = {"price": self.price, "value": self.value, "opinion_index": x,
                      "trend": tr, "a1": self.a1, "a2": self.a2}
        obs = []
        for t in self.traders:
            ctx = dict(ctx_common, stance=t.stance)
            obs.append(Observation(agent_id=t.agent_id, state={"stance": t.stance},
                                   context=ctx, rendered=self._render(t.stance, x, tr)))
        return obs

    def _render(self, stance: str, x: float, tr: float) -> str:
        mood = "bullish" if x > 0 else "bearish" if x < 0 else "balanced"
        return (f"You are a {stance} trader. Price={self.price:.2f}, fundamental "
                f"value={self.value:.2f}, crowd mood index={x:+.2f} ({mood}), recent "
                f"return={tr:+.3%}. Your stance next (optimist/pessimist/fundamentalist):")

    def apply(self, actions: List[Action]) -> None:
        by_id = {a.agent_id: a.kind for a in actions}
        for t in self.traders:
            t.stance = by_id.get(t.agent_id, t.stance)
        # Clear the market: excess demand -> price move.
        n_opt, n_pess, n_fund = self._counts()
        excess_demand = self.tc * (n_opt - n_pess) + self.gamma * n_fund * (self.value - self.price)
        self.prev_price = self.price
        self.price = max(1e-6, self.price + self.beta * self.dt * excess_demand)
        self.value = max(1e-6, self.value + self.rng.gauss(0.0, self.value_sigma * self.value))

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        import math
        ret = math.log(self.price / self.prev_price) if self.prev_price > 0 else 0.0
        return {"t": self.t, "price": self.price, "value": self.value,
                "opinion_index": self.opinion_index(), "log_return": ret}
