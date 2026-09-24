"""E — the Chicago tract environment.

Wraps the vendored ``TractEnvironment`` (781 census tracts, Queen-contiguity graph,
dynamic per-tract racial population) and the model's own step phases, exposing the
kernel's duck-typed env protocol: reset / observe_batch / apply / advance / snapshot.

Split of legacy ``SegregationModel.step()`` across the kernel loop
(observe → decide → apply → snapshot):

    observe_batch()  → one Observation per household (rendered tract context)
    behavior(f)      → decide: sets group_would_move / move_targets  (llm_f or rule_f)
    apply(actions)   → snapshot prev, reset inflow, _apply_move_budget + shuffle_do("step")
    snapshot()       → _compute_metrics (+ movers + direction), kept in lockstep with
                       legacy via the len(metrics_history) step counter.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from socioverse_abm.behavior_engine.action import Action, Observation

_RACE_COLS = ["nh_white", "nh_black", "nh_asian", "hispanic", "nh_other"]


class ChicagoTractEnv:
    def __init__(self, engine, cfg: dict):
        self.engine = engine
        self.cfg = cfg
        self.threshold = cfg["behavior"].get("homophily", 0.5)
        self._prev_race_df = None
        self.reset()

    @property
    def model(self):
        return self.engine.model

    def reset(self) -> dict:
        self.engine.ensure_built()
        self.t = 0
        self._prev_race_df = None
        return self.snapshot()

    # --- perception (E projection fed to f) ---
    def observe_batch(self) -> List[Observation]:
        m = self.model
        env = m.environment
        city = dict(getattr(m, "city_race_share", {}))
        desc_cache: dict[str, str] = {}
        surround_cache: dict[str, dict] = {}
        obs: List[Observation] = []
        for agent in self.engine.agent_order:
            tid = agent.tract_id
            if tid not in surround_cache:
                surround_cache[tid] = env.get_surrounding_area_summary(tid)
                desc_cache[tid] = env.describe_tract_with_context_for_llm(tid)
            surrounding = surround_cache[tid]
            own_share = surrounding.get(f"pct_{agent.race}", 0.0) / 100.0
            ctx = {
                "tract_id": tid,
                "race": agent.race,
                "own_race_share": own_share,     # own-race share of the 1-hop neighbourhood
                "threshold": self.threshold,     # τ (rule mode)
                "surrounding": surrounding,
                "city_race_share": city,
            }
            obs.append(Observation(
                agent_id=agent._sv_id,
                state={
                    "tract_id": tid,
                    "race": agent.race,
                    "income_bracket": agent.archetype.income_bracket,
                    "satisfaction": round(float(agent.satisfaction), 3),
                    "pop_count": int(agent.pop_count),
                },
                context=ctx,
                rendered=desc_cache[tid],
            ))
        return obs

    # --- execution (validated legacy machinery, reused verbatim for rule AND llm) ---
    def apply(self, actions: List[Action]) -> None:
        m = self.model
        # Composition BEFORE this step's moves — for the direction metric.
        self._prev_race_df = m.environment.get_racial_composition_array().copy()
        m._step_inflow = defaultdict(int)
        m._apply_move_budget()          # per-race budget gate (same budget for rule & llm)
        m.agents.shuffle_do("step")     # move execution (capacity + inflow caps)

    def advance(self) -> None:
        self.t += 1

    # --- metrics (kept in len(metrics_history) lockstep with legacy) ---
    def snapshot(self) -> dict:
        m = self.model
        if self.t == 0:
            # __init__ already appended the step-0 metrics; return verbatim and DO NOT
            # append, so len(metrics_history) stays 1 and the first move-step sees
            # current_step == 1 exactly like legacy (keeps the 3-step cascade aligned).
            return dict(m.metrics_history[0]) if m.metrics_history else {"step": 0}

        metrics = m._compute_metrics()
        agents = list(m.agents)
        movers = [a for a in agents if a.moved_this_step]
        total_pop = sum(a.pop_count for a in agents) or 1
        metrics["n_movers"] = len(movers)
        metrics["pct_movers"] = len(movers) / len(agents) if agents else 0.0
        metrics["pop_moved"] = sum(a.pop_count for a in movers)
        metrics["pct_pop_moved"] = metrics["pop_moved"] / total_pop
        if self._prev_race_df is not None:
            metrics.update(m._compute_direction_metrics(self._prev_race_df, movers))

        # Advance the model's own history: step == len BEFORE append (matches legacy).
        metrics["step"] = len(m.metrics_history)
        m.metrics_history.append(metrics)
        return metrics
