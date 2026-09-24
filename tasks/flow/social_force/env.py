"""
E — bounded evacuation room, built on ContinuousField2D with wrap=False (agents
clamp at the walls, they do not wrap). This exercises the field base's non-toroidal
path, complementing Boids' toroidal use.

Pedestrians that reach the exit are marked evacuated and drop out of observation/
dynamics. The behavior function returns each active pedestrian's new velocity; the
env integrates positions and checks the exit.
"""
from __future__ import annotations

from typing import List

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation
from socioverse_abm.social_env_engine import ContinuousField2D

from tasks.flow.social_force.agents import Population


class SocialForceEnv:
    def __init__(self, population: Population, cfg: dict):
        d = cfg["dynamics"]
        self.field = ContinuousField2D(cfg["population"]["width"],
                                       cfg["population"]["height"], wrap=False)
        self._pop = population
        self.goal = population.goal
        self.desired_speed = population.desired_speed
        self.dt = d["dt"]
        self.tau = d["tau"]
        self.v_max = d["v_max"]
        self.ped_radius = d["ped_radius"]
        self.interaction_range = d["interaction_range"]
        self.A = d["A"]
        self.B = d["B"]
        self.exit_radius = d["exit_radius"]
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.pos = self._pop.positions.astype(float).copy()
        self.vel = self._pop.velocities.astype(float).copy()
        self.evacuated = np.zeros(len(self.pos), dtype=bool)
        return self.snapshot()

    def _active(self) -> np.ndarray:
        return np.nonzero(~self.evacuated)[0]

    def _params(self) -> dict:
        return {"tau": self.tau, "dt": self.dt, "v_max": self.v_max,
                "ped_radius": self.ped_radius, "A": self.A, "B": self.B}

    def observe_batch(self) -> List[Observation]:
        obs = []
        for i in self._active():
            nbrs = self.field.neighbors(self.pos, i, self.interaction_range)
            nbrs = nbrs[~self.evacuated[nbrs]]                 # ignore evacuated
            rel = self.field.displacement(self.pos[i], self.pos[nbrs]) if len(nbrs) else np.empty((0, 2))
            dist = np.linalg.norm(rel, axis=1) if len(nbrs) else np.empty(0)
            ctx = {
                "pos": self.pos[i].copy(),
                "vel": self.vel[i].copy(),
                "desired_speed": float(self.desired_speed[i]),
                "goal": self.goal.copy(),
                "neighbor_rel": rel,
                "neighbor_dist": dist,
                "params": self._params(),
            }
            obs.append(Observation(agent_id=int(i), state={"pos": self.pos[i].tolist()},
                                   context=ctx, rendered=self._render(i, len(nbrs))))
        return obs

    def _render(self, i: int, n_neighbors: int) -> str:
        d = float(np.linalg.norm(self.goal - self.pos[i]))
        return (f"You are evacuating toward the exit, {d:.1f} m away, with {n_neighbors} "
                f"people crowding around you. Which way do you move?")

    def apply(self, actions: List[Action]) -> None:
        for a in actions:
            v = np.array([a.payload["vx"], a.payload["vy"]], dtype=float)
            speed = np.linalg.norm(v)
            if speed > self.v_max:
                v = v / speed * self.v_max
            self.vel[a.agent_id] = v
            self.pos[a.agent_id] = self.field.wrap_positions(self.pos[a.agent_id] + v * self.dt)
            if np.linalg.norm(self.goal - self.pos[a.agent_id]) <= self.exit_radius:
                self.evacuated[a.agent_id] = True

    def advance(self) -> None:
        self.t += 1

    def all_evacuated(self) -> bool:
        return bool(self.evacuated.all())

    def snapshot(self) -> dict:
        active = self._active()
        dists = np.linalg.norm(self.goal - self.pos[active], axis=1) if len(active) else np.empty(0)
        return {
            "t": self.t,
            "n": len(self.pos),
            "evacuated": int(self.evacuated.sum()),
            "evac_fraction": float(self.evacuated.mean()),
            "mean_dist_to_goal": float(dists.mean()) if len(dists) else 0.0,
        }
