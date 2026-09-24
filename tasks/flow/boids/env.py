"""
E — the continuous 2-D flocking field for Boids, built on the kernel's reusable
ContinuousField2D (toroidal box + radius neighbour queries).

A third kind of E after SIR's network and NaSch's lattice: continuous space with
vector state. The behavior function returns each boid's *new velocity vector*; the
env integrates positions. Boids update synchronously (all decide from the current
frame, then all move), so one kernel step is observe -> decide -> apply.
"""
from __future__ import annotations

from typing import List

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation
from socioverse_abm.social_env_engine import ContinuousField2D

from tasks.flow.boids.agents import Population


class BoidsField:
    def __init__(self, population: Population, cfg: dict):
        d = cfg["dynamics"]
        self.field = ContinuousField2D(cfg["population"]["width"],
                                       cfg["population"]["height"], wrap=True)
        self.pos = population.positions.astype(float).copy()
        self.vel = population.velocities.astype(float).copy()
        self._pop = population
        self.dt = d["dt"]
        self.max_speed = d["max_speed"]
        self.max_force = d["max_force"]
        self.perception_radius = d["perception_radius"]
        self.separation_radius = d["separation_radius"]
        self.weights = d["weights"]
        self.t = 0

    def reset(self) -> dict:
        self.t = 0
        self.pos = self._pop.positions.astype(float).copy()
        self.vel = self._pop.velocities.astype(float).copy()
        return self.snapshot()

    def _params(self) -> dict:
        return {
            "max_speed": self.max_speed,
            "max_force": self.max_force,
            "separation_radius": self.separation_radius,
            "sep_w": self.weights["separation"],
            "align_w": self.weights["alignment"],
            "coh_w": self.weights["cohesion"],
        }

    def observe_batch(self) -> List[Observation]:
        obs = []
        for i in range(len(self.pos)):
            nbrs = self.field.neighbors(self.pos, i, self.perception_radius)
            rel = self.field.displacement(self.pos[i], self.pos[nbrs])  # i->j vectors
            dist = np.linalg.norm(rel, axis=1) if len(nbrs) else np.empty(0)
            ctx = {
                "vel": self.vel[i].copy(),
                "neighbor_rel": rel,            # (k, 2)  vectors from i to each neighbour
                "neighbor_vel": self.vel[nbrs], # (k, 2)
                "neighbor_dist": dist,          # (k,)
                "params": self._params(),
            }
            obs.append(Observation(agent_id=i, state={"speed": float(np.linalg.norm(self.vel[i]))},
                                   context=ctx, rendered=self._render(len(nbrs), self.vel[i])))
        return obs

    def _render(self, n_neighbors: int, vel: np.ndarray) -> str:
        heading = float(np.degrees(np.arctan2(vel[1], vel[0]))) % 360
        return (f"You are a bird flying at heading {heading:.0f} deg with {n_neighbors} "
                f"flockmates nearby. Choose a new heading to flock (separate, align, cohere).")

    def apply(self, actions: List[Action]) -> None:
        for a in actions:
            v = np.array([a.payload["vx"], a.payload["vy"]], dtype=float)
            speed = np.linalg.norm(v)
            if speed > self.max_speed:                       # clamp speed
                v = v / speed * self.max_speed
            self.vel[a.agent_id] = v
        self.pos = self.field.wrap_positions(self.pos + self.vel * self.dt)

    def advance(self) -> None:
        self.t += 1

    def snapshot(self) -> dict:
        speeds = np.linalg.norm(self.vel, axis=1)
        moving = speeds > 1e-9
        if moving.any():
            units = self.vel[moving] / speeds[moving, None]
            polarization = float(np.linalg.norm(units.mean(axis=0)))  # order parameter in [0,1]
        else:
            polarization = 0.0
        return {
            "t": self.t,
            "n": len(self.pos),
            "polarization": polarization,
            "mean_speed": float(speeds.mean()),
        }
