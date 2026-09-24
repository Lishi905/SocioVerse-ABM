"""
f (rule) — the Helbing social force: a driving force pulling each pedestrian toward
the exit at their desired speed, plus exponential repulsion from nearby pedestrians.
Deterministic; returns the new velocity (the env integrates the position).

  driving   = (desired_speed * e_goal - v) / tau
  repulsion = sum_j A * exp((2*r - d_ij) / B) * n_ij      (n_ij points away from j)
  v_new     = clip(v + (driving + repulsion) * dt, v_max)
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation


def _unit(vec: np.ndarray) -> np.ndarray:
    mag = float(np.linalg.norm(vec))
    return vec / mag if mag > 0 else vec


def social_force_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        p = c["params"]
        pos = np.asarray(c["pos"], float)
        vel = np.asarray(c["vel"], float)
        rel = np.asarray(c["neighbor_rel"], float)    # i->j
        dist = np.asarray(c["neighbor_dist"], float)

        e_goal = _unit(np.asarray(c["goal"], float) - pos)
        driving = (c["desired_speed"] * e_goal - vel) / p["tau"]

        repulsion = np.zeros(2)
        if len(rel):
            d = np.maximum(dist, 1e-6)
            mag = p["A"] * np.exp((2 * p["ped_radius"] - d) / p["B"])   # (k,)
            away = -rel / d[:, None]                                    # unit away from each j
            repulsion = (mag[:, None] * away).sum(axis=0)

        new_vel = vel + (driving + repulsion) * p["dt"]
        speed = float(np.linalg.norm(new_vel))
        if speed > p["v_max"]:
            new_vel = new_vel / speed * p["v_max"]
        actions.append(Action(agent_id=o.agent_id, kind="set_velocity",
                              payload={"vx": float(new_vel[0]), "vy": float(new_vel[1])}))
    return actions
