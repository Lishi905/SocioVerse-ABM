"""
f (rule) — Reynolds flocking, the parity baseline. Textbook steering form
(Reynolds / Shiffman): for each of separation, alignment, cohesion compute a
*desired* velocity, the steering = limit(desired - velocity, max_force); the
weighted sum (clamped to max_force) is the acceleration. Deterministic.

The behavior function returns each boid's new velocity as Action.payload {vx, vy};
the env clamps speed and integrates the position.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation


def _limit(vec: np.ndarray, max_mag: float) -> np.ndarray:
    mag = float(np.linalg.norm(vec))
    if mag > max_mag and mag > 0:
        return vec / mag * max_mag
    return vec


def _normalize(vec: np.ndarray, to: float) -> np.ndarray:
    mag = float(np.linalg.norm(vec))
    return vec / mag * to if mag > 0 else vec


def boids_rule_f(observations: Sequence[Observation]) -> List[Action]:
    actions: List[Action] = []
    for o in observations:
        c = o.context
        vel = np.asarray(c["vel"], dtype=float)
        rel = np.asarray(c["neighbor_rel"], dtype=float)     # (k,2) i->j
        nvel = np.asarray(c["neighbor_vel"], dtype=float)    # (k,2)
        dist = np.asarray(c["neighbor_dist"], dtype=float)   # (k,)
        p = c["params"]
        vmax, fmax = p["max_speed"], p["max_force"]

        accel = np.zeros(2)
        if len(rel):
            # Cohesion: steer toward the neighbourhood centroid (mean of i->j).
            desired_c = _normalize(rel.mean(axis=0), vmax)
            steer_c = _limit(desired_c - vel, fmax)
            # Alignment: match average neighbour velocity.
            desired_a = _normalize(nvel.mean(axis=0), vmax)
            steer_a = _limit(desired_a - vel, fmax)
            # Separation: push away from neighbours within separation_radius.
            close = dist < p["separation_radius"]
            if close.any():
                away = -(rel[close] / dist[close, None]).sum(axis=0)  # sum of unit away vectors
                desired_s = _normalize(away, vmax)
                steer_s = _limit(desired_s - vel, fmax)
            else:
                steer_s = np.zeros(2)
            accel = p["sep_w"] * steer_s + p["align_w"] * steer_a + p["coh_w"] * steer_c
            accel = _limit(accel, fmax)

        new_vel = _limit(vel + accel, vmax)
        actions.append(Action(agent_id=o.agent_id, kind="set_velocity",
                              payload={"vx": float(new_vel[0]), "vy": float(new_vel[1])}))
    return actions
