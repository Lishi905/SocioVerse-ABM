"""
f (LLM) — same contract as rule_f: each boid outputs a new velocity vector. The
LLM picks a heading (degrees); we keep the boid's current speed and convert to a
velocity. The openai import is lazy so import / rule-mode / fake-LLM tests need no key.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You are a bird in a flock. Based on your flockmates, choose a heading in "
    "degrees [0,360) that balances three rules: avoid crowding neighbours "
    "(separation), steer toward their average heading (alignment), and move toward "
    "their center (cohesion). Respond with ONLY an integer heading. No other text."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _build_prompt(o: Observation) -> str:
    c = o.context
    vel = np.asarray(c["vel"], float)
    own_heading = np.degrees(np.arctan2(vel[1], vel[0])) % 360
    nvel = np.asarray(c["neighbor_vel"], float)
    rel = np.asarray(c["neighbor_rel"], float)
    if len(nvel):
        avg = nvel.mean(axis=0)
        avg_heading = np.degrees(np.arctan2(avg[1], avg[0])) % 360
        center = rel.mean(axis=0)
        center_dir = np.degrees(np.arctan2(center[1], center[0])) % 360
        summary = (f"{len(nvel)} flockmates; their average heading is {avg_heading:.0f} deg; "
                   f"their center is toward {center_dir:.0f} deg.")
    else:
        summary = "no flockmates nearby."
    return f"Your heading is {own_heading:.0f} deg. {summary}\nYour new heading:"


def _parse_heading(text: str, default: float) -> float:
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(m.group()) % 360 if m else default


def boids_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        vel = np.asarray(o.context["vel"], float)
        speed = float(np.linalg.norm(vel)) or o.context["params"]["max_speed"] * 0.5
        own_heading = float(np.degrees(np.arctan2(vel[1], vel[0])) % 360)
        raw = llm.generate(_MODEL, _build_prompt(o), sys_prompt=_SYS_PROMPT,
                           max_tokens=8, temperature=0.2)
        heading = np.radians(_parse_heading(raw, default=own_heading))
        new_vel = speed * np.array([np.cos(heading), np.sin(heading)])
        actions.append(Action(agent_id=o.agent_id, kind="set_velocity",
                              payload={"vx": float(new_vel[0]), "vy": float(new_vel[1])}, raw=raw))
    return actions
