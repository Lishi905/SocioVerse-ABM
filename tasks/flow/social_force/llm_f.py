"""
f (LLM) — same contract as rule_f: each pedestrian outputs a new velocity. The LLM
picks a heading (degrees) given the exit direction and local crowding; we move at the
pedestrian's desired speed along it. openai is imported lazily.
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

import numpy as np

from socioverse_abm.behavior_engine.action import Action, Observation

_MODEL = os.getenv("SV_LLM_MODEL", "gpt-4o")

_SYS_PROMPT = (
    "You are a pedestrian evacuating a room. Head toward the exit but avoid crowding "
    "into the people nearest you. Respond with ONLY an integer heading in degrees "
    "[0,360). No other text."
)


def configure(model: str) -> None:
    global _MODEL
    _MODEL = model


def _build_prompt(o: Observation) -> str:
    c = o.context
    pos = np.asarray(c["pos"], float)
    to_goal = np.asarray(c["goal"], float) - pos
    goal_heading = np.degrees(np.arctan2(to_goal[1], to_goal[0])) % 360
    rel = np.asarray(c["neighbor_rel"], float)
    if len(rel):
        crowd = rel.mean(axis=0)
        crowd_heading = np.degrees(np.arctan2(crowd[1], crowd[0])) % 360
        summary = f"{len(rel)} people are crowded toward {crowd_heading:.0f} deg."
    else:
        summary = "no one is close."
    return (f"The exit is toward {goal_heading:.0f} deg. {summary}\n"
            f"Your heading to move toward the exit while avoiding the crowd:")


def _parse_heading(text: str, default: float) -> float:
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(m.group()) % 360 if m else default


def social_force_llm_f(observations: Sequence[Observation], llm=None) -> List[Action]:
    if llm is None:
        from socioverse_abm.behavior_engine import llm_f as llm  # lazy: needs openai
    actions: List[Action] = []
    for o in observations:
        c = o.context
        pos = np.asarray(c["pos"], float)
        to_goal = np.asarray(c["goal"], float) - pos
        goal_heading = float(np.degrees(np.arctan2(to_goal[1], to_goal[0])) % 360)
        raw = llm.generate(_MODEL, _build_prompt(o), sys_prompt=_SYS_PROMPT,
                           max_tokens=8, temperature=0.2)
        heading = np.radians(_parse_heading(raw, default=goal_heading))
        new_vel = c["desired_speed"] * np.array([np.cos(heading), np.sin(heading)])
        actions.append(Action(agent_id=o.agent_id, kind="set_velocity",
                              payload={"vx": float(new_vel[0]), "vy": float(new_vel[1])}, raw=raw))
    return actions
