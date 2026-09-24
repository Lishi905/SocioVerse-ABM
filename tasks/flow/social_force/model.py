"""
Social Force evacuation task — assembles P (crowd) + E (bounded room) + f
(rule | llm | hybrid), runs the loop until everyone is out or steps run out,
registers with the kernel.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import SocialForceEnv

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = SocialForceEnv(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.social_force_rule_f
    if mode == "llm":
        return llm_f.social_force_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.social_force_rule_f,
                                         llm_f.social_force_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    if mode in ("llm", "hybrid"):
        llm_f.configure(cfg["behavior"]["llm_model"])

    env, _ = build(cfg)
    behavior = _behavior_fn(mode)

    snapshots = [env.snapshot()]
    for _t in range(1, cfg["dynamics"]["steps"] + 1):
        obs = env.observe_batch()
        if obs:
            env.apply(list(behavior(obs)))
        env.advance()
        snapshots.append(env.snapshot())
        if env.all_evacuated():
            break

    metrics = evaluate.social_force_metrics(snapshots)
    print(
        f"[social_force/{mode}] steps={metrics['steps']} n={metrics['n']} "
        f"evacuated={metrics['evac_fraction']:.0%} t_evac={metrics['evacuation_time']} "
        f"mean_dist={metrics['mean_dist_to_goal']:.1f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="social_force",
    family="flow",
    source_abm="Helbing Social Force (2000)",
    socioverse2_study="abm_social_force",
    build=build,
    rule_f=rule_f.social_force_rule_f,
    llm_f=llm_f.social_force_llm_f,
    evaluate=evaluate.social_force_metrics,
    run=run,
)
