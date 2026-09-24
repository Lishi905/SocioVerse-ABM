"""
Boids flocking task — assembles P (boids) + E (continuous field) + f
(rule | llm | hybrid), runs the synchronous step loop, registers with the kernel.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import BoidsField

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = BoidsField(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.boids_rule_f
    if mode == "llm":
        return llm_f.boids_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.boids_rule_f, llm_f.boids_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    if mode in ("llm", "hybrid"):
        llm_f.configure(cfg["behavior"]["llm_model"])

    env, _ = build(cfg)
    behavior = _behavior_fn(mode)

    snapshots = [env.snapshot()]
    for _t in range(1, cfg["dynamics"]["steps"] + 1):
        actions = list(behavior(env.observe_batch()))
        env.apply(actions)
        env.advance()
        snapshots.append(env.snapshot())

    metrics = evaluate.boids_metrics(snapshots)
    print(
        f"[boids/{mode}] steps={metrics['steps']} n={metrics['n']} "
        f"polarization={metrics['polarization_final']:.2f} "
        f"(mean {metrics['polarization_mean']:.2f}) mean_speed={metrics['mean_speed']:.2f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="boids",
    family="flow",
    source_abm="Reynolds Boids (1987)",
    socioverse2_study="abm_boids",
    build=build,
    rule_f=rule_f.boids_rule_f,
    llm_f=llm_f.boids_llm_f,
    evaluate=evaluate.boids_metrics,
    run=run,
)
