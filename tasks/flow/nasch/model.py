"""
NaSch single-lane traffic task — assembles P (vehicles on a ring) + E (the road)
+ f (rule | llm | hybrid), runs the synchronous step loop, registers with the kernel.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import NaSchLaneEnv

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = NaSchLaneEnv(
        population,
        vmax=cfg["dynamics"]["vmax"],
        randomization_prob=cfg["dynamics"]["randomization_prob"],
    )
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.nasch_rule_f
    if mode == "llm":
        return llm_f.nasch_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.nasch_rule_f, llm_f.nasch_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    rule_f.seed(cfg["seed"])
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

    metrics = evaluate.nasch_metrics(snapshots)
    print(
        f"[nasch/{mode}] steps={metrics['steps']} n={metrics['n']} "
        f"rho={metrics['density']:.2f} mean_speed={metrics['mean_speed']:.2f} "
        f"flow={metrics['flow']:.3f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="nasch",
    family="flow",
    source_abm="Nagel-Schreckenberg (1992)",
    socioverse2_study="abm_nasch",
    build=build,
    rule_f=rule_f.nasch_rule_f,
    llm_f=llm_f.nasch_llm_f,
    evaluate=evaluate.nasch_metrics,
    run=run,
)
