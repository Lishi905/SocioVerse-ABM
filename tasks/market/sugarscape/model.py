"""
Sugarscape task — assembles P (citizens) + E (sugar grid) + f (rule | llm | hybrid),
runs the loop, registers with the kernel.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import SugarscapeGrid

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = SugarscapeGrid(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.sugarscape_rule_f
    if mode == "llm":
        return llm_f.sugarscape_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.sugarscape_rule_f,
                                         llm_f.sugarscape_llm_f).actions
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
        obs = env.observe_batch()
        if not obs:
            break
        env.apply(list(behavior(obs)))
        env.advance()
        snapshots.append(env.snapshot())

    metrics = evaluate.sugarscape_metrics(snapshots)
    print(
        f"[sugarscape/{mode}] steps={metrics['steps']} "
        f"alive={metrics['alive_start']}->{metrics['alive_final']} "
        f"gini={metrics['gini']:.3f} mean_sugar={metrics['mean_sugar']:.1f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="sugarscape",
    family="market",
    source_abm="Sugarscape / Epstein-Axtell (1996)",
    socioverse2_study="abm_sugarscape",
    build=build,
    rule_f=rule_f.sugarscape_rule_f,
    llm_f=llm_f.sugarscape_llm_f,
    evaluate=evaluate.sugarscape_metrics,
    run=run,
)
