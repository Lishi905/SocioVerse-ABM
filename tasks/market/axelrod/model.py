"""
Axelrod IPD tournament task — assembles P (agents+strategies) + E (round-robin arena)
+ f (rule | llm | hybrid), runs the rounds, registers with the kernel.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import AxelrodArena

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = AxelrodArena(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.axelrod_rule_f
    if mode == "llm":
        return llm_f.axelrod_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.axelrod_rule_f, llm_f.axelrod_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    rule_f.seed(cfg["seed"])
    if mode in ("llm", "hybrid"):
        llm_f.configure(cfg["behavior"]["llm_model"])

    env, _ = build(cfg)
    behavior = _behavior_fn(mode)

    snapshots = [env.snapshot()]
    for _t in range(1, cfg["dynamics"]["rounds"] + 1):
        env.apply(list(behavior(env.observe_batch())))
        env.advance()
        snapshots.append(env.snapshot())

    metrics = evaluate.axelrod_metrics(snapshots)
    winner = metrics["strategy_ranking"][0]
    print(
        f"[axelrod/{mode}] rounds={metrics['rounds']} n={metrics['n']} "
        f"coop_rate={metrics['cooperation_rate']:.2f} "
        f"top_strategy={winner[0]}({winner[1]:.0f})"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="axelrod",
    family="market",
    source_abm="Axelrod iterated Prisoner's Dilemma (1984)",
    socioverse2_study="abm_axelrod",
    build=build,
    rule_f=rule_f.axelrod_rule_f,
    llm_f=llm_f.axelrod_llm_f,
    evaluate=evaluate.axelrod_metrics,
    run=run,
)
