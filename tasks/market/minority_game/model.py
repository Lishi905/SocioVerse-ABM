"""
Minority Game task — assembles P (agents+strategies) + E (the game) + f
(rule | llm | hybrid), runs the loop, registers with the kernel.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import MinorityGameEnv

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = MinorityGameEnv(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.minority_game_rule_f
    if mode == "llm":
        return llm_f.minority_game_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.minority_game_rule_f,
                                         llm_f.minority_game_llm_f).actions
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
        env.apply(list(behavior(env.observe_batch())))
        env.advance()
        snapshots.append(env.snapshot())

    metrics = evaluate.minority_game_metrics(snapshots)
    print(
        f"[minority_game/{mode}] steps={metrics['steps']} n={metrics['n']} "
        f"volatility(sigma^2/N)={metrics['volatility']:.3f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="minority_game",
    family="market",
    source_abm="Minority Game / Challet-Zhang (1997)",
    socioverse2_study="abm_minority_game",
    build=build,
    rule_f=rule_f.minority_game_rule_f,
    llm_f=llm_f.minority_game_llm_f,
    evaluate=evaluate.minority_game_metrics,
    run=run,
)
