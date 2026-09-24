"""
Schelling segregation task — assembles P (residents) + E (grid) + f
(rule | llm | hybrid), runs until everyone is content or steps run out, registers
with the kernel. The ocm/lcm and tbf/lbf variants are read from config.behavior.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import SchellingGrid

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = SchellingGrid(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.schelling_rule_f
    if mode == "llm":
        return llm_f.schelling_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.schelling_rule_f, llm_f.schelling_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    b = cfg["behavior"]
    if mode in ("llm", "hybrid"):
        llm_f.configure(b["llm_model"], b.get("llm_behavior", "tbf"))

    env, _ = build(cfg)
    behavior = _behavior_fn(mode)

    snapshots = [env.snapshot()]
    for _t in range(1, cfg["dynamics"]["steps"] + 1):
        actions = list(behavior(env.observe_batch()))
        env.apply(actions)
        env.advance()
        snapshots.append(env.snapshot())
        if all(a.kind == "stay" for a in actions):    # everyone content
            break

    metrics = evaluate.schelling_metrics(snapshots)
    print(
        f"[schelling/{mode} ctx={b.get('context')}/{b.get('llm_behavior')}] "
        f"steps={metrics['steps']} "
        f"segregation={metrics['segregation_start']:.2f}->{metrics['segregation_final']:.2f} "
        f"happy={metrics['happy_fraction']:.2f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="schelling",
    family="organization",
    source_abm="Schelling (1971) segregation",
    socioverse2_study="abm_schelling",
    build=build,
    rule_f=rule_f.schelling_rule_f,
    llm_f=llm_f.schelling_llm_f,
    evaluate=evaluate.schelling_metrics,
    run=run,
)
