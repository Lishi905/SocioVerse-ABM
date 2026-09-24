"""
Civil Violence task — assembles P (citizens + cops) + E (grid + enforcement) + f
(rule | llm | hybrid), runs the loop, registers with the kernel. The ocm/lcm and
tbf/lbf variants are read from config.behavior.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import CivilViolenceGrid

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = CivilViolenceGrid(population, cfg)
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.civil_violence_rule_f
    if mode == "llm":
        return llm_f.civil_violence_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.civil_violence_rule_f,
                                         llm_f.civil_violence_llm_f).actions
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
        env.apply(list(behavior(env.observe_batch())))
        env.advance()
        snapshots.append(env.snapshot())

    metrics = evaluate.civil_violence_metrics(snapshots)
    print(
        f"[civil_violence/{mode} ctx={b.get('context')}/{b.get('llm_behavior')}] "
        f"steps={metrics['steps']} peak_active={metrics['peak_active']} "
        f"mean_active={metrics['mean_active']:.1f} jailed_final={metrics['jailed_final']}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="civil_violence",
    family="organization",
    source_abm="Epstein (2002) civil violence",
    socioverse2_study="abm_civil_violence",
    build=build,
    rule_f=rule_f.civil_violence_rule_f,
    llm_f=llm_f.civil_violence_llm_f,
    evaluate=evaluate.civil_violence_metrics,
    run=run,
)
