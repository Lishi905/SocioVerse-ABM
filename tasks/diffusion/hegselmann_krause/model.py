"""
Hegselmann-Krause opinion-dynamics task — assembles P (opinions) + E (bounded-
confidence field) + f (rule | llm | hybrid), runs the synchronous update loop,
registers with the kernel. Maps to the SocioVerse2 study `abm_hegselmann_krause`.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import HKOpinionEnv

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    cfg = cfg or load_config()
    population = agents.build_population(cfg, seed)
    env = HKOpinionEnv(population, epsilon=cfg["dynamics"]["epsilon"])
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.hk_rule_f
    if mode == "llm":
        return llm_f.hk_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.hk_rule_f, llm_f.hk_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    if mode in ("llm", "hybrid"):
        llm_f.configure(cfg["behavior"]["llm_model"])

    env, _ = build(cfg)
    behavior = _behavior_fn(mode)

    snapshots = [env.snapshot()]
    prev = env.opinions.copy()
    for _t in range(1, cfg["dynamics"]["steps"] + 1):
        env.apply(list(behavior(env.observe_batch())))
        env.advance()
        snapshots.append(env.snapshot())
        if np.allclose(env.opinions, prev, atol=1e-6):   # converged
            break
        prev = env.opinions.copy()

    metrics = evaluate.hk_metrics(snapshots)
    print(
        f"[hegselmann_krause/{mode}] steps={metrics['steps']} n={metrics['n']} "
        f"clusters={metrics['converged_from']}->{metrics['num_clusters']} "
        f"final_spread={metrics['final_spread']:.3f}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="hegselmann_krause",
    family="diffusion",
    source_abm="Hegselmann-Krause (2002) bounded confidence",
    socioverse2_study="abm_hegselmann_krause",
    build=build,
    rule_f=rule_f.hk_rule_f,
    llm_f=llm_f.hk_llm_f,
    evaluate=evaluate.hk_metrics,
    run=run,
)
