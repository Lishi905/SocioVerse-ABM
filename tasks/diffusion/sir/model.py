"""
SIR rumor-spreading task — assembles P (network population) + E (network env) + f
(rule | llm | hybrid) and runs the SocioVerse2-style step loop. Registers itself with the
kernel so `sv-abm run sir --mode rule` works and the SocioVerse2 adapter can build it.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry

from . import agents, evaluate, llm_f, rule_f
from .env import SIRNetworkEnv

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def build(cfg: Optional[dict] = None, seed: Optional[int] = None):
    """PopulationProvider + EnvironmentProvider assembly -> (env, population)."""
    cfg = cfg or load_config()
    d = cfg["dynamics"]
    beta_eff = d["p"] * (1.0 - d["q"])
    population = agents.build_population(cfg, seed)
    env = SIRNetworkEnv(population, beta_eff=beta_eff, gamma=d["gamma"])
    return env, population


def _behavior_fn(mode: str):
    if mode == "rule":
        return rule_f.sir_rule_f
    if mode == "llm":
        return llm_f.sir_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.sir_rule_f, llm_f.sir_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None) -> dict:
    cfg = load_config(config)
    rule_f.seed(cfg["seed"])
    if mode in ("llm", "hybrid"):
        llm_f.configure(cfg["behavior"]["llm_model"])

    env, _ = build(cfg)
    behavior = _behavior_fn(mode)

    snapshots = [env.snapshot()]
    action_log = []
    for _t in range(1, cfg["dynamics"]["steps"] + 1):
        actions = list(behavior(env.observe_batch()))
        env.apply(actions)
        env.advance()
        snapshots.append(env.snapshot())
        action_log.append(actions)
        if snapshots[-1]["I"] == 0:  # absorbing: no spreaders left
            break

    metrics = evaluate.sir_metrics(snapshots)
    print(
        f"[sir/{mode}] steps={metrics['steps']} "
        f"reach={metrics['final_reach']}/{metrics['n']} "
        f"({metrics['final_reach_frac']:.1%}) "
        f"peak_I={metrics['peak_spreaders']}@t{metrics['time_to_peak']}"
    )
    return {"snapshots": snapshots, "actions": action_log, "metrics": metrics}


registry.register(
    name="sir",
    family="diffusion",
    source_abm="SIR rumor (Ignorant-Spreader-Stifler)",
    socioverse2_study="abm_sir",
    build=build,
    rule_f=rule_f.sir_rule_f,
    llm_f=llm_f.sir_llm_f,
    evaluate=evaluate.sir_metrics,
    run=run,
)
