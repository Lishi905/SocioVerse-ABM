"""Chicago residential-segregation task — assembles P (archetype households) + E
(2010 tract graph) + f (rule | llm | hybrid) by wrapping the vendored SegregationModel,
runs the observe → decide → apply → snapshot loop, and registers with the kernel.

rule mode = the classical Schelling-on-Chicago baseline (no API key needed);
llm mode  = the validated LLM behaviour function (needs a key, or a DeterministicLLMClient).
Both run on the SAME environment and the SAME per-step move budget.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import yaml

from socioverse_abm.behavior_engine.hybrid import hybrid_decide
from socioverse_abm.scenario_engine import registry
from socioverse_abm.scenario_engine.registry import TaskSetupError

from . import agents, evaluate, llm_f, rule_f
from ._engine import ChicagoEngine, _load_dotenv, default_llm_config, set_active
from .env import ChicagoTractEnv

CONFIG_PATH = pathlib.Path(__file__).with_name("config.yaml")


def load_config(path: Optional[str] = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _engine_from_cfg(cfg: dict, seed: Optional[int], llm_client, llm_config) -> ChicagoEngine:
    pop = cfg.get("population", {})
    dyn = cfg.get("dynamics", {})
    # Forward config → SegregationModel kwargs (anything omitted keeps the legacy default).
    model_kwargs = dict(cfg.get("model") or {})
    if "perturb_pct" in pop:
        model_kwargs.setdefault("perturb_pct", pop["perturb_pct"])
    if "perturb_radius_hops" in pop:
        model_kwargs.setdefault("perturb_radius_hops", pop["perturb_radius_hops"])
    if "move_budget_pct" in dyn:
        model_kwargs.setdefault("move_budget_pct", dyn["move_budget_pct"])
    return ChicagoEngine(
        scale=pop.get("scale", "small"),
        tract_ids=pop.get("tract_ids"),
        init_mode=pop.get("init_mode", "census"),
        seed=cfg.get("seed", 42) if seed is None else seed,
        model_kwargs=model_kwargs,
        llm_client=llm_client,
        llm_config=llm_config,
    )


def build(cfg: Optional[dict] = None, seed: Optional[int] = None,
          *, llm_client=None, llm_config=None):
    """Assemble (env, population) on a single shared SegregationModel instance."""
    cfg = cfg or load_config()
    eng = _engine_from_cfg(cfg, seed, llm_client, llm_config)
    eng.ensure_built()
    set_active(eng)                                # llm_f / rule_f read this engine
    ab = cfg.get("ablations") or {}
    if any(bool(v) for v in ab.values()):          # supplementary ablations (E0-E4)
        from . import ablations
        ablations.apply(eng, ab, seed=cfg.get("seed") if seed is None else seed)
    b = cfg.get("behavior", {})
    rule_f.configure(b.get("homophily", 0.5), b.get("search_radius_hops", 1))
    env = ChicagoTractEnv(eng, cfg)
    population = agents.build_population(eng)
    return env, population


def _behavior_fn(mode: str, rule_variant: str = "classical"):
    if mode == "rule":
        if rule_variant == "classical":
            return rule_f.chicago_rule_f
        from . import rule_variants                # classical_random | calibrated_multigroup | logit (E0)
        return rule_variants.get(rule_variant)
    if mode == "llm":
        return llm_f.chicago_llm_f
    if mode == "hybrid":
        return lambda obs: hybrid_decide(obs, rule_f.chicago_rule_f, llm_f.chicago_llm_f).actions
    raise ValueError(f"unknown mode '{mode}' (rule|llm|hybrid)")


def run(mode: str = "rule", config: Optional[str] = None, *, llm_client=None, seed: Optional[int] = None) -> dict:
    cfg = load_config(config)
    b = cfg.get("behavior", {})

    # llm needs a client: a live LLMConfig from env, unless a fake one is injected.
    # hybrid builds one when a key is available; its default routing (always_rule)
    # never calls the LLM, and chicago_llm_f refuses to run on an engine without one.
    llm_config = None
    if mode == "llm" and llm_client is None:
        llm_config = default_llm_config(b.get("llm_model"))
    elif mode == "hybrid" and llm_client is None:
        try:
            llm_config = default_llm_config(b.get("llm_model"))
        except TaskSetupError:
            llm_config = None

    env, _ = build(cfg, seed=seed, llm_client=llm_client, llm_config=llm_config)
    behavior = _behavior_fn(mode, b.get("rule_variant", "classical"))

    snapshots = [env.snapshot()]                    # t=0 (post-init / post-perturbation)
    for _t in range(1, cfg["dynamics"]["steps"] + 1):
        actions = list(behavior(env.observe_batch()))
        env.apply(actions)
        env.advance()
        snapshots.append(env.snapshot())
        if all(a.kind == "stay" for a in actions):  # everyone content — converged
            break

    metrics = evaluate.chicago_metrics(snapshots)
    d0 = snapshots[0].get("D_black_white")
    dN = snapshots[-1].get("D_black_white")
    r2 = snapshots[-1].get("R2_recovery")
    scale = cfg.get("population", {}).get("scale", "small")
    print(
        f"[chicago_segregation/{mode} scale={scale} init={cfg.get('population', {}).get('init_mode')}] "
        f"steps={metrics.get('steps')} "
        f"D_bw={d0:.4f}->{dN:.4f} "
        f"R2_recovery={r2:.4f} "
        f"avg_movers/step={metrics.get('avg_movers_per_step', 0):.1f}"
        if d0 is not None and dN is not None and r2 is not None
        else f"[chicago_segregation/{mode}] steps={metrics.get('steps')}"
    )
    return {"snapshots": snapshots, "metrics": metrics}


registry.register(
    name="chicago_segregation",
    family="organization",
    source_abm="Schelling segregation — Chicago 2010 real geography (vendored Chicago segregation model @ a1964b8)",
    socioverse2_study="chicago_schelling",
    build=build,
    rule_f=rule_f.chicago_rule_f,
    llm_f=llm_f.chicago_llm_f,
    evaluate=evaluate.chicago_metrics,
    run=run,
    meta={
        # read by `sv-abm run` to check the key before an llm run (and to explain a
        # key-less hybrid run)
        "llm_key_envs": ("SV_LLM_API_KEY", "OPENAI_API_KEY"),
        "llm_base_url_envs": ("SV_LLM_BASE_URL", "OPENAI_BASE_URL"),
        "load_env": _load_dotenv,
        "load_env_hint": "both may also live in a gitignored .env at the SocioVerse-ABM repository root",
    },
)
