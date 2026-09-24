"""
Task registry: the single place that maps a task *name* to everything the runner
(and the SocioVerse2 adapter) needs to build and run it.

Each task module under `tasks/<family>/<task>/` registers itself once, e.g.

    from socioverse_abm.scenario_engine import registry

    registry.register(
        name="schelling",
        family="organization",
        source_abm="Schelling (1971)",
        socioverse2_study="abm_schelling",
        build=build,                 # (cfg, seed) -> (env, population)
        rule_f=schelling_rule_f,     # Sequence[Observation] -> Sequence[Action]
        llm_f=schelling_llm_f,
        evaluate=segregation_metrics,
        run=run,                     # (mode, config) -> results  (CLI entry)
    )

Dependency-free; importing it never imports a task or its heavy deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

FAMILIES = ("flow", "market", "organization", "diffusion")


class TaskSetupError(RuntimeError):
    """A task cannot run because its local setup is incomplete (data files, credentials).

    Raised by a task's ``build``/``run``; ``sv-abm run`` prints the message without a
    traceback and exits with status 2.
    """


@dataclass
class TaskSpec:
    name: str
    family: str
    source_abm: str = ""
    socioverse2_study: str = ""
    build: Optional[Callable] = None
    rule_f: Optional[Callable] = None
    llm_f: Optional[Callable] = None
    evaluate: Optional[Callable] = None
    run: Optional[Callable] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def behavior_fn(self, mode: str) -> Callable:
        """Resolve the behavior function for a run mode, with clear errors."""
        if mode == "rule":
            fn = self.rule_f
        elif mode == "llm":
            fn = self.llm_f
        elif mode == "hybrid":
            if self.rule_f is None or self.llm_f is None:
                raise ValueError(f"task '{self.name}' lacks rule_f or llm_f for hybrid mode")
            return self  # hybrid handled by caller via rule_f + llm_f
        else:
            raise ValueError(f"unknown mode '{mode}' (expected rule|llm|hybrid)")
        if fn is None:
            raise ValueError(f"task '{self.name}' has no {mode}_f registered")
        return fn


_REGISTRY: Dict[str, TaskSpec] = {}


def register(name: str, family: str, **kwargs) -> TaskSpec:
    if family not in FAMILIES:
        raise ValueError(f"family '{family}' not in {FAMILIES}")
    spec = TaskSpec(name=name, family=family, **kwargs)
    _REGISTRY[name] = spec
    return spec


def get(name: str) -> TaskSpec:
    if name not in _REGISTRY:
        raise KeyError(f"task '{name}' not registered (have: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def all_tasks() -> Dict[str, TaskSpec]:
    return dict(_REGISTRY)


def by_family(family: str) -> Dict[str, TaskSpec]:
    return {n: s for n, s in _REGISTRY.items() if s.family == family}
