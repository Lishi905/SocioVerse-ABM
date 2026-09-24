"""
SocioVerse-ABM kernel — shared f / E / P / config engines for the
"LLM agents replace rule-based ABM agents" experiments, built on Lewin's
B = f(P, E) (same anchor as the SocioVerse2 runtime).

Only the dependency-free core (typed actions, task registry, consistency
evaluator) is re-exported here so `import socioverse_abm` stays cheap and never
imports openai. The LLM helper (`behavior_engine.llm_f`) is imported on demand
by the tasks that need it.
"""
from socioverse_abm import eval  # noqa: F401  (dep-light: only stdlib + action)
from socioverse_abm.behavior_engine.action import Action, Decision, Observation  # noqa: F401
from socioverse_abm.scenario_engine import registry  # noqa: F401

__version__ = "0.2.0"

__all__ = ["Action", "Decision", "Observation", "registry", "eval", "__version__"]
