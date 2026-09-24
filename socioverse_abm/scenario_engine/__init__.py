"""
Scenario engine — the task registry.

    registry.py       name -> TaskSpec; how the runner and the SocioVerse2 adapter
                      discover and build each task.

Dependency-free and safe to import eagerly.
"""
from socioverse_abm.scenario_engine import registry  # noqa: F401

__all__ = ["registry"]
