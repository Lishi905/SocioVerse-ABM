"""
Task packages. Importing `tasks` imports each family, and importing a family
imports each task module, which registers itself with
`socioverse_abm.scenario_engine.registry`.

Discovery is defensive: a task that fails to import (e.g. a missing optional dep)
is warned about but does not hide the other tasks from `sv-abm list`.
"""
import importlib
import warnings

# Task families (see socioverse_abm.scenario_engine.registry.FAMILIES).
_FAMILIES = ["diffusion", "flow", "market", "organization"]

for _family in _FAMILIES:
    try:
        importlib.import_module(f"tasks.{_family}")
    except Exception as exc:  # pragma: no cover - discovery robustness
        warnings.warn(f"tasks.{_family} failed to import: {exc}")
