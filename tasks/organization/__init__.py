"""Organization family: segregation / collective unrest (Schelling, Chicago, Civil Violence)."""
import importlib
import warnings

_TASKS = ["schelling", "chicago_segregation", "civil_violence"]

for _task in _TASKS:
    try:
        importlib.import_module(f"tasks.organization.{_task}")
    except Exception as exc:  # pragma: no cover - discovery robustness
        warnings.warn(f"tasks.organization.{_task} failed to import: {exc}")
