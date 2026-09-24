"""Flow family: traffic / crowd / flocking (NaSch, Social Force, Boids)."""
import importlib
import warnings

_TASKS = ["nasch", "boids", "social_force"]

for _task in _TASKS:
    try:
        importlib.import_module(f"tasks.flow.{_task}")
    except Exception as exc:  # pragma: no cover - discovery robustness
        warnings.warn(f"tasks.flow.{_task} failed to import: {exc}")
