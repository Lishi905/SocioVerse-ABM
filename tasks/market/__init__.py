"""Market family: wealth / finance / games (Sugarscape, Lux-Marchesi, Minority, Axelrod)."""
import importlib
import warnings

_TASKS = ["sugarscape", "lux_marchesi", "minority_game", "axelrod"]

for _task in _TASKS:
    try:
        importlib.import_module(f"tasks.market.{_task}")
    except Exception as exc:  # pragma: no cover - discovery robustness
        warnings.warn(f"tasks.market.{_task} failed to import: {exc}")
