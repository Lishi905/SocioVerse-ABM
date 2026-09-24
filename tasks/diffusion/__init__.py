"""Diffusion family: opinion / rumor spreading on networks (HK, SIR)."""
import importlib
import warnings

_TASKS = ["sir", "hegselmann_krause"]

for _task in _TASKS:
    try:
        importlib.import_module(f"tasks.diffusion.{_task}")
    except Exception as exc:  # pragma: no cover - discovery robustness
        warnings.warn(f"tasks.diffusion.{_task} failed to import: {exc}")
