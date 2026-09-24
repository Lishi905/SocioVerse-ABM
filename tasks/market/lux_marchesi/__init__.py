"""Lux-Marchesi market task. Importing registers it with the kernel registry."""
from . import model  # noqa: F401  (side effect: registry.register)
