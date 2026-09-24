"""
Social-environment engine — the E side of B = f(P, E).

    field.py   ContinuousField2D: a NumPy-only continuous 2-D space (toroidal or
               bounded) shared by the continuous-space tasks (boids, social_force).

Task-specific environments (networks, lattices, grids, markets) live with each
task in `tasks/<family>/<task>/env.py`.
"""
from socioverse_abm.social_env_engine.field import ContinuousField2D  # noqa: F401

__all__ = ["ContinuousField2D"]
