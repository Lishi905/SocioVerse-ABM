"""Tests for the reusable continuous 2-D field environment."""
import numpy as np

from socioverse_abm.social_env_engine import ContinuousField2D


def test_wrap_toroidal():
    f = ContinuousField2D(100, 100, wrap=True)
    out = f.wrap_positions(np.array([[105.0, -5.0]]))
    assert np.allclose(out, [[5.0, 95.0]])


def test_minimum_image_displacement():
    f = ContinuousField2D(100, 100, wrap=True)
    # 2 -> 98 the short way is -4 (wrap), not +96
    d = f.displacement(np.array([2.0, 50.0]), np.array([98.0, 50.0]))
    assert np.allclose(d, [-4.0, 0.0])


def test_neighbors_within_radius_and_wrap():
    f = ContinuousField2D(100, 100, wrap=True)
    pos = np.array([[1.0, 1.0], [3.0, 1.0], [99.0, 1.0], [50.0, 50.0]])
    nbrs = set(f.neighbors(pos, 0, radius=5.0).tolist())
    assert nbrs == {1, 2}        # 3 (dist 2) and 99 (dist 2 via wrap); 50,50 too far
    assert 0 not in nbrs          # never include self


def test_non_wrap_clamps():
    f = ContinuousField2D(100, 100, wrap=False)
    out = f.wrap_positions(np.array([[105.0, -5.0]]))
    assert np.allclose(out, [[100.0, 0.0]])
