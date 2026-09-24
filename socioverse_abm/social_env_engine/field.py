"""
ContinuousField2D — a reusable continuous 2-D environment (the third kind of E,
alongside SIR's network and NaSch's lattice).

It owns the geometry only: positions in a width x height box, optionally toroidal
(wrap-around), with vectorized radius neighbour queries. Tasks layer their own
dynamics on top (Boids flocking, Social Force pedestrians, ...). NumPy-only.
"""
from __future__ import annotations

import numpy as np


class ContinuousField2D:
    def __init__(self, width: float, height: float, wrap: bool = True):
        self.width = float(width)
        self.height = float(height)
        self.wrap = wrap
        self.size = np.array([self.width, self.height], dtype=float)

    def wrap_positions(self, pos: np.ndarray) -> np.ndarray:
        """Keep positions inside the box: modulo if toroidal, else clamp to edges."""
        pos = np.asarray(pos, dtype=float)
        if self.wrap:
            return np.mod(pos, self.size)
        return np.clip(pos, [0.0, 0.0], self.size)

    def displacement(self, frm: np.ndarray, to: np.ndarray) -> np.ndarray:
        """Shortest displacement vector(s) frm -> to (minimum-image if toroidal)."""
        d = np.asarray(to, dtype=float) - np.asarray(frm, dtype=float)
        if self.wrap:
            d = (d + self.size / 2.0) % self.size - self.size / 2.0
        return d

    def distances_from(self, positions: np.ndarray, i: int) -> np.ndarray:
        """Distance from agent i to every agent (i->i is 0)."""
        d = self.displacement(positions[i], positions)
        return np.linalg.norm(d, axis=1)

    def neighbors(self, positions: np.ndarray, i: int, radius: float) -> np.ndarray:
        """Indices j != i with distance(i, j) <= radius."""
        dist = self.distances_from(positions, i)
        mask = (dist <= radius)
        mask[i] = False
        return np.nonzero(mask)[0]
