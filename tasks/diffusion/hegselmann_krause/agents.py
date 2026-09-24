"""
P — the HK population: N agents each holding a continuous opinion in [0, 1].
Classic HK is fully mixed (a complete interaction graph), so the population is
just the opinion vector; who-influences-whom is decided dynamically by the
confidence bound, not by a fixed network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    opinions: np.ndarray  # (N,) in [0, 1]

    def __len__(self) -> int:
        return len(self.opinions)


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    opinions = rng.uniform(0.0, 1.0, size=cfg["population"]["n"])
    return Population(opinions=opinions)
