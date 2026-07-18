from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

Array = np.ndarray


@dataclass
class QueryCounter:
    """Wrap a callable objective and count scalar evaluations."""

    function: Callable[[Array], float]
    count: int = 0
    history_x: list[Array] = field(default_factory=list)
    history_y: list[float] = field(default_factory=list)
    store_history: bool = False

    def __call__(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        value = float(self.function(x))
        self.count += 1
        if self.store_history:
            self.history_x.append(x.copy())
            self.history_y.append(value)
        return value

    def reset(self) -> None:
        self.count = 0
        self.history_x.clear()
        self.history_y.clear()


@dataclass
class NoisyOracle:
    """Function-value oracle with independent additive noise.

    Parameters
    ----------
    function:
        Deterministic objective.
    noise_std:
        Standard deviation of additive Gaussian noise.
    rng:
        NumPy random generator.
    heteroscedastic:
        Optional nonnegative function returning a multiplicative scale.
    contamination_prob:
        Probability of replacing the Gaussian perturbation by a larger one.
    contamination_scale:
        Standard-deviation multiplier for contaminated observations.
    """

    function: Callable[[Array], float]
    noise_std: float = 0.0
    rng: Optional[np.random.Generator] = None
    heteroscedastic: Optional[Callable[[Array], float]] = None
    contamination_prob: float = 0.0
    contamination_scale: float = 10.0
    count: int = 0

    def __post_init__(self) -> None:
        if self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative")
        if not 0 <= self.contamination_prob <= 1:
            raise ValueError("contamination_prob must lie in [0, 1]")
        if self.rng is None:
            self.rng = np.random.default_rng()

    def __call__(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        base = float(self.function(x))
        scale = 1.0 if self.heteroscedastic is None else float(self.heteroscedastic(x))
        sd = self.noise_std * max(scale, 0.0)
        if self.rng.random() < self.contamination_prob:
            sd *= self.contamination_scale
        noise = 0.0 if sd == 0.0 else float(self.rng.normal(0.0, sd))
        self.count += 1
        return base + noise

    def reset(self) -> None:
        self.count = 0
