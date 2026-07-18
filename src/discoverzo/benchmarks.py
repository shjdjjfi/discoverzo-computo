from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

Array = np.ndarray


def random_orthonormal(d: int, r: int, rng: np.random.Generator) -> Array:
    if not (1 <= r <= d):
        raise ValueError("require 1 <= r <= d")
    q, _ = np.linalg.qr(rng.normal(size=(d, r)))
    return q[:, :r]


def sphere(z: Array) -> float:
    z = np.asarray(z)
    return float(np.dot(z, z))


def ellipsoid(z: Array, condition: float = 1e4) -> float:
    z = np.asarray(z)
    if z.size == 1:
        weights = np.ones(1)
    else:
        weights = condition ** (np.arange(z.size) / (z.size - 1))
    return float(np.dot(weights, z * z))


def discus(z: Array, condition: float = 1e6) -> float:
    z = np.asarray(z)
    return float(condition * z[0] ** 2 + np.dot(z[1:], z[1:]))


def bent_cigar(z: Array, condition: float = 1e6) -> float:
    z = np.asarray(z)
    return float(z[0] ** 2 + condition * np.dot(z[1:], z[1:]))


def rastrigin(z: Array) -> float:
    z = np.asarray(z)
    return float(10.0 * z.size + np.sum(z * z - 10.0 * np.cos(2 * np.pi * z)))


def rosenbrock(z: Array) -> float:
    z = np.asarray(z)
    if z.size < 2:
        return float((z[0] - 1.0) ** 2)
    return float(np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2))


def different_powers(z: Array) -> float:
    z = np.asarray(z)
    exponents = 2.0 + 4.0 * np.arange(z.size) / max(z.size - 1, 1)
    return float(np.sqrt(np.sum(np.abs(z) ** exponents)))


BENCHMARKS: dict[str, Callable[[Array], float]] = {
    "sphere": sphere,
    "ellipsoid": ellipsoid,
    "discus": discus,
    "bent_cigar": bent_cigar,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "different_powers": different_powers,
}


@dataclass
class EmbeddedBenchmark:
    """Embed a public low-dimensional test function into R^d.

    The objective is

        f(x) = phi(U^T (x - shift)) + tau ||(I-UU^T)(x-shift)||^2 / 2.

    Setting tau=0 gives an exact ridge function. The optional orthogonal
    penalty produces a controlled approximate-active-subspace problem.
    """

    base_function: Callable[[Array], float]
    U: Array
    shift: Array
    tau: float = 0.0
    name: str = "embedded"
    lower: float = -5.0
    upper: float = 5.0

    def __post_init__(self) -> None:
        self.U = np.asarray(self.U, dtype=float)
        self.shift = np.asarray(self.shift, dtype=float)
        d, _ = self.U.shape
        if self.shift.shape != (d,):
            raise ValueError("shift has incompatible shape")
        if self.tau < 0:
            raise ValueError("tau must be nonnegative")

    @property
    def dimension(self) -> int:
        return int(self.U.shape[0])

    @property
    def intrinsic_dimension(self) -> int:
        return int(self.U.shape[1])

    @property
    def projector(self) -> Array:
        return self.U @ self.U.T

    def __call__(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        centered = x - self.shift
        z = self.U.T @ centered
        residual = centered - self.U @ z
        return float(self.base_function(z) + 0.5 * self.tau * np.dot(residual, residual))

    def optimum(self) -> Array:
        if self.name == "rosenbrock":
            z_star = np.ones(self.intrinsic_dimension)
        else:
            z_star = np.zeros(self.intrinsic_dimension)
        # Minimum-norm ambient representative.
        return self.shift + self.U @ z_star

    def optimum_value(self) -> float:
        return float(self(self.optimum()))


def make_benchmark(
    name: str,
    d: int,
    r: int,
    seed: int,
    tau: float = 0.0,
    random_shift: bool = True,
    lower: float = -5.0,
    upper: float = 5.0,
) -> EmbeddedBenchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}")
    rng = np.random.default_rng(seed)
    U = random_orthonormal(d, r, rng)
    shift = rng.uniform(-1.0, 1.0, size=d) if random_shift else np.zeros(d)
    return EmbeddedBenchmark(
        BENCHMARKS[name], U=U, shift=shift, tau=tau, name=name, lower=lower, upper=upper
    )


def try_make_coco_problem(
    function_id: int,
    dimension: int,
    instance: int = 1,
):
    """Return an official COCO BBOB problem when cocoex is installed.

    This optional adapter is deliberately isolated so the default artifact
    remains runnable on Python versions for which cocoex wheels are absent.
    """

    try:
        import cocoex  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install the optional dependency with `pip install -e .[coco]` "
            "under a cocoex-supported Python version."
        ) from exc
    suite = cocoex.Suite(
        "bbob",
        f"instances: {instance}",
        f"dimensions: {dimension} function_indices: {function_id}",
    )
    return next(iter(suite))
