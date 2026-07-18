from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .subspace import CrossMomentEstimator, estimate_active_subspace, sample_unit_sphere

Array = np.ndarray


@dataclass
class OptimizationTrace:
    x_best: Array
    f_best: float
    query_count: int
    values: list[float]
    best_values: list[float]
    points: list[Array]


@dataclass
class TwoPointOptimizer:
    """Projected two-point optimizer with Adam-style stabilization.

    The update direction is restricted to columns of ``subspace``. With an
    identity subspace this becomes a standard full-dimensional two-point
    zeroth-order method. The practical Adam normalization is used only in the
    experiments; the paper's basic convergence theorem is stated for the
    constant-step gradient-descent version.
    """

    step_size: float = 0.05
    smoothing: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    lower: float = -5.0
    upper: float = 5.0
    decay: float = 0.0

    def run(
        self,
        oracle: Callable[[Array], float],
        x0: Array,
        budget: int,
        rng: np.random.Generator,
        subspace: Optional[Array] = None,
        evaluate_initial: bool = True,
    ) -> OptimizationTrace:
        x = np.asarray(x0, dtype=float).copy()
        d = x.size
        A = np.eye(d) if subspace is None else np.asarray(subspace, dtype=float)
        A, _ = np.linalg.qr(A)
        q = A.shape[1]
        m = np.zeros(q)
        v = np.zeros(q)
        queries = 0
        if evaluate_initial:
            f_best = float(oracle(x))
            queries += 1
        else:
            f_best = np.inf
        x_best = x.copy()
        values: list[float] = []
        best_values: list[float] = []
        points: list[Array] = []
        t = 0
        while queries + 2 <= budget:
            t += 1
            u = sample_unit_sphere(q, rng)
            direction = A @ u
            x_old = x.copy()
            x_plus = np.clip(x_old + self.smoothing * direction, self.lower, self.upper)
            x_minus = np.clip(x_old - self.smoothing * direction, self.lower, self.upper)
            yp = float(oracle(x_plus))
            ym = float(oracle(x_minus))
            queries += 2
            grad = q * (yp - ym) / (2.0 * self.smoothing) * u
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
            mhat = m / (1.0 - self.beta1**t)
            vhat = v / (1.0 - self.beta2**t)
            eta = self.step_size / np.sqrt(1.0 + self.decay * t)
            reduced_step = eta * mhat / (np.sqrt(vhat) + self.epsilon)
            x = np.clip(x - A @ reduced_step, self.lower, self.upper)
            # Reuse the lower of the two observed probes as an incumbent; this
            # avoids an extra query and makes query accounting exact.
            if yp <= ym:
                current_value = yp
                current_point = x_plus
            else:
                current_value = ym
                current_point = x_minus
            if current_value < f_best:
                f_best = current_value
                x_best = current_point.copy()
            values.append(current_value)
            best_values.append(f_best)
            points.append(x.copy())
        return OptimizationTrace(x_best, float(f_best), queries, values, best_values, points)


def discover_then_optimize(
    oracle: Callable[[Array], float],
    d: int,
    total_budget: int,
    discovery_anchors: int,
    rng: np.random.Generator,
    rank: Optional[int] = None,
    max_rank: Optional[int] = None,
    estimator: Optional[CrossMomentEstimator] = None,
    optimizer: Optional[TwoPointOptimizer] = None,
    x0: Optional[Array] = None,
) -> tuple[OptimizationTrace, Array, int, Array]:
    estimator = CrossMomentEstimator() if estimator is None else estimator
    optimizer = TwoPointOptimizer() if optimizer is None else optimizer
    x0 = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float)
    discovery = estimator.estimate(oracle, d=d, n_anchors=discovery_anchors, rng=rng)
    A, selected_rank = estimate_active_subspace(discovery, rank=rank, max_rank=max_rank)
    remaining = total_budget - discovery.query_count
    if remaining < 3:
        raise ValueError("total budget is too small after discovery")
    trace = optimizer.run(oracle, x0=x0, budget=remaining, rng=rng, subspace=A)
    trace.query_count += discovery.query_count
    return trace, A, selected_rank, discovery.eigenvalues
