from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

Array = np.ndarray


def sample_unit_sphere(d: int, rng: np.random.Generator) -> Array:
    u = rng.normal(size=d)
    norm = np.linalg.norm(u)
    while norm == 0:
        u = rng.normal(size=d)
        norm = np.linalg.norm(u)
    return u / norm


def sample_orthonormal_directions(d: int, m: int, rng: np.random.Generator) -> Array:
    """Sample m orthonormal directions (or independent blocks if m>d)."""
    if m < 1:
        raise ValueError("m must be positive")
    blocks = []
    remaining = m
    while remaining > 0:
        take = min(d, remaining)
        q, _ = np.linalg.qr(rng.normal(size=(d, take)))
        blocks.append(q[:, :take])
        remaining -= take
    return np.concatenate(blocks, axis=1)


def batched_directional_gradient(
    oracle: Callable[[Array], float],
    x: Array,
    directions: Array,
    h: float,
) -> Array:
    """Average spherical gradient proxies over a direction batch."""
    d, m = directions.shape
    acc = np.zeros(d)
    for j in range(m):
        u = directions[:, j]
        diff = oracle(x + h * u) - oracle(x - h * u)
        acc += (d * diff / (2.0 * h)) * u
    return acc / m


def central_directional_gradient(
    oracle: Callable[[Array], float],
    x: Array,
    u: Array,
    h: float,
) -> Array:
    """Two-point spherical gradient proxy d * directional derivative * u."""

    d = x.size
    diff = oracle(x + h * u) - oracle(x - h * u)
    return (d * diff / (2.0 * h)) * u


def symmetrize(a: Array) -> Array:
    return 0.5 * (a + a.T)


def project_psd(a: Array) -> Array:
    vals, vecs = np.linalg.eigh(symmetrize(a))
    vals = np.maximum(vals, 0.0)
    return (vecs * vals) @ vecs.T


def spectral_clip(a: Array, clip: Optional[float]) -> Array:
    if clip is None:
        return a
    norm = np.linalg.norm(a, ord=2)
    if norm <= clip or norm == 0:
        return a
    return a * (clip / norm)


@dataclass
class MomentEstimate:
    matrix: Array
    eigenvalues: Array
    eigenvectors: Array
    anchors: Array
    query_count: int


@dataclass
class CrossMomentEstimator:
    """Cross-fitted moment estimator from independent random directions.

    For each anchor x, two independent two-point gradient proxies g and h are
    computed and the symmetric cross product (gh^T + hg^T)/2 is averaged.
    Conditional independence removes the isotropic second-moment inflation
    of the usual outer-product estimator.
    """

    h: float = 1e-2
    anchor_radius: float = 1.0
    clip: Optional[float] = None
    psd_projection: bool = True
    directions_per_side: int = 4

    def estimate(
        self,
        oracle: Callable[[Array], float],
        d: int,
        n_anchors: int,
        rng: np.random.Generator,
        anchors: Optional[Array] = None,
    ) -> MomentEstimate:
        if self.h <= 0 or self.anchor_radius <= 0:
            raise ValueError("h and anchor_radius must be positive")
        if n_anchors < 1:
            raise ValueError("n_anchors must be positive")
        if anchors is None:
            anchors = rng.uniform(-self.anchor_radius, self.anchor_radius, size=(n_anchors, d))
        anchors = np.asarray(anchors, dtype=float)
        if anchors.shape != (n_anchors, d):
            raise ValueError("anchors have incompatible shape")
        acc = np.zeros((d, d), dtype=float)
        for x in anchors:
            dirs_g = sample_orthonormal_directions(d, self.directions_per_side, rng)
            dirs_h = sample_orthonormal_directions(d, self.directions_per_side, rng)
            g = batched_directional_gradient(oracle, x, dirs_g, self.h)
            hvec = batched_directional_gradient(oracle, x, dirs_h, self.h)
            term = 0.5 * (np.outer(g, hvec) + np.outer(hvec, g))
            acc += spectral_clip(term, self.clip)
        matrix = symmetrize(acc / n_anchors)
        if self.psd_projection:
            matrix = project_psd(matrix)
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        return MomentEstimate(matrix, values[order], vectors[:, order], anchors, 4 * self.directions_per_side * n_anchors)


@dataclass
class OuterMomentEstimator:
    """Conventional outer-product estimator used as a diagnostic baseline."""

    h: float = 1e-2
    anchor_radius: float = 1.0
    clip: Optional[float] = None
    debias_trace: bool = False
    psd_projection: bool = True
    directions_per_anchor: int = 8

    def estimate(
        self,
        oracle: Callable[[Array], float],
        d: int,
        n_anchors: int,
        rng: np.random.Generator,
        anchors: Optional[Array] = None,
    ) -> MomentEstimate:
        if anchors is None:
            anchors = rng.uniform(-self.anchor_radius, self.anchor_radius, size=(n_anchors, d))
        anchors = np.asarray(anchors, dtype=float)
        acc = np.zeros((d, d), dtype=float)
        for x in anchors:
            dirs = sample_orthonormal_directions(d, self.directions_per_anchor, rng)
            g = batched_directional_gradient(oracle, x, dirs, self.h)
            acc += spectral_clip(np.outer(g, g), self.clip)
        matrix = symmetrize(acc / n_anchors)
        if self.debias_trace:
            # Heuristic isotropic correction; retained as a transparent baseline.
            matrix -= np.trace(matrix) / (d * (d + 2.0)) * np.eye(d)
        if self.psd_projection:
            matrix = project_psd(matrix)
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        return MomentEstimate(matrix, values[order], vectors[:, order], anchors, 2 * self.directions_per_anchor * n_anchors)


def select_rank_eigengap(
    eigenvalues: Array,
    max_rank: Optional[int] = None,
    min_rank: int = 1,
    relative_floor: float = 1e-3,
) -> int:
    """Select rank using a scale-normalized eigengap.

    Negative values are discarded. Candidate gaps are divided by the leading
    eigenvalue and only candidates above a relative spectral floor are used.
    """

    vals = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("eigenvalues must be a non-empty vector")
    if vals[0] <= 0:
        return min_rank
    upper = vals.size - 1 if max_rank is None else min(max_rank, vals.size - 1)
    upper = max(upper, min_rank)
    candidates = []
    for k in range(min_rank, upper + 1):
        if vals[k - 1] < relative_floor * vals[0]:
            continue
        next_val = vals[k] if k < vals.size else 0.0
        denominator = max(next_val, relative_floor * vals[0])
        ratio_gap = vals[k - 1] / denominator
        candidates.append((ratio_gap, k))
    if not candidates:
        positive = int(np.sum(vals >= relative_floor * vals[0]))
        return max(min_rank, min(positive, upper))
    return max(candidates)[1]


def estimate_active_subspace(
    estimate: MomentEstimate,
    rank: Optional[int] = None,
    max_rank: Optional[int] = None,
) -> tuple[Array, int]:
    selected = select_rank_eigengap(estimate.eigenvalues, max_rank=max_rank) if rank is None else rank
    if not 1 <= selected <= estimate.eigenvectors.shape[1]:
        raise ValueError("invalid rank")
    return estimate.eigenvectors[:, :selected], selected


def principal_angle_error(U: Array, V: Array, mode: Literal["op", "fro"] = "op") -> float:
    """Return ||sin Theta(U,V)|| in operator or Frobenius norm."""

    U, _ = np.linalg.qr(np.asarray(U, dtype=float))
    V, _ = np.linalg.qr(np.asarray(V, dtype=float))
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    k = min(U.shape[1], V.shape[1])
    s = np.clip(s[:k], 0.0, 1.0)
    sin_values = np.sqrt(np.maximum(0.0, 1.0 - s * s))
    if U.shape[1] != V.shape[1]:
        pad = abs(U.shape[1] - V.shape[1])
        sin_values = np.concatenate([sin_values, np.ones(pad)])
    return float(np.max(sin_values) if mode == "op" else np.linalg.norm(sin_values))
