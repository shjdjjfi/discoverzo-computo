import numpy as np

from discoverzo.benchmarks import random_orthonormal
from discoverzo.oracles import NoisyOracle
from discoverzo.subspace import (
    CrossMomentEstimator,
    estimate_active_subspace,
    principal_angle_error,
    select_rank_eigengap,
)


def test_random_orthonormal():
    rng = np.random.default_rng(1)
    U = random_orthonormal(20, 4, rng)
    np.testing.assert_allclose(U.T @ U, np.eye(4), atol=1e-12)


def test_principal_angle_identity_and_orthogonal():
    U = np.eye(4)[:, :2]
    assert principal_angle_error(U, U) < 1e-12
    V = np.eye(4)[:, 2:]
    assert abs(principal_angle_error(U, V) - 1.0) < 1e-12


def test_cross_estimator_recovers_quadratic_ridge():
    rng = np.random.default_rng(4)
    d, r = 12, 2
    U = random_orthonormal(d, r, rng)
    weights = np.array([1.0, 2.0])

    def f(x):
        z = U.T @ x
        return 0.5 * np.dot(weights, z * z)

    oracle = NoisyOracle(f, noise_std=0.0, rng=np.random.default_rng(5))
    result = CrossMomentEstimator(h=1e-2, directions_per_side=8).estimate(
        oracle, d=d, n_anchors=1000, rng=np.random.default_rng(6)
    )
    A, _ = estimate_active_subspace(result, rank=r)
    assert principal_angle_error(U, A) < 0.18
    assert result.query_count == 32000


def test_rank_selection():
    vals = np.array([10.0, 4.0, 3.0, 0.08, 0.05])
    assert select_rank_eigengap(vals, max_rank=4) == 3
