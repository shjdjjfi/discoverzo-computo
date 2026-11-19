from itertools import product

import numpy as np
import pytest

from discoverzo.oracles import QueryCounter
from discoverzo.subspace import OuterMomentEstimator, sample_orthonormal_directions


def coefficients(d: int, m: int) -> tuple[float, float]:
    alpha = d * (d * (m + 1) - 2) / (m * (d + 2) * (d - 1))
    beta = d * (d - m) / (m * (d + 2) * (d - 1))
    return alpha, beta


def test_orthogonal_batch_outer_moment_formula() -> None:
    d, m, n = 7, 3, 12000
    rng = np.random.default_rng(20260718)
    g = rng.normal(size=d)
    acc = np.zeros((d, d))
    for _ in range(n):
        Q = sample_orthonormal_directions(d, m, rng)
        gh = (d / m) * (Q @ (Q.T @ g))
        acc += np.outer(gh, gh)
    empirical = acc / n
    alpha, beta = coefficients(d, m)
    target = alpha * np.outer(g, g) + beta * np.dot(g, g) * np.eye(d)
    assert np.linalg.norm(empirical - target, ord=2) / np.linalg.norm(target, ord=2) < 0.035


def test_orthogonal_batch_endpoint_is_exact() -> None:
    d = 11
    alpha, beta = coefficients(d, d)
    assert np.isclose(alpha, 1.0)
    assert np.isclose(beta, 0.0)


@pytest.mark.parametrize("noise_std", [0.0, 0.25])
def test_calibrated_self_moment_inverts_projection_and_noise(monkeypatch, noise_std):
    # Four lines integrate fourth moments on the circle exactly. Enumerate
    # all four pairs of independent two-point errors for each line.
    d, h = 2, 0.125
    gradient = np.array([2.0, -1.0])
    angles = np.arange(4) * np.pi / 4
    frames = np.column_stack([np.cos(angles), np.sin(angles)])
    frames = iter(np.repeat(frames[:, :, None], 4, axis=0))
    noise_pairs = list(product([-noise_std, noise_std], repeat=2))
    errors = iter(np.array(noise_pairs * 4).ravel())
    monkeypatch.setattr("discoverzo.subspace.sample_orthonormal_directions", lambda *args: next(frames))

    oracle = QueryCounter(lambda x: gradient @ x + next(errors))
    anchors = np.zeros((16, d))
    estimate = OuterMomentEstimator(
        h=h, directions_per_anchor=1, calibrate=True,
        noise_variance=noise_std**2, psd_projection=False,
    ).estimate(oracle, d, len(anchors), np.random.default_rng(0), anchors=anchors)

    np.testing.assert_allclose(estimate.matrix, np.outer(gradient, gradient), atol=1e-12)
    assert estimate.query_count == oracle.count == 2 * len(anchors)


def test_calibrated_self_moment_supports_one_dimension():
    estimate = OuterMomentEstimator(directions_per_anchor=1, calibrate=True).estimate(
        lambda x: 2 * x[0], d=1, n_anchors=1, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(estimate.matrix, [[4.0]])
