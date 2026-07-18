import numpy as np

from discoverzo.subspace import sample_orthonormal_directions


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
