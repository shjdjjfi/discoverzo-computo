import numpy as np

from discoverzo.benchmarks import BENCHMARKS, make_benchmark


def test_public_benchmarks_are_finite_at_origin():
    for name, fun in BENCHMARKS.items():
        value = fun(np.zeros(5))
        assert np.isfinite(value), name


def test_embedded_optimum():
    for name in ["sphere", "ellipsoid", "rastrigin", "rosenbrock"]:
        bench = make_benchmark(name, d=20, r=4, seed=7, tau=0.1)
        assert abs(bench(bench.optimum()) - bench.optimum_value()) < 1e-12
        assert bench.optimum_value() < 1e-12
