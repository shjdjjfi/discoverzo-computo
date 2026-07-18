import numpy as np

from discoverzo.benchmarks import make_benchmark
from discoverzo.optimizers import TwoPointOptimizer, discover_then_optimize
from discoverzo.oracles import NoisyOracle
from discoverzo.subspace import CrossMomentEstimator


def test_two_point_optimizer_reduces_sphere():
    bench = make_benchmark("sphere", d=10, r=3, seed=11, random_shift=False)
    rng = np.random.default_rng(12)
    oracle = NoisyOracle(bench, noise_std=0.0, rng=rng)
    x0 = bench.U @ np.array([2.0, -1.0, 1.5])
    initial = bench(x0)
    trace = TwoPointOptimizer(step_size=0.08, smoothing=0.02).run(
        oracle, x0=x0, budget=401, rng=rng, subspace=bench.U
    )
    assert trace.query_count <= 401
    assert bench(trace.x_best) < 0.05 * initial


def test_discover_then_optimize_accounting():
    bench = make_benchmark("sphere", d=12, r=2, seed=20, random_shift=False)
    rng = np.random.default_rng(21)
    oracle = NoisyOracle(bench, noise_std=0.0, rng=rng)
    trace, A, rank, eig = discover_then_optimize(
        oracle,
        d=12,
        total_budget=1200,
        discovery_anchors=50,
        rng=rng,
        rank=2,
        estimator=CrossMomentEstimator(h=0.02),
        optimizer=TwoPointOptimizer(step_size=0.08, smoothing=0.02),
    )
    assert trace.query_count <= 1200
    assert oracle.count == trace.query_count
    assert A.shape == (12, 2)
    assert rank == 2
    assert eig.shape == (12,)
