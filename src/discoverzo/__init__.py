"""discoverzo: active-subspace discovery from noisy function values."""

from .oracles import NoisyOracle, QueryCounter
from .subspace import (
    CrossMomentEstimator,
    OuterMomentEstimator,
    estimate_active_subspace,
    principal_angle_error,
    select_rank_eigengap,
)
from .optimizers import TwoPointOptimizer, discover_then_optimize
from .benchmarks import EmbeddedBenchmark, make_benchmark, random_orthonormal

__all__ = [
    "NoisyOracle",
    "QueryCounter",
    "CrossMomentEstimator",
    "OuterMomentEstimator",
    "estimate_active_subspace",
    "principal_angle_error",
    "select_rank_eigengap",
    "TwoPointOptimizer",
    "discover_then_optimize",
    "EmbeddedBenchmark",
    "make_benchmark",
    "random_orthonormal",
]
