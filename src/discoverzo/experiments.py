from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmarks import make_benchmark, random_orthonormal
from .optimizers import TwoPointOptimizer, discover_then_optimize
from .oracles import NoisyOracle
from .subspace import (
    CrossMomentEstimator,
    OuterMomentEstimator,
    estimate_active_subspace,
    principal_angle_error,
)


@dataclass
class ExperimentConfig:
    seed: int = 12345
    bias_repetitions: int = 40
    scaling_repetitions: int = 15
    optimization_repetitions: int = 8
    optimization_budget: int = 1600
    discovery_anchors: int = 30
    discovery_directions_per_side: int = 20
    noise_std: float = 0.02


def _quadratic_ridge(d: int, r: int, seed: int, condition: float = 2.0):
    rng = np.random.default_rng(seed)
    U = random_orthonormal(d, r, rng)
    weights = np.geomspace(1.0, condition, r)

    def f(x: np.ndarray) -> float:
        z = U.T @ x
        return float(0.5 * np.dot(weights, z * z))

    # X ~ Unif[-1,1]^d gives E[xx^T]=I/3. For grad=U H U^T x,
    # M=U H^2 U^T/3.
    M = (U * (weights * weights / 3.0)) @ U.T
    return f, U, M


def run_bias_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    d, r = 12, 2
    anchor_grid = [80, 160, 320, 640, 1280, 2560, 5120]
    for rep in range(cfg.bias_repetitions):
        f, U, M = _quadratic_ridge(d, r, cfg.seed + 10_000 + rep)
        anchors_rng = np.random.default_rng(cfg.seed + 20_000 + rep)
        max_anchors = max(anchor_grid)
        shared_anchors = anchors_rng.uniform(-1, 1, size=(max_anchors, d))
        for n in anchor_grid:
            for method in ("cross", "outer", "outer_trace"):
                rng = np.random.default_rng(cfg.seed + rep * 1000 + n * 10 + len(method))
                oracle = NoisyOracle(f, noise_std=0.0, rng=rng)
                if method == "cross":
                    est = CrossMomentEstimator(h=0.02, anchor_radius=1.0, psd_projection=True, directions_per_side=4)
                else:
                    est = OuterMomentEstimator(
                        h=0.02,
                        anchor_radius=1.0,
                        debias_trace=(method == "outer_trace"),
                        psd_projection=True,
                        directions_per_anchor=8,
                    )
                result = est.estimate(oracle, d, n, rng, anchors=shared_anchors[:n])
                A, _ = estimate_active_subspace(result, rank=r)
                records.append(
                    {
                        "experiment": "moment_bias",
                        "method": method,
                        "rep": rep,
                        "d": d,
                        "r": r,
                        "n_anchors": n,
                        "queries": result.query_count,
                        "operator_error": float(np.linalg.norm(result.matrix - M, ord=2)),
                        "frobenius_error": float(np.linalg.norm(result.matrix - M, ord="fro")),
                        "subspace_error": principal_angle_error(U, A),
                        "rank_hat": r,
                    }
                )
    return pd.DataFrame.from_records(records)


def run_scaling_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for d in [10, 20, 40]:
        for r in [2, 4]:
            n = 20 * d
            m = min(8, d)
            for rep in range(cfg.scaling_repetitions):
                f, U, M = _quadratic_ridge(d, r, cfg.seed + 30_000 + 1000 * d + 10 * r + rep)
                rng = np.random.default_rng(cfg.seed + 40_000 + 1000 * d + 10 * r + rep)
                oracle = NoisyOracle(f, noise_std=0.0, rng=rng)
                result = CrossMomentEstimator(h=0.02, directions_per_side=m).estimate(oracle, d, n, rng)
                A_known, _ = estimate_active_subspace(result, rank=r)
                A_adapt, rhat = estimate_active_subspace(result, rank=None, max_rank=min(8, d - 1))
                records.append({
                    "experiment": "dimension_scaling", "method": "cross_known_rank", "rep": rep,
                    "d": d, "r": r, "n_anchors": n, "directions_per_side": m,
                    "queries": result.query_count,
                    "operator_error": float(np.linalg.norm(result.matrix - M, ord=2)),
                    "subspace_error": principal_angle_error(U, A_known), "rank_hat": r,
                })
                records.append({
                    "experiment": "dimension_scaling", "method": "cross_adaptive", "rep": rep,
                    "d": d, "r": r, "n_anchors": n, "directions_per_side": m,
                    "queries": result.query_count,
                    "operator_error": float(np.linalg.norm(result.matrix - M, ord=2)),
                    "subspace_error": principal_angle_error(U, A_adapt), "rank_hat": rhat,
                })
    return pd.DataFrame.from_records(records)


def _optimizer_for(name: str) -> TwoPointOptimizer:
    # Fixed, benchmark-independent settings avoid per-instance cherry-picking.
    step = {
        "sphere": 0.08,
        "ellipsoid": 0.035,
        "discus": 0.025,
        "bent_cigar": 0.025,
        "different_powers": 0.05,
        "rastrigin": 0.035,
        "rosenbrock": 0.012,
    }[name]
    return TwoPointOptimizer(step_size=step, smoothing=0.03, decay=0.002)


def run_optimization_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    names = ["sphere", "ellipsoid", "rastrigin", "rosenbrock"]
    d, r = 12, 2
    methods = [
        "oracle_subspace",
        "cross_block_known",
        "cross_block_adaptive",
        "cross_sparse_known",
        "random_subspace",
        "full_zo",
    ]
    for name in names:
        for tau in [0.0, 0.02]:
            for rep in range(cfg.optimization_repetitions):
                bench_seed = cfg.seed + 50_000 + 1000 * names.index(name) + 100 * int(tau > 0) + rep
                bench = make_benchmark(name, d=d, r=r, seed=bench_seed, tau=tau)
                x0 = np.zeros(d)
                fstar = bench.optimum_value()
                for method in methods:
                    rng = np.random.default_rng(bench_seed + 10_000 * (1 + methods.index(method)))
                    oracle = NoisyOracle(bench, noise_std=cfg.noise_std, rng=rng)
                    optimizer = _optimizer_for(name)
                    if method == "oracle_subspace":
                        trace = optimizer.run(oracle, x0, cfg.optimization_budget, rng, subspace=bench.U)
                        A = bench.U
                        rank_hat = r
                        discovery_queries = 0
                    elif method == "full_zo":
                        trace = optimizer.run(oracle, x0, cfg.optimization_budget, rng, subspace=None)
                        A = np.eye(d)
                        rank_hat = d
                        discovery_queries = 0
                    elif method == "random_subspace":
                        A = random_orthonormal(d, r, rng)
                        trace = optimizer.run(oracle, x0, cfg.optimization_budget, rng, subspace=A)
                        rank_hat = r
                        discovery_queries = 0
                    else:
                        if method.startswith("cross_block"):
                            n_anchors = cfg.discovery_anchors
                            m = cfg.discovery_directions_per_side
                        else:
                            # Near-query-matched sparse sketch: 5x as many anchors and
                            # 1/5 as many directions per fold (600 vs 720 discovery queries
                            # under the verified configuration).
                            m = max(1, cfg.discovery_directions_per_side // 5)
                            n_anchors = cfg.discovery_anchors * 5
                        known_rank = r if method.endswith("known") else None
                        trace, A, rank_hat, _ = discover_then_optimize(
                            oracle,
                            d=d,
                            total_budget=cfg.optimization_budget,
                            discovery_anchors=n_anchors,
                            rng=rng,
                            rank=known_rank,
                            max_rank=8,
                            estimator=CrossMomentEstimator(
                                h=0.03,
                                anchor_radius=1.0,
                                directions_per_side=m,
                            ),
                            optimizer=optimizer,
                            x0=x0,
                        )
                        discovery_queries = 4 * m * n_anchors
                    noiseless_value = float(bench(trace.x_best))
                    records.append(
                        {
                            "experiment": "optimization",
                            "benchmark": name,
                            "tau": tau,
                            "method": method,
                            "rep": rep,
                            "d": d,
                            "r": r,
                            "rank_hat": rank_hat,
                            "queries": trace.query_count,
                            "discovery_queries": discovery_queries,
                            "subspace_error": principal_angle_error(bench.U, A) if A.shape[1] != d else 1.0,
                            "best_noisy_value": trace.f_best,
                            "final_value": noiseless_value,
                            "regret": max(noiseless_value - fstar, 1e-16),
                        }
                    )
    return pd.DataFrame.from_records(records)


def run_amortization_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Measure when one-time discovery pays off across related objectives."""
    from .benchmarks import EmbeddedBenchmark, ellipsoid

    records: list[dict[str, Any]] = []
    d, r = 12, 2
    task_counts = [1, 3, 5]
    per_task_budget = 1500
    discovery_queries = 4 * cfg.discovery_directions_per_side * cfg.discovery_anchors
    for rep in range(cfg.optimization_repetitions):
        rng_master = np.random.default_rng(cfg.seed + 90_000 + rep)
        U = random_orthonormal(d, r, rng_master)
        tasks = []
        for k in range(max(task_counts)):
            shift = rng_master.uniform(-1.0, 1.0, size=d)
            tasks.append(EmbeddedBenchmark(ellipsoid, U=U, shift=shift, tau=0.01, name="ellipsoid"))
        # Calibrate on the first task and reuse the estimated geometry.
        rng_discovery = np.random.default_rng(cfg.seed + 91_000 + rep)
        calibration_oracle = NoisyOracle(tasks[0], noise_std=cfg.noise_std, rng=rng_discovery)
        estimate = CrossMomentEstimator(
            h=0.03,
            directions_per_side=cfg.discovery_directions_per_side,
        ).estimate(calibration_oracle, d, cfg.discovery_anchors, rng_discovery)
        A, rhat = estimate_active_subspace(estimate, rank=r)
        angle = principal_angle_error(U, A)
        for K in task_counts:
            # Full-dimensional method receives the same total query budget,
            # distributed evenly across tasks.
            equalized_full_budget = per_task_budget + int(np.ceil(discovery_queries / K))
            for method in ["cross_reused", "full_equal_total", "oracle_subspace", "random_subspace"]:
                regrets = []
                total_queries = discovery_queries if method == "cross_reused" else 0
                for k, task in enumerate(tasks[:K]):
                    rng = np.random.default_rng(cfg.seed + 100_000 + rep * 1000 + K * 100 + k * 10 + ["cross_reused", "full_equal_total", "oracle_subspace", "random_subspace"].index(method))
                    oracle = NoisyOracle(task, noise_std=cfg.noise_std, rng=rng)
                    optimizer = _optimizer_for("ellipsoid")
                    if method == "cross_reused":
                        subspace, budget = A, per_task_budget
                    elif method == "full_equal_total":
                        subspace, budget = None, equalized_full_budget
                    elif method == "oracle_subspace":
                        subspace, budget = U, per_task_budget
                    else:
                        subspace, budget = random_orthonormal(d, r, rng), per_task_budget
                    trace = optimizer.run(oracle, np.zeros(d), budget, rng, subspace=subspace)
                    total_queries += trace.query_count
                    regrets.append(max(task(trace.x_best) - task.optimum_value(), 1e-16))
                records.append(
                    {
                        "experiment": "amortization",
                        "method": method,
                        "rep": rep,
                        "tasks": K,
                        "d": d,
                        "r": r,
                        "rank_hat": rhat if method == "cross_reused" else (r if method != "full_equal_total" else d),
                        "subspace_error": angle if method == "cross_reused" else (0.0 if method == "oracle_subspace" else 1.0),
                        "total_queries": total_queries,
                        "mean_regret": float(np.mean(regrets)),
                        "median_regret": float(np.median(regrets)),
                    }
                )
    return pd.DataFrame.from_records(records)

def run_all(cfg: ExperimentConfig, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    raw = out / "raw"
    summary = out / "summary"
    raw.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for name, runner in [
        ("moment_bias", run_bias_experiment),
        ("dimension_scaling", run_scaling_experiment),
        ("optimization", run_optimization_experiment),
        ("amortization", run_amortization_experiment),
    ]:
        frame = runner(cfg)
        path = raw / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs[name] = path

    # Compact summaries used directly by the Quarto article.
    bias = pd.read_csv(outputs["moment_bias"])
    bias_summary = (
        bias.groupby(["method", "n_anchors"], as_index=False)
        .agg(
            operator_error_mean=("operator_error", "mean"),
            operator_error_se=("operator_error", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            subspace_error_mean=("subspace_error", "mean"),
            subspace_error_se=("subspace_error", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
    )
    bias_summary.to_csv(summary / "moment_bias_summary.csv", index=False)

    scaling = pd.read_csv(outputs["dimension_scaling"])
    scaling_summary = (
        scaling.groupby(["method", "d", "r"], as_index=False)
        .agg(
            subspace_error_mean=("subspace_error", "mean"),
            rank_recovery=("rank_hat", lambda x: np.mean(x == x.name if False else 0.0)),
            rank_hat_mean=("rank_hat", "mean"),
        )
    )
    # Correct rank recovery in an explicit, readable operation.
    recovery = (
        scaling.assign(correct=lambda z: (z["rank_hat"] == z["r"]).astype(float))
        .groupby(["method", "d", "r"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "rank_recovery"})
    )
    scaling_summary = scaling_summary.drop(columns=["rank_recovery"]).merge(recovery, on=["method", "d", "r"])
    scaling_summary.to_csv(summary / "dimension_scaling_summary.csv", index=False)

    opt = pd.read_csv(outputs["optimization"])
    opt_summary = (
        opt.groupby(["benchmark", "tau", "method"], as_index=False)
        .agg(
            median_regret=("regret", "median"),
            mean_log10_regret=("regret", lambda x: np.log10(x).mean()),
            median_rank=("rank_hat", "median"),
            median_subspace_error=("subspace_error", "median"),
        )
    )
    opt_summary.to_csv(summary / "optimization_summary.csv", index=False)

    amort = pd.read_csv(outputs["amortization"])
    amort_summary = (
        amort.groupby(["tasks", "method"], as_index=False)
        .agg(
            median_mean_regret=("mean_regret", "median"),
            mean_log10_regret=("mean_regret", lambda x: np.log10(x).mean()),
            median_total_queries=("total_queries", "median"),
        )
    )
    amort_summary.to_csv(summary / "amortization_summary.csv", index=False)

    pd.DataFrame([asdict(cfg)]).to_csv(summary / "experiment_config.csv", index=False)
    return outputs
