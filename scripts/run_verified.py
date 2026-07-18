"""Reproduce the cached, manuscript-facing experiment suite.

The optimization matrix is intentionally split into five deterministic batches
of two repetitions.  This keeps peak runtime and failure recovery manageable
without changing the random design.  The script overwrites manuscript-facing
CSV files, summaries, figures, and validation metadata.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from discoverzo.experiments import (
    ExperimentConfig,
    run_amortization_experiment,
    run_bias_experiment,
    run_optimization_experiment,
    run_scaling_experiment,
)
from discoverzo.plotting import plot_all

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"
FIGURES = ROOT / "figures"


def _se(series: pd.Series) -> float:
    return float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else 0.0


def summarize() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    bias = pd.read_csv(RAW / "moment_bias.csv")
    bias.groupby(["method", "n_anchors"], as_index=False).agg(
        operator_error_mean=("operator_error", "mean"),
        operator_error_se=("operator_error", _se),
        subspace_error_mean=("subspace_error", "mean"),
        subspace_error_se=("subspace_error", _se),
    ).to_csv(SUMMARY / "moment_bias_summary.csv", index=False)

    scaling = pd.read_csv(RAW / "dimension_scaling.csv")
    scaling["correct"] = (scaling["rank_hat"] == scaling["r"]).astype(float)
    scaling.groupby(["method", "d", "r"], as_index=False).agg(
        subspace_error_mean=("subspace_error", "mean"),
        subspace_error_se=("subspace_error", _se),
        rank_hat_mean=("rank_hat", "mean"),
        rank_recovery=("correct", "mean"),
        queries=("queries", "median"),
    ).to_csv(SUMMARY / "dimension_scaling_summary.csv", index=False)

    opt = pd.read_csv(RAW / "optimization.csv")
    opt.groupby(["benchmark", "tau", "method"], as_index=False).agg(
        median_regret=("regret", "median"),
        q25_regret=("regret", lambda x: x.quantile(0.25)),
        q75_regret=("regret", lambda x: x.quantile(0.75)),
        mean_log10_regret=("regret", lambda x: np.log10(x).mean()),
        median_rank=("rank_hat", "median"),
        median_subspace_error=("subspace_error", "median"),
        queries=("queries", "median"),
    ).to_csv(SUMMARY / "optimization_summary.csv", index=False)

    amort = pd.read_csv(RAW / "amortization.csv")
    amort.groupby(["tasks", "method"], as_index=False).agg(
        median_mean_regret=("mean_regret", "median"),
        q25=("mean_regret", lambda x: x.quantile(0.25)),
        q75=("mean_regret", lambda x: x.quantile(0.75)),
        median_total_queries=("total_queries", "median"),
        median_subspace_error=("subspace_error", "median"),
    ).to_csv(SUMMARY / "amortization_summary.csv", index=False)


def bootstrap_paired(seed: int = 8675309, draws: int = 20000) -> None:
    rng = np.random.default_rng(seed)
    opt = pd.read_csv(RAW / "optimization.csv")
    rows = []
    for (benchmark, tau), frame in opt.groupby(["benchmark", "tau"]):
        pivot = frame.pivot(index="rep_global", columns="method", values="regret")
        full = np.log10(pivot["full_zo"].to_numpy())
        for method in [c for c in pivot.columns if c != "full_zo"]:
            diff = np.log10(pivot[method].to_numpy()) - full
            boot = np.mean(rng.choice(diff, size=(draws, len(diff)), replace=True), axis=1)
            rows.append({
                "benchmark": benchmark,
                "tau": tau,
                "method": method,
                "n": len(diff),
                "mean_log10_regret_difference_vs_full": float(diff.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "win_rate": float(np.mean(diff < 0)),
            })
    pd.DataFrame(rows).to_csv(SUMMARY / "paired_comparisons.csv", index=False)

    amort = pd.read_csv(RAW / "amortization.csv")
    rows = []
    for tasks, frame in amort.groupby("tasks"):
        pivot = frame.pivot(index="rep", columns="method", values="mean_regret")
        diff = np.log10(pivot["cross_reused"].to_numpy()) - np.log10(pivot["full_equal_total"].to_numpy())
        boot = np.mean(rng.choice(diff, size=(draws, len(diff)), replace=True), axis=1)
        rows.append({
            "tasks": tasks,
            "n": len(diff),
            "mean_log10_difference_cross_minus_full": float(diff.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "cross_win_rate": float(np.mean(diff < 0)),
        })
    pd.DataFrame(rows).to_csv(SUMMARY / "amortization_paired.csv", index=False)


def main() -> None:
    with (ROOT / "configs" / "verified.yaml").open(encoding="utf-8") as handle:
        cfg = ExperimentConfig(**yaml.safe_load(handle))
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)

    print("[1/4] moment calibration")
    run_bias_experiment(cfg).to_csv(RAW / "moment_bias.csv", index=False)
    print("[2/4] dimension/rank scaling")
    run_scaling_experiment(cfg).to_csv(RAW / "dimension_scaling.csv", index=False)
    print("[3/4] optimization benchmark (5 deterministic batches)")
    batches = []
    for batch in range(5):
        batch_cfg = replace(cfg, seed=cfg.seed + 1_000_000 * batch, optimization_repetitions=2)
        frame = run_optimization_experiment(batch_cfg)
        frame["batch"] = batch
        frame["rep_global"] = frame["rep"] + 2 * batch
        batches.append(frame)
    pd.concat(batches, ignore_index=True).to_csv(RAW / "optimization.csv", index=False)
    print("[4/4] shared-subspace amortization")
    run_amortization_experiment(cfg).to_csv(RAW / "amortization.csv", index=False)

    summarize()
    bootstrap_paired()
    pd.DataFrame([asdict(cfg)]).to_csv(SUMMARY / "experiment_config.csv", index=False)
    plot_all(ROOT / "results", FIGURES)

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": asdict(cfg),
        "raw_rows": {
            p.stem: int(len(pd.read_csv(p)))
            for p in sorted(RAW.glob("*.csv"))
        },
    }
    (SUMMARY / "validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Verified experiment suite completed.")


if __name__ == "__main__":
    main()
