from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

bias = pd.read_csv(RAW / "moment_bias.csv")
bias_summary = (
    bias.groupby(["method", "n_anchors"], as_index=False)
    .agg(
        operator_error_mean=("operator_error", "mean"),
        operator_error_se=("operator_error", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        subspace_error_mean=("subspace_error", "mean"),
        subspace_error_se=("subspace_error", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )
)
bias_summary.to_csv(SUMMARY / "moment_bias_summary.csv", index=False)

scaling = pd.read_csv(RAW / "dimension_scaling.csv")
scaling = scaling.assign(correct=(scaling["rank_hat"] == scaling["r"]).astype(float))
scaling_summary = (
    scaling.groupby(["method", "d", "r"], as_index=False)
    .agg(
        subspace_error_mean=("subspace_error", "mean"),
        subspace_error_se=("subspace_error", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        rank_hat_mean=("rank_hat", "mean"),
        rank_recovery=("correct", "mean"),
        queries=("queries", "median"),
    )
)
scaling_summary.to_csv(SUMMARY / "dimension_scaling_summary.csv", index=False)

opt = pd.read_csv(RAW / "optimization.csv")
opt_summary = (
    opt.groupby(["benchmark", "tau", "method"], as_index=False)
    .agg(
        median_regret=("regret", "median"),
        q25_regret=("regret", lambda x: x.quantile(0.25)),
        q75_regret=("regret", lambda x: x.quantile(0.75)),
        mean_log10_regret=("regret", lambda x: np.log10(x).mean()),
        median_rank=("rank_hat", "median"),
        median_subspace_error=("subspace_error", "median"),
        queries=("queries", "median"),
    )
)
opt_summary.to_csv(SUMMARY / "optimization_summary.csv", index=False)

amort = pd.read_csv(RAW / "amortization.csv")
amort_summary = (
    amort.groupby(["tasks", "method"], as_index=False)
    .agg(
        median_mean_regret=("mean_regret", "median"),
        q25=("mean_regret", lambda x: x.quantile(0.25)),
        q75=("mean_regret", lambda x: x.quantile(0.75)),
        median_total_queries=("total_queries", "median"),
        median_subspace_error=("subspace_error", "median"),
    )
)
amort_summary.to_csv(SUMMARY / "amortization_summary.csv", index=False)

print("Wrote summaries to", SUMMARY)
