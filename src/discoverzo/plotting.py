from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(results_dir: str | Path, figure_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    bias = pd.read_csv(results_dir / "raw" / "moment_bias.csv")
    summary = bias.groupby(["method", "n_anchors"], as_index=False).agg(
        mean=("operator_error", "mean"),
        se=("operator_error", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for method, group in summary.groupby("method"):
        group = group.sort_values("n_anchors")
        ax.errorbar(group["n_anchors"], group["mean"], yerr=1.96 * group["se"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Operator-norm error")
    ax.set_title("Bias–variance crossover of moment estimators")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "moment_operator_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    sub = bias.groupby(["method", "n_anchors"], as_index=False)["subspace_error"].mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for method, group in sub.groupby("method"):
        ax.plot(group["n_anchors"], group["subspace_error"], marker="o", label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel(r"$\|\sin\Theta(\widehat U,U_\star)\|_{op}$")
    ax.set_title("Finite-sample eigenvector recovery")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "moment_subspace_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    scaling = pd.read_csv(results_dir / "raw" / "dimension_scaling.csv")
    adapt = scaling[scaling["method"] == "cross_adaptive"].copy()
    adapt["correct"] = (adapt["rank_hat"] == adapt["r"]).astype(float)
    recovery = adapt.groupby(["d", "r"], as_index=False)["correct"].mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r, group in recovery.groupby("r"):
        ax.plot(group["d"], group["correct"], marker="o", label=f"r={r}")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Ambient dimension d")
    ax.set_ylabel("Exact rank-recovery frequency")
    ax.set_title("Adaptive rank selection at N=20d anchors")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "rank_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    known = scaling[scaling["method"] == "cross_known_rank"]
    known_summary = known.groupby(["d", "r"], as_index=False)["subspace_error"].mean()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for r, group in known_summary.groupby("r"):
        ax.plot(group["d"], group["subspace_error"], marker="o", label=f"r={r}")
    ax.set_xlabel("Ambient dimension d")
    ax.set_ylabel("Mean principal-angle error")
    ax.set_title("Discovery error under linear anchor scaling")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "dimension_scaling.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    opt = pd.read_csv(results_dir / "raw" / "optimization.csv")
    exact = opt[opt["tau"] == 0.0]
    order = [
        "oracle_subspace",
        "cross_block_known",
        "cross_block_adaptive",
        "cross_sparse_known",
        "random_subspace",
        "full_zo",
    ]
    names = ["sphere", "ellipsoid", "rosenbrock", "rastrigin"]
    pivot = exact.groupby(["benchmark", "method"])["regret"].median().unstack()
    matrix = np.log10(pivot.reindex(index=names, columns=order).to_numpy())
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(order)), order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_title("Median log10 regret on embedded public benchmarks")
    fig.colorbar(image, ax=ax, label="log10 regret")
    fig.tight_layout()
    path = figure_dir / "optimization_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    approx = opt[opt["tau"] == 0.02]
    med = approx.groupby("method", as_index=False)["regret"].median().set_index("method").reindex(order)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(len(order)), np.log10(med["regret"].to_numpy()))
    ax.set_xticks(np.arange(len(order)), order, rotation=30, ha="right")
    ax.set_ylabel("Median log10 regret")
    ax.set_title("Approximate active-subspace robustness")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "approximate_subspace_robustness.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    amort = pd.read_csv(results_dir / "raw" / "amortization.csv")
    am = amort.groupby(["tasks", "method"], as_index=False)["mean_regret"].median()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for method, group in am.groupby("method"):
        ax.plot(group["tasks"], group["mean_regret"], marker="o", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Number of related optimization tasks")
    ax.set_ylabel("Median mean regret")
    ax.set_title("Amortizing one-time structure discovery")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "amortization.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    return outputs
