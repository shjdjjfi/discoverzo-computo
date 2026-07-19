#!/usr/bin/env python3
"""Generate publication-quality manuscript figures and static tables.

All plotted values are read from the cached CSV files under ``results/``.
The script never reruns an optimizer. It writes PDF, SVG, and PNG versions
of each figure and Markdown/CSV versions of each table.
"""
from __future__ import annotations

from pathlib import Path
import math
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary"
RAW = ROOT / "results" / "raw"
FIGDIR = ROOT / "figures" / "manuscript"
TABDIR = ROOT / "tables" / "manuscript"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

# Okabe--Ito / neutral palette, chosen for color-vision accessibility.
COLORS = {
    "cross": "#D55E00",
    "outer": "#0072B2",
    "outer_trace": "#009E73",
    "cross_block_adaptive": "#E69F00",
    "cross_block_known": "#D55E00",
    "cross_sparse_known": "#CC79A7",
    "full_zo": "#0072B2",
    "oracle_subspace": "#000000",
    "random_subspace": "#8A8A8A",
    "cross_reused": "#D55E00",
    "full_equal_total": "#0072B2",
}
LABELS = {
    "cross": "Cross-fitted",
    "outer": "Self outer product",
    "outer_trace": "Trace-corrected outer",
    "cross_block_adaptive": "Cross, adaptive rank",
    "cross_block_known": "Cross, known rank",
    "cross_sparse_known": "Cross, sparse sketch",
    "full_zo": "Full-dimensional ZO",
    "oracle_subspace": "Oracle subspace",
    "random_subspace": "Random subspace",
    "cross_reused": "Cross, reused",
    "full_equal_total": "Full ZO, equal total budget",
}
MARKERS = {
    "cross": "o", "outer": "s", "outer_trace": "^",
    "cross_block_adaptive": "o", "cross_block_known": "s",
    "cross_sparse_known": "^", "full_zo": "D",
    "oracle_subspace": "*", "random_subspace": "X",
    "cross_reused": "o", "full_equal_total": "s",
}

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def finish(fig: plt.Figure, stem: str) -> None:
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold", va="top")


def make_schematic() -> None:
    # 略微增加画布宽度，并统一四个框的宽度
    fig, ax = plt.subplots(figsize=(7.4, 2.45))
    ax.set_axis_off()

    box_y, box_h, box_w = 0.34, 0.40, 0.205
    box_x = [0.025, 0.273, 0.521, 0.769]

    stages = [
        (
            box_x[0], box_y, box_w, box_h,
            "Anchor design",
            "Draw $X_i\\sim\\nu$\nand two frames",
        ),
        (
            box_x[1], box_y, box_w, box_h,
            "Cross moment",
            "$4m$ calls per anchor\n$\\widetilde M_N$",
        ),
        (
            box_x[2], box_y, box_w, box_h,
            "Spectral step",
            "Estimate $\\widehat U$\nand optionally $\\widehat r$",
        ),
        (
            box_x[3], box_y, box_w, box_h,
            "Restricted search",
            "Optimize in\n$x_0+\\operatorname{span}(\\widehat U)$",
        ),
    ]

    # 不再继承全局的较大字体，单独控制流程图字号
    title_fs = 8.7
    body_fs = 8.2

    for i, (x, y, w, h, title, body) in enumerate(stages):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor="#333333",
            facecolor="#F7F7F7",
        )
        ax.add_patch(box)

        ax.text(
            x + w / 2,
            y + h * 0.72,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
        )

        ax.text(
            x + w / 2,
            y + h * 0.34,
            body,
            ha="center",
            va="center",
            fontsize=body_fs,
            linespacing=1.18,
        )

        if i < len(stages) - 1:
            nx = stages[i + 1][0]

            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.006, y + h / 2),
                    (nx - 0.006, y + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color="#444444",
                )
            )

    # 横线和文字分别设置纵坐标，避免二者贴在一起
    budget_y = 0.205
    label_y = 0.155

    # Discovery budget
    ax.plot(
        [box_x[0], box_x[2] + box_w],
        [budget_y, budget_y],
        color="#777777",
        lw=0.7,
    )

    ax.text(
        (box_x[0] + box_x[2] + box_w) / 2,
        label_y,
        "Discovery budget",
        ha="center",
        va="top",
        fontsize=8.6,
        fontstyle="italic",
    )

    # Optimization budget
    ax.plot(
        [box_x[3], box_x[3] + box_w],
        [budget_y, budget_y],
        color="#777777",
        lw=0.7,
    )

    ax.text(
        box_x[3] + box_w / 2,
        label_y,
        "Optimization budget",
        ha="center",
        va="top",
        fontsize=8.6,
        fontstyle="italic",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    finish(fig, "fig01_workflow")


def make_calibration() -> None:
    df = pd.read_csv(SUMMARY / "moment_bias_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), sharex=True)
    specs = [
        ("operator_error_mean", "operator_error_se", r"Operator error $\|\widehat M-M\|_{\mathrm{op}}$"),
        ("subspace_error_mean", "subspace_error_se", r"Subspace error $\|\sin\Theta\|_{\mathrm{op}}$"),
    ]
    for ax, (mean_col, se_col, ylabel) in zip(axes, specs):
        for method in ["cross", "outer", "outer_trace"]:
            d = df[df.method == method].sort_values("n_anchors")
            x = d.n_anchors.to_numpy(); y = d[mean_col].to_numpy(); se = d[se_col].to_numpy()
            ax.plot(x, y, marker=MARKERS[method], ms=4.2, lw=1.35,
                    color=COLORS[method], label=LABELS[method])
            ax.fill_between(x, np.maximum(y-se, 1e-5), y+se, color=COLORS[method], alpha=0.13, linewidth=0)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.grid(True, which="both")
        ax.set_xlabel("Number of anchors $N$")
        ax.set_ylabel(ylabel)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.51, 1.07))
    fig.subplots_adjust(top=0.78, wspace=0.28)
    finish(fig, "fig02_calibration")


def make_scaling() -> None:
    df = pd.read_csv(SUMMARY / "dimension_scaling_summary.csv")
    adaptive = df[df.method == "cross_adaptive"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0))

    # -------------------------
    # (a) 左图：误差曲线
    # -------------------------
    for r, marker in [(2, "o"), (4, "s")]:
        for method, ls, color in [
            ("cross_known_rank", "-", "#0072B2"),
            ("cross_adaptive", "--", "#D55E00"),
        ]:
            d = df[(df.method == method) & (df.r == r)].sort_values("d")
            axes[0].errorbar(
                d.d,
                d.subspace_error_mean,
                yerr=d.subspace_error_se,
                marker=marker,
                ms=4,
                lw=1.25,
                capsize=2,
                linestyle=ls,
                color=color,
                label=f"{'Known' if method.endswith('known_rank') else 'Adaptive'}, r={r}",
            )

    axes[0].set_xlabel("Ambient dimension $d$")
    axes[0].set_ylabel(r"Subspace error $\|\sin\Theta\|_{\mathrm{op}}$")
    axes[0].set_xticks([10, 20, 30, 40])   # 强制保留 30
    axes[0].set_xlim(8.5, 41.5)
    axes[0].grid(True)

    # 面板字母上移
    axes[0].text(
        -0.14, 1.12, "a",
        transform=axes[0].transAxes,
        fontweight="bold",
        va="top"
    )

    # 图例放到子图外上方
    handles0, labels0 = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles0,
        labels0,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.18),
        borderaxespad=0.0,
    )

    # -------------------------
    # (b) 右图：rank recovery 柱状图
    # -------------------------
    width = 2.8
    dims_all = [10, 20, 30, 40]   # 强制显示 30
    for i, r in enumerate([2, 4]):
        d = adaptive[adaptive.r == r].sort_values("d")
        axes[1].bar(
            d.d + (i - 0.5) * width,
            100 * d.rank_recovery,
            width=width,
            edgecolor="white",
            linewidth=0.5,
            label=f"True rank {r}",
        )

    axes[1].set_xlabel("Ambient dimension $d$")
    axes[1].set_ylabel("Exact rank recovery (%)")
    axes[1].set_xticks(dims_all)          # 显示 10,20,30,40
    axes[1].set_xlim(6, 44)               # 给 30 留出明确空间
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, axis="y")

    # 面板字母上移
    axes[1].text(
        -0.14, 1.12, "b",
        transform=axes[1].transAxes,
        fontweight="bold",
        va="top"
    )

    # 图例放到子图外上方
    handles1, labels1 = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles1,
        labels1,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.18),
        borderaxespad=0.0,
    )

    # 给顶部图例留空间
    fig.subplots_adjust(top=0.73, wspace=0.30)

    finish(fig, "fig03_dimension_scaling")


def make_regret_distribution(tau: float, stem: str) -> None:
    raw = pd.read_csv(RAW / "optimization.csv")
    raw = raw[np.isclose(raw.tau, tau)].copy()
    benchmarks = ["sphere", "ellipsoid", "rosenbrock", "rastrigin"]
    methods = ["oracle_subspace", "cross_block_known", "cross_block_adaptive",
               "cross_sparse_known", "full_zo", "random_subspace"]
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 2.7), sharey=True)
    for j, (ax, bench) in enumerate(zip(axes, benchmarks)):
        b = raw[raw.benchmark == bench]
        values = [np.log10(np.maximum(b[b.method == m].regret.to_numpy(), 1e-12)) for m in methods]
        bp = ax.boxplot(values, positions=np.arange(len(methods)), widths=0.62,
                        patch_artist=True, showfliers=False, whis=(5,95),
                        medianprops={"color":"black", "linewidth":1.0},
                        whiskerprops={"linewidth":0.7}, capprops={"linewidth":0.7},
                        boxprops={"linewidth":0.7})
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.78)
        # deterministic jitter by point index
        for k, m in enumerate(methods):
            y = values[k]
            x = k + np.linspace(-0.12,0.12,len(y))
            ax.scatter(x, y, s=7, color="black", alpha=0.35, linewidths=0, zorder=3)
        ax.axhline(np.median(values[methods.index("full_zo")]), color=COLORS["full_zo"], lw=0.8, ls=":")
        ax.set_title(bench.capitalize())
        ax.set_xticks(np.arange(len(methods)))
        short = ["Oracle", "Cross\nknown", "Cross\nadapt.", "Sparse", "Full", "Random"]
        ax.set_xticklabels(short, rotation=45, ha="right")
        ax.grid(True, axis="y")
        panel_label(ax, chr(ord('a')+j))
        if j == 0:
            ax.set_ylabel(r"$\log_{10}$ final normalized regret")
    fig.suptitle("Exact ridge structure" if tau == 0 else r"Approximate ridge structure ($\tau=0.02$)", y=1.03, fontsize=10.5)
    fig.subplots_adjust(wspace=0.12, top=0.85, bottom=0.29)
    finish(fig, stem)


def make_paired_forest() -> None:
    df = pd.read_csv(SUMMARY / "paired_comparisons.csv")
    keep = ["oracle_subspace", "cross_block_known", "cross_block_adaptive", "cross_sparse_known", "random_subspace"]
    benches = ["sphere", "ellipsoid", "rosenbrock", "rastrigin"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharex=True, sharey=True)
    for ax, tau in zip(axes, [0.0, 0.02]):
        sub = df[np.isclose(df.tau, tau)]
        y_positions=[]; labels=[]; cur=0
        for b in benches:
            for m in keep:
                row=sub[(sub.benchmark==b)&(sub.method==m)].iloc[0]
                y=-cur
                ax.errorbar(row.mean_log10_regret_difference_vs_full, y,
                            xerr=[[row.mean_log10_regret_difference_vs_full-row.ci95_low],
                                  [row.ci95_high-row.mean_log10_regret_difference_vs_full]],
                            fmt=MARKERS[m], ms=4.2, color=COLORS[m], capsize=2, lw=0.9)
                y_positions.append(y); labels.append(f"{b.capitalize()} — {LABELS[m]}")
                cur += 1
            cur += 0.75
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.grid(True, axis="x")
        ax.set_title("Exact ridge" if tau==0 else r"Approximate ridge ($\tau=0.02$)")
        ax.text(0.02, 0.02, r"favors subspace $\leftarrow$", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        ax.text(0.98, 0.02, r"$\rightarrow$ favors full ZO", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    axes[0].set_yticks(y_positions); axes[0].set_yticklabels(labels, fontsize=6.7)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.supxlabel(r"Mean paired difference in $\log_{10}$ regret relative to full-dimensional ZO", y=0.03, fontsize=9.5)
    fig.subplots_adjust(left=0.29, right=0.99, wspace=0.10, bottom=0.15, top=0.90)
    finish(fig, "fig06_paired_forest")


def make_amortization() -> None:
    summ = pd.read_csv(SUMMARY / "amortization_summary.csv")
    paired = pd.read_csv(SUMMARY / "amortization_paired.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75))
    methods = ["cross_reused", "full_equal_total", "oracle_subspace", "random_subspace"]
    for m in methods:
        d=summ[summ.method==m].sort_values("tasks")
        axes[0].plot(d.tasks, d.median_mean_regret, marker=MARKERS.get(m,"o"), lw=1.3, ms=4,
                     color=COLORS.get(m,"#666666"), label=LABELS.get(m,m))
        axes[0].fill_between(d.tasks, d.q25, d.q75, color=COLORS.get(m,"#666666"), alpha=0.10, linewidth=0)
    axes[0].set_yscale("log"); axes[0].set_xticks([1,3,5]); axes[0].grid(True, which="both")
    axes[0].set_xlabel("Number of related tasks")
    axes[0].set_ylabel("Median mean regret (IQR band)")
    axes[0].legend(frameon=False, fontsize=7.3)
    panel_label(axes[0], "a")
    x=paired.tasks.to_numpy(); y=paired.mean_log10_difference_cross_minus_full.to_numpy()
    lo=paired.ci95_low.to_numpy(); hi=paired.ci95_high.to_numpy()
    axes[1].errorbar(x,y,yerr=[y-lo,hi-y],fmt="o-",color=COLORS["cross_reused"],lw=1.2,ms=4,capsize=3)
    axes[1].axhline(0,color="black",ls="--",lw=0.8); axes[1].grid(True)
    axes[1].set_xticks([1,3,5]); axes[1].set_xlabel("Number of related tasks")
    axes[1].set_ylabel(r"Cross minus full, mean $\log_{10}$ regret")
    axes[1].text(0.03,0.04,"negative favors reuse",transform=axes[1].transAxes,fontsize=8)
    panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=0.32)
    finish(fig, "fig07_amortization")


def md_table(df: pd.DataFrame, path: Path, formats: dict[str, str] | None = None) -> None:
    out=df.copy()
    formats=formats or {}
    for c, fmt in formats.items():
        if c in out.columns:
            out[c]=out[c].map(lambda x: fmt.format(x))
    path.write_text(out.to_markdown(index=False)+"\n", encoding="utf-8")
    out.to_csv(path.with_suffix(".csv"), index=False)


def make_tables() -> None:
    # Table 1: estimator calibration at endpoints.
    d=pd.read_csv(SUMMARY/"moment_bias_summary.csv")
    d=d[d.n_anchors.isin([80,5120])].copy()
    d["method"]=d.method.map(LABELS)
    d=d.rename(columns={"method":"Estimator","n_anchors":"Anchors",
        "operator_error_mean":"Operator error","operator_error_se":"Operator SE",
        "subspace_error_mean":"Subspace error","subspace_error_se":"Subspace SE"})
    md_table(d, TABDIR/"table01_calibration.md", {c:"{:.4f}" for c in d.columns if "error" in c.lower() or "SE" in c})

    # Table 2: dimension and rank recovery.
    d=pd.read_csv(SUMMARY/"dimension_scaling_summary.csv")
    d=d[d.method=="cross_adaptive"].copy()
    d=d[["d","r","queries","subspace_error_mean","subspace_error_se","rank_recovery"]]
    d.columns=["Ambient d","True r","Queries","Subspace error","SE","Rank recovery"]
    d["Rank recovery"]*=100
    md_table(d,TABDIR/"table02_scaling.md",{"Queries":"{:.0f}","Subspace error":"{:.4f}","SE":"{:.4f}","Rank recovery":"{:.0f}%"})

    # Table 3: selected optimization medians for interpretable methods.
    d=pd.read_csv(SUMMARY/"optimization_summary.csv")
    d=d[d.method.isin(["oracle_subspace","cross_block_known","full_zo","random_subspace"])].copy()
    piv=d.pivot_table(index=["benchmark","tau"],columns="method",values="median_regret").reset_index()
    piv=piv.rename(columns={"benchmark":"Benchmark","tau":r"tau","oracle_subspace":"Oracle subspace",
                            "cross_block_known":"Cross known rank","full_zo":"Full ZO","random_subspace":"Random subspace"})
    md_table(piv,TABDIR/"table03_optimization.md",{c:"{:.4g}" for c in piv.columns if c not in ["Benchmark",r"tau"]})

    # Table 4: strongest paired findings; include all intervals excluding zero plus null oracle rows.
    d=pd.read_csv(SUMMARY/"paired_comparisons.csv")
    sig=d[(d.ci95_low>0)|(d.ci95_high<0)].copy()
    sig["method"]=sig.method.map(LABELS)
    sig=sig[["benchmark","tau","method","mean_log10_regret_difference_vs_full","ci95_low","ci95_high","win_rate"]]
    sig.columns=["Benchmark",r"tau","Method","Mean log10 diff","CI low","CI high","Win rate"]
    sig["Win rate"]*=100
    md_table(sig,TABDIR/"table04_paired.md",{"Mean log10 diff":"{:.3f}","CI low":"{:.3f}","CI high":"{:.3f}","Win rate":"{:.0f}%"})

    # Table 5: amortization paired comparison.
    d=pd.read_csv(SUMMARY/"amortization_paired.csv").copy()
    d.columns=["Tasks","n","Mean log10 diff","CI low","CI high","Cross win rate"]
    d["Cross win rate"]*=100
    md_table(d,TABDIR/"table05_amortization.md",{"Mean log10 diff":"{:.3f}","CI low":"{:.3f}","CI high":"{:.3f}","Cross win rate":"{:.0f}%"})

    design=pd.DataFrame([
        ["Moment calibration","Quadratic ridge","d=12, r=2","N=80,...,5120","20","16 calls/anchor"],
        ["Dimension scaling","Quadratic ridge","d=10,20,40; r=2,4","N=20d","10","4mN, m=min(8,d)"],
        ["Single-task optimization","Embedded BBOB formulas","d=12, r=2; tau=0,0.02","4000 total calls","10","Discovery charged"],
        ["Amortization","Five related ellipsoids","d=12, r=2; K=1,3,5","1500/task + discovery","10","Equal-total full baseline"],
    ],columns=["Study","Objective family","Dimensions","Budget / anchors","Replicates","Budget rule"])
    md_table(design,TABDIR/"table00_design.md")

    index_lines=["# Manuscript table index","",
        "All tables are generated by `scripts/make_manuscript_assets.py` from cached CSV files in `results/summary/`.",""]
    for p in sorted(TABDIR.glob("table*.md")):
        index_lines += [f"## {p.stem}","",p.read_text(encoding="utf-8"),""]
    (TABDIR/"TABLES.md").write_text("\n".join(index_lines),encoding="utf-8")


def main() -> None:
    make_schematic(); make_calibration(); make_scaling()
    make_regret_distribution(0.0,"fig04_exact_regret")
    make_regret_distribution(0.02,"fig05_approx_regret")
    make_paired_forest(); make_amortization(); make_tables()
    print(f"Wrote manuscript assets to {FIGDIR} and {TABDIR}")

if __name__ == "__main__":
    main()
