from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "summary"
rng = np.random.default_rng(20260718)

opt = pd.read_csv(RAW / "optimization.csv")
key = ["benchmark", "tau", "batch", "rep"] if "batch" in opt.columns else ["benchmark", "tau", "rep"]
wide = opt.pivot_table(index=key, columns="method", values="regret").reset_index()
rows = []
for (benchmark, tau), group in wide.groupby(["benchmark", "tau"]):
    for method in [c for c in wide.columns if c not in key and c != "full_zo"]:
        delta = np.log10(group[method].to_numpy()) - np.log10(group["full_zo"].to_numpy())
        boots = np.empty(20000)
        for b in range(20000):
            boots[b] = rng.choice(delta, size=len(delta), replace=True).mean()
        rows.append({
            "benchmark": benchmark,
            "tau": tau,
            "method": method,
            "n": len(delta),
            "mean_log10_regret_difference_vs_full": delta.mean(),
            "ci95_low": np.quantile(boots, 0.025),
            "ci95_high": np.quantile(boots, 0.975),
            "win_rate": np.mean(delta < 0),
        })
pd.DataFrame(rows).to_csv(OUT / "paired_comparisons.csv", index=False)

am = pd.read_csv(RAW / "amortization.csv")
wide_am = am.pivot_table(index=["tasks", "rep"], columns="method", values="mean_regret").reset_index()
rows = []
for tasks, group in wide_am.groupby("tasks"):
    delta = np.log10(group["cross_reused"].to_numpy()) - np.log10(group["full_equal_total"].to_numpy())
    boots = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(20000)])
    rows.append({
        "tasks": tasks,
        "n": len(delta),
        "mean_log10_difference_cross_minus_full": delta.mean(),
        "ci95_low": np.quantile(boots, 0.025),
        "ci95_high": np.quantile(boots, 0.975),
        "cross_win_rate": np.mean(delta < 0),
    })
pd.DataFrame(rows).to_csv(OUT / "amortization_paired.csv", index=False)
print("Wrote paired analyses")
