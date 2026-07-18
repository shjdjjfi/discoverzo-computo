# 全部表格：代码、输入数据与生成结果

本文件是项目中**唯一的表格说明与代码入口**。每张表的输入 CSV 位于 `tables/data/`。下列代码均可从项目根目录直接运行。

## 通用依赖

```python
from pathlib import Path
import pandas as pd
ROOT = Path("tables/data")
```

## 表 1：矩估计校准与子空间恢复

输入：`tables/data/table_01_moment_calibration.csv`

```python
df = pd.read_csv(ROOT / "table_01_moment_calibration.csv")
view = df.rename(columns={
    "operator_error_mean": "op_error_mean",
    "operator_error_se": "op_error_se",
    "subspace_error_mean": "subspace_mean",
    "subspace_error_se": "subspace_se",
})
print(view.to_markdown(index=False, floatfmt=".4f"))
```

| method      |   n_anchors |   op_error_mean |   op_error_se |   subspace_mean |   subspace_se |
|:------------|------------:|----------------:|--------------:|----------------:|--------------:|
| cross       |          80 |          0.4797 |        0.0394 |          0.4405 |        0.031  |
| cross       |        5120 |          0.0514 |        0.003  |          0.0468 |        0.0019 |
| outer       |          80 |          0.3247 |        0.037  |          0.2177 |        0.0136 |
| outer       |        5120 |          0.1186 |        0.005  |          0.0237 |        0.0013 |
| outer_trace |          80 |          0.318  |        0.0369 |          0.2083 |        0.0095 |
| outer_trace |        5120 |          0.1023 |        0.0047 |          0.0225 |        0.0009 |

## 表 2：环境维数、真实秩和恢复误差

输入：`tables/data/table_02_dimension_rank_scaling.csv`

```python
df = pd.read_csv(ROOT / "table_02_dimension_rank_scaling.csv")
view = df[["method", "d", "r", "subspace_error_mean",
           "rank_hat_mean", "rank_recovery", "queries"]]
print(view.to_markdown(index=False, floatfmt=".4f"))
```

| method           |   d |   r |   subspace_error_mean |   rank_hat_mean |   rank_recovery |   queries |
|:-----------------|----:|----:|----------------------:|----------------:|----------------:|----------:|
| cross_adaptive   |  10 |   2 |                0.0677 |             2   |             1   |      6400 |
| cross_adaptive   |  10 |   4 |                0.0843 |             4   |             1   |      6400 |
| cross_adaptive   |  20 |   2 |                0.1334 |             2   |             1   |     12800 |
| cross_adaptive   |  20 |   4 |                0.1835 |             4   |             1   |     12800 |
| cross_adaptive   |  40 |   2 |                0.2556 |             1.9 |             0.9 |     25600 |
| cross_adaptive   |  40 |   4 |                0.2307 |             4   |             1   |     25600 |
| cross_known_rank |  10 |   2 |                0.0677 |             2   |             1   |      6400 |
| cross_known_rank |  10 |   4 |                0.0843 |             4   |             1   |      6400 |
| cross_known_rank |  20 |   2 |                0.1334 |             2   |             1   |     12800 |
| cross_known_rank |  20 |   4 |                0.1835 |             4   |             1   |     12800 |
| cross_known_rank |  40 |   2 |                0.1726 |             2   |             1   |     25600 |
| cross_known_rank |  40 |   4 |                0.2307 |             4   |             1   |     25600 |

## 表 3：精确低维结构下的单任务中位遗憾

输入：`tables/data/table_03_exact_optimization.csv`

```python
df = pd.read_csv(ROOT / "table_03_exact_optimization.csv")
view = df.pivot(index="benchmark", columns="method", values="median_regret")
print(view.to_markdown(floatfmt=".6f"))
```

| benchmark   |   cross_block_adaptive |   cross_block_known |   cross_sparse_known |   full_zo |   oracle_subspace |   random_subspace |
|:------------|-----------------------:|--------------------:|---------------------:|----------:|------------------:|------------------:|
| ellipsoid   |               0.452012 |            0.359151 |             0.328307 |  0.196188 |          0.212801 |          0.357772 |
| rastrigin   |               1.00621  |            1.00616  |             0.996197 |  0.506746 |          1.00703  |          0.50653  |
| rosenbrock  |               0.094335 |            0.059636 |             0.181645 |  0.025812 |          0.035392 |          0.236891 |
| sphere      |               0.003869 |            0.005112 |             0.002228 |  0.005033 |          0.001889 |          0.006426 |

## 表 4：近似低维结构下的单任务中位遗憾

输入：`tables/data/table_04_approximate_optimization.csv`

```python
df = pd.read_csv(ROOT / "table_04_approximate_optimization.csv")
view = df.pivot(index="benchmark", columns="method", values="median_regret")
print(view.to_markdown(floatfmt=".6f"))
```

| benchmark   |   cross_block_adaptive |   cross_block_known |   cross_sparse_known |   full_zo |   oracle_subspace |   random_subspace |
|:------------|-----------------------:|--------------------:|---------------------:|----------:|------------------:|------------------:|
| ellipsoid   |               0.20456  |            0.093827 |             0.179374 |  0.150347 |          0.103512 |          0.397706 |
| rastrigin   |               1.02189  |            0.559422 |             0.549375 |  0.056179 |          0.05313  |          0.615418 |
| rosenbrock  |               0.186196 |            0.124767 |             0.351426 |  0.107646 |          0.097036 |          0.549042 |
| sphere      |               0.037668 |            0.036426 |             0.044118 |  0.048079 |          0.041673 |          0.109598 |

## 表 5：共享子空间的发现成本摊销

输入：`tables/data/table_05_amortization.csv`

```python
df = pd.read_csv(ROOT / "table_05_amortization.csv")
view = df.pivot(index="tasks", columns="method", values="median_mean_regret")
print(view.to_markdown(floatfmt=".6f"))
```

|   tasks |   cross_reused |   full_equal_total |   oracle_subspace |   random_subspace |
|--------:|---------------:|-------------------:|------------------:|------------------:|
|       1 |       0.373196 |           0.287078 |          0.282168 |          0.187205 |
|       3 |       0.291635 |           0.432325 |          0.37523  |          0.430715 |
|       5 |       0.378172 |           0.454373 |          0.289778 |          0.991279 |

## 表 6：摊销实验的配对 bootstrap

输入：`tables/data/table_06_amortization_paired.csv`

```python
df = pd.read_csv(ROOT / "table_06_amortization_paired.csv")
print(df.to_markdown(index=False, floatfmt=".4f"))
```

|   tasks |   n |   mean_log10_difference_cross_minus_full |   ci95_low |   ci95_high |   cross_win_rate |
|--------:|----:|-----------------------------------------:|-----------:|------------:|-----------------:|
|       1 |  10 |                                   0.0555 |    -0.1311 |      0.2614 |              0.6 |
|       3 |  10 |                                  -0.1204 |    -0.2941 |      0.0321 |              0.6 |
|       5 |  10 |                                  -0.0597 |    -0.1568 |      0.03   |              0.6 |

## 表 7：所有优化方法相对全维零阶方法的配对比较

输入：`tables/data/table_07_optimization_paired.csv`

```python
df = pd.read_csv(ROOT / "table_07_optimization_paired.csv")
print(df.to_markdown(index=False, floatfmt=".4f"))
```

完整表有 40 行，直接运行上面的代码可生成。为避免本 Markdown 过长，下面展示前 12 行：

| benchmark   |   tau | method               |   n |   mean_log10_regret_difference_vs_full |   ci95_low |   ci95_high |   win_rate |
|:------------|------:|:---------------------|----:|---------------------------------------:|-----------:|------------:|-----------:|
| ellipsoid   |  0    | cross_block_adaptive |  10 |                                 0.304  |     0.1679 |      0.4444 |        0   |
| ellipsoid   |  0    | cross_block_known    |  10 |                                -0.0602 |    -0.5332 |      0.3383 |        0.4 |
| ellipsoid   |  0    | cross_sparse_known   |  10 |                                 0.3258 |     0.1912 |      0.4694 |        0.1 |
| ellipsoid   |  0    | oracle_subspace      |  10 |                                 0.0023 |    -0.2403 |      0.2289 |        0.5 |
| ellipsoid   |  0    | random_subspace      |  10 |                                 0.3237 |     0.0469 |      0.5918 |        0.2 |
| ellipsoid   |  0.02 | cross_block_adaptive |  10 |                                 0.1416 |    -0.0365 |      0.3367 |        0.3 |
| ellipsoid   |  0.02 | cross_block_known    |  10 |                                -0.089  |    -0.3403 |      0.1905 |        0.7 |
| ellipsoid   |  0.02 | cross_sparse_known   |  10 |                                 0.0134 |    -0.1344 |      0.1876 |        0.5 |
| ellipsoid   |  0.02 | oracle_subspace      |  10 |                                -0.0292 |    -0.2288 |      0.1568 |        0.5 |
| ellipsoid   |  0.02 | random_subspace      |  10 |                                 0.4691 |     0.1716 |      0.7673 |        0.2 |
| rastrigin   |  0    | cross_block_adaptive |  10 |                                 0.2887 |    -0.0449 |      0.7379 |        0.2 |
| rastrigin   |  0    | cross_block_known    |  10 |                                 0.4047 |     0.0728 |      0.843  |        0.2 |

## 一键导出所有表格

下面的代码将 7 张表写入 `tables/generated_tables.md`：

```python
from pathlib import Path
import pandas as pd
root = Path("tables/data")
out = []
for path in sorted(root.glob("table_*.csv")):
    df = pd.read_csv(path)
    out.append(f"## {path.stem}\n\n" + df.to_markdown(index=False, floatfmt=".6g"))
Path("tables/generated_tables.md").write_text("\n\n".join(out), encoding="utf-8")
```
