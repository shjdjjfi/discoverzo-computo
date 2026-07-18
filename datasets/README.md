# 数据集说明

本项目没有使用私有数据，也没有使用大模型。实验数据由两部分组成：

1. **公开标准函数**：Sphere、Ellipsoid、Rastrigin 和 Rosenbrock，采用公开 BBOB/经典优化基准公式；实现位于 `src/discoverzo/benchmarks.py`。
2. **确定性生成的实验实例**：每个实例由公开公式、随机正交子空间 `U`、位移 `shift`、近似低维强度 `tau` 和固定种子唯一确定。

目录内容：

- `benchmark_catalog.csv`：全部公开函数、公式和类别。
- `generated_inputs/moment_calibration_inputs.npz`：20 个矩估计实验的真实 `U`、真实矩阵 `M` 和全部 5,120 个共享锚点。
- `generated_inputs/dimension_scaling_inputs.npz`：所有维数/秩组合的真实 `U`、`M` 和锚点。
- `generated_inputs/optimization_benchmark_instances.npz`：80 个优化实例的真实 `U` 和 `shift`。
- `generated_inputs/amortization_instances.npz`：10 次共享子空间实验中的真实 `U` 和 5 个任务位移。
- `seed_ledger.csv`：所有实验的随机种子账本。
- `raw_outputs/`：逐重复、逐方法的原始实验结果，共 1,140 行。
- `summary_outputs/`：聚合统计、四分位数、标准误和配对 bootstrap 区间。

随机方向属于算法内部随机性，不作为静态数据文件存储；它们由 `seed_ledger.csv` 和源码中的确定性公式逐位重建。这样避免存储数百 MB 的冗余随机矩阵，同时仍保持精确复现。

读取 NPZ 示例：

```python
import numpy as np
bundle = np.load("datasets/generated_inputs/optimization_benchmark_instances.npz")
U = bundle["sphere_tau0.00_rep0_U"]
shift = bundle["sphere_tau0.00_rep0_shift"]
```
