# Experiment protocol

## General rules

1. Every function evaluation, including structure discovery, counts against the stated budget.
2. Hyperparameters are fixed by benchmark family, not selected per random instance.
3. No large language model, neural surrogate, or private dataset is used.
4. All benchmark formulas, shifts, embeddings, and noise draws are seed controlled.
5. Every manuscript number is derived from committed repetition-level CSV files.
6. Negative outcomes are retained and discussed.

## Experiment A: moment calibration

- Ambient dimension: 12
- True rank: 2
- Objective: random quadratic ridge function
- Anchor distribution: uniform on `[-1,1]^12`
- Noise: none, to isolate moment bias and variance
- Anchors: 80, 160, 320, 640, 1,280, 2,560, 5,120
- Repetitions: 20
- Cross estimator: 4 orthogonal directions per side
- Outer estimators: 8 orthogonal directions
- Equal cost: 16 function evaluations per anchor
- Metrics: operator error, Frobenius error, principal-angle error

## Experiment B: ambient-dimension scaling

- Dimensions: 10, 20, 40
- Ranks: 2, 4
- Anchors: `20*d`
- Directions per side: `min(8,d)`
- Repetitions: 10
- Metrics: known-rank angle error, adaptive rank, exact rank recovery

## Experiment C: single-task optimization

- Ambient dimension: 12
- Intrinsic dimension: 2
- Public formulas: Sphere, Ellipsoid, Rosenbrock, Rastrigin
- Exact/approximate ridge parameter: 0 and 0.02
- Total budget: 4,000 calls
- Oracle noise standard deviation: 0.02
- Repetitions: 10, generated as five restartable batches of two
- Methods: oracle subspace, cross block known rank, cross block adaptive rank, near-cost-matched sparse cross sketch, random rank-two subspace, full-dimensional two-point optimization
- Primary metric: noiseless regret of the best observed point
- Secondary metrics: projector error, selected rank, discovery-query share

## Experiment D: discovery amortization

- Five related shifted Ellipsoid objectives share one rank-two subspace
- Evaluated task counts: 1, 3, 5
- Discovery performed once and reused
- Per-task optimization budget: 1,500
- The full-dimensional comparator receives the same total calls, including the discovery allowance
- Repetitions: 10
- Metrics: mean regret over tasks, total query count, projector error

## Statistical summaries

- Medians and interquartile ranges for optimization outcomes
- Means and standard errors for estimation outcomes
- Paired bootstrap intervals on log10 regret differences
- 20,000 bootstrap resamples, fixed RNG seed

The small number of optimization repetitions is treated as a limitation. Bootstrap intervals are uncertainty summaries, not evidence of universal dominance.
