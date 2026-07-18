# Theory audit checklist

## Estimator target

- Raw cross moment targets the smoothed-gradient moment `M_h`.
- PSD projection is a numerical post-processing operation and is not called unbiased.
- For quadratic objectives, centered finite differences eliminate smoothing bias exactly.

## Independence requirements

- The two frames at an anchor are independent.
- Oracle noises used by the two sketches are independent and mean zero.
- Anchors are independent in the stated concentration theorem.
- Reusing common random numbers would require a different covariance correction.

## Self-product population formula

- Proposition 2 covers an arbitrary Haar orthogonal batch of size `m`, matching the implementation.
- The coefficients reduce to the spherical one-direction identity at `m=1` and to the exact gradient outer product at `m=d`.
- A deterministic Monte Carlo unit test checks the formula at an intermediate `(d,m)`.
- The trace-adjusted baseline remains a heuristic and is never described as unbiased.

## Concentration theorem

- Assumes naturally bounded sketches. If clipping is used, Bernstein concentration is around the clipped population moment and a separate clipping-bias term is required.
- Matrix Bernstein gives a conservative `G^2 sqrt(log(d)/N)` scale.
- Effective-rank, sub-exponential, and robust heavy-tail improvements are future work.

## Eigenspace theorem

- Uses an eigengap and Davis--Kahan.
- Proposition 3 proves exact recovery for a threshold lying between the perturbation radius and the smallest signal eigenvalue.
- The implemented normalized eigengap is an empirical surrogate and is not silently covered by that proposition.
- A decaying spectrum is not forced into a single “true” dimension.

## Optimization transfer

- The approximation result assumes convexity, smoothness, an interior stationary minimizer, and that the displacement from the anchor lies in the reference subspace.
- The stochastic two-point statement is for projected SGD on the smoothed objective.
- The experimental Adam stabilization is not covered by that theorem.

## Lower bound

- Family: rank-one quadratic objectives on the unit ball with Gaussian additive noise.
- Method: local spherical packing, adaptive KL chain rule, Fano inequality.
- Conclusion: ambient-dimensional structure-discovery cost.
- Not claimed: a universal direct-sum lower bound for every subsequent optimization problem.

## Recommended independent mathematical review

1. Verify constants and geometric packing radius in Theorem 3.
2. Check boundary feasibility in Theorem 2 for general convex domains; the current statement implicitly requires the projected representative to remain feasible.
3. Decide whether to state the stochastic two-point result as a theorem or retain it as a corollary under a second-moment assumption.
4. Add a robust median-of-means theorem if heavy-tailed noise becomes part of the claimed scope.
5. Conduct a fresh, exhaustive novelty search immediately before submission.
