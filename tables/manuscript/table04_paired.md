| Benchmark   |   tau | Method               |   Mean log10 diff |   CI low |   CI high | Win rate   |
|:------------|------:|:---------------------|------------------:|---------:|----------:|:-----------|
| ellipsoid   |  0    | Cross, adaptive rank |             0.304 |    0.172 |     0.446 | 0%         |
| ellipsoid   |  0    | Cross, sparse sketch |             0.326 |    0.193 |     0.469 | 10%        |
| ellipsoid   |  0    | Random subspace      |             0.324 |    0.045 |     0.599 | 20%        |
| ellipsoid   |  0.02 | Random subspace      |             0.469 |    0.162 |     0.764 | 20%        |
| rastrigin   |  0    | Cross, known rank    |             0.405 |    0.063 |     0.839 | 20%        |
| rastrigin   |  0.02 | Cross, adaptive rank |             0.323 |    0.033 |     0.69  | 40%        |
| rastrigin   |  0.02 | Cross, known rank    |             0.213 |    0.032 |     0.467 | 30%        |
| rastrigin   |  0.02 | Cross, sparse sketch |             0.181 |    0.034 |     0.418 | 0%         |
| rastrigin   |  0.02 | Random subspace      |             0.274 |    0.036 |     0.558 | 40%        |
| rosenbrock  |  0    | Cross, known rank    |             0.169 |    0.019 |     0.33  | 30%        |
| rosenbrock  |  0    | Cross, sparse sketch |             0.81  |    0.53  |     1.112 | 0%         |
| rosenbrock  |  0    | Random subspace      |             0.769 |    0.501 |     1.048 | 0%         |
| rosenbrock  |  0.02 | Cross, adaptive rank |             0.615 |    0.184 |     1.121 | 10%        |
| rosenbrock  |  0.02 | Cross, known rank    |             0.087 |    0.049 |     0.128 | 0%         |
| rosenbrock  |  0.02 | Cross, sparse sketch |             0.502 |    0.274 |     0.719 | 10%        |
| rosenbrock  |  0.02 | Random subspace      |             0.76  |    0.305 |     1.234 | 20%        |
| sphere      |  0    | Random subspace      |             0.339 |    0.067 |     0.619 | 20%        |
| sphere      |  0.02 | Random subspace      |             0.44  |    0.2   |     0.697 | 20%        |
