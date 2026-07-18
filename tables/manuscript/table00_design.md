| Study                    | Objective family        | Dimensions            | Budget / anchors      |   Replicates | Budget rule               |
|:-------------------------|:------------------------|:----------------------|:----------------------|-------------:|:--------------------------|
| Moment calibration       | Quadratic ridge         | d=12, r=2             | N=80,...,5120         |           20 | 16 calls/anchor           |
| Dimension scaling        | Quadratic ridge         | d=10,20,40; r=2,4     | N=20d                 |           10 | 4mN, m=min(8,d)           |
| Single-task optimization | Embedded BBOB formulas  | d=12, r=2; tau=0,0.02 | 4000 total calls      |           10 | Discovery charged         |
| Amortization             | Five related ellipsoids | d=12, r=2; K=1,3,5    | 1500/task + discovery |           10 | Equal-total full baseline |
