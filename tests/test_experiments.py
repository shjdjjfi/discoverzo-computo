from discoverzo.experiments import ExperimentConfig, run_bias_experiment, run_scaling_experiment


def test_small_experiments_return_expected_columns():
    cfg = ExperimentConfig(bias_repetitions=1, scaling_repetitions=1, optimization_repetitions=1)
    bias = run_bias_experiment(cfg)
    scaling = run_scaling_experiment(cfg)
    assert {"method", "operator_error", "subspace_error"}.issubset(bias.columns)
    assert {"d", "r", "rank_hat"}.issubset(scaling.columns)
    assert len(bias) == 21
    assert len(scaling) == 12


def test_small_optimization_and_amortization_execute():
    from discoverzo.experiments import run_amortization_experiment, run_optimization_experiment

    cfg = ExperimentConfig(
        optimization_repetitions=1,
        optimization_budget=500,
        discovery_anchors=3,
        discovery_directions_per_side=4,
        noise_std=0.0,
    )
    opt = run_optimization_experiment(cfg)
    amort = run_amortization_experiment(cfg)
    assert len(opt) == 48  # 4 functions x 2 tau values x 6 methods
    assert len(amort) == 12  # 3 task counts x 4 methods
    assert opt["queries"].max() <= cfg.optimization_budget
    assert (amort["total_queries"] > 0).all()
