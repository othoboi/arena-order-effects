import numpy as np
from arena_order.bt import fit_side_effect, voter_heterogeneity, cluster_bootstrap_gamma
from arena_order.simulate import simulate_votes


def test_recovers_known_side_effect():
    left, right, y, _ = simulate_votes(n_votes=40000, gamma=0.20, seed=1)
    fit = fit_side_effect(left, right, y)
    assert fit["ci_low"] < 0.20 < fit["ci_high"]
    assert fit["p_value"] < 0.001


def test_no_side_effect_is_not_flagged_too_often():
    rejections = 0
    for seed in range(40):
        left, right, y, _ = simulate_votes(n_votes=4000, gamma=0.0, seed=seed)
        if fit_side_effect(left, right, y)["p_value"] < 0.05:
            rejections += 1
    assert rejections <= 6


def test_strengths_are_recovered_in_order():
    left, right, y, _ = simulate_votes(n_votes=60000, n_models=6, gamma=0.1, seed=3)
    fit = fit_side_effect(left, right, y)
    rng = np.random.default_rng(3)
    true = rng.normal(0.0, 0.7, size=6)
    est = np.array([fit["strengths"][f"model_{i:02d}"] for i in range(6)])
    assert np.corrcoef(true, est)[0, 1] > 0.98


def test_voter_heterogeneity_detects_side_clickers():
    left, right, y, voter = simulate_votes(n_votes=30000, n_voters=300, seed=4)
    y = np.array(y)
    y[np.isin(voter, np.arange(15))] = 1
    out = voter_heterogeneity(y, voter, min_votes=20)
    assert out["p_value"] < 0.001
    assert out["voters_above_70pct_left"] >= 15


def test_bootstrap_interval_covers_truth():
    left, right, y, voter = simulate_votes(n_votes=6000, gamma=0.15, n_voters=200, seed=5)
    low, high = cluster_bootstrap_gamma(left, right, y, voter, n_boot=60, seed=5)
    assert low < 0.15 < high


def test_covariate_effect_is_recovered_alongside_side_effect():
    left, right, y, _ = simulate_votes(n_votes=40000, gamma=0.0, seed=6)
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 1.0, size=len(y))
    y = (rng.random(len(y)) < 1.0 / (1.0 + np.exp(-(0.10 + 0.50 * x)))).astype(int)
    fit = fit_side_effect(left, right, y, covariates={"x": x})
    assert abs(fit["gamma"] - 0.10) < 0.03
    assert abs(fit["covariates"]["x"]["coef"] - 0.50) < 0.04
