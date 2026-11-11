"""Quantile definition and coverage on exchangeable scores. No trained model."""

import numpy as np

from cgg.conformal import chi2_threshold, conformal_quantile, squared_mahalanobis


def test_squared_mahalanobis_of_a_known_shift() -> None:
    scores = squared_mahalanobis(
        np.array([[3.0, 4.0]]),
        np.zeros(2),
        np.eye(2),
        noise_var=0.0,
    )
    assert abs(float(scores[0]) - 25.0) < 1e-8


def test_conformal_quantile_is_the_stated_order_statistic() -> None:
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    # (4 + 1) * 0.5 = 2.5, ceil = 3, so the third smallest score.
    assert conformal_quantile(scores, 0.5) == 2.0
    assert conformal_quantile(np.array([1.0]), 0.1) == float("inf")
    assert chi2_threshold(128, 0.05) > chi2_threshold(128, 0.5)


def test_exchangeable_conformal_coverage_meets_the_nominal() -> None:
    rng = np.random.default_rng(0)
    n_cal = 100
    n_trials = 400
    tolerance = 0.05
    for alpha in (0.1, 0.5):
        covered = 0
        for _ in range(n_trials):
            scores = rng.random(n_cal + 1)
            threshold = conformal_quantile(scores[:-1], alpha)
            covered += int(scores[-1] <= threshold)
        rate = covered / n_trials
        assert rate >= (1.0 - alpha) - tolerance, (alpha, rate)
