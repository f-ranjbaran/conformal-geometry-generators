"""Exact checks for the Gaussian population metrics."""

import numpy as np

from cgg.uncertainty import covariance_frobenius, frechet_distance, population_metrics


def test_frechet_reduces_to_the_mean_shift_when_covariances_match() -> None:
    cov = np.eye(2)
    origin = np.zeros(2)
    shifted = np.array([3.0, 4.0])
    assert frechet_distance(origin, cov, origin, cov, ridge=0.0) == 0.0
    distance = frechet_distance(origin, cov, shifted, cov, ridge=1e-4)
    assert abs(distance - 5.0) < 1e-6
    assert abs(covariance_frobenius(np.eye(2), 2.0 * np.eye(2)) - np.sqrt(2.0)) < 1e-8


def test_two_model_disagreement() -> None:
    quiet = np.zeros((4, 2))
    shifted = np.tile(np.array([2.0, 0.0]), (4, 1))
    metrics = population_metrics([quiet, shifted], true_samples=quiet, ridge=0.0)
    assert metrics["aleatoric_trace_mean"] == 0.0
    assert abs(float(metrics["epistemic_mean_trace"]) - 2.0) < 1e-8
    assert metrics["epistemic_covariance_rms"] == 0.0
    assert abs(float(metrics["mean_l2"]) - 1.0) < 1e-8
    assert abs(float(metrics["covariance_frobenius"]) - (8.0 / 7.0)) < 1e-8
