"""Aggregate-posterior moments and the W2 tie break. No training."""

import numpy as np

from cgg.diagnostics import aggregate_posterior_moments, choose_lowest_score
from cgg.train import load_run_config


def test_mixture_of_standard_normals_has_mean_zero_and_std_one() -> None:
    mu = np.zeros((6, 3))
    variance = np.ones((6, 3))
    mean, std = aggregate_posterior_moments(mu, variance)
    assert np.allclose(mean, 0.0)
    assert np.allclose(std, 1.0)


def test_separated_point_masses_put_their_spread_in_the_mixture_std() -> None:
    mu = np.array([[-1.0], [1.0]])
    variance = np.zeros((2, 1))
    mean, std = aggregate_posterior_moments(mu, variance)
    assert mean[0] == 0.0
    assert abs(std[0] - 1.0) < 1e-12


def test_lowest_score_breaks_ties_toward_the_earlier_index() -> None:
    assert choose_lowest_score([2.0, 1.0, 1.0]) == 1
    assert choose_lowest_score([0.4]) == 0


def test_run_config_reads_the_w2_sweep_lists(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  latent_dim: 8",
                "  beta: 0.003",
                "train:",
                "  beta_candidates_v2: [0.5, 0.001]",
                "  latent_dim_candidates: [4, 8]",
                "  sweep_conditions: [0.1, 0.5, 0.9]",
                "ensemble:",
                "  seeds: [0, 1]",
                "uncertainty: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_run_config(path)
    assert config.beta_candidates_v2 == (0.5, 0.001)
    assert config.latent_dim_candidates == (4, 8)
    assert config.sweep_conditions == (0.1, 0.5, 0.9)
