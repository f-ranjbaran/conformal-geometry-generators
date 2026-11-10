"""Checks for the synthetic contour population."""

import numpy as np

from cgg.data import (
    PopulationConfig,
    generate_dataset,
    sample_shapes,
    sample_true_conditional,
    shape_elongation,
    split_dataset,
    split_indices,
)


def test_shape_layout_and_fixed_condition() -> None:
    shapes, conditions = sample_shapes(8, c=0.3, rng=np.random.default_rng(0))
    assert shapes.shape == (8, 128)
    assert conditions.shape == (8,)
    assert np.all(conditions == 0.3)
    assert np.isfinite(shapes).all()

    drawn, drawn_c = sample_shapes(20, rng=np.random.default_rng(1))
    assert drawn.shape == (20, 128)
    assert np.all((drawn_c >= 0.0) & (drawn_c < 1.0))


def test_matches_specified_formula_when_latents_are_fixed() -> None:
    config = PopulationConfig(
        radius_std=0.0,
        amplitude_low=0.1,
        amplitude_high=0.1,
        phase_std=0.0,
        elongation_std=0.0,
        noise_std=0.0,
    )
    c = 0.25
    shapes, conditions = sample_shapes(3, c=c, rng=np.random.default_rng(0), config=config)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radial = 1.0 * (1.0 + 0.1 * np.cos(3.0 * theta))
    expected = np.concatenate([(1.0 + c) * radial * np.cos(theta), radial * np.sin(theta)])
    assert np.all(conditions == c)
    np.testing.assert_allclose(shapes, np.broadcast_to(expected, shapes.shape))


def test_sampling_and_split_are_deterministic() -> None:
    first, first_c = sample_shapes(16, rng=np.random.default_rng(123))
    second, second_c = sample_shapes(16, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_c, second_c)

    conditional = sample_true_conditional(0.4, 10, rng=np.random.default_rng(7))
    again = sample_true_conditional(0.4, 10, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(conditional, again)

    shapes, conditions = sample_shapes(100, rng=np.random.default_rng(5))
    split = split_dataset(shapes, conditions, seed=0)
    repeat = generate_dataset(100, seed=5)
    train_idx, cal_idx, test_idx = split_indices(100, seed=0)
    np.testing.assert_array_equal(split.train_shapes, shapes[train_idx])
    np.testing.assert_array_equal(split.calibration_c, conditions[cal_idx])
    np.testing.assert_array_equal(split.test_shapes, shapes[test_idx])
    np.testing.assert_array_equal(split_indices(100, seed=0)[0], train_idx)
    np.testing.assert_array_equal(repeat.train_shapes, generate_dataset(100, seed=5).train_shapes)
    np.testing.assert_array_equal(repeat.test_c, generate_dataset(100, seed=5).test_c)


def test_split_is_a_partition_and_remainder_goes_to_test() -> None:
    train, calibration, test = split_indices(10, seed=0)
    assert (train.shape, calibration.shape, test.shape) == ((7,), (1,), (2,))
    joined = np.concatenate([train, calibration, test])
    assert sorted(joined.tolist()) == list(range(10))

    full = split_indices(4000, seed=0)
    assert tuple(part.shape[0] for part in full) == (2800, 600, 600)
    assert sorted(np.concatenate(full).tolist()) == list(range(4000))


def test_mean_elongation_increases_with_c() -> None:
    rng = np.random.default_rng(0)
    low = shape_elongation(sample_true_conditional(0.1, 400, rng=rng))
    mid = shape_elongation(sample_true_conditional(0.5, 400, rng=rng))
    high = shape_elongation(sample_true_conditional(0.9, 400, rng=rng))
    assert low.mean() < mid.mean() < high.mean()
    # E[e | c] rises by 0.8 from c=0.1 to c=0.9. The RMS proxy tracks e.
    assert high.mean() - low.mean() > 0.5
