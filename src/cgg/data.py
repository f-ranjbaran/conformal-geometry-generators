"""Synthetic population of 2D closed contours with a known conditional law.

A shape is 64 points stored as a vector of length 128: all x coordinates, then
all y coordinates. Angles are ``theta_k = 2 * pi * k / 64`` for ``k = 0..63``
(the first point is not repeated).

Independent draws for each shape:

    R   ~ Normal(mean=1,   std=0.05)
    a   ~ Uniform[0.05, 0.2)
    phi ~ Normal(mean=0,   std=0.3)
    c   ~ Uniform[0, 1)           if the caller does not fix c
    e   ~ Normal(mean=1 + c, std=0.1)

    r(theta) = R * (1 + a * cos(3 * theta + phi))
    x = e * r * cos(theta) + eps_x
    y =     r * sin(theta) + eps_y

Every coordinate of ``eps`` is Normal(mean=0, std=0.01), i.e. variance
``0.01**2``. The second parameter of each normal is a standard deviation.
``R`` is not clipped; with std 0.05 it is positive with overwhelming
probability. ``c`` controls the conditional law only through ``e``.

The dataset size ``n_shapes`` is a design choice (not fixed by the generative
specification) so that a 70/15/15 split is exact at the default of 4000.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CONDITIONS_TO_SHOW: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class PopulationConfig:
    """Parameters of the true conditional distribution p(shape | c)."""

    n_points: int = 64
    radius_mean: float = 1.0
    radius_std: float = 0.05
    amplitude_low: float = 0.05
    amplitude_high: float = 0.2
    phase_std: float = 0.3
    elongation_std: float = 0.1
    noise_std: float = 0.01

    def __post_init__(self) -> None:
        if self.n_points < 3:
            raise ValueError("n_points must be at least 3")
        if self.amplitude_low > self.amplitude_high:
            raise ValueError("amplitude_low must be <= amplitude_high")
        for name in ("radius_std", "phase_std", "elongation_std", "noise_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class DataConfig:
    """Dataset size, split, seed, and the population those rows are drawn from."""

    seed: int = 0
    n_shapes: int = 4000
    train_fraction: float = 0.70
    calibration_fraction: float = 0.15
    test_fraction: float = 0.15
    population: PopulationConfig = field(default_factory=PopulationConfig)

    def __post_init__(self) -> None:
        if self.n_shapes < 3:
            raise ValueError("n_shapes must be at least 3")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def fractions(self) -> tuple[float, float, float]:
        """Train, calibration, and test fractions in that order."""
        return (self.train_fraction, self.calibration_fraction, self.test_fraction)


@dataclass(frozen=True)
class Split:
    """One dataset partitioned into train, calibration, and test."""

    train_shapes: FloatArray
    train_c: FloatArray
    calibration_shapes: FloatArray
    calibration_c: FloatArray
    test_shapes: FloatArray
    test_c: FloatArray

    @property
    def counts(self) -> dict[str, int]:
        """Number of rows in each split."""
        return {
            "train": int(self.train_shapes.shape[0]),
            "calibration": int(self.calibration_shapes.shape[0]),
            "test": int(self.test_shapes.shape[0]),
        }


def theta_grid(n_points: int = 64) -> FloatArray:
    """Contour angles on ``[0, 2*pi)``, without repeating the first point."""
    return np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)


def sample_shapes(
    n: int,
    c: float | FloatArray | None = None,
    *,
    rng: np.random.Generator,
    config: PopulationConfig | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Draw ``n`` shapes and the conditioning value used for each.

    If ``c`` is ``None``, each shape gets an independent ``c ~ Uniform[0, 1)``.
    A scalar ``c`` is shared by every shape. An array ``c`` must have length ``n``.

    Random draws, in order, from ``rng``: ``c`` (only when it is drawn), then
    ``R``, ``a``, ``phi``, ``e``, then x-noise, then y-noise. Passing ``c`` does
    not consume a draw for the condition.
    """
    cfg = _population(config)
    if n < 0:
        raise ValueError("n must be non-negative")
    conditions = _conditions(n, c, rng)
    if n == 0:
        return np.zeros((0, 2 * cfg.n_points), dtype=np.float64), conditions

    radius, amplitude, phase, elongation = _sample_latents(conditions, rng, cfg)
    shapes = _contours(radius, amplitude, phase, elongation, cfg, rng)
    return shapes, conditions


def sample_true_conditional(
    c: float,
    n: int,
    *,
    rng: np.random.Generator,
    config: PopulationConfig | None = None,
) -> FloatArray:
    """Draw ``n`` fresh shapes from the true conditional p(shape | c).

    ``c`` is held fixed, so these rows are a reference sample from the
    data-generating process rather than a reuse of a stored dataset.
    """
    shapes, _conditions = sample_shapes(n, c=c, rng=rng, config=config)
    return shapes


def shape_elongation(shapes: FloatArray, n_points: int = 64) -> FloatArray:
    """Horizontal-over-vertical RMS radius of each shape.

    For a noise-free draw, ``x = e * r * cos(theta)`` and ``y = r * sin(theta)``,
    so this ratio estimates the latent elongation ``e``.
    """
    expected = 2 * n_points
    if shapes.ndim != 2 or shapes.shape[1] != expected:
        raise ValueError(f"expected shape (n, {expected}), got {shapes.shape}")
    x = shapes[:, :n_points]
    y = shapes[:, n_points:]
    x_rms = np.sqrt(np.mean(x * x, axis=1))
    y_rms = np.sqrt(np.mean(y * y, axis=1))
    return x_rms / np.maximum(y_rms, 1e-12)


def split_indices(
    n: int,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> tuple[IntArray, IntArray, IntArray]:
    """Permute ``n`` rows into train, calibration, and test index arrays.

    Train and calibration lengths are ``floor(n * fraction)``. A tolerance of
    ``1e-8`` is added before the floor so that binary rounding of a decimal
    fraction cannot drop a row. The test split receives whatever remains.
    """
    n_train, n_cal, _n_test = _split_counts(n, fractions)
    permutation = np.random.default_rng(seed).permutation(n)
    train = permutation[:n_train]
    calibration = permutation[n_train : n_train + n_cal]
    test = permutation[n_train + n_cal :]
    return train, calibration, test


def split_dataset(
    shapes: FloatArray,
    conditions: FloatArray,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> Split:
    """Split rows into train, calibration, and test with a fixed permutation."""
    if shapes.ndim != 2:
        raise ValueError("shapes must have shape (n, dim)")
    if conditions.shape != (shapes.shape[0],):
        raise ValueError("conditions must have shape (n,)")
    train, calibration, test = split_indices(shapes.shape[0], fractions, seed)
    return Split(
        train_shapes=shapes[train],
        train_c=conditions[train],
        calibration_shapes=shapes[calibration],
        calibration_c=conditions[calibration],
        test_shapes=shapes[test],
        test_c=conditions[test],
    )


def generate_dataset(
    n: int,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    population: PopulationConfig | None = None,
) -> Split:
    """Sample ``n`` shapes with ``c ~ Uniform[0, 1)`` and split them.

    Sampling and the split each construct ``default_rng(seed)``. They do not
    share a stream, so a change inside the sampler does not reshuffle the split.
    """
    shapes, conditions = sample_shapes(
        n,
        c=None,
        rng=np.random.default_rng(seed),
        config=population,
    )
    return split_dataset(shapes, conditions, fractions=fractions, seed=seed)


def mean_elongation_by_c(
    conditions: tuple[float, ...] = CONDITIONS_TO_SHOW,
    n: int = 2000,
    seed: int = 1,
    config: PopulationConfig | None = None,
) -> list[dict[str, float | int]]:
    """Mean observed elongation of fresh conditional samples at each ``c``."""
    cfg = _population(config)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for condition in conditions:
        shapes = sample_true_conditional(float(condition), n, rng=rng, config=cfg)
        rows.append(
            {
                "c": float(condition),
                "n": int(n),
                "mean_elongation": float(np.mean(shape_elongation(shapes, cfg.n_points))),
            }
        )
    return rows


def plot_conditional_samples(
    path: Path,
    conditions: tuple[float, ...] = CONDITIONS_TO_SHOW,
    n_shown: int = 15,
    n_mean: int = 500,
    seed: int = 2,
    config: PopulationConfig | None = None,
) -> Path:
    """Plot fresh draws from p(shape | c). Shared limits keep elongation visible.

    Faint curves are ``n_shown`` draws. The dark curve is the coordinate-wise
    mean of a separate sample of ``n_mean`` draws. For each ``c``, the shown
    draws are taken before the mean sample, from one generator seeded by ``seed``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = _population(config)
    rng = np.random.default_rng(seed)
    shown: list[FloatArray] = []
    means: list[FloatArray] = []
    for condition in conditions:
        shown.append(sample_true_conditional(float(condition), n_shown, rng=rng, config=cfg))
        mean_sample = sample_true_conditional(float(condition), n_mean, rng=rng, config=cfg)
        means.append(np.mean(mean_sample, axis=0))

    flat = np.concatenate([block.ravel() for block in shown + means])
    limit = float(np.max(np.abs(flat))) * 1.05
    if limit == 0.0:
        limit = 1.0

    fig, axes = plt.subplots(1, len(conditions), figsize=(9.2, 3.4), constrained_layout=True)
    if len(conditions) == 1:
        axes = [axes]
    sample_color = "#2F6FAD"
    for axis, condition, samples, mean in zip(axes, conditions, shown, means, strict=True):
        for index, shape in enumerate(samples):
            x, y = _closed_polyline(shape, cfg.n_points)
            axis.plot(
                x,
                y,
                color=sample_color,
                alpha=0.45,
                linewidth=0.9,
                label="draws" if index == 0 else None,
            )
        mean_x, mean_y = _closed_polyline(mean, cfg.n_points)
        axis.plot(mean_x, mean_y, color="#111111", linewidth=1.6, label=f"mean of {n_mean}")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"c = {condition:.1f}")
        axis.set_xlabel("x")
        axis.axhline(0.0, color="0.75", linewidth=0.5)
        axis.axvline(0.0, color="0.75", linewidth=0.5)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("True conditional population  p(shape | c)", y=1.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def load_data_config(path: Path) -> DataConfig:
    """Read ``configs/default.yaml`` (or another file with the same keys)."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    known = {
        "seed",
        "n_shapes",
        "train_fraction",
        "calibration_fraction",
        "test_fraction",
        "population",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    population_raw = raw.get("population", {})
    if not isinstance(population_raw, dict):
        raise ValueError("population must be a mapping")
    known_population = {item.name for item in fields(PopulationConfig)}
    unknown_population = set(population_raw) - known_population
    if unknown_population:
        raise ValueError(f"unknown population keys: {sorted(unknown_population)}")
    return DataConfig(
        seed=int(raw.get("seed", 0)),
        n_shapes=int(raw.get("n_shapes", 4000)),
        train_fraction=float(raw.get("train_fraction", 0.70)),
        calibration_fraction=float(raw.get("calibration_fraction", 0.15)),
        test_fraction=float(raw.get("test_fraction", 0.15)),
        population=PopulationConfig(**population_raw),
    )


def write_data_summary(config: DataConfig, out_dir: Path) -> Path:
    """Write split counts and conditional elongation, both computed here."""
    split = generate_dataset(
        config.n_shapes,
        seed=config.seed,
        fractions=config.fractions,
        population=config.population,
    )
    elongation_seed = config.seed + 1
    rows = mean_elongation_by_c(n=2000, seed=elongation_seed, config=config.population)
    means = [row["mean_elongation"] for row in rows]
    summary = {
        "seed": config.seed,
        "n_shapes": config.n_shapes,
        "shape_dim": 2 * config.population.n_points,
        "coordinate_layout": "x[0:n_points] then y[n_points:2*n_points]",
        "normal_second_parameter": "standard deviation",
        "population": asdict(config.population),
        "split_fractions": {
            "train": config.train_fraction,
            "calibration": config.calibration_fraction,
            "test": config.test_fraction,
        },
        "split_rule": (
            "train and calibration counts are floor(n * fraction + 1e-8); "
            "the test split receives the remainder"
        ),
        "split_counts": split.counts,
        "elongation_proxy": "rms(x) / rms(y)",
        "elongation_check": {
            "seed": elongation_seed,
            "n_per_c": 2000,
            "by_c": rows,
            "increases_with_c": bool(means[0] < means[1] < means[2]),
        },
        "figure": {
            "path": "figures/conditional_samples.png",
            "seed": config.seed + 2,
            "n_shown": 15,
            "n_mean": 500,
            "conditions": list(CONDITIONS_TO_SHOW),
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """Generate the summary JSON and the conditional-sample figure."""
    parser = argparse.ArgumentParser(description="Sample the synthetic contour population.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    config = load_data_config(args.config)
    figure_path = plot_conditional_samples(
        args.out / "figures" / "conditional_samples.png",
        seed=config.seed + 2,
        config=config.population,
    )
    summary_path = write_data_summary(config, args.out)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["elongation_check"]["by_c"]
    rendered = ", ".join(
        f"c={row['c']}: {row['mean_elongation']:.6f}" for row in rows
    )
    print(f"wrote {summary_path}")
    print(f"wrote {figure_path}")
    print(f"split counts: {summary['split_counts']}")
    print(f"mean elongation (rms x / rms y): {rendered}")
    if not summary["elongation_check"]["increases_with_c"]:
        raise SystemExit("mean elongation did not increase with c; see data_summary.json")


def _population(config: PopulationConfig | None) -> PopulationConfig:
    """Return ``config``, or the default population if it is ``None``."""
    return config if config is not None else PopulationConfig()


def _conditions(n: int, c: float | FloatArray | None, rng: np.random.Generator) -> FloatArray:
    """Broadcast a supplied condition, or draw one per shape."""
    if c is None:
        return rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    values = np.asarray(c, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(n, float(values), dtype=np.float64)
    elif values.shape != (n,):
        raise ValueError(f"c must be a scalar or shape ({n},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("c must be finite")
    return values


def _sample_latents(
    conditions: FloatArray,
    rng: np.random.Generator,
    config: PopulationConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Draw ``R``, ``a``, ``phi``, and ``e`` for each supplied condition."""
    n = conditions.shape[0]
    radius = rng.normal(config.radius_mean, config.radius_std, size=n)
    amplitude = rng.uniform(config.amplitude_low, config.amplitude_high, size=n)
    phase = rng.normal(0.0, config.phase_std, size=n)
    elongation = rng.normal(1.0 + conditions, config.elongation_std, size=n)
    return radius, amplitude, phase, elongation


def _contours(
    radius: FloatArray,
    amplitude: FloatArray,
    phase: FloatArray,
    elongation: FloatArray,
    config: PopulationConfig,
    rng: np.random.Generator,
) -> FloatArray:
    """Map latent draws to shape vectors of length ``2 * n_points``."""
    theta = theta_grid(config.n_points)
    angle = 3.0 * theta[None, :] + phase[:, None]
    radial = radius[:, None] * (1.0 + amplitude[:, None] * np.cos(angle))
    x = elongation[:, None] * radial * np.cos(theta)[None, :]
    y = radial * np.sin(theta)[None, :]
    x = x + rng.normal(0.0, config.noise_std, size=x.shape)
    y = y + rng.normal(0.0, config.noise_std, size=y.shape)
    return np.concatenate([x, y], axis=1)


def _split_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert fractions to counts. Test receives the leftover rows."""
    train_f, cal_f, test_f = fractions
    if n < 3:
        raise ValueError("need at least 3 rows to form three splits")
    if min(train_f, cal_f, test_f) <= 0.0:
        raise ValueError("split fractions must be positive")
    if abs((train_f + cal_f + test_f) - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to 1")
    n_train = int(np.floor(n * train_f + 1e-8))
    n_cal = int(np.floor(n * cal_f + 1e-8))
    n_test = n - n_train - n_cal
    if min(n_train, n_cal, n_test) < 1:
        raise ValueError(f"fractions {fractions} leave an empty split for n={n}")
    return n_train, n_cal, n_test


def _closed_polyline(shape: FloatArray, n_points: int) -> tuple[FloatArray, FloatArray]:
    """Return x and y with the first point appended, so the contour closes."""
    x = shape[:n_points]
    y = shape[n_points:]
    return np.concatenate([x, x[:1]]), np.concatenate([y, y[:1]])


if __name__ == "__main__":
    main()
