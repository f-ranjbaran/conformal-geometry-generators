"""Figures for training, reconstructions, prior samples, and uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def plot_training_curves(path: Path, models: list[dict]) -> Path:
    """Plot every seed's train loss (dashed) and validation loss (solid)."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for index, model in enumerate(models):
        color = f"C{index}"
        epochs = [row["epoch"] for row in model["history"]]
        axis.plot(
            epochs,
            [row["train_loss"] for row in model["history"]],
            color=color,
            linestyle="--",
            linewidth=1.0,
            label=f"train {model['seed']}",
        )
        axis.plot(
            epochs,
            [row["val_loss"] for row in model["history"]],
            color=color,
            linestyle="-",
            linewidth=1.4,
            label=f"val {model['seed']}",
        )
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_yscale("log")
    axis.legend(ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Conditional VAE learning curves", y=1.02)
    return _save(fig, path)


def plot_reconstructions(
    path: Path,
    truths: FloatArray,
    recons: FloatArray,
    conditions: FloatArray,
    n_points: int = 64,
) -> Path:
    """Overlay validation contours and their posterior-mean reconstructions."""
    plt = _pyplot()
    limit = _shared_limit(np.concatenate([truths, recons], axis=0))
    fig, axes = plt.subplots(1, truths.shape[0], figsize=(2.15 * truths.shape[0], 3.2), constrained_layout=True)
    if truths.shape[0] == 1:
        axes = [axes]
    for index, (axis, truth, recon, condition) in enumerate(
        zip(axes, truths, recons, conditions, strict=True)
    ):
        _draw(axis, truth, n_points, color="#222222", width=1.4, label="validation" if index == 0 else None)
        _draw(
            axis,
            recon,
            n_points,
            color="#C65A2E",
            width=1.3,
            style="--",
            label="reconstruction" if index == 0 else None,
        )
        _panel(axis, limit, f"c = {condition:.2f}")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Posterior-mean reconstructions on the validation split", y=1.02)
    return _save(fig, path)


def plot_generated_samples(
    path: Path,
    shown_by_c: list[FloatArray],
    means_by_c: list[FloatArray],
    conditions: tuple[float, ...],
    n_mean: int,
    n_points: int = 64,
) -> Path:
    """Overlay prior samples from one trained model at several conditions."""
    plt = _pyplot()
    mean_rows = [np.reshape(mean, (1, -1)) for mean in means_by_c]
    stacked = np.concatenate(list(shown_by_c) + mean_rows, axis=0)
    limit = _shared_limit(stacked)
    fig, axes = plt.subplots(1, len(conditions), figsize=(9.2, 3.4), constrained_layout=True)
    if len(conditions) == 1:
        axes = [axes]
    for index, (axis, samples, mean, condition) in enumerate(
        zip(axes, shown_by_c, means_by_c, conditions, strict=True)
    ):
        for draw_index, shape in enumerate(samples):
            _draw(
                axis,
                shape,
                n_points,
                color="#2F6FAD",
                width=0.9,
                alpha=0.45,
                label="prior samples" if draw_index == 0 else None,
            )
        _draw(axis, np.ravel(mean), n_points, color="#111111", width=1.6, label=f"mean of {n_mean}")
        _panel(axis, limit, f"c = {condition:.1f}")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Prior samples from the first ensemble member", y=1.02)
    return _save(fig, path)


def plot_diagnostics(
    path: Path,
    latent_mean: FloatArray,
    latent_std: FloatArray,
    shown: list[list[FloatArray]],
    means: list[list[FloatArray]],
    conditions: tuple[float, ...],
    method_names: tuple[str, str],
    n_mean: int,
    n_points: int = 64,
) -> Path:
    """Plot validation latent moments and the two conditional populations."""
    plt = _pyplot()
    stacked = [block for method in shown for block in method]
    stacked.extend(np.reshape(mean, (1, -1)) for method in means for mean in method)
    limit = _shared_limit(np.concatenate(stacked, axis=0))
    fig = plt.figure(figsize=(9.6, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1:])]
    dims = np.arange(latent_mean.shape[0])
    axes[0].axhline(0.0, color="0.45", linewidth=0.8, label="N(0, I)")
    axes[0].plot(dims, latent_mean, marker="o", color="#2F6FAD", linestyle="None")
    axes[0].set_title("Aggregate-posterior mean")
    axes[0].set_ylabel("mean of z")
    axes[1].axhline(1.0, color="0.45", linewidth=0.8, label="N(0, I)")
    axes[1].plot(dims, latent_std, marker="o", color="#C65A2E", linestyle="None")
    axes[1].set_title("Aggregate-posterior std")
    axes[1].set_ylabel("std of z")
    for axis in axes:
        axis.set_xlabel("latent dimension")
        axis.set_xticks(dims)
        axis.legend(frameon=False, fontsize=8)
    colors = ("#2F6FAD", "#C65A2E")
    for row, (name, color) in enumerate(zip(method_names, colors, strict=True)):
        for col, condition in enumerate(conditions):
            axis = fig.add_subplot(grid[row + 1, col])
            for draw_index, shape in enumerate(shown[row][col]):
                _draw(
                    axis,
                    shape,
                    n_points,
                    color=color,
                    width=0.8,
                    alpha=0.45,
                    label="samples" if draw_index == 0 else None,
                )
            _draw(axis, np.ravel(means[row][col]), n_points, color="#111111", width=1.5, label=f"mean of {n_mean}")
            _panel(axis, limit, f"c = {condition:.1f}")
            if col == 0:
                axis.set_ylabel(name)
                axis.legend(loc="upper right", frameon=False, fontsize=7)
    fig.suptitle("Validation latents and the two ways of sampling z", y=1.02)
    return _save(fig, path)


def plot_uncertainty(path: Path, rows: list[dict]) -> Path:
    """Plot aleatoric spread, epistemic disagreement, and error against c."""
    plt = _pyplot()
    c = [row["c"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    axes[0].plot(c, [row["aleatoric_trace_mean"] for row in rows], marker="o", color="#2F6FAD", label="model")
    axes[0].plot(c, [row["true_trace"] for row in rows], marker="o", color="#222222", linestyle="--", label="truth")
    axes[0].set_title("Aleatoric-like spread")
    axes[0].set_ylabel("trace of covariance")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(c, [row["epistemic_mean_trace"] for row in rows], marker="o", label="mean shape")
    axes[1].plot(c, [row["epistemic_covariance_rms"] for row in rows], marker="s", label="covariance")
    axes[1].set_title("Epistemic-like disagreement")
    axes[1].set_ylabel("trace or RMS Frobenius")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(c, [row["frechet_distance"] for row in rows], marker="o", label="Frechet")
    axes[2].plot(
        c,
        [row["frechet_true_vs_true"] for row in rows],
        marker="o",
        linestyle="--",
        color="#222222",
        label="true vs true",
    )
    axes[2].plot(c, [row["mean_l2"] for row in rows], marker="s", label="mean L2")
    axes[2].plot(c, [row["covariance_frobenius"] for row in rows], marker="^", label="cov Frobenius")
    axes[2].set_title("Accuracy against the truth")
    axes[2].set_ylabel("distance")
    axes[2].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.set_xlabel("c")
    return _save(fig, path)


def plot_calibration(path: Path, methods: list[dict]) -> Path:
    """Nominal versus empirical test coverage, with the diagonal."""
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0], color="0.45", linewidth=1.0, label="nominal")
    for method in methods:
        nominal = [row["nominal"] for row in method["by_alpha"]]
        empirical = [row["test_coverage"] for row in method["by_alpha"]]
        axis.plot(
            nominal,
            empirical,
            marker="o",
            linewidth=1.4,
            color=method["color"],
            label=method["label"],
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("nominal coverage")
    axis.set_ylabel("empirical coverage")
    axis.set_title("Test coverage")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, path)


def plot_conditional_coverage(path: Path, methods: list[dict], n_bins: int) -> Path:
    """Empirical coverage in equal-width bins of c, one panel per alpha."""
    plt = _pyplot()
    alphas = [row["alpha"] for row in methods[0]["by_alpha"]]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), constrained_layout=True, sharex=True, sharey=True)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    for axis, alpha_index, alpha in zip(axes.ravel(), range(len(alphas)), alphas, strict=True):
        nominal = methods[0]["by_alpha"][alpha_index]["nominal"]
        axis.axhline(nominal, color="0.45", linewidth=1.0, label="nominal" if alpha_index == 0 else None)
        for method in methods:
            coverage = [bin_row["coverage"] for bin_row in method["by_alpha"][alpha_index]["by_bin"]]
            axis.plot(
                centers,
                coverage,
                marker="o",
                linewidth=1.3,
                color=method["color"],
                label=method["label"] if alpha_index == 0 else None,
            )
        axis.set_title(f"1 − α = {nominal:.2f}")
        axis.set_ylim(0.0, 1.02)
    for axis in axes[1, :]:
        axis.set_xlabel("c bin centre")
    for axis in axes[:, 0]:
        axis.set_ylabel("empirical coverage")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    return _save(fig, path)


def _draw(
    axis,
    shape: FloatArray,
    n_points: int,
    color: str,
    width: float,
    label: str | None = None,
    style: str = "-",
    alpha: float = 1.0,
) -> None:
    """Plot one closed contour."""
    x = np.concatenate([shape[:n_points], shape[:1]])
    y = np.concatenate([shape[n_points:], shape[n_points : n_points + 1]])
    axis.plot(x, y, color=color, linewidth=width, linestyle=style, alpha=alpha, label=label)


def _panel(axis, limit: float, title: str) -> None:
    """Apply the shared limits used so elongation is visible."""
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.axhline(0.0, color="0.75", linewidth=0.5)
    axis.axvline(0.0, color="0.75", linewidth=0.5)


def _shared_limit(shapes: FloatArray) -> float:
    """Square limit, slightly outside the largest absolute coordinate."""
    limit = float(np.max(np.abs(shapes))) * 1.05
    return limit if limit > 0.0 else 1.0


def _pyplot():
    """Import pyplot with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    """Write a figure, including a suptitle placed just above the axes."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
