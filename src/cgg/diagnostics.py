"""Latent diagnostics for one trained conditional VAE.

The aggregate posterior is the equal mixture of q(z | x, c) on the validation
cut of the train split. Calibration and test rows are not read. Encoding those
validation shapes again with the condition replaced by a target c is not used:
that asks the encoder to explain a shape under a condition it was not drawn
from, so the decoded cloud is not a sample from a conditional population. The
second population draws z from the diagonal Gaussian with the mixture's
per-dimension mean and variance, then decodes at the target c.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from cgg.data import generate_dataset, sample_true_conditional, shape_elongation
from cgg.model import LOGVAR_MAX, LOGVAR_MIN
from cgg.plots import plot_diagnostics
from cgg.train import load_model, load_run_config, split_train_val
from cgg.uncertainty import fit_gaussian, frechet_distance

FloatArray = NDArray[np.float64]
N_SHOWN = 15
DIAGNOSTIC_CONDITIONS = (0.1, 0.5, 0.9)


def aggregate_posterior_moments(mu: FloatArray, variance: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-dimension mean and std of an equal mixture of diagonal Gaussians.

    ``mu`` and ``variance`` have shape (n, d). On each coordinate the mixture
    variance is the average component variance plus the average squared
    deviation of the component means. Both averages are over the n rows.
    """
    if mu.ndim != 2 or mu.shape != variance.shape or mu.shape[0] < 1:
        raise ValueError(f"mu and variance must share shape (n, d), got {mu.shape} and {variance.shape}")
    if np.any(variance < 0.0):
        raise ValueError("variance must be non-negative")
    mean = np.mean(mu, axis=0)
    mixture_var = np.mean(variance, axis=0) + np.mean((mu - mean) ** 2, axis=0)
    return np.asarray(mean, dtype=np.float64), np.sqrt(np.asarray(mixture_var, dtype=np.float64))


def choose_lowest_score(scores: list[float]) -> int:
    """Index of the smallest score. The earliest index wins a tie."""
    if not scores:
        raise ValueError("scores must not be empty")
    return min(range(len(scores)), key=lambda index: (scores[index], index))


def run_diagnostics(config_path: Path, out_dir: Path, checkpoint_dir: Path) -> None:
    """Score the seed-0 checkpoint and write ``results/diagnostics.json``."""
    config = load_run_config(config_path)
    seed = int(config.ensemble.seeds[0])
    checkpoint = Path(checkpoint_dir) / f"seed_{seed}" / "model.pt"
    metadata_path = checkpoint.parent / "history.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    latent_dim = int(metadata["latent_dim"])
    beta = float(metadata["beta"])
    model = load_model(checkpoint, latent_dim)
    split = generate_dataset(
        config.data.n_shapes,
        seed=config.data.seed,
        fractions=config.data.fractions,
        population=config.data.population,
    )
    _fit_x, _fit_c, val_x, val_c = split_train_val(
        split.train_shapes,
        split.train_c,
        config.train.val_fraction,
        seed=config.data.seed,
    )
    moments = _encode_moments(model, val_x, val_c)
    true_samples = _true_samples(config, DIAGNOSTIC_CONDITIONS, config.uncertainty.sample_seed + 11)
    prior_rng = np.random.default_rng(config.uncertainty.sample_seed)
    fitted_rng = np.random.default_rng(config.uncertainty.sample_seed + 3)
    methods = []
    shown: list[list[FloatArray]] = []
    mean_shapes: list[list[FloatArray]] = []
    for name, description, sampler in (
        (
            "prior",
            "z ~ N(0, I), then decode at c",
            lambda condition, rng: _prior_samples(model, condition, config.ensemble.samples_per_model, rng),
        ),
        (
            "aggregate_posterior_gaussian",
            (
                "z ~ diagonal Gaussian matching the validation aggregate posterior, "
                "then decode at c. Correlations between latent dimensions are not kept."
            ),
            lambda condition, rng: _fitted_samples(
                model,
                condition,
                config.ensemble.samples_per_model,
                moments["mean"],
                moments["std"],
                rng,
            ),
        ),
    ):
        rows = []
        method_shown: list[FloatArray] = []
        method_means: list[FloatArray] = []
        for condition, truth in zip(DIAGNOSTIC_CONDITIONS, true_samples, strict=True):
            generated = sampler(float(condition), prior_rng if name == "prior" else fitted_rng)
            rows.append(_score_population(generated, truth, config.uncertainty.ridge, float(condition)))
            method_shown.append(generated[:N_SHOWN])
            method_means.append(np.mean(generated, axis=0))
        methods.append({"source": name, "description": description, "by_c": rows})
        shown.append(method_shown)
        mean_shapes.append(method_means)
    out_dir = Path(out_dir)
    figure = out_dir / "figures" / "diagnostics.png"
    plot_diagnostics(
        figure,
        moments["mean"],
        moments["std"],
        shown,
        mean_shapes,
        DIAGNOSTIC_CONDITIONS,
        ("z ~ N(0, I)", "z ~ fitted q(z)"),
        config.ensemble.samples_per_model,
    )
    payload = {
        "checkpoint": str(checkpoint).replace("\\", "/"),
        "seed": seed,
        "beta": beta,
        "latent_dim": latent_dim,
        "kl_warmup_epochs": int(metadata["kl_warmup_epochs"]),
        "best_epoch": int(metadata["best_epoch"]),
        "n_val": int(val_x.shape[0]),
        "val_split_seed": config.data.seed,
        "conditions": list(DIAGNOSTIC_CONDITIONS),
        "n_generated": config.ensemble.samples_per_model,
        "n_true": config.uncertainty.n_true,
        "n_shown": N_SHOWN,
        "ridge": config.uncertainty.ridge,
        "prior_sample_seed": config.uncertainty.sample_seed,
        "fitted_sample_seed": config.uncertainty.sample_seed + 3,
        "true_sample_seed": config.uncertainty.sample_seed + 11,
        "draw_order": (
            "For each source, c loops in (0.1, 0.5, 0.9) and one Generator draws "
            "n_generated latents at each c. The plotted curves are the first "
            "n_shown of those draws. The true samples are one Generator, true "
            "population then the next c."
        ),
        "why_not_replaced_c": (
            "Encoding a validation shape with its condition replaced by the target c "
            "reconstructs that shape under a condition it was not drawn from. "
            "The decoded set is not a sample from p(shape | c)."
        ),
        "aggregate_posterior": {
            "definition": (
                "Equal mixture of the encoder Gaussians q(z | x, c) on the validation "
                "cut, using each row's own c. Log-variance is clamped to [-30, 20] "
                "before the exponential, matching the training KL. mean is the average "
                "of the posterior means. std is the square root of (average posterior "
                "variance + variance of the posterior means) on each coordinate."
            ),
            "target": "A standard normal prior has mean 0 and std 1 on every coordinate.",
            "mean": _as_list(moments["mean"]),
            "std": _as_list(moments["std"]),
            "posterior_mean_std": _as_list(moments["posterior_mean_std"]),
            "mean_component_std": _as_list(moments["mean_component_std"]),
            "max_abs_mean": float(np.max(np.abs(moments["mean"]))),
            "rms_std_error": float(np.sqrt(np.mean((moments["std"] - 1.0) ** 2))),
        },
        "elongation_proxy": "mean over shapes of rms(x) / rms(y)",
        "trace": "trace of the unbiased sample covariance (ddof=1)",
        "frechet_distance": (
            "2-Wasserstein distance between Gaussian fits, with ridge * I added "
            "to both covariances before the square root. The same definition as "
            "the ensemble score."
        ),
        "figure": "figures/diagnostics.png",
        "populations": methods,
    }
    _write(out_dir / "diagnostics.json", payload)
    print(f"wrote {out_dir / 'diagnostics.json'}")
    print(
        "aggregate posterior max |mean| "
        f"{payload['aggregate_posterior']['max_abs_mean']:.6f}, "
        f"rms std error {payload['aggregate_posterior']['rms_std_error']:.6f}"
    )
    for method in methods:
        for row in method["by_c"]:
            print(
                f"{method['source']} c={row['c']}: elongation {row['mean_elongation']:.6f} "
                f"(true {row['true_mean_elongation']:.6f}), "
                f"trace {row['generated_trace']:.6f} (true {row['true_trace']:.6f}), "
                f"frechet {row['frechet_distance']:.6f}"
            )


def _encode_moments(model, shapes: FloatArray, conditions: FloatArray) -> dict[str, FloatArray]:
    """Posterior moments of every validation row, then the mixture moments."""
    model.eval()
    x = torch.from_numpy(np.asarray(shapes, dtype=np.float32))
    c = torch.from_numpy(np.asarray(conditions, dtype=np.float32).reshape(-1, 1))
    with torch.no_grad():
        mu_t, logvar_t = model.encode(x, c)
    mu = mu_t.numpy().astype(np.float64)
    variance = np.exp(np.clip(logvar_t.numpy().astype(np.float64), LOGVAR_MIN, LOGVAR_MAX))
    mean, std = aggregate_posterior_moments(mu, variance)
    return {
        "mean": mean,
        "std": std,
        "posterior_mean_std": np.std(mu, axis=0, ddof=1),
        "mean_component_std": np.mean(np.sqrt(variance), axis=0),
    }


def _prior_samples(model, condition: float, n: int, rng: np.random.Generator) -> FloatArray:
    """Draw from the conditional prior. Imported lazily to avoid an import cycle."""
    from cgg.ensemble import sample_model

    return sample_model(model, condition, n, rng)


def _fitted_samples(
    model,
    condition: float,
    n: int,
    mean: FloatArray,
    std: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    """Decode diagonal-Gaussian draws whose moments match the aggregate posterior."""
    latent = rng.normal(loc=mean, scale=std, size=(n, mean.shape[0])).astype(np.float32)
    condition_col = np.full((n, 1), float(condition), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        shapes = model.decode(torch.from_numpy(latent), torch.from_numpy(condition_col))
    return shapes.numpy().astype(np.float64)


def _true_samples(config, conditions: tuple[float, ...], seed: int) -> list[FloatArray]:
    """One fresh true population at each condition, from a single generator."""
    rng = np.random.default_rng(seed)
    return [
        sample_true_conditional(
            float(condition),
            config.uncertainty.n_true,
            rng=rng,
            config=config.data.population,
        )
        for condition in conditions
    ]


def _score_population(generated: FloatArray, truth: FloatArray, ridge: float, condition: float) -> dict:
    """Elongation, covariance trace, and 2-Wasserstein distance at one c."""
    generated_mean, generated_cov = fit_gaussian(generated)
    true_mean, true_cov = fit_gaussian(truth)
    return {
        "c": float(condition),
        "n": int(generated.shape[0]),
        "n_true": int(truth.shape[0]),
        "mean_elongation": float(np.mean(shape_elongation(generated))),
        "true_mean_elongation": float(np.mean(shape_elongation(truth))),
        "generated_trace": float(np.trace(generated_cov)),
        "true_trace": float(np.trace(true_cov)),
        "frechet_distance": frechet_distance(generated_mean, generated_cov, true_mean, true_cov, ridge),
    }


def _as_list(values: FloatArray) -> list[float]:
    """Json-ready Python floats."""
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _write(path: Path, payload: dict) -> None:
    """Write one results record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
