"""Population distances between generated shapes and the true conditional.

Each population is summarised by its sample mean and unbiased sample
covariance. Aleatoric-like spread is the trace of one model's covariance.
Epistemic-like disagreement is how those means and covariances differ across
models. Accuracy is the 2-Wasserstein distance between the pooled Gaussian
fit and the Gaussian fit to fresh draws from the true conditional, plus the
Euclidean mean error and the Frobenius covariance error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm

FloatArray = NDArray[np.float64]


def fit_gaussian(samples: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Sample mean and unbiased covariance of an (n, d) population."""
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError(f"need at least two d-dimensional rows, got {samples.shape}")
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False, ddof=1)
    return np.asarray(mean, dtype=np.float64), np.atleast_2d(np.asarray(cov, dtype=np.float64))


def frechet_distance(
    mean_a: FloatArray,
    cov_a: FloatArray,
    mean_b: FloatArray,
    cov_b: FloatArray,
    ridge: float = 0.0,
) -> float:
    """2-Wasserstein distance between two Gaussian fits.

    ``ridge`` times the identity is added to both covariances before the
    matrix square root. The same shift is used on both sides, so it cancels
    when the covariances are equal and the distance reduces to the Euclidean
    distance between the means. A tiny negative trace from rounding is clipped
    to zero before the square root.
    """
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    dim = mean_a.shape[0]
    eye = np.eye(dim)
    left = _symmetrize(cov_a) + ridge * eye
    right = _symmetrize(cov_b) + ridge * eye
    root_left = _matrix_sqrt(left)
    cross = _matrix_sqrt(root_left @ right @ root_left)
    trace_term = float(np.trace(left) + np.trace(right) - 2.0 * np.trace(cross))
    squared = float(np.sum((mean_a - mean_b) ** 2) + max(trace_term, 0.0))
    return float(np.sqrt(max(squared, 0.0)))


def mean_l2(mean_a: FloatArray, mean_b: FloatArray) -> float:
    """Euclidean distance between two mean shapes."""
    return float(np.linalg.norm(mean_a - mean_b))


def covariance_frobenius(cov_a: FloatArray, cov_b: FloatArray) -> float:
    """Frobenius distance between two covariance matrices. No ridge is added."""
    return float(np.linalg.norm(_symmetrize(cov_a) - _symmetrize(cov_b), ord="fro"))


def population_metrics(
    generated_by_model: list[FloatArray],
    true_samples: FloatArray,
    ridge: float,
) -> dict[str, float | list[float]]:
    """Compare M generated populations at one condition with the true sample.

    The pooled generated sample is the concatenation of the per-model draws.
    That pool is the ensemble population used for the Frechet, mean, and
    covariance comparisons.
    """
    if len(generated_by_model) < 2:
        raise ValueError("disagreement across models needs at least two models")
    fitted = [fit_gaussian(samples) for samples in generated_by_model]
    means = np.stack([mean for mean, _cov in fitted])
    covs = np.stack([cov for _mean, cov in fitted])
    pooled = np.concatenate(generated_by_model, axis=0)
    pooled_mean, pooled_cov = fit_gaussian(pooled)
    true_mean, true_cov = fit_gaussian(true_samples)
    traces = [float(np.trace(cov)) for _mean, cov in fitted]
    return {
        "aleatoric_trace_per_model": traces,
        "aleatoric_trace_mean": float(np.mean(traces)),
        "epistemic_mean_trace": _trace_of_mean_disagreement(means),
        "epistemic_covariance_rms": _covariance_disagreement(covs),
        "frechet_distance": frechet_distance(pooled_mean, pooled_cov, true_mean, true_cov, ridge),
        "mean_l2": mean_l2(pooled_mean, true_mean),
        "covariance_frobenius": covariance_frobenius(pooled_cov, true_cov),
        "true_trace": float(np.trace(true_cov)),
    }


def _trace_of_mean_disagreement(means: FloatArray) -> float:
    """Trace of the unbiased covariance of the M model-wise mean shapes."""
    return float(np.sum(np.var(means, axis=0, ddof=1)))


def _covariance_disagreement(covs: FloatArray) -> float:
    """RMS Frobenius distance of each model covariance from their average."""
    center = np.mean(covs, axis=0)
    delta = covs - center
    mean_square = float(np.mean(np.sum(delta * delta, axis=(1, 2))))
    return float(np.sqrt(mean_square))


def _symmetrize(matrix: FloatArray) -> FloatArray:
    """Return the symmetric part of a covariance estimate."""
    return 0.5 * (matrix + matrix.T)


def _matrix_sqrt(matrix: FloatArray) -> FloatArray:
    """Principal square root of a symmetric covariance.

    SciPy's Schur square root is used when it is accurate. A singular sample
    covariance falls back to the eigenvalue square root, with negative
    eigenvalues clipped to zero.
    """
    sym = _symmetrize(matrix)
    if float(np.linalg.norm(sym, ord="fro")) <= 1e-12:
        return np.zeros_like(sym, dtype=np.float64)
    root, residual = sqrtm(sym, disp=False)
    residual_ok = bool(np.isfinite(residual) and residual <= 1e-5)
    imaginary = float(np.max(np.abs(np.imag(root)))) if np.iscomplexobj(root) else 0.0
    if residual_ok and imaginary <= 1e-6:
        if np.iscomplexobj(root):
            root = np.real(root)
        return np.asarray(root, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(sym)
    clipped = np.clip(np.real(eigvals), 0.0, None)
    return (eigvecs * np.sqrt(clipped)) @ eigvecs.T
