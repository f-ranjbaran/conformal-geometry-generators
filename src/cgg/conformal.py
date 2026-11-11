"""Split conformal prediction for one conditional population.

The nonconformity score of a shape is its squared Mahalanobis distance from
the Gaussian summary of a population at the same condition. Under a correct
Gaussian model that squared distance is chi-square with 128 degrees of freedom,
which is the naive threshold. The unsquared Mahalanobis distance is a monotone
transform, so the prediction sets would be the same if the threshold were
transformed with it.

The ensemble covariance is far too small to invert raw. Every covariance, model
and oracle, is replaced by ``cov + noise_std**2 * I``. ``noise_std`` is the
coordinate noise of the synthetic generator (0.01), so the floor is the
observation noise already in the data definition. It is not chosen by looking
at calibration or test coverage. The same floor is applied to the oracle so the
three methods differ only in where the Gaussian and the threshold come from.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import chi2

from cgg.data import generate_dataset, sample_shapes
from cgg.model import X_DIM
from cgg.plots import plot_calibration, plot_conditional_coverage
from cgg.train import load_model, load_run_config
from cgg.uncertainty import fit_gaussian

FloatArray = NDArray[np.float64]
ALPHAS = (0.5, 0.3, 0.1, 0.05)
N_BINS = 5
CHUNK = 8

METHODS = (
    {
        "key": "naive_chi2",
        "label": "naive chi-square",
        "color": "#C65A2E",
    },
    {
        "key": "split_conformal",
        "label": "split conformal",
        "color": "#2F6FAD",
    },
    {
        "key": "oracle_conformal",
        "label": "oracle conformal",
        "color": "#2E7D4F",
    },
)


def conformal_quantile(scores: FloatArray, alpha: float) -> float:
    """Order statistic ``ceil((n+1)(1-alpha))`` of the calibration scores.

    The index is 1-based and capped at ``n``. If the formula asks for an order
    past ``n``, the threshold is infinite and the prediction set is everything.
    A test score is inside when it is less than or equal to this value.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(values.shape[0])
    if n < 1:
        raise ValueError("calibration scores must be non-empty")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    ordered = np.sort(values)
    return float(ordered[k - 1])


def chi2_threshold(dimension: int, alpha: float) -> float:
    """Chi-square quantile at survival ``alpha``, with ``dimension`` degrees of freedom."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return float(chi2.ppf(1.0 - alpha, df=dimension))


def squared_mahalanobis(x: FloatArray, mean: FloatArray, cov: FloatArray, noise_var: float) -> FloatArray:
    """Squared Mahalanobis distances of each row from one regularised Gaussian."""
    if noise_var < 0.0:
        raise ValueError("noise_var must be non-negative")
    rows = np.asarray(x, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64).reshape(-1)
    if rows.ndim != 2 or rows.shape[1] != center.shape[0]:
        raise ValueError("x and mean have incompatible shapes")
    reg = _regularise(np.asarray(cov, dtype=np.float64), noise_var)
    diff = rows - center
    solved = np.linalg.solve(reg, diff.T)
    scores = np.sum(diff.T * solved, axis=0)
    return np.asarray(scores, dtype=np.float64)


def run(config_path: Path, out_dir: Path, checkpoint_dir: Path) -> None:
    """Score calibration and test, write ``results/conformal.json``, and plot."""
    config = load_run_config(config_path)
    _check_checkpoints(config, checkpoint_dir)
    models = [
        load_model(Path(checkpoint_dir) / f"seed_{seed}" / "model.pt", config.model.latent_dim)
        for seed in config.ensemble.seeds
    ]
    split = generate_dataset(
        config.data.n_shapes,
        seed=config.data.seed,
        fractions=config.data.fractions,
        population=config.data.population,
    )
    noise_var = float(config.data.population.noise_std) ** 2
    cal_x, cal_c = split.calibration_shapes, split.calibration_c
    test_x, test_c = split.test_shapes, split.test_c
    model_rng = np.random.default_rng(config.uncertainty.sample_seed + 30)
    oracle_rng = np.random.default_rng(config.uncertainty.sample_seed + 31)
    print("scoring the ensemble population at each calibration and test condition")
    model_cal, model_cal_cov = _population_gaussians(
        models, cal_c, config.ensemble.samples_per_model, model_rng
    )
    model_test, _model_test_cov = _population_gaussians(
        models, test_c, config.ensemble.samples_per_model, model_rng
    )
    print("scoring the true conditional at each calibration and test condition")
    oracle_cal, oracle_cal_cov = _oracle_gaussians(
        cal_c, config.uncertainty.n_true, oracle_rng, config.data.population
    )
    oracle_test, _oracle_test_cov = _oracle_gaussians(
        test_c, config.uncertainty.n_true, oracle_rng, config.data.population
    )
    model_cal_scores = _scores(cal_x, model_cal, noise_var)
    model_test_scores = _scores(test_x, model_test, noise_var)
    oracle_cal_scores = _scores(cal_x, oracle_cal, noise_var)
    oracle_test_scores = _scores(test_x, oracle_test, noise_var)
    _require_finite(model_cal_scores, "ensemble calibration")
    _require_finite(model_test_scores, "ensemble test")
    _require_finite(oracle_cal_scores, "oracle calibration")
    _require_finite(oracle_test_scores, "oracle test")
    payload = _payload(
        config,
        noise_var,
        model_cal_scores,
        model_test_scores,
        oracle_cal_scores,
        oracle_test_scores,
        cal_c,
        test_c,
        _covariance_summary(model_cal_cov, noise_var),
        _covariance_summary(oracle_cal_cov, noise_var),
    )
    out_dir = Path(out_dir)
    plot_methods = _plot_methods(payload)
    plot_calibration(out_dir / "figures" / "calibration_curve.png", plot_methods)
    plot_conditional_coverage(out_dir / "figures" / "conditional_coverage.png", plot_methods, N_BINS)
    _write(out_dir / "conformal.json", payload)
    _report(payload)


def _population_gaussians(models, conditions, n_per_model: int, rng: np.random.Generator):
    """Pooled ensemble mean and covariance at each condition."""
    conditions = np.asarray(conditions, dtype=np.float64)
    means = np.empty((conditions.shape[0], X_DIM))
    covs = np.empty((conditions.shape[0], X_DIM, X_DIM))
    for start in range(0, conditions.shape[0], CHUNK):
        stop = min(start + CHUNK, conditions.shape[0])
        block = _ensemble_samples(models, conditions[start:stop], n_per_model, rng)
        for index, sample in enumerate(block):
            mean, cov = fit_gaussian(sample)
            means[start + index] = mean
            covs[start + index] = cov
        print(f"  ensemble conditions {stop}/{conditions.shape[0]}")
    return list(zip(means, covs, strict=True)), covs


def _ensemble_samples(models, conditions, n_per_model: int, rng: np.random.Generator) -> FloatArray:
    """Draw ``n_per_model`` shapes from every member and concatenate on axis 1."""
    m = int(conditions.shape[0])
    blocks = []
    for model in models:
        model.eval()
        latent = rng.standard_normal((m * n_per_model, model.latent_dim)).astype(np.float32)
        condition = np.repeat(conditions.astype(np.float32), n_per_model)[:, None]
        with torch.no_grad():
            shapes = model.decode(torch.from_numpy(latent), torch.from_numpy(condition))
        blocks.append(shapes.numpy().astype(np.float64).reshape(m, n_per_model, X_DIM))
    return np.concatenate(blocks, axis=1)


def _oracle_gaussians(conditions, n_true: int, rng: np.random.Generator, population):
    """Gaussian fit of fresh true draws at each condition."""
    conditions = np.asarray(conditions, dtype=np.float64)
    means = np.empty((conditions.shape[0], X_DIM))
    covs = np.empty((conditions.shape[0], X_DIM, X_DIM))
    for start in range(0, conditions.shape[0], CHUNK):
        stop = min(start + CHUNK, conditions.shape[0])
        repeated = np.repeat(conditions[start:stop], n_true)
        shapes, _used = sample_shapes(repeated.shape[0], c=repeated, rng=rng, config=population)
        block = shapes.reshape(stop - start, n_true, X_DIM)
        for index, sample in enumerate(block):
            mean, cov = fit_gaussian(sample)
            means[start + index] = mean
            covs[start + index] = cov
        print(f"  oracle conditions {stop}/{conditions.shape[0]}")
    return list(zip(means, covs, strict=True)), covs


def _scores(shapes, gaussians, noise_var: float) -> FloatArray:
    """One squared Mahalanobis score per shape, using that shape's Gaussian."""
    scores = np.empty(shapes.shape[0], dtype=np.float64)
    for index, (shape, (mean, cov)) in enumerate(zip(shapes, gaussians, strict=True)):
        scores[index] = squared_mahalanobis(shape.reshape(1, -1), mean, cov, noise_var)[0]
    return scores


def _regularise(cov: FloatArray, noise_var: float) -> FloatArray:
    """Symmetrise and add the observation-noise floor on the diagonal."""
    sym = 0.5 * (cov + cov.T)
    return sym + noise_var * np.eye(sym.shape[0])


def _covariance_summary(covs: FloatArray, noise_var: float) -> dict:
    """Median size of the calibration covariances, before and after the floor."""
    traces = []
    eig_min = []
    eig_max = []
    reg_condition = []
    for cov in covs:
        evals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        traces.append(float(np.sum(evals)))
        eig_min.append(float(evals[0]))
        eig_max.append(float(evals[-1]))
        floored = evals + noise_var
        reg_condition.append(float(floored[-1] / floored[0]))
    raw = np.asarray(traces)
    return {
        "n": int(covs.shape[0]),
        "trace_raw_median": float(np.median(raw)),
        "trace_raw_min": float(np.min(raw)),
        "trace_raw_max": float(np.max(raw)),
        "trace_regularised_median": float(np.median(raw + covs.shape[1] * noise_var)),
        "eigenvalue_min_median": float(np.median(eig_min)),
        "eigenvalue_max_median": float(np.median(eig_max)),
        "regularised_condition_median": float(np.median(reg_condition)),
    }


def _payload(
    config,
    noise_var: float,
    model_cal: FloatArray,
    model_test: FloatArray,
    oracle_cal: FloatArray,
    oracle_test: FloatArray,
    cal_c: FloatArray,
    test_c: FloatArray,
    model_cov: dict,
    oracle_cov: dict,
) -> dict:
    """Json record. Thresholds for the two conformal methods use calibration only."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    test_bins = _bin_index(test_c)
    naive_rows = []
    conformal_rows = []
    oracle_rows = []
    for alpha in ALPHAS:
        naive_q = chi2_threshold(X_DIM, alpha)
        conformal_q = conformal_quantile(model_cal, alpha)
        oracle_q = conformal_quantile(oracle_cal, alpha)
        naive_rows.append(_alpha_row(alpha, naive_q, model_test, test_bins, edges, "chi2"))
        conformal_rows.append(_alpha_row(alpha, conformal_q, model_test, test_bins, edges, "calibration"))
        oracle_rows.append(_alpha_row(alpha, oracle_q, oracle_test, test_bins, edges, "calibration"))
    return {
        "beta": config.model.beta,
        "latent_dim": config.model.latent_dim,
        "seeds": list(config.ensemble.seeds),
        "dimension": X_DIM,
        "alphas": list(ALPHAS),
        "n_calibration": int(model_cal.shape[0]),
        "n_test": int(model_test.shape[0]),
        "samples_per_model": config.ensemble.samples_per_model,
        "n_true": config.uncertainty.n_true,
        "model_sample_seed": config.uncertainty.sample_seed + 30,
        "oracle_sample_seed": config.uncertainty.sample_seed + 31,
        "score": (
            "Squared Mahalanobis distance of the real shape from the Gaussian "
            "fit of a population at the same c. The ensemble population is the "
            "pool of samples_per_model prior draws from every member. The oracle "
            "population is n_true fresh draws from the true conditional."
        ),
        "regularisation": {
            "rule": "cov + noise_std**2 * I before the Mahalanobis solve",
            "noise_std": config.data.population.noise_std,
            "noise_var": noise_var,
            "tuned_on_test": False,
            "tuned_on_calibration_coverage": False,
            "why": (
                "noise_std is the synthetic coordinate noise. Adding its variance "
                "on the diagonal stops near-zero eigenvalues of a collapsed "
                "covariance from dominating the score, and it does not inflate "
                "the trace up to the true conditional trace. The value is part of "
                "the data definition. Coverage was not used to pick it. The naive "
                "chi-square quantile still uses 128 degrees of freedom; the floor "
                "means that quantile is not exact even for a correct Gaussian."
            ),
            "ensemble_on_calibration": model_cov,
            "oracle_on_calibration": oracle_cov,
        },
        "conformal_rule": (
            "k = ceil((n_calibration + 1) * (1 - alpha)). The threshold is the "
            "k-th smallest calibration score. A test shape is covered when its "
            "score is less than or equal to the threshold. k > n would make the "
            "threshold infinite."
        ),
        "naive_rule": "threshold = chi2.ppf(1 - alpha, df=128) on the ensemble scores",
        "oracle_rule": (
            "Same conformal rule as split conformal, with scores from the true "
            "conditional mean and covariance instead of the ensemble."
        ),
        "bins": {
            "n_bins": N_BINS,
            "edges": [round(float(edge), 10) for edge in edges],
            "rule": "equal width on [0, 1); the rightmost edge is inclusive",
        },
        "outside_nominal_pm_3pct": (
            "abs(empirical - nominal) > 0.03 + 1e-9. The tolerance keeps a gap of "
            "exactly 0.03 from being flagged by binary rounding. This flag is a "
            "comparison of the numbers below, not a second estimator."
        ),
        "figures": ["figures/calibration_curve.png", "figures/conditional_coverage.png"],
        "methods": {
            "naive_chi2": {"by_alpha": naive_rows},
            "split_conformal": {"by_alpha": conformal_rows},
            "oracle_conformal": {"by_alpha": oracle_rows},
        },
        "score_scale": {
            "ensemble_calibration_median": float(np.median(model_cal)),
            "ensemble_test_median": float(np.median(model_test)),
            "oracle_calibration_median": float(np.median(oracle_cal)),
            "oracle_test_median": float(np.median(oracle_test)),
        },
    }


def _alpha_row(alpha, threshold, test_scores, test_bins, edges, threshold_source: str) -> dict:
    """Coverage of one threshold on the test scores, overall and by bin."""
    nominal = 1.0 - alpha
    inside = test_scores <= threshold
    rows = []
    for index in range(N_BINS):
        mask = test_bins == index
        n = int(np.sum(mask))
        n_covered = int(np.sum(inside[mask])) if n else 0
        rows.append(
            {
                "bin": index,
                "c_low": round(float(edges[index]), 10),
                "c_high": round(float(edges[index + 1]), 10),
                "n": n,
                "n_covered": n_covered,
                "coverage": None if n == 0 else float(n_covered / n),
            }
        )
    coverage = float(np.mean(inside))
    return {
        "alpha": float(alpha),
        "nominal": float(nominal),
        "threshold": None if threshold == float("inf") else float(threshold),
        "threshold_source": threshold_source,
        "n_test": int(test_scores.shape[0]),
        "n_covered": int(np.sum(inside)),
        "test_coverage": coverage,
        "gap": float(coverage - nominal),
        "outside_nominal_pm_3pct": bool(abs(coverage - nominal) > 0.03 + 1e-9),
        "by_bin": rows,
    }


def _bin_index(conditions: FloatArray) -> NDArray[np.int64]:
    """Bin index on equal-width intervals of ``[0, 1)``."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    return np.digitize(conditions, edges[1:-1], right=False).astype(np.int64)


def _plot_methods(payload: dict) -> list[dict]:
    """Attach colours and labels for the two figures."""
    plotted = []
    for spec in METHODS:
        plotted.append(
            {
                "label": spec["label"],
                "color": spec["color"],
                "by_alpha": payload["methods"][spec["key"]]["by_alpha"],
            }
        )
    return plotted


def _check_checkpoints(config, checkpoint_dir: Path) -> None:
    """Refuse to score a checkpoint whose saved beta or width disagrees with the config."""
    for seed in config.ensemble.seeds:
        path = Path(checkpoint_dir) / f"seed_{seed}" / "history.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        if int(meta["latent_dim"]) != int(config.model.latent_dim):
            raise RuntimeError(f"{path} has latent_dim {meta['latent_dim']}, config has {config.model.latent_dim}")
        if not np.isclose(float(meta["beta"]), float(config.model.beta)):
            raise RuntimeError(f"{path} has beta {meta['beta']}, config has {config.model.beta}")


def _require_finite(scores: FloatArray, name: str) -> None:
    """Stop if a score is missing or negative beyond rounding."""
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{name} scores are not finite")
    if np.any(scores < -1e-6):
        raise RuntimeError(f"{name} scores have a negative Mahalanobis value")


def _report(payload: dict) -> None:
    """Print coverage against nominal. The json file is the record."""
    scale = payload["score_scale"]
    print(
        "ensemble score median "
        f"calibration {scale['ensemble_calibration_median']:.4f}, "
        f"test {scale['ensemble_test_median']:.4f}"
    )
    print(
        "oracle score median "
        f"calibration {scale['oracle_calibration_median']:.4f}, "
        f"test {scale['oracle_test_median']:.4f}"
    )
    cov = payload["regularisation"]["ensemble_on_calibration"]
    print(
        "calibration ensemble trace median "
        f"{cov['trace_raw_median']:.6f} raw, "
        f"{cov['trace_regularised_median']:.6f} after the noise floor"
    )
    for key in ("naive_chi2", "split_conformal", "oracle_conformal"):
        for row in payload["methods"][key]["by_alpha"]:
            flag = " OUTSIDE +/- 3pct" if row["outside_nominal_pm_3pct"] else ""
            print(
                f"{key} nominal {row['nominal']:.2f}: coverage {row['test_coverage']:.4f}, "
                f"threshold {row['threshold']:.4f}{flag}"
            )


def _write(path: Path, payload: dict) -> None:
    """Write one results record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    """Score the saved ensemble and write the conformal figures."""
    import argparse

    parser = argparse.ArgumentParser(description="Split conformal evaluation of the ensemble.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args(argv)
    run(args.config, args.out, args.checkpoints)


if __name__ == "__main__":
    main()
