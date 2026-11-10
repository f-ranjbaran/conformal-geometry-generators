"""Train the ensemble and score its conditional populations.

Calibration and test rows are not read. Every ensemble member uses the same
fit/validation cut of the train split. The member seed changes only the
initial weights and the order of training batches.
"""

from __future__ import annotations

import os

# PyTorch and MKL each ship an OpenMP runtime on Windows. Without this, scoring
# aborts after training when SciPy touches the second copy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import torch

from cgg.data import CONDITIONS_TO_SHOW, generate_dataset, sample_true_conditional, shape_elongation
from cgg.model import ConditionalVAE
from cgg.plots import plot_generated_samples, plot_reconstructions, plot_training_curves, plot_uncertainty
from cgg.train import TrainResult, load_model, load_run_config, split_train_val, train_model
from cgg.uncertainty import fit_gaussian, frechet_distance, population_metrics

N_SHOWN = 15
N_MEAN = 500
N_RECON = 6


def sample_model(model: ConditionalVAE, c: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` shapes from the conditional prior p_θ(x | c)."""
    model.eval()
    latent = rng.standard_normal((n, model.latent_dim)).astype(np.float32)
    condition = np.full((n, 1), float(c), dtype=np.float32)
    with torch.no_grad():
        shapes = model.decode(torch.from_numpy(latent), torch.from_numpy(condition))
    return shapes.numpy().astype(np.float64)


def reconstruct_shapes(model: ConditionalVAE, shapes: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    """Posterior-mean reconstructions of a numpy batch."""
    model.eval()
    x = torch.from_numpy(np.asarray(shapes, dtype=np.float32))
    c = torch.from_numpy(np.asarray(conditions, dtype=np.float32).reshape(-1, 1))
    with torch.no_grad():
        recon = model.reconstruct(x, c)
    return recon.numpy().astype(np.float64)


def run(config_path: Path, out_dir: Path, checkpoint_dir: Path) -> None:
    """Train every seed, write results, and save the figures."""
    config = load_run_config(config_path)
    split = generate_dataset(
        config.data.n_shapes,
        seed=config.data.seed,
        fractions=config.data.fractions,
        population=config.data.population,
    )
    fit_x, fit_c, val_x, val_c = split_train_val(
        split.train_shapes,
        split.train_c,
        config.train.val_fraction,
        seed=config.data.seed,
    )
    results = [
        train_model(
            fit_x,
            fit_c,
            val_x,
            val_c,
            config.model,
            config.train,
            seed=seed,
            out_dir=checkpoint_dir / f"seed_{seed}",
        )
        for seed in config.ensemble.seeds
    ]
    models = [load_model(result.checkpoint, config.model.latent_dim) for result in results]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    training = _training_payload(config, results, fit_x.shape[0], val_x.shape[0])
    _write(out_dir / "training.json", training)
    plot_training_curves(out_dir / "figures" / "training_loss.png", training["models"])

    first_seed = int(config.ensemble.seeds[0])
    recon_payload, truths, recons = _reconstruction_payload(models[0], val_x, val_c, first_seed)
    plot_reconstructions(out_dir / "figures" / "reconstructions.png", truths, recons, recon_payload["conditions_array"])
    recon_public = {key: value for key, value in recon_payload.items() if key != "conditions_array"}
    sample_payload, shown, means = _sample_payload(models[0], config.uncertainty.sample_seed, first_seed)
    plot_generated_samples(
        out_dir / "figures" / "generated_samples.png",
        shown,
        means,
        CONDITIONS_TO_SHOW,
        N_MEAN,
    )
    _write(out_dir / "generated_samples.json", {**recon_public, **sample_payload})
    uncertainty = _uncertainty_payload(config, models)
    plot_uncertainty(out_dir / "figures" / "uncertainty_vs_c.png", uncertainty["by_c"])
    _write(out_dir / "uncertainty.json", uncertainty)
    _report(out_dir)


def _training_payload(config, results: list[TrainResult], n_fit: int, n_val: int) -> dict:
    """Json-ready training record built from the returned histories."""
    models = []
    for result in results:
        best = min(result.history, key=lambda row: (row["val_loss"], row["epoch"]))
        models.append(
            {
                "seed": result.seed,
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "best_val_recon": best["val_recon"],
                "best_val_kl": best["val_kl"],
                "first_train_loss": result.history[0]["train_loss"],
                "last_train_loss": result.history[-1]["train_loss"],
                "first_val_loss": result.history[0]["val_loss"],
                "history": result.history,
            }
        )
    return {
        "device": "cpu",
        "latent_dim": config.model.latent_dim,
        "beta": config.model.beta,
        "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "lr": config.train.lr,
        "val_fraction": config.train.val_fraction,
        "val_split_seed": config.data.seed,
        "n_fit": n_fit,
        "n_val": n_val,
        "seeds": list(config.ensemble.seeds),
        "loss_definition": (
            "batch mean of (sum of squared errors over 128 coordinates + beta * KL). "
            "KL is the diagonal Gaussian closed form, also averaged over the batch."
        ),
        "protocol": (
            "One validation cut of the train split is shared by every member. "
            "Its seed is the dataset seed. Each model seed sets initialization and "
            "batch order only. Calibration and test rows are not used."
        ),
        "selection": "lowest validation loss; the earliest epoch wins a tie",
        "figure": "figures/training_loss.png",
        "models": models,
    }


def _reconstruction_payload(model, val_x, val_c, model_seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Six validation shapes spread across c, reconstructed by the first model."""
    order = np.argsort(val_c)
    picks = np.linspace(0, order.shape[0] - 1, N_RECON).astype(int)
    chosen = order[picks]
    truths = val_x[chosen]
    conditions = val_c[chosen]
    recons = reconstruct_shapes(model, truths, conditions)
    payload = {
        "reconstruction_model_seed": model_seed,
        "reconstruction_source": "posterior mean on six validation quantiles of c",
        "reconstruction_conditions": [float(value) for value in conditions],
        "reconstruction_summed_mse": float(np.mean(np.sum((truths - recons) ** 2, axis=1))),
        "reconstruction_figure": "figures/reconstructions.png",
        "conditions_array": conditions,
    }
    return payload, truths, recons


def _sample_payload(model, sample_seed: int, model_seed: int) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    """Prior samples at the same conditions as the true-population figure."""
    rng = np.random.default_rng(sample_seed + 2)
    shown: list[np.ndarray] = []
    mean_draws: list[np.ndarray] = []
    rows = []
    for condition in CONDITIONS_TO_SHOW:
        shown.append(sample_model(model, float(condition), N_SHOWN, rng))
        draws = sample_model(model, float(condition), N_MEAN, rng)
        mean_draws.append(np.mean(draws, axis=0))
        rows.append(
            {
                "c": float(condition),
                "n": N_MEAN,
                "mean_elongation": float(np.mean(shape_elongation(draws))),
            }
        )
    means = [row["mean_elongation"] for row in rows]
    payload = {
        "sample_model_seed": model_seed,
        "sample_seed": sample_seed + 2,
        "n_shown": N_SHOWN,
        "n_mean": N_MEAN,
        "draw_order": "for each c, n_shown prior draws then n_mean prior draws",
        "elongation_proxy": "rms(x) / rms(y)",
        "by_c": rows,
        "increases_with_c": bool(means[0] < means[1] < means[2]),
        "sample_figure": "figures/generated_samples.png",
    }
    return payload, shown, mean_draws


def _uncertainty_payload(config, models: list[ConditionalVAE]) -> dict:
    """Score the pooled ensemble on a grid of conditions."""
    grid = config.uncertainty.c_grid
    model_rng = np.random.default_rng(config.uncertainty.sample_seed)
    true_rng = np.random.default_rng(config.uncertainty.sample_seed + 1)
    rows = []
    for condition in grid:
        generated = [
            sample_model(model, condition, config.ensemble.samples_per_model, model_rng) for model in models
        ]
        true_a = sample_true_conditional(condition, config.uncertainty.n_true, rng=true_rng)
        true_b = sample_true_conditional(condition, config.uncertainty.n_true, rng=true_rng)
        metrics = population_metrics(generated, true_a, config.uncertainty.ridge)
        mean_a, cov_a = fit_gaussian(true_a)
        mean_b, cov_b = fit_gaussian(true_b)
        row = {
            "c": float(condition),
            "frechet_true_vs_true": frechet_distance(mean_a, cov_a, mean_b, cov_b, config.uncertainty.ridge),
            "mean_l2_true_vs_true": float(np.linalg.norm(mean_a - mean_b)),
            "covariance_frobenius_true_vs_true": float(np.linalg.norm(cov_a - cov_b, ord="fro")),
        }
        for key, value in metrics.items():
            if isinstance(value, list):
                row[key] = [float(item) for item in value]
            else:
                row[key] = float(value)
        rows.append(row)
    ordered = sorted(rows, key=lambda item: item["c"])
    edge = 0.5 * (ordered[0]["epistemic_mean_trace"] + ordered[-1]["epistemic_mean_trace"])
    center = min(ordered, key=lambda item: abs(item["c"] - 0.5))
    return {
        "seeds": list(config.ensemble.seeds),
        "samples_per_model": config.ensemble.samples_per_model,
        "n_true": config.uncertainty.n_true,
        "ridge": config.uncertainty.ridge,
        "sample_seed": config.uncertainty.sample_seed,
        "draw_order": (
            "Generator(sample_seed) draws models in seed order inside each c, "
            "c looping in grid order. Generator(sample_seed + 1) draws two independent "
            "true populations of n_true at each c, true_a then true_b."
        ),
        "covariance": "unbiased sample covariance (ddof=1)",
        "aleatoric_trace_mean": "mean over models of the trace of that model's sample covariance",
        "epistemic_mean_trace": "trace of the unbiased covariance, across models, of their mean shapes",
        "epistemic_covariance_rms": "RMS Frobenius distance of each model covariance from the average covariance",
        "frechet_distance": (
            "2-Wasserstein distance between the Gaussian fit of the pooled ensemble "
            "and the Gaussian fit of true_a. ridge * I is added to both covariances "
            "before the square root."
        ),
        "frechet_true_vs_true": (
            "Same Frechet distance between true_a and true_b. This is the sampling "
            "floor at this n_true, not a model score."
        ),
        "mean_l2": "Euclidean distance between the pooled mean and the true_a mean",
        "covariance_frobenius": "Frobenius distance between those covariances, without ridge",
        "true_trace": "trace of the unbiased covariance of true_a",
        "epistemic_edge_vs_center": {
            "endpoint_c": [ordered[0]["c"], ordered[-1]["c"]],
            "endpoint_mean_epistemic_mean_trace": edge,
            "center_c": center["c"],
            "center_epistemic_mean_trace": center["epistemic_mean_trace"],
            "endpoints_larger_than_center": bool(edge > center["epistemic_mean_trace"]),
        },
        "figure": "figures/uncertainty_vs_c.png",
        "by_c": rows,
    }


def _report(out_dir: Path) -> None:
    """Print the numbers just written, then stop if a training check failed."""
    training = json.loads((out_dir / "training.json").read_text(encoding="utf-8"))
    generated = json.loads((out_dir / "generated_samples.json").read_text(encoding="utf-8"))
    uncertainty = json.loads((out_dir / "uncertainty.json").read_text(encoding="utf-8"))
    print(f"wrote {out_dir / 'training.json'}")
    print(f"wrote {out_dir / 'generated_samples.json'}")
    print(f"wrote {out_dir / 'uncertainty.json'}")
    for model in training["models"]:
        print(
            f"seed {model['seed']}: first val {model['first_val_loss']:.6f}, "
            f"best val {model['best_val_loss']:.6f} at epoch {model['best_epoch']}, "
            f"last train {model['last_train_loss']:.6f}"
        )
    elong = ", ".join(f"c={row['c']}: {row['mean_elongation']:.6f}" for row in generated["by_c"])
    print(f"prior elongation: {elong}")
    edge = uncertainty["epistemic_edge_vs_center"]
    print(
        "epistemic mean-trace endpoints "
        f"{edge['endpoint_mean_epistemic_mean_trace']:.6f} vs center "
        f"{edge['center_epistemic_mean_trace']:.6f}"
    )
    problems = []
    for model in training["models"]:
        if model["best_val_loss"] >= model["first_val_loss"]:
            problems.append(f"seed {model['seed']}: validation loss did not fall")
        if model["last_train_loss"] >= model["first_train_loss"]:
            problems.append(f"seed {model['seed']}: training loss did not fall")
    if not generated["increases_with_c"]:
        problems.append("prior samples from seed 0 do not elongate as c increases")
    traces = [row["epistemic_mean_trace"] for row in uncertainty["by_c"]]
    if all(value == 0.0 for value in traces):
        problems.append("ensemble disagreement on the mean shape is zero")
    if problems:
        for problem in problems:
            print(f"problem: {problem}")
        raise SystemExit(1)


def _write(path: Path, payload: dict) -> None:
    """Write one results record."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Train the ensemble and write results next to the existing data summary."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the ensemble and score populations.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args(argv)
    run(args.config, args.out, args.checkpoints)


if __name__ == "__main__":
    main()
