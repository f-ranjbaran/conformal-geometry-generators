"""Train one conditional VAE on the training split, with a validation holdout.

The holdout is cut from the training rows only. Calibration and test rows are
never arguments of ``train_model``. The KL weight increases linearly from 0 to
the configured beta over ``kl_warmup_epochs``. The checkpoint is the lowest
validation loss among epochs that have reached the full beta; an earlier epoch
wins a tie. Warm-up epochs are not eligible, because their loss uses a smaller
KL weight.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cgg.data import DataConfig, load_data_config
from cgg.model import ConditionalVAE, elbo_loss

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ModelConfig:
    """Width of the latent code and the KL weight."""

    latent_dim: int = 8
    beta: float = 0.5

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings. Validation is a fraction of the train split."""

    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    val_fraction: float = 0.15
    kl_warmup_epochs: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        if self.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be non-negative")


@dataclass(frozen=True)
class EnsembleConfig:
    """Seeds and the number of prior draws taken from each trained model."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    samples_per_model: int = 4096

    def __post_init__(self) -> None:
        if len(self.seeds) < 2:
            raise ValueError("an ensemble needs at least two seeds")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if self.samples_per_model < 2:
            raise ValueError("samples_per_model must be at least 2")


@dataclass(frozen=True)
class UncertaintyConfig:
    """Grid of conditions and the sample sizes used to score populations."""

    c_min: float = 0.0
    c_max: float = 1.0
    c_count: int = 9
    n_true: int = 4096
    ridge: float = 1e-4
    sample_seed: int = 1000

    def __post_init__(self) -> None:
        if self.c_count < 2:
            raise ValueError("c_count must be at least 2")
        if self.c_max <= self.c_min:
            raise ValueError("c_max must be greater than c_min")
        if self.n_true < 2:
            raise ValueError("n_true must be at least 2")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if self.sample_seed < 0:
            raise ValueError("sample_seed must be non-negative")

    @property
    def c_grid(self) -> tuple[float, ...]:
        """Inclusive grid from ``c_min`` to ``c_max``."""
        return tuple(float(value) for value in np.linspace(self.c_min, self.c_max, self.c_count))


@dataclass(frozen=True)
class RunConfig:
    """Data, model, training, ensemble, and uncertainty settings for one run."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    ensemble: EnsembleConfig
    uncertainty: UncertaintyConfig
    beta_candidates: tuple[float, ...] = (0.1, 0.03, 0.01, 0.003)
    sweep_c: float = 0.5
    beta_candidates_v2: tuple[float, ...] = (0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    latent_dim_candidates: tuple[int, ...] = (4, 8)
    sweep_conditions: tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass
class TrainResult:
    """History and the checkpoint path for one seed."""

    seed: int
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float | int]]
    checkpoint: Path


def load_run_config(path: Path) -> RunConfig:
    """Read the experiment yaml. Data keys are parsed by ``load_data_config``."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    for name, cls in (
        ("model", ModelConfig),
        ("train", TrainConfig),
        ("ensemble", EnsembleConfig),
        ("uncertainty", UncertaintyConfig),
    ):
        if not isinstance(raw.get(name), dict):
            raise ValueError(f"{name} must be a mapping")
    ensemble_raw = dict(raw["ensemble"])
    seeds = ensemble_raw.get("seeds", [0, 1, 2, 3, 4])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("ensemble.seeds must be a non-empty list")
    ensemble_raw["seeds"] = tuple(int(seed) for seed in seeds)
    train_raw = dict(raw["train"])
    candidates = train_raw.pop("beta_candidates", [0.1, 0.03, 0.01, 0.003])
    sweep_c = float(train_raw.pop("sweep_c", 0.5))
    candidates_v2 = train_raw.pop(
        "beta_candidates_v2",
        [0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    )
    latent_candidates = train_raw.pop("latent_dim_candidates", [4, 8])
    sweep_conditions = train_raw.pop("sweep_conditions", [0.1, 0.5, 0.9])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("train.beta_candidates must be a non-empty list")
    if not isinstance(candidates_v2, list) or not candidates_v2:
        raise ValueError("train.beta_candidates_v2 must be a non-empty list")
    if not isinstance(latent_candidates, list) or not latent_candidates:
        raise ValueError("train.latent_dim_candidates must be a non-empty list")
    if not isinstance(sweep_conditions, list) or not sweep_conditions:
        raise ValueError("train.sweep_conditions must be a non-empty list")
    return RunConfig(
        data=load_data_config(path),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**train_raw),
        ensemble=EnsembleConfig(**ensemble_raw),
        uncertainty=UncertaintyConfig(**raw["uncertainty"]),
        beta_candidates=tuple(float(beta) for beta in candidates),
        sweep_c=sweep_c,
        beta_candidates_v2=tuple(float(beta) for beta in candidates_v2),
        latent_dim_candidates=tuple(int(dim) for dim in latent_candidates),
        sweep_conditions=tuple(float(condition) for condition in sweep_conditions),
    )


def split_train_val(
    shapes: FloatArray,
    conditions: FloatArray,
    val_fraction: float,
    seed: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Hold out validation rows from the array that is passed in.

    The validation count is ``floor(n * val_fraction + 1e-8)``. The permutation
    uses ``Generator(seed)`` and does not touch any other split.
    """
    n = shapes.shape[0]
    if conditions.shape != (n,):
        raise ValueError("conditions must have shape (n,)")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    n_val = int(np.floor(n * val_fraction + 1e-8))
    if n_val < 1 or n_val >= n:
        raise ValueError(f"val_fraction {val_fraction} leaves an empty split for n={n}")
    permutation = np.random.default_rng(seed).permutation(n)
    val_idx = permutation[:n_val]
    fit_idx = permutation[n_val:]
    return shapes[fit_idx], conditions[fit_idx], shapes[val_idx], conditions[val_idx]


def effective_beta(beta: float, epoch: int, warmup_epochs: int) -> float:
    """KL coefficient at a 1-based epoch.

    With a positive warm-up, the coefficient is ``beta * epoch / warmup_epochs``
    until epoch ``warmup_epochs``, and ``beta`` after that. Epoch
    ``warmup_epochs`` is the first epoch at the full weight. A warm-up of 0
    uses ``beta`` from epoch 1.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if epoch < 1:
        raise ValueError("epoch is 1-based")
    if warmup_epochs == 0:
        return float(beta)
    return float(beta) * min(1.0, epoch / warmup_epochs)


def train_model(
    train_x: FloatArray,
    train_c: FloatArray,
    val_x: FloatArray,
    val_c: FloatArray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    seed: int,
    out_dir: Path,
) -> TrainResult:
    """Fit one model. ``seed`` sets initialisation and the batch order."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    _check_shapes(train_x, train_c)
    _check_shapes(val_x, val_c)
    torch.manual_seed(seed)
    model = ConditionalVAE(latent_dim=model_config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    train_loader = _loader(train_x, train_c, train_config.batch_size, seed, shuffle=True)
    val_loader = _loader(val_x, val_c, train_config.batch_size, seed, shuffle=False)
    history: list[dict[str, float | int]] = []
    best_state = _clone_state(model)
    best_epoch = 0
    best_val = float("inf")
    warmup = train_config.kl_warmup_epochs
    for epoch in tqdm(range(1, train_config.epochs + 1), desc=f"seed {seed}"):
        beta_now = effective_beta(model_config.beta, epoch, warmup)
        train_stats = _epoch(model, train_loader, optimizer, beta_now, update=True)
        val_stats = _epoch(model, val_loader, optimizer, beta_now, update=False)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "beta": beta_now,
            "train_loss": train_stats["loss"],
            "train_recon": train_stats["recon"],
            "train_kl": train_stats["kl"],
            "val_loss": val_stats["loss"],
            "val_recon": val_stats["recon"],
            "val_kl": val_stats["kl"],
        }
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"non-finite loss at epoch {epoch} for seed {seed}")
        history.append(row)
        at_full_beta = warmup == 0 or epoch >= warmup
        if at_full_beta and row["val_loss"] < best_val:
            best_val = float(row["val_loss"])
            best_epoch = epoch
            best_state = _clone_state(model)
    if best_epoch < 1:
        raise RuntimeError("epochs must be at least kl_warmup_epochs so a full-beta checkpoint exists")
    model.load_state_dict(best_state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    metadata = {
        "seed": seed,
        "latent_dim": model_config.latent_dim,
        "beta": model_config.beta,
        "kl_warmup_epochs": train_config.kl_warmup_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "device": "cpu",
        "history": history,
    }
    (out_dir / "history.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return TrainResult(seed, best_epoch, best_val, history, checkpoint)


def load_model(path: Path, latent_dim: int) -> ConditionalVAE:
    """Load a CPU checkpoint saved by ``train_model``."""
    model = ConditionalVAE(latent_dim=latent_dim)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> None:
    """Train the single seed given on the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Train one conditional VAE.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from cgg.data import generate_dataset

    config = load_run_config(args.config)
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
    result = train_model(
        fit_x,
        fit_c,
        val_x,
        val_c,
        config.model,
        config.train,
        seed=args.seed,
        out_dir=args.out,
    )
    print(f"seed {result.seed}: best epoch {result.best_epoch}, best val loss {result.best_val_loss}")
    print(f"wrote {result.checkpoint}")


def _loader(x: FloatArray, c: FloatArray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    """Build a CPU loader. The shuffle generator is independent of model init."""
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(c, dtype=np.float32).reshape(-1, 1)),
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _epoch(model, loader, optimizer, beta: float, update: bool) -> dict[str, float]:
    """One pass. Losses are batch-size-weighted means over rows."""
    model.train(update)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    count = 0
    context = torch.enable_grad() if update else torch.no_grad()
    with context:
        for batch_x, batch_c in loader:
            if update:
                optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch_x, batch_c)
            loss, recon_loss, kl = elbo_loss(recon, batch_x, mu, logvar, beta)
            if update:
                loss.backward()
                optimizer.step()
            batch_n = int(batch_x.shape[0])
            totals["loss"] += float(loss.detach()) * batch_n
            totals["recon"] += float(recon_loss.detach()) * batch_n
            totals["kl"] += float(kl.detach()) * batch_n
            count += batch_n
    if count == 0:
        raise ValueError("loader produced no rows")
    return {key: value / count for key, value in totals.items()}


def _clone_state(model: ConditionalVAE) -> dict[str, torch.Tensor]:
    """Detach a copy of the weights so later steps cannot change the checkpoint."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _check_shapes(shapes: FloatArray, conditions: FloatArray) -> None:
    """Require the contour layout this architecture was specified for."""
    if shapes.ndim != 2 or shapes.shape[1] != 128:
        raise ValueError(f"shapes must have shape (n, 128), got {shapes.shape}")
    if conditions.shape != (shapes.shape[0],):
        raise ValueError("conditions must have shape (n,)")


if __name__ == "__main__":
    main()
