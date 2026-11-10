"""Shapes, loss definition, seeding, and a short training run."""

import numpy as np
import torch

from cgg.data import sample_shapes
from cgg.model import ConditionalVAE, elbo_loss
from cgg.train import ModelConfig, TrainConfig, split_train_val, train_model


def test_layer_sizes_and_forward() -> None:
    model = ConditionalVAE(latent_dim=8)
    assert model.encoder[0].in_features == 129
    assert model.encoder[0].out_features == 128
    assert model.encoder[2].in_features == 128
    assert model.encoder[2].out_features == 64
    assert model.mu_layer.out_features == 8
    assert model.logvar_layer.out_features == 8
    assert model.decoder[0].in_features == 9
    assert model.decoder[0].out_features == 64
    assert model.decoder[2].out_features == 128
    assert model.decoder[4].out_features == 128
    recon, mu, logvar = model(torch.zeros(4, 128), torch.zeros(4, 1))
    assert recon.shape == (4, 128)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)


def test_elbo_sums_squared_error_over_coordinates() -> None:
    recon = torch.zeros(2, 4)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mu = torch.zeros(2, 1)
    logvar = torch.zeros(2, 1)
    loss, recon_loss, kl = elbo_loss(recon, target, mu, logvar, beta=0.5)
    assert abs(float(recon_loss) - 0.5) < 1e-6
    assert abs(float(kl)) < 1e-6
    assert abs(float(loss) - 0.5) < 1e-6

    shifted = torch.ones(2, 1)
    _loss, _recon, kl_shifted = elbo_loss(recon, target, shifted, logvar, beta=0.5)
    assert abs(float(kl_shifted) - 0.5) < 1e-6
    assert abs(float(_loss) - 0.75) < 1e-6


def test_initialization_follows_the_seed() -> None:
    def first_weight(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return ConditionalVAE().encoder[0].weight.detach().clone()

    assert torch.equal(first_weight(0), first_weight(0))
    assert not torch.equal(first_weight(0), first_weight(1))


def test_validation_split_is_a_partition_of_its_input() -> None:
    shapes = np.arange(100, dtype=np.float64).reshape(100, 1).repeat(128, axis=1)
    conditions = np.linspace(0.0, 1.0, 100, endpoint=False)
    fit_x, _fit_c, val_x, val_c = split_train_val(shapes, conditions, val_fraction=0.15, seed=0)
    assert fit_x.shape == (85, 128)
    assert val_x.shape == (15, 128)
    combined = np.concatenate([fit_x[:, 0], val_x[:, 0]])
    assert sorted(combined.tolist()) == list(range(100))
    assert val_c.shape == (15,)


def test_short_training_lowers_the_loss(tmp_path) -> None:
    shapes, conditions = sample_shapes(256, rng=np.random.default_rng(0))
    result = train_model(
        shapes[:192],
        conditions[:192],
        shapes[192:],
        conditions[192:],
        ModelConfig(),
        TrainConfig(epochs=12, batch_size=64, lr=1e-3, val_fraction=0.25),
        seed=0,
        out_dir=tmp_path,
    )
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert result.best_val_loss < result.history[0]["val_loss"]
    assert (tmp_path / "model.pt").is_file()
