"""Conditional variational autoencoder for closed contours.

The encoder reads the 128 shape coordinates and the condition. The decoder
reads the latent code and the condition. Hidden layers use ReLU. The last
encoder heads and the last decoder layer are linear.

The loss is the batch mean of the summed squared error over the 128
coordinates, plus ``beta`` times the diagonal-Gaussian KL.
"""

from __future__ import annotations

import torch
from torch import nn

X_DIM = 128
LOGVAR_MIN = -30.0
LOGVAR_MAX = 20.0


class ConditionalVAE(nn.Module):
    """Encoder 129→128→64→(μ, log σ²), decoder 9→64→128→128."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(X_DIM + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, X_DIM),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and log-variance of q(z | x, c)."""
        if x.ndim != 2 or x.shape[1] != X_DIM:
            raise ValueError(f"x must have shape (n, {X_DIM}), got {tuple(x.shape)}")
        hidden = self.encoder(torch.cat([x, _column(c, x.shape[0])], dim=1))
        return self.mu_layer(hidden), self.logvar_layer(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw z = μ + σ ε with ε ~ N(0, I), using clamped log-variance."""
        std = torch.exp(0.5 * logvar.clamp(LOGVAR_MIN, LOGVAR_MAX))
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Map latent codes and conditions to shape vectors."""
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape (n, {self.latent_dim}), got {tuple(z.shape)}")
        return self.decoder(torch.cat([z, _column(c, z.shape[0])], dim=1))

    def reconstruct(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode the posterior mean. This reconstruction does not sample z."""
        mu, _logvar = self.encode(x, c)
        return self.decode(mu, c)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a reparameterized reconstruction, plus μ and log σ²."""
        mu, logvar = self.encode(x, c)
        return self.decode(self.reparameterize(mu, logvar), c), mu, logvar


def elbo_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``recon + beta * KL``, the summed squared error, and the KL.

    Squared error is summed over coordinates and averaged over the batch.
    KL is the standard closed form for a diagonal Gaussian against N(0, I),
    averaged over the batch. Log-variance is clamped to [-30, 20] before the
    exponential; the clamp is numerical, not a learned parameter.
    """
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    safe = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    recon_loss = torch.sum((recon - target) ** 2, dim=1).mean()
    kl = -0.5 * torch.sum(1.0 + safe - mu.pow(2) - safe.exp(), dim=1).mean()
    return recon_loss + beta * kl, recon_loss, kl


def _column(c: torch.Tensor, n: int) -> torch.Tensor:
    """Reshape a condition vector to (n, 1)."""
    if c.ndim == 1:
        c = c.unsqueeze(1)
    if c.shape != (n, 1):
        raise ValueError(f"c must have shape ({n}, 1), got {tuple(c.shape)}")
    return c
