from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rotations import (
    matrix_to_axis_angle,
    rot6d_to_matrix,
    rotation_geodesic_distance,
)


@dataclass
class JointLimitStatistics:
    """Summary statistics for a joint-limit prior in axis-angle space."""

    mean: torch.Tensor
    std: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class JointLimitPrior(nn.Module):
    """Soft joint-limit prior learned from a corpus of valid poses."""

    def __init__(
        self,
        stats: JointLimitStatistics,
        *,
        barrier_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", stats.mean)
        self.register_buffer("std", stats.std)
        self.register_buffer("lower", stats.lower)
        self.register_buffer("upper", stats.upper)
        self.barrier_scale = float(barrier_scale)

    @staticmethod
    def fit(
        body_pose_rotmats: torch.Tensor,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        std_floor: float = 1e-3,
    ) -> "JointLimitPrior":
        """Fit joint-limit statistics from rotation matrices."""

        if body_pose_rotmats.ndim != 4 or body_pose_rotmats.shape[-2:] != (3, 3):
            raise ValueError("body_pose_rotmats must have shape [N, J, 3, 3].")
        axis_angle = matrix_to_axis_angle(body_pose_rotmats)
        mean = axis_angle.mean(dim=0)
        std = axis_angle.std(dim=0).clamp_min(std_floor)
        lower = torch.quantile(axis_angle, lower_quantile, dim=0)
        upper = torch.quantile(axis_angle, upper_quantile, dim=0)
        return JointLimitPrior(
            JointLimitStatistics(mean=mean, std=std, lower=lower, upper=upper)
        )

    def forward(self, body_pose_rotmats: torch.Tensor) -> torch.Tensor:
        """Compute the joint-limit penalty for a batch of body rotations."""

        axis_angle = matrix_to_axis_angle(body_pose_rotmats)
        zscore = ((axis_angle - self.mean) / self.std).pow(2).mean()
        lower_violation = F.softplus(self.lower - axis_angle, beta=self.barrier_scale)
        upper_violation = F.softplus(axis_angle - self.upper, beta=self.barrier_scale)
        return zscore + lower_violation.mean() + upper_violation.mean()


class PoseVAE(nn.Module):
    """Variational autoencoder over non-root joint rotations in 6D form."""

    def __init__(
        self,
        num_joints: int,
        *,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.latent_dim = int(latent_dim)
        self.pose_dim = self.num_joints * 6

        encoder_layers: list[nn.Module] = []
        in_features = self.pose_dim
        for _ in range(num_hidden_layers):
            encoder_layers.append(nn.Linear(in_features, hidden_dim))
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.GELU())
            in_features = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        in_features = latent_dim
        for _ in range(num_hidden_layers):
            decoder_layers.append(nn.Linear(in_features, hidden_dim))
            decoder_layers.append(nn.LayerNorm(hidden_dim))
            decoder_layers.append(nn.GELU())
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dim, self.pose_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(
        self, body_pose_rot6d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode body pose rotations into latent mean and log-variance."""

        flat = body_pose_rot6d.reshape(body_pose_rot6d.shape[:-2] + (self.pose_dim,))
        hidden = self.encoder(flat)
        return self.encoder_mu(hidden), self.encoder_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent code using the reparameterization trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent codes into joint rotations in 6D form."""

        flat = self.decoder(latent)
        return flat.reshape(latent.shape[:-1] + (self.num_joints, 6))

    def forward(self, body_pose_rot6d: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode and decode a batch of body poses."""

        mu, logvar = self.encode(body_pose_rot6d)
        latent = self.reparameterize(mu, logvar)
        decoded = self.decode(latent)
        return {"decoded": decoded, "mu": mu, "logvar": logvar, "latent": latent}

    def prior_loss(
        self,
        body_pose_rot6d: torch.Tensor,
        *,
        kl_weight: float = 1e-4,
        recon_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted VAE training loss for body-pose inputs."""

        output = self(body_pose_rot6d)
        pred_rot = rot6d_to_matrix(output["decoded"])
        target_rot = rot6d_to_matrix(body_pose_rot6d)
        recon = rotation_geodesic_distance(pred_rot, target_rot, reduction="mean")
        kl = (
            -0.5
            * (1.0 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp())
            .sum(dim=-1)
            .mean()
        )
        total = recon_weight * recon + kl_weight * kl
        return total, {"recon": recon.detach(), "kl": kl.detach()}

    def latent_regularization(self, latent: torch.Tensor) -> torch.Tensor:
        """Apply a simple zero-mean Gaussian latent penalty."""

        return latent.pow(2).mean()


class TemporalPrior(nn.Module):
    """GRU-based autoregressive prior over body-pose sequences."""

    def __init__(
        self,
        pose_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.rnn = nn.GRU(
            input_size=pose_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, pose_dim)

    def forward(self, body_pose_rot6d_seq: torch.Tensor) -> torch.Tensor:
        """Predict the next-step pose sequence in 6D rotation space."""

        if body_pose_rot6d_seq.ndim != 4:
            raise ValueError("body_pose_rot6d_seq must have shape [B, T, J, 6].")
        flat = body_pose_rot6d_seq.reshape(
            body_pose_rot6d_seq.shape[0], body_pose_rot6d_seq.shape[1], -1
        )
        hidden, _ = self.rnn(flat[:, :-1])
        pred = self.output(hidden)
        return pred.reshape(
            body_pose_rot6d_seq.shape[0],
            body_pose_rot6d_seq.shape[1] - 1,
            body_pose_rot6d_seq.shape[2],
            6,
        )

    def loss(self, body_pose_rot6d_seq: torch.Tensor) -> torch.Tensor:
        """Compute geodesic prediction loss over a pose sequence."""

        pred = self(body_pose_rot6d_seq)
        target = body_pose_rot6d_seq[:, 1:]
        pred_rot = rot6d_to_matrix(pred)
        target_rot = rot6d_to_matrix(target)
        return rotation_geodesic_distance(pred_rot, target_rot, reduction="mean")


class MotionSmoothnessPrior(nn.Module):
    """Finite-difference smoothness prior for pose and translation sequences."""

    def __init__(
        self, *, velocity_weight: float = 1.0, acceleration_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)

    def forward(
        self, body_pose_rot6d_seq: torch.Tensor, transl_seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Penalize velocity and acceleration in pose and translation space."""

        loss = torch.zeros(
            (), dtype=body_pose_rot6d_seq.dtype, device=body_pose_rot6d_seq.device
        )
        if body_pose_rot6d_seq.shape[1] >= 2:
            vel = body_pose_rot6d_seq[:, 1:] - body_pose_rot6d_seq[:, :-1]
            loss = loss + self.velocity_weight * vel.pow(2).mean()
        if body_pose_rot6d_seq.shape[1] >= 3:
            acc = (
                body_pose_rot6d_seq[:, 2:]
                - 2.0 * body_pose_rot6d_seq[:, 1:-1]
                + body_pose_rot6d_seq[:, :-2]
            )
            loss = loss + self.acceleration_weight * acc.pow(2).mean()
        if transl_seq is not None and transl_seq.shape[1] >= 2:
            loss = (
                loss
                + self.velocity_weight
                * 0.1
                * (transl_seq[:, 1:] - transl_seq[:, :-1]).pow(2).mean()
            )
        return loss
