from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ...model import SkeletalModel
from ...rotations import rot6d_to_matrix
from .data import prepare_frame_batch, prepare_sequence_batch
from .priors import JointLimitPrior, PoseVAE, TemporalPrior


@dataclass
class TrainerState:
    """Summary statistics for a single training epoch."""

    epoch: int
    loss: float
    metrics: dict[str, float]


class JointLimitTrainer:
    """Fit a joint-limit prior from joint observations for a given skeleton."""

    def __init__(self, *, model: SkeletalModel) -> None:
        self.model = model

    def fit_from_joints(self, joints_3d: torch.Tensor) -> JointLimitPrior:
        """Estimate joint-limit statistics from a tensor of 3D joints."""

        prepared = prepare_frame_batch(joints_3d, model=self.model)
        return JointLimitPrior.fit(rot6d_to_matrix(prepared.body_pose_rot6d))

    def fit_from_loader(
        self, loader: Iterable[dict[str, torch.Tensor]]
    ) -> JointLimitPrior:
        """Estimate joint-limit statistics from a loader of `.npz`-style batches."""

        rotmats = []
        for batch in loader:
            prepared = prepare_frame_batch(batch["joints_3d"], model=self.model)
            rotmats.append(rot6d_to_matrix(prepared.body_pose_rot6d).cpu())
        if not rotmats:
            raise ValueError("loader must yield at least one batch.")
        return JointLimitPrior.fit(torch.cat(rotmats, dim=0))


class PoseVAETrainer:
    """Train a pose VAE for a given skeleton model."""

    def __init__(
        self,
        prior: PoseVAE,
        *,
        model: SkeletalModel,
        device: torch.device | str = "cpu",
        lr: float = 1e-3,
        kl_weight: float = 1e-4,
        recon_weight: float = 1.0,
    ) -> None:
        self.prior = prior.to(device)
        self.model = model
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.prior.parameters(), lr=lr)
        self.kl_weight = float(kl_weight)
        self.recon_weight = float(recon_weight)

    def train_epoch(self, loader: Iterable[dict[str, torch.Tensor]]) -> TrainerState:
        """Run one training epoch over frame-wise joint batches."""

        self.prior.train()
        losses: list[float] = []
        recons: list[float] = []
        kls: list[float] = []
        for batch in loader:
            prepared = prepare_frame_batch(
                batch["joints_3d"].to(self.device), model=self.model
            )
            loss, metrics = self.prior.prior_loss(
                prepared.body_pose_rot6d,
                kl_weight=self.kl_weight,
                recon_weight=self.recon_weight,
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            recons.append(float(metrics["recon"].cpu()))
            kls.append(float(metrics["kl"].cpu()))
        return TrainerState(
            epoch=0,
            loss=sum(losses) / max(1, len(losses)),
            metrics={
                "recon": sum(recons) / max(1, len(recons)),
                "kl": sum(kls) / max(1, len(kls)),
            },
        )


class TemporalPriorTrainer:
    """Train a temporal prior for a given skeleton model."""

    def __init__(
        self,
        prior: TemporalPrior,
        *,
        model: SkeletalModel,
        device: torch.device | str = "cpu",
        lr: float = 1e-3,
    ) -> None:
        self.prior = prior.to(device)
        self.model = model
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.prior.parameters(), lr=lr)

    def train_epoch(self, loader: Iterable[dict[str, torch.Tensor]]) -> TrainerState:
        """Run one training epoch over sequence joint batches."""

        self.prior.train()
        losses: list[float] = []
        for batch in loader:
            prepared = prepare_sequence_batch(
                batch["joints_3d"].to(self.device), model=self.model
            )
            loss = self.prior.loss(prepared.body_pose_rot6d)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return TrainerState(epoch=0, loss=sum(losses) / max(1, len(losses)), metrics={})
