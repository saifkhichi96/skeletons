from __future__ import annotations

import torch
import torch.nn as nn

from ..model import SkeletalModel


class ForwardKinematicsLoss(nn.Module):
    def __init__(
        self, model: SkeletalModel, *, p: int = 2, reduction: str = "mean"
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        self.model = model
        self.p = p
        self.reduction = reduction

    def forward(
        self,
        target_joints: torch.Tensor,
        *,
        global_orient: torch.Tensor | None = None,
        body_pose: torch.Tensor | None = None,
        full_pose: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
        transl: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = self.model(
            global_orient=global_orient,
            body_pose=body_pose,
            full_pose=full_pose,
            scales=scales,
            transl=transl,
        )
        diff = pred.joints - target_joints
        loss = torch.linalg.vector_norm(diff, ord=self.p, dim=-1)
        if weights is not None:
            loss = loss * weights
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
