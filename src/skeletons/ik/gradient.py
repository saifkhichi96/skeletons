from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..model import SkeletalModel
from ._common import IKSolution, _prepare_targets


class GradientIK:
    """Adam-based skeletal IK with optional extra differentiable loss terms."""

    def __init__(
        self,
        model: SkeletalModel,
        frame_names: Sequence[str],
        lr: float = 5e-2,
        max_iter: int = 100,
        joint_limit_weight: float = 1e-3,
    ) -> None:
        self.model = model
        self.frame_names = list(frame_names)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.joint_limit_weight = float(joint_limit_weight)
        if self.lr <= 0 or self.max_iter < 0 or self.joint_limit_weight < 0:
            raise ValueError(
                "lr must be positive; max_iter and joint_limit_weight nonnegative."
            )

    def solve(
        self,
        q_init: torch.Tensor,
        target_positions: torch.Tensor,
        root: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        extra_loss: Callable[
            [SkeletalModel, torch.Tensor, torch.Tensor | None], torch.Tensor
        ]
        | None = None,
    ) -> IKSolution:
        target_positions, confidence = _prepare_targets(
            self.model, q_init, target_positions, self.frame_names, confidence
        )
        q = torch.nn.Parameter(q_init.detach().clone())
        opt = torch.optim.Adam([q], lr=self.lr)
        history: list[float] = []
        for _ in range(self.max_iter):
            opt.zero_grad(set_to_none=True)
            pred = self.model.frame_positions(q, root, self.frame_names)
            residual = (pred - target_positions).norm(dim=-1)
            loss = (residual * confidence).sum() / confidence.sum().clamp_min(1e-8)
            loss = loss + self.joint_limit_weight * self.model.joint_limit_loss(q)
            if extra_loss is not None:
                loss = loss + extra_loss(self.model, q, root)
            if not loss.requires_grad:
                break
            loss.backward()
            opt.step()
            with torch.no_grad():
                q.data = self.model.clamp_q(q.data)
            history.append(float(loss.detach().cpu()))
        q_out = q.detach()
        final_current = self.model.frame_positions(q_out, root, self.frame_names)
        final_error = (target_positions - final_current).norm(dim=-1)
        return IKSolution(
            q=q_out, final_error=final_error, iterations=len(history), history=history
        )
