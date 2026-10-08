from __future__ import annotations

from typing import Sequence

import torch

from ..model import SkeletalModel
from ._common import IKSolution, _prepare_targets


class DampedLeastSquaresIK:
    """Batched damped least-squares IK over any rig's named frames."""

    def __init__(
        self,
        model: SkeletalModel,
        frame_names: Sequence[str],
        damping: float = 1e-3,
        step_size: float = 1.0,
        max_iter: int = 20,
        tolerance: float = 1e-4,
        joint_limits: bool = True,
    ) -> None:
        self.model = model
        self.frame_names = list(frame_names)
        self.damping = float(damping)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.joint_limits = bool(joint_limits)
        if self.damping <= 0 or self.step_size <= 0 or self.max_iter < 0:
            raise ValueError(
                "Damping and step_size must be positive; max_iter nonnegative."
            )

    def solve(
        self,
        q_init: torch.Tensor,
        target_positions: torch.Tensor,
        root: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> IKSolution:
        """Solve for a skeletal pose that matches target frame positions."""
        target_positions, confidence = _prepare_targets(
            self.model, q_init, target_positions, self.frame_names, confidence
        )
        if q_init.ndim != 2:
            raise ValueError("q_init must have shape [B, D].")
        if target_positions.shape[:2] != (q_init.shape[0], len(self.frame_names)):
            raise ValueError(
                f"target_positions must have shape [B, {len(self.frame_names)}, 3], "
                f"got {tuple(target_positions.shape)}."
            )

        q = q_init.detach().clone()
        history: list[float] = []
        batch, frames = target_positions.shape[:2]
        device, dtype = q.device, q.dtype

        row_weight = confidence.sqrt().repeat_interleave(3, dim=-1)

        eye_cache: torch.Tensor | None = None
        iterations = 0
        for iteration in range(self.max_iter):
            current, jac = self.model.frame_position_jacobian(q, root, self.frame_names)
            current = current.to(device=device, dtype=dtype)
            jac = jac.to(device=device, dtype=dtype)

            residual = (target_positions - current).reshape(batch, -1)
            weighted_residual = residual * row_weight
            weighted_jac = jac * row_weight.unsqueeze(-1)
            error = (
                weighted_residual.reshape(batch, frames, 3).norm(dim=-1).mean(dim=-1)
            )
            mean_error = float(error.mean().detach().cpu())
            history.append(mean_error)
            iterations = iteration + 1
            if mean_error < self.tolerance:
                break

            jj_t = weighted_jac @ weighted_jac.transpose(-1, -2)
            if eye_cache is None or eye_cache.shape[-1] != jj_t.shape[-1]:
                eye_cache = torch.eye(
                    jj_t.shape[-1], device=device, dtype=dtype
                ).expand(batch, -1, -1)
            lhs = jj_t + (self.damping**2) * eye_cache
            step_task = torch.linalg.solve(
                lhs, weighted_residual.unsqueeze(-1)
            ).squeeze(-1)
            dq = weighted_jac.transpose(-1, -2) @ step_task.unsqueeze(-1)
            q = q + self.step_size * dq.squeeze(-1)
            if self.joint_limits:
                q = self.model.clamp_q(q)

        final_current = self.model.frame_positions(q, root, self.frame_names)
        final_error = (target_positions - final_current).norm(dim=-1)
        return IKSolution(
            q=q, final_error=final_error, iterations=iterations, history=history
        )
