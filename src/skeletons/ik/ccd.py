from __future__ import annotations

from typing import Sequence

import torch

from ..model import SkeletalModel
from ..rotations import matrix_to_axis_angle
from ._common import IKSolution, _normalize_vector, _prepare_targets


def _rotation_between_vectors(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    a_n = _normalize_vector(a, eps=eps)
    b_n = _normalize_vector(b, eps=eps)
    cross = torch.cross(a_n, b_n, dim=-1)
    dot = (a_n * b_n).sum(dim=-1).clamp(-1.0, 1.0)
    skew = torch.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype, device=a.device)
    skew[..., 0, 1] = -cross[..., 2]
    skew[..., 0, 2] = cross[..., 1]
    skew[..., 1, 0] = cross[..., 2]
    skew[..., 1, 2] = -cross[..., 0]
    skew[..., 2, 0] = -cross[..., 1]
    skew[..., 2, 1] = cross[..., 0]
    eye = torch.eye(3, dtype=a.dtype, device=a.device).expand(a.shape[:-1] + (3, 3))
    cross_norm_sq = (cross * cross).sum(dim=-1).clamp_min(eps)
    rotation = (
        eye + skew + (skew @ skew) * ((1.0 - dot) / cross_norm_sq)[..., None, None]
    )
    # Antiparallel vectors need a half-turn around any perpendicular axis.
    basis = torch.nn.functional.one_hot(a_n.abs().argmin(dim=-1), 3).to(a)
    axis = _normalize_vector(torch.cross(a_n, basis, dim=-1), eps=eps)
    half_turn = 2 * axis.unsqueeze(-1) * axis.unsqueeze(-2) - eye
    rotation = torch.where((dot < -1.0 + 1e-6)[..., None, None], half_turn, rotation)
    return torch.where((dot > 1.0 - 1e-6)[..., None, None], eye, rotation)


class CyclicCoordinateDescentIK:
    """Tree-aware CCD IK over any rig's joints or attached frames."""

    def __init__(
        self,
        model: SkeletalModel,
        frame_names: Sequence[str],
        max_iter: int = 10,
        step_size: float = 1.0,
        tolerance: float = 1e-4,
        joint_limits: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.frame_names = list(frame_names)
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.tolerance = float(tolerance)
        self.joint_limits = bool(joint_limits)
        self.eps = float(eps)
        if self.max_iter < 0 or self.step_size <= 0 or self.eps <= 0:
            raise ValueError(
                "max_iter must be nonnegative; step_size and eps positive."
            )
        self._joint_targets = self._build_joint_targets()
        self._solve_joint_indices = [
            joint_i
            for joint_i in reversed(model.topological_order)
            if self._joint_targets[joint_i]
        ]

    def solve(
        self,
        q_init: torch.Tensor,
        target_positions: torch.Tensor,
        root: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> IKSolution:
        target_positions, confidence = _prepare_targets(
            self.model, q_init, target_positions, self.frame_names, confidence
        )
        q = q_init.detach().clone()
        history: list[float] = []
        iterations = 0

        for iteration in range(self.max_iter):
            for joint_i in self._solve_joint_indices:
                q = self._optimize_joint(q, root, target_positions, confidence, joint_i)
                if self.joint_limits:
                    q = self.model.clamp_q(q)
            current = self.model.frame_positions(q, root, self.frame_names)
            err = (
                (target_positions - current).norm(dim=-1) * confidence
            ).sum() / confidence.sum().clamp_min(1e-8)
            history.append(float(err.detach().cpu()))
            iterations = iteration + 1
            if float(err.detach().cpu()) < self.tolerance:
                break
        final_current = self.model.frame_positions(q, root, self.frame_names)
        final_error = (target_positions - final_current).norm(dim=-1)
        return IKSolution(
            q=q, final_error=final_error, iterations=iterations, history=history
        )

    def _target_parent_link(self, frame_name: str) -> int:
        return self.model.frame_parent_index(frame_name)

    def _build_joint_targets(self) -> list[list[int]]:
        joint_targets: list[list[int]] = [[] for _ in self.model.joints]
        for target_i, frame in enumerate(self.frame_names):
            current = self._target_parent_link(frame)
            while current != self.model.root_index and current >= 0:
                joint_targets[current].append(target_i)
                current = self.model.parents[current]
        return joint_targets

    def _optimize_joint(
        self,
        q: torch.Tensor,
        root: torch.Tensor | None,
        target_positions: torch.Tensor,
        confidence: torch.Tensor,
        joint_i: int,
    ) -> torch.Tensor:
        joint = self.model.joints[joint_i]
        joint_dof = self.model.spec.joint_dof(joint_i)
        if joint_dof == 0 or joint.name not in self.model._q_slices:
            return q
        s = self.model.dof_slice(joint.name)
        fk = self.model.forward_kinematics(q, root=root)
        parent_link_i = self.model.parents[joint_i]
        if parent_link_i < 0:
            return q
        joint_pos = fk.joints[..., joint_i, :]
        child_rot = fk.global_rotations[..., joint_i, :, :]
        parent_rot = fk.global_rotations[..., parent_link_i, :, :]
        current_targets = self.model.frame_positions(q, root, self.frame_names)
        weighted_delta_q = torch.zeros(
            q.shape[0], joint_dof, device=q.device, dtype=q.dtype
        )
        weight_sum = torch.zeros(q.shape[0], 1, device=q.device, dtype=q.dtype)

        for target_i in self._joint_targets[joint_i]:
            w = confidence[:, target_i : target_i + 1]
            if torch.all(w <= 0):
                continue
            current_vec = current_targets[:, target_i, :] - joint_pos
            target_vec = target_positions[:, target_i, :] - joint_pos
            valid = (current_vec.norm(dim=-1, keepdim=True) > self.eps) & (
                target_vec.norm(dim=-1, keepdim=True) > self.eps
            )
            if joint.joint_type == "ball":
                delta_rot = _rotation_between_vectors(
                    current_vec, target_vec, eps=self.eps
                )
                desired_axis_angle = matrix_to_axis_angle(
                    parent_rot.transpose(-1, -2) @ delta_rot @ child_rot
                )
                dq = desired_axis_angle - q[:, s]
            elif joint.joint_type == "hinge":
                axis = (self.model.spec.joint_axes or (None,) * self.model.NUM_JOINTS)[
                    joint_i
                ] or (1.0, 0.0, 0.0)
                axis_local = torch.tensor(axis, device=q.device, dtype=q.dtype)
                axis_local = axis_local / axis_local.norm()
                axis_global = torch.matmul(
                    parent_rot, axis_local.view(1, 3, 1).expand(q.shape[0], -1, -1)
                ).squeeze(-1)
                curr_proj = (
                    current_vec
                    - (current_vec * axis_global).sum(dim=-1, keepdim=True)
                    * axis_global
                )
                targ_proj = (
                    target_vec
                    - (target_vec * axis_global).sum(dim=-1, keepdim=True) * axis_global
                )
                valid = (
                    valid
                    & (curr_proj.norm(dim=-1, keepdim=True) > self.eps)
                    & (targ_proj.norm(dim=-1, keepdim=True) > self.eps)
                )
                curr_proj = curr_proj / curr_proj.norm(dim=-1, keepdim=True).clamp_min(
                    self.eps
                )
                targ_proj = targ_proj / targ_proj.norm(dim=-1, keepdim=True).clamp_min(
                    self.eps
                )
                sin = (axis_global * torch.cross(curr_proj, targ_proj, dim=-1)).sum(
                    dim=-1, keepdim=True
                )
                cos = (curr_proj * targ_proj).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                dq = torch.atan2(sin, cos)
            else:
                continue
            dq = torch.where(valid.expand_as(dq), dq, torch.zeros_like(dq))
            weighted_delta_q = weighted_delta_q + dq * w
            weight_sum = weight_sum + torch.where(valid, w, torch.zeros_like(w))

        if torch.all(weight_sum <= 0):
            return q
        out = q.clone()
        out[:, s] = out[
            :, s
        ] + self.step_size * weighted_delta_q / weight_sum.clamp_min(self.eps)
        return out
