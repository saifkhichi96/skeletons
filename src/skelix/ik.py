from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import SkeletalModel


@dataclass
class InverseKinematicsResult:
    local_rotations: torch.Tensor
    global_rotations: torch.Tensor
    bone_scales: torch.Tensor


def _normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(eps)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def shortest_arc_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError('source and target must end in 3.')

    a = _normalize_vector(source)
    b = _normalize_vector(target)
    cross = torch.cross(a, b, dim=-1)
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    eye = torch.eye(3, dtype=source.dtype, device=source.device).expand(a.shape[:-1] + (3, 3))
    cross_norm_sq = (cross * cross).sum(dim=-1, keepdim=True)
    K = _skew(cross)

    general = eye + K + (K @ K) * ((1.0 - dot).unsqueeze(-1) / cross_norm_sq.clamp_min(1e-8).unsqueeze(-1))

    opposite = dot[..., 0] < -0.9999
    same = dot[..., 0] > 0.9999

    if opposite.any():
        axis_candidate = torch.zeros_like(a)
        axis_candidate[..., 0] = 1.0
        alt = torch.cross(a, axis_candidate, dim=-1)
        alt_small = alt.norm(dim=-1) < 1e-4
        axis_candidate_y = torch.zeros_like(a)
        axis_candidate_y[..., 1] = 1.0
        alt = torch.where(alt_small.unsqueeze(-1), torch.cross(a, axis_candidate_y, dim=-1), alt)
        axis = _normalize_vector(alt)
        K_pi = _skew(axis)
        rot_pi = eye + 2.0 * (K_pi @ K_pi)
        general = torch.where(opposite.unsqueeze(-1).unsqueeze(-1), rot_pi, general)

    general = torch.where(same.unsqueeze(-1).unsqueeze(-1), eye, general)
    return general


def kabsch_rotation(source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if source.shape[-2:] != target.shape[-2:] or source.shape[-1] != 3:
        raise ValueError('source and target must have shape [..., N, 3] and match.')
    if source.shape[-2] == 1:
        return shortest_arc_rotation(source[..., 0, :], target[..., 0, :])

    source_unit = _normalize_vector(source)
    target_unit = _normalize_vector(target)
    if weights is None:
        weights = torch.ones(source.shape[:-1], dtype=source.dtype, device=source.device)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    covariance = source_unit.transpose(-2, -1) @ (target_unit * weights.unsqueeze(-1))
    u, _, vh = torch.linalg.svd(covariance)
    vt = vh.transpose(-2, -1)
    det = torch.det(vt @ u.transpose(-2, -1))
    correction = torch.eye(3, dtype=source.dtype, device=source.device).expand(source.shape[:-2] + (3, 3)).clone()
    correction[..., 2, 2] = torch.where(det < 0.0, -1.0, 1.0)
    rotation = vt @ correction @ u.transpose(-2, -1)
    return rotation


def estimate_bone_scales_from_joints(joints: torch.Tensor, model: SkeletalModel, *, clamp: tuple[float, float] | None = (0.5, 1.5)) -> torch.Tensor:
    if joints.shape[-2:] != (model.num_joints, 3):
        raise ValueError(f'joints must have shape [..., {model.num_joints}, 3].')

    scales = torch.ones(joints.shape[:-2] + (model.num_joints - 1,), dtype=joints.dtype, device=joints.device)
    for out_idx, joint_idx in enumerate(model.non_root_joint_indices):
        parent_idx = model.parents[joint_idx]
        observed = (joints[..., joint_idx, :] - joints[..., parent_idx, :]).norm(dim=-1)
        rest_length = model.rest_offsets[joint_idx].to(joints).norm()
        scales[..., out_idx] = observed / rest_length.clamp_min(1e-8)
    if clamp is not None:
        scales = scales.clamp(clamp[0], clamp[1])
    return scales


def estimate_rotations_from_joints(
    joints: torch.Tensor,
    model: SkeletalModel,
    *,
    bone_scales: torch.Tensor | None = None,
) -> InverseKinematicsResult:
    if joints.shape[-2:] != (model.num_joints, 3):
        raise ValueError(f'joints must have shape [..., {model.num_joints}, 3].')

    batch_shape = joints.shape[:-2]
    dtype = joints.dtype
    device = joints.device

    if bone_scales is None:
        bone_scales_full = model._canonicalize_bone_scales(None, dtype=dtype, device=device)
        bone_scales_full = bone_scales_full.expand(batch_shape + (model.num_joints,)).clone()
        estimated_non_root = estimate_bone_scales_from_joints(joints, model)
        bone_scales_full[..., list(model.non_root_joint_indices)] = estimated_non_root
    else:
        bone_scales_full = model._canonicalize_bone_scales(bone_scales, dtype=dtype, device=device)
        if bone_scales_full.shape[:-1] != batch_shape:
            bone_scales_full = bone_scales_full.expand(batch_shape + (model.num_joints,))

    global_rotations = torch.zeros(batch_shape + (model.num_joints, 3, 3), dtype=dtype, device=device)
    local_rotations = torch.zeros_like(global_rotations)
    identity = torch.eye(3, dtype=dtype, device=device).expand(batch_shape + (3, 3))

    children = [[] for _ in range(model.num_joints)]
    for child_idx, parent_idx in enumerate(model.parents):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)

    scaled_offsets = model._scaled_offsets(
        bone_scales_full,
        batch_shape=batch_shape,
        dtype=dtype,
        device=device,
    )

    for joint_idx in model.topological_order:
        child_indices = children[joint_idx]
        if not child_indices:
            if model.parents[joint_idx] == -1:
                global_rotations[..., joint_idx, :, :] = identity
                local_rotations[..., joint_idx, :, :] = identity
            else:
                parent_idx = model.parents[joint_idx]
                global_rotations[..., joint_idx, :, :] = global_rotations[..., parent_idx, :, :]
                local_rotations[..., joint_idx, :, :] = identity
            continue

        rest_children = torch.stack([scaled_offsets[..., child_idx, :] for child_idx in child_indices], dim=-2)
        observed_children = torch.stack(
            [joints[..., child_idx, :] - joints[..., joint_idx, :] for child_idx in child_indices],
            dim=-2,
        )
        global_rotation = kabsch_rotation(rest_children, observed_children)
        global_rotations[..., joint_idx, :, :] = global_rotation

        parent_idx = model.parents[joint_idx]
        if parent_idx == -1:
            local_rotations[..., joint_idx, :, :] = global_rotation
        else:
            parent_global = global_rotations[..., parent_idx, :, :]
            local_rotations[..., joint_idx, :, :] = parent_global.transpose(-2, -1) @ global_rotation

    return InverseKinematicsResult(
        local_rotations=local_rotations,
        global_rotations=global_rotations,
        bone_scales=bone_scales_full[..., list(model.non_root_joint_indices)],
    )
