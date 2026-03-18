from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

PoseRepr = Literal['axis_angle', 'rotmat', 'rot6d', 'quat']


def normalize_pose_repr(pose_repr: str) -> PoseRepr:
    key = pose_repr.lower().replace('-', '_')
    aliases = {
        'axis_angle': 'axis_angle',
        'aa': 'axis_angle',
        'rotvec': 'axis_angle',
        'rotation_vector': 'axis_angle',
        'rotmat': 'rotmat',
        'matrix': 'rotmat',
        'rotation_matrix': 'rotmat',
        'rot6d': 'rot6d',
        'rotation_6d': 'rot6d',
        'quat': 'quat',
        'quaternion': 'quat',
    }
    if key not in aliases:
        raise ValueError(f'Unsupported pose representation: {pose_repr!r}')
    return aliases[key]  # type: ignore[return-value]


def pose_repr_size(pose_repr: str) -> int:
    normalized = normalize_pose_repr(pose_repr)
    if normalized == 'axis_angle':
        return 3
    if normalized == 'rot6d':
        return 6
    if normalized == 'quat':
        return 4
    raise ValueError('Rotation matrices do not have a flat feature size.')


def identity_pose(num_joints: int, pose_repr: str, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    normalized = normalize_pose_repr(pose_repr)
    if normalized == 'axis_angle':
        return torch.zeros(num_joints, 3, dtype=dtype, device=device)
    if normalized == 'rot6d':
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=dtype, device=device)
        return ident.view(1, 6).repeat(num_joints, 1)
    if normalized == 'quat':
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        return ident.view(1, 4).repeat(num_joints, 1)
    if normalized == 'rotmat':
        ident = torch.eye(3, dtype=dtype, device=device)
        return ident.view(1, 3, 3).repeat(num_joints, 1, 1)
    raise ValueError(f'Unsupported pose representation: {pose_repr!r}')


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Axis-angle input must end in 3, got {tuple(axis_angle.shape)}.')

    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    safe_theta = theta.clamp_min(1e-8)
    axis = axis_angle / safe_theta

    zeros = torch.zeros_like(axis[..., 0])
    kx, ky, kz = axis.unbind(dim=-1)
    K = torch.stack(
        [
            zeros, -kz, ky,
            kz, zeros, -kx,
            -ky, kx, zeros,
        ],
        dim=-1,
    ).reshape(axis.shape[:-1] + (3, 3))

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(axis.shape[:-1] + (3, 3))

    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]
    outer = axis[..., :, None] * axis[..., None, :]
    rot = cos_theta * eye + (1.0 - cos_theta) * outer + sin_theta * K

    small = (theta[..., 0] < 1e-6).unsqueeze(-1).unsqueeze(-1)
    first_order = eye + K * theta[..., None]
    return torch.where(small, first_order, rot)


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f'Quaternion input must end in 4, got {tuple(quat.shape)}.')

    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError(f'6D rotation input must end in 6, got {tuple(rot6d.shape)}.')

    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def to_rotation_matrix(pose: torch.Tensor, pose_repr: str) -> torch.Tensor:
    normalized = normalize_pose_repr(pose_repr)
    if normalized == 'axis_angle':
        return axis_angle_to_matrix(pose)
    if normalized == 'rot6d':
        return rot6d_to_matrix(pose)
    if normalized == 'quat':
        return quaternion_to_matrix(pose)
    if normalized == 'rotmat':
        if pose.shape[-2:] != (3, 3):
            raise ValueError(f'Rotation-matrix input must end in (3, 3), got {tuple(pose.shape)}.')
        return pose
    raise ValueError(f'Unsupported pose representation: {pose_repr!r}')
