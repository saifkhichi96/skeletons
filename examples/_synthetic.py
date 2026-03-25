from __future__ import annotations

import torch

from differential_skeletons import SkeletalModel
from differential_skeletons.fitting import PerspectiveCamera, WeakPerspectiveCamera


def make_random_pose_batch(
    model: SkeletalModel,
    *,
    batch_size: int,
    pose_std: float,
    transl_std: float = 25.0,
    depth: float = 2500.0,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    full_pose = torch.randn(batch_size, model.NUM_JOINTS, 3, device=device) * pose_std
    transl = torch.randn(batch_size, 3, device=device) * transl_std
    transl[..., 2] += depth
    scales = torch.ones(batch_size, model.NUM_JOINTS, 3, device=device)
    joints = model(full_pose=full_pose, transl=transl, scales=scales).joints.detach()
    return full_pose, transl, scales, joints


def make_random_walk_sequences(
    model: SkeletalModel,
    *,
    batch_size: int,
    seq_len: int,
    pose_step_std: float,
    transl_step_std: float,
    depth: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pose_steps = (
        torch.randn(batch_size, seq_len, model.NUM_JOINTS, 3, device=device)
        * pose_step_std
    )
    full_pose = pose_steps.cumsum(dim=1)

    transl_steps = torch.randn(batch_size, seq_len, 3, device=device) * transl_step_std
    transl = transl_steps.cumsum(dim=1)
    transl[..., 2] += depth
    scales = torch.ones(batch_size, model.NUM_JOINTS, 3, device=device)

    flat_out = model(
        full_pose=full_pose.reshape(-1, model.NUM_JOINTS, 3),
        transl=transl.reshape(-1, 3),
        scales=scales.repeat_interleave(seq_len, dim=0),
    )
    joints = flat_out.joints.reshape(batch_size, seq_len, model.NUM_JOINTS, 3).detach()
    return full_pose, transl, scales, joints


def make_perspective_camera(device: torch.device) -> PerspectiveCamera:
    return PerspectiveCamera(
        fx=torch.tensor(1000.0, device=device),
        fy=torch.tensor(1000.0, device=device),
        cx=torch.tensor(512.0, device=device),
        cy=torch.tensor(512.0, device=device),
    )


def make_weak_perspective_camera(device: torch.device) -> WeakPerspectiveCamera:
    return WeakPerspectiveCamera(
        scale=torch.tensor(180.0, device=device),
        tx=torch.tensor(0.0, device=device),
        ty=torch.tensor(0.0, device=device),
    )
