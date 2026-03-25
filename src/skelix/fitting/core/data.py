from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ...ik import estimate_bone_scales_from_joints, estimate_rotations_from_joints
from ...model import SkeletalModel
from ...rotations import matrix_to_rot6d

CAMERA_KEYS = ('fx', 'fy', 'cx', 'cy', 'camera_translation', 'camera_rotation')


def _clone_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.copy()
    return value


def _validate_joint_tensor(
    name: str,
    joints: torch.Tensor,
    *,
    ndim: int,
    expected_num_joints: int | None = None,
) -> None:
    if joints.ndim != ndim or joints.shape[-1] != 3:
        raise ValueError(f'{name} must have shape [..., J, 3] with {ndim} dimensions, got {tuple(joints.shape)}.')
    if expected_num_joints is not None and joints.shape[-2] != expected_num_joints:
        raise ValueError(f'{name} must contain {expected_num_joints} joints, got {joints.shape[-2]}.')


@dataclass
class FittingDataBatch:
    """Prepared fitting targets for a concrete skeleton model.

    Attributes:
        joints_3d: Input joints in model joint order with shape `[..., J, 3]`.
        body_pose_rot6d: Non-root joint rotations with shape `[..., J - 1, 6]`.
        global_orient_rot6d: Root rotation with shape `[..., 6]`.
        bone_scales: OpenSim-style per-body scale factors with shape `[..., J, 3]`.
    """

    joints_3d: torch.Tensor
    body_pose_rot6d: torch.Tensor
    global_orient_rot6d: torch.Tensor
    bone_scales: torch.Tensor


class FrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Frame-wise fitting dataset loaded from a shared `.npz` schema.

    The project assumes all fitting datasets are stored as `.npz` files with at
    least a `joints_3d` array of shape `[N, J, 3]` in the target skeleton's
    joint order. Optional arrays are:

    - `joints_2d`: shape `[N, J, 2]`
    - `confidences`: shape `[N, J]` or broadcast-compatible to the 2D/3D loss
    - Camera tensors named `fx`, `fy`, `cx`, `cy`, `camera_translation`,
      `camera_rotation`

    Any other keys are preserved in `metadata`.

    Args:
        joints_3d: World-space 3D joints with shape `[N, J, 3]`.
        joints_2d: Optional image-space joints with shape `[N, J, 2]`.
        confidences: Optional per-joint confidence weights.
        cameras: Optional camera tensors keyed by the shared dataset names.
        metadata: Additional arrays or scalars from the source dataset.
        expected_num_joints: Optional joint-count check for a specific skeleton.
    """

    def __init__(
        self,
        joints_3d: torch.Tensor,
        *,
        joints_2d: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        cameras: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
        expected_num_joints: int | None = None,
    ) -> None:
        _validate_joint_tensor('joints_3d', joints_3d, ndim=3, expected_num_joints=expected_num_joints)
        self.joints_3d = joints_3d.float()
        self.joints_2d = joints_2d.float() if joints_2d is not None else None
        self.confidences = confidences.float() if confidences is not None else None
        self.cameras = {key: value.float() for key, value in (cameras or {}).items()}
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return int(self.joints_3d.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {'joints_3d': self.joints_3d[index]}
        if self.joints_2d is not None:
            item['joints_2d'] = self.joints_2d[index]
        if self.confidences is not None:
            item['confidences'] = self.confidences[index]
        for key, value in self.cameras.items():
            item[key] = value[index] if value.shape[0] == len(self) else value
        return item

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_num_joints: int | None = None,
    ) -> 'FrameDataset':
        """Construct a frame dataset from the shared `.npz` fitting schema.

        Args:
            path: Path to the `.npz` payload.
            expected_num_joints: Optional joint-count check for the payload.

        Returns:
            A dataset instance backed by tensors loaded from the archive.
        """

        with np.load(path, allow_pickle=True) as payload:
            joints_3d_key = 'joints_3d' if 'joints_3d' in payload else 'S' if 'S' in payload else None
            if joints_3d_key is None:
                raise ValueError('No joints_3d or S array found in the .npz payload.')
            joints_3d = torch.from_numpy(payload[joints_3d_key]).float()
            if joints_3d.shape[-1] == 4:
                joints_3d = joints_3d[..., :3]
            joints_2d = torch.from_numpy(payload['joints_2d']).float() if 'joints_2d' in payload else None
            confidences = torch.from_numpy(payload['confidences']).float() if 'confidences' in payload else None
            cameras = {
                key: torch.from_numpy(payload[key]).float()
                for key in CAMERA_KEYS
                if key in payload
            }
            metadata = {
                key: _clone_metadata_value(payload[key])
                for key in payload.files
                if key not in {'joints_3d', 'joints_2d', 'confidences', *CAMERA_KEYS}
            }
        return cls(
            joints_3d,
            joints_2d=joints_2d,
            confidences=confidences,
            cameras=cameras,
            metadata=metadata,
            expected_num_joints=expected_num_joints,
        )


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Sequence fitting dataset loaded from the shared `.npz` schema.

    The `.npz` layout mirrors :class:`FrameDataset`, except `joints_3d` is
    stored as `[N, T, J, 3]` and optional `joints_2d` / `confidences` follow
    the same leading dimensions.

    Args:
        joints_3d: World-space 3D joints with shape `[N, T, J, 3]`.
        joints_2d: Optional image-space joints with shape `[N, T, J, 2]`.
        confidences: Optional per-joint confidence weights.
        cameras: Optional camera tensors keyed by the shared dataset names.
        metadata: Additional arrays or scalars from the source dataset.
        expected_num_joints: Optional joint-count check for a specific skeleton.
    """

    def __init__(
        self,
        joints_3d: torch.Tensor,
        *,
        joints_2d: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        cameras: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
        expected_num_joints: int | None = None,
    ) -> None:
        _validate_joint_tensor('joints_3d', joints_3d, ndim=4, expected_num_joints=expected_num_joints)
        self.joints_3d = joints_3d.float()
        self.joints_2d = joints_2d.float() if joints_2d is not None else None
        self.confidences = confidences.float() if confidences is not None else None
        self.cameras = {key: value.float() for key, value in (cameras or {}).items()}
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return int(self.joints_3d.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {'joints_3d': self.joints_3d[index]}
        if self.joints_2d is not None:
            item['joints_2d'] = self.joints_2d[index]
        if self.confidences is not None:
            item['confidences'] = self.confidences[index]
        for key, value in self.cameras.items():
            item[key] = value[index] if value.shape[0] == len(self) else value
        return item

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_num_joints: int | None = None,
    ) -> 'SequenceDataset':
        """Construct a sequence dataset from the shared `.npz` fitting schema.

        Args:
            path: Path to the `.npz` payload.
            expected_num_joints: Optional joint-count check for the payload.

        Returns:
            A dataset instance backed by tensors loaded from the archive.
        """

        with np.load(path, allow_pickle=True) as payload:
            joints_3d = torch.from_numpy(payload['joints_3d']).float()
            joints_2d = torch.from_numpy(payload['joints_2d']).float() if 'joints_2d' in payload else None
            confidences = torch.from_numpy(payload['confidences']).float() if 'confidences' in payload else None
            cameras = {
                key: torch.from_numpy(payload[key]).float()
                for key in CAMERA_KEYS
                if key in payload
            }
            metadata = {
                key: _clone_metadata_value(payload[key])
                for key in payload.files
                if key not in {'joints_3d', 'joints_2d', 'confidences', *CAMERA_KEYS}
            }
        return cls(
            joints_3d,
            joints_2d=joints_2d,
            confidences=confidences,
            cameras=cameras,
            metadata=metadata,
            expected_num_joints=expected_num_joints,
        )


def prepare_frame_batch(
    joints_3d: torch.Tensor,
    *,
    model: SkeletalModel,
    estimate_bone_scales: bool = True,
) -> FittingDataBatch:
    """Convert frame-wise joint targets into model-aligned pose parameters.

    Args:
        joints_3d: Joint targets with shape `[..., J, 3]` using `model` order.
        model: Skeleton model defining joint topology and rest offsets.
        estimate_bone_scales: Whether to estimate per-body scale factors from
            the target joints. When `False`, unit body scales are used.

    Returns:
        A :class:`FittingDataBatch` with root pose, body pose, and scales.
    """

    if joints_3d.shape[-2:] != (model.num_joints, 3):
        raise ValueError(f'joints_3d must have shape [..., {model.num_joints}, 3].')
    joints_3d = joints_3d.float()
    if estimate_bone_scales:
        bone_scales = estimate_bone_scales_from_joints(joints_3d, model)
    else:
        bone_scales = torch.ones(
            joints_3d.shape[:-2] + (model.num_joints, 3),
            dtype=joints_3d.dtype,
            device=joints_3d.device,
        )
    ik = estimate_rotations_from_joints(joints_3d, model, bone_scales=bone_scales)
    local_rot6d = matrix_to_rot6d(ik.local_rotations)
    return FittingDataBatch(
        joints_3d=joints_3d,
        body_pose_rot6d=local_rot6d[..., list(model.non_root_joint_indices), :],
        global_orient_rot6d=local_rot6d[..., model.root_index, :],
        bone_scales=ik.bone_scales,
    )


def prepare_sequence_batch(
    joints_3d: torch.Tensor,
    *,
    model: SkeletalModel,
    estimate_bone_scales: bool = True,
) -> FittingDataBatch:
    """Convert sequence joint targets into model-aligned pose parameters.

    Args:
        joints_3d: Joint targets with shape `[..., T, J, 3]` using `model`
            order.
        model: Skeleton model defining joint topology and rest offsets.
        estimate_bone_scales: Whether to estimate per-body scale factors from
            the target joints. When `False`, unit body scales are used.

    Returns:
        A :class:`FittingDataBatch` with root pose, body pose, and scales.
    """

    if joints_3d.ndim < 4 or joints_3d.shape[-2:] != (model.num_joints, 3):
        raise ValueError(f'joints_3d must have shape [..., T, {model.num_joints}, 3].')
    flat = joints_3d.reshape(-1, joints_3d.shape[-2], joints_3d.shape[-1])
    batch = prepare_frame_batch(flat, model=model, estimate_bone_scales=estimate_bone_scales)
    seq_shape = joints_3d.shape[:-2]
    return FittingDataBatch(
        joints_3d=joints_3d.float(),
        body_pose_rot6d=batch.body_pose_rot6d.reshape(seq_shape + batch.body_pose_rot6d.shape[-2:]),
        global_orient_rot6d=batch.global_orient_rot6d.reshape(seq_shape + batch.global_orient_rot6d.shape[-1:]),
        bone_scales=batch.bone_scales.reshape(seq_shape + batch.bone_scales.shape[-2:]),
    )
