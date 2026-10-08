from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..ik import estimate_rotations_from_joints, estimate_scales_from_joints
from ..model import SkeletalModel
from ..rotations import matrix_to_rot6d

NPZ_INPUT_KEY_CANDIDATES = {
    "joints_3d": (
        "joints_3d",
        "joints3d",
        "keypoints_3d",
        "keypoints3d",
        "pose_3d",
        "pose3d",
        "positions_3d",
        "positions3d",
        "part_3d",
        "part3d",
        "S",
    ),
    "joints_2d": (
        "joints_2d",
        "joints2d",
        "keypoints_2d",
        "keypoints2d",
        "pose_2d",
        "pose2d",
        "part_2d",
        "part2d",
        "part",
    ),
    "confidences": (
        "confidences",
        "confidence",
        "conf",
        "confs",
        "scores",
        "score",
        "weights",
        "weight",
        "visibility",
        "visibilities",
    ),
    "fx": ("fx", "f_x", "focal_length_x", "focalx"),
    "fy": ("fy", "f_y", "focal_length_y", "focaly"),
    "cx": ("cx", "c_x", "principal_point_x", "principalx"),
    "cy": ("cy", "c_y", "principal_point_y", "principaly"),
    "camera_translation": (
        "camera_translation",
        "camera_transl",
        "camera_t",
        "cam_t",
        "cam_translation",
    ),
    "camera_rotation": (
        "camera_rotation",
        "camera_rot",
        "camera_r",
        "cam_r",
        "cam_rotation",
    ),
}

CAMERA_KEYS = ("fx", "fy", "cx", "cy", "camera_translation", "camera_rotation")


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
        raise ValueError(
            f"{name} must have shape [..., J, 3] with {ndim} dimensions, got {tuple(joints.shape)}."
        )
    if expected_num_joints is not None and joints.shape[-2] != expected_num_joints:
        raise ValueError(
            f"{name} must contain {expected_num_joints} joints, got {joints.shape[-2]}."
        )


def _normalize_npz_key(key: str) -> str:
    return "".join(char for char in key.lower() if char.isalnum())


def _build_normalized_payload_key_map(payload: Any) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in payload.files:
        normalized_key = _normalize_npz_key(key)
        existing = normalized.get(normalized_key)
        if existing is not None and existing != key:
            raise ValueError(
                "The .npz payload contains ambiguous keys after normalization: "
                f"{existing!r} and {key!r} both map to {normalized_key!r}."
            )
        normalized[normalized_key] = key
    return normalized


def _resolve_npz_input_keys(payload: Any) -> dict[str, str | None]:
    normalized_keys = _build_normalized_payload_key_map(payload)
    resolved: dict[str, str | None] = {}
    for canonical_name, candidates in NPZ_INPUT_KEY_CANDIDATES.items():
        resolved[canonical_name] = next(
            (
                normalized_keys[_normalize_npz_key(candidate)]
                for candidate in candidates
                if _normalize_npz_key(candidate) in normalized_keys
            ),
            None,
        )
    return resolved


def _require_npz_input_key(
    resolved_keys: dict[str, str | None],
    canonical_name: str,
) -> str:
    resolved_key = resolved_keys[canonical_name]
    if resolved_key is not None:
        return resolved_key
    candidates = ", ".join(NPZ_INPUT_KEY_CANDIDATES[canonical_name])
    raise ValueError(
        f"No {canonical_name} array found in the .npz payload. Tried aliases: {candidates}."
    )


def _load_npz_tensor(payload: Any, key: str | None) -> torch.Tensor | None:
    if key is None:
        return None
    return torch.from_numpy(payload[key]).float()


def _slice_or_keep_constant(
    value: torch.Tensor,
    *,
    index: int,
    length: int,
) -> torch.Tensor:
    if value.ndim > 0 and value.shape[0] == length:
        return value[index]
    return value


def _canonicalize_loaded_joint_inputs(
    joints_3d: torch.Tensor,
    joints_2d: torch.Tensor | None,
    confidences: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if joints_3d.shape[-1] == 4:
        joints_3d = joints_3d[..., :3]

    if joints_2d is not None and joints_2d.shape[-1] == 3:
        if confidences is None:
            confidences = joints_2d[..., 2]
        joints_2d = joints_2d[..., :2]

    return joints_3d, joints_2d, confidences


@dataclass
class FittingDataBatch:
    """Prepared fitting targets for a concrete skeleton model."""

    joints_3d: torch.Tensor
    body_pose_rot6d: torch.Tensor
    global_orient_rot6d: torch.Tensor
    scales: torch.Tensor


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
        _validate_joint_tensor(
            "joints_3d", joints_3d, ndim=3, expected_num_joints=expected_num_joints
        )
        self.joints_3d = joints_3d.float()
        self.joints_2d = joints_2d.float() if joints_2d is not None else None
        self.confidences = confidences.float() if confidences is not None else None
        self.cameras = {key: value.float() for key, value in (cameras or {}).items()}
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return int(self.joints_3d.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"joints_3d": self.joints_3d[index]}
        if self.joints_2d is not None:
            item["joints_2d"] = self.joints_2d[index]
        if self.confidences is not None:
            item["confidences"] = self.confidences[index]
        for key, value in self.cameras.items():
            item[key] = _slice_or_keep_constant(value, index=index, length=len(self))
        return item

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_num_joints: int | None = None,
    ) -> "FrameDataset":
        """Construct a frame dataset from the shared `.npz` fitting schema.

        Args:
            path: Path to the `.npz` payload.
            expected_num_joints: Optional joint-count check for the payload.

        Returns:
            A dataset instance backed by tensors loaded from the archive.
        """

        with np.load(path, allow_pickle=True) as payload:
            resolved_keys = _resolve_npz_input_keys(payload)
            joints_3d = _load_npz_tensor(
                payload,
                _require_npz_input_key(resolved_keys, "joints_3d"),
            )
            assert joints_3d is not None
            joints_2d = _load_npz_tensor(payload, resolved_keys["joints_2d"])
            confidences = _load_npz_tensor(payload, resolved_keys["confidences"])
            joints_3d, joints_2d, confidences = _canonicalize_loaded_joint_inputs(
                joints_3d, joints_2d, confidences
            )
            cameras = {
                canonical_key: camera_tensor
                for canonical_key in CAMERA_KEYS
                if (
                    camera_tensor := _load_npz_tensor(
                        payload, resolved_keys[canonical_key]
                    )
                )
                is not None
            }
            used_keys = {
                resolved_key
                for resolved_key in resolved_keys.values()
                if resolved_key is not None
            }
            metadata = {
                key: _clone_metadata_value(payload[key])
                for key in payload.files
                if key not in used_keys
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
        _validate_joint_tensor(
            "joints_3d", joints_3d, ndim=4, expected_num_joints=expected_num_joints
        )
        self.joints_3d = joints_3d.float()
        self.joints_2d = joints_2d.float() if joints_2d is not None else None
        self.confidences = confidences.float() if confidences is not None else None
        self.cameras = {key: value.float() for key, value in (cameras or {}).items()}
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return int(self.joints_3d.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"joints_3d": self.joints_3d[index]}
        if self.joints_2d is not None:
            item["joints_2d"] = self.joints_2d[index]
        if self.confidences is not None:
            item["confidences"] = self.confidences[index]
        for key, value in self.cameras.items():
            item[key] = _slice_or_keep_constant(value, index=index, length=len(self))

        return item

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        expected_num_joints: int | None = None,
    ) -> "SequenceDataset":
        """Construct a sequence dataset from the shared `.npz` fitting schema.

        Args:
            path: Path to the `.npz` payload.
            expected_num_joints: Optional joint-count check for the payload.

        Returns:
            A dataset instance backed by tensors loaded from the archive.
        """

        with np.load(path, allow_pickle=True) as payload:
            resolved_keys = _resolve_npz_input_keys(payload)
            joints_3d = _load_npz_tensor(
                payload,
                _require_npz_input_key(resolved_keys, "joints_3d"),
            )
            assert joints_3d is not None
            joints_2d = _load_npz_tensor(payload, resolved_keys["joints_2d"])
            confidences = _load_npz_tensor(payload, resolved_keys["confidences"])
            joints_3d, joints_2d, confidences = _canonicalize_loaded_joint_inputs(
                joints_3d, joints_2d, confidences
            )
            cameras = {
                canonical_key: camera_tensor
                for canonical_key in CAMERA_KEYS
                if (
                    camera_tensor := _load_npz_tensor(
                        payload, resolved_keys[canonical_key]
                    )
                )
                is not None
            }
            used_keys = {
                resolved_key
                for resolved_key in resolved_keys.values()
                if resolved_key is not None
            }
            metadata = {
                key: _clone_metadata_value(payload[key])
                for key in payload.files
                if key not in used_keys
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
    estimate_scales: bool = True,
) -> FittingDataBatch:
    """Convert frame-wise joint targets into model-aligned pose parameters.

    Args:
        joints_3d: Joint targets with shape `[..., J, 3]` using `model` order.
        model: Skeleton model defining joint topology and rest offsets.
        estimate_scales: Whether to estimate per-body scale factors from
            the target joints. When `False`, unit body scales are used.

    Returns:
        A :class:`FittingDataBatch` with root pose, body pose, and scales.
    """

    if joints_3d.shape[-2:] != (model.NUM_JOINTS, 3):
        raise ValueError(f"joints_3d must have shape [..., {model.NUM_JOINTS}, 3].")
    joints_3d = joints_3d.float()
    if estimate_scales:
        scales = estimate_scales_from_joints(joints_3d, model)
    else:
        scales = torch.ones(
            joints_3d.shape[:-2] + (model.NUM_JOINTS, 3),
            dtype=joints_3d.dtype,
            device=joints_3d.device,
        )
    ik = estimate_rotations_from_joints(joints_3d, model, scales=scales)
    local_rot6d = matrix_to_rot6d(ik.local_rotations)
    return FittingDataBatch(
        joints_3d=joints_3d,
        body_pose_rot6d=local_rot6d[..., list(model.non_root_joint_indices), :],
        global_orient_rot6d=local_rot6d[..., model.root_index, :],
        scales=ik.scales,
    )


def prepare_sequence_batch(
    joints_3d: torch.Tensor,
    *,
    model: SkeletalModel,
    estimate_scales: bool = True,
) -> FittingDataBatch:
    """Convert sequence joint targets into model-aligned pose parameters.

    Args:
        joints_3d: Joint targets with shape `[..., T, J, 3]` using `model`
            order.
        model: Skeleton model defining joint topology and rest offsets.
        estimate_scales: Whether to estimate per-body scale factors from
            the target joints. When `False`, unit body scales are used.

    Returns:
        A :class:`FittingDataBatch` with root pose, body pose, and scales.
    """

    if joints_3d.ndim < 4 or joints_3d.shape[-2:] != (model.NUM_JOINTS, 3):
        raise ValueError(f"joints_3d must have shape [..., T, {model.NUM_JOINTS}, 3].")
    flat = joints_3d.reshape(-1, joints_3d.shape[-2], joints_3d.shape[-1])
    batch = prepare_frame_batch(flat, model=model, estimate_scales=estimate_scales)
    seq_shape = joints_3d.shape[:-2]
    return FittingDataBatch(
        joints_3d=joints_3d.float(),
        body_pose_rot6d=batch.body_pose_rot6d.reshape(
            seq_shape + batch.body_pose_rot6d.shape[-2:]
        ),
        global_orient_rot6d=batch.global_orient_rot6d.reshape(
            seq_shape + batch.global_orient_rot6d.shape[-1:]
        ),
        scales=batch.scales.reshape(seq_shape + batch.scales.shape[-2:]),
    )
