from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..models import Human36MModel
from .core.data import (
    FittingDataBatch,
    FrameDataset,
    SequenceDataset,
    prepare_frame_batch,
    prepare_sequence_batch,
)
from .core.fitter import SkeletalFitter
from .core.priors import JointLimitPrior, MotionSmoothnessPrior, PoseVAE, TemporalPrior
from .core.training import JointLimitTrainer, PoseVAETrainer, TemporalPriorTrainer


def _default_h36m_model() -> Human36MModel:
    return Human36MModel(create_global_orient=False, create_body_pose=False)


class H36MFrameDataset(FrameDataset):
    """Human3.6M frame dataset wrapper for the 17-joint layout."""

    def __init__(
        self,
        joints_3d: torch.Tensor,
        *,
        joints_2d: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        cameras: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            joints_3d,
            joints_2d=joints_2d,
            confidences=confidences,
            cameras=cameras,
            metadata=metadata,
            expected_num_joints=17,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "H36MFrameDataset":
        dataset = FrameDataset.from_npz(path, expected_num_joints=17)
        return cls(
            dataset.joints_3d,
            joints_2d=dataset.joints_2d,
            confidences=dataset.confidences,
            cameras=dataset.cameras,
            metadata=dataset.metadata,
        )


class H36MSequenceDataset(SequenceDataset):
    """Human3.6M sequence dataset wrapper for the 17-joint layout."""

    def __init__(
        self,
        joints_3d: torch.Tensor,
        *,
        joints_2d: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        cameras: dict[str, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            joints_3d,
            joints_2d=joints_2d,
            confidences=confidences,
            cameras=cameras,
            metadata=metadata,
            expected_num_joints=17,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "H36MSequenceDataset":
        dataset = SequenceDataset.from_npz(path, expected_num_joints=17)
        return cls(
            dataset.joints_3d,
            joints_2d=dataset.joints_2d,
            confidences=dataset.confidences,
            cameras=dataset.cameras,
            metadata=dataset.metadata,
        )


def prepare_h36m_frame_batch(
    joints_3d: torch.Tensor,
    *,
    model: Human36MModel | None = None,
    estimate_bone_scales: bool = True,
) -> FittingDataBatch:
    """Prepare Human3.6M frame targets for fitting."""

    return prepare_frame_batch(
        joints_3d,
        model=model or _default_h36m_model(),
        estimate_bone_scales=estimate_bone_scales,
    )


def prepare_h36m_sequence_batch(
    joints_3d: torch.Tensor,
    *,
    model: Human36MModel | None = None,
    estimate_bone_scales: bool = True,
) -> FittingDataBatch:
    """Prepare Human3.6M sequence targets for fitting."""

    return prepare_sequence_batch(
        joints_3d,
        model=model or _default_h36m_model(),
        estimate_bone_scales=estimate_bone_scales,
    )


class H36MFitter(SkeletalFitter):
    """Human3.6M convenience wrapper around :class:`SkeletalFitter`."""

    def __init__(
        self,
        *,
        model: Human36MModel | None = None,
        pose_prior: PoseVAE | None = None,
        joint_limit_prior: JointLimitPrior | None = None,
        temporal_prior: TemporalPrior | None = None,
        smoothness_prior: MotionSmoothnessPrior | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            model=model or _default_h36m_model(),
            pose_prior=pose_prior,
            joint_limit_prior=joint_limit_prior,
            temporal_prior=temporal_prior,
            smoothness_prior=smoothness_prior,
            device=device,
        )


class H36MJointLimitTrainer(JointLimitTrainer):
    """Human3.6M convenience wrapper around :class:`JointLimitTrainer`."""

    def __init__(self, *, model: Human36MModel | None = None) -> None:
        super().__init__(model=model or _default_h36m_model())


class H36MPoseVAETrainer(PoseVAETrainer):
    """Human3.6M convenience wrapper around :class:`PoseVAETrainer`."""

    def __init__(
        self,
        prior: PoseVAE,
        *,
        model: Human36MModel | None = None,
        device: torch.device | str = "cpu",
        lr: float = 1e-3,
        kl_weight: float = 1e-4,
        recon_weight: float = 1.0,
    ) -> None:
        super().__init__(
            prior,
            model=model or _default_h36m_model(),
            device=device,
            lr=lr,
            kl_weight=kl_weight,
            recon_weight=recon_weight,
        )


class H36MTemporalPriorTrainer(TemporalPriorTrainer):
    """Human3.6M convenience wrapper around :class:`TemporalPriorTrainer`."""

    def __init__(
        self,
        prior: TemporalPrior,
        *,
        model: Human36MModel | None = None,
        device: torch.device | str = "cpu",
        lr: float = 1e-3,
    ) -> None:
        super().__init__(
            prior, model=model or _default_h36m_model(), device=device, lr=lr
        )


__all__ = [
    "H36MFrameDataset",
    "H36MSequenceDataset",
    "prepare_h36m_frame_batch",
    "prepare_h36m_sequence_batch",
    "H36MFitter",
    "H36MJointLimitTrainer",
    "H36MPoseVAETrainer",
    "H36MTemporalPriorTrainer",
]
