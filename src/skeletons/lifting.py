from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from .losses import ForwardKinematicsLoss
from .metrics import mean_per_joint_position_error
from .model import SkeletalModel

Tensor = torch.Tensor


class PoseLifter(nn.Module):
    """Feed-forward 2D-to-3D pose lifter baseline.

    Parameters
    ----------
    num_joints : int
        Number of input and output joints.
    hidden_dim : int, optional
        Hidden layer width.
    """

    def __init__(self, num_joints: int, hidden_dim: int = 256) -> None:
        super().__init__()
        input_dim = num_joints * 2
        output_dim = num_joints * 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.num_joints = num_joints

    def forward(self, keypoints_2d: Tensor) -> Tensor:
        """Predict axis-angle pose values from 2D keypoints.

        Parameters
        ----------
        keypoints_2d : torch.Tensor
            2D keypoints with shape ``[batch, joints, 2]``.

        Returns
        -------
        torch.Tensor
            Axis-angle pose tensor with shape ``[batch, joints, 3]``.
        """

        batch_size = keypoints_2d.shape[0]
        flat = keypoints_2d.reshape(batch_size, -1)
        return self.net(flat).reshape(batch_size, self.num_joints, 3)


class TemporalPoseLifter(nn.Module):
    """Temporal 2D-to-3D lifter baseline using 1D convolutions.

    Parameters
    ----------
    num_joints : int
        Number of input and output joints.
    hidden_dim : int, optional
        Hidden channel count.
    kernel_size : int, optional
        Temporal convolution kernel size.
    """

    def __init__(
        self, num_joints: int, hidden_dim: int = 256, kernel_size: int = 3
    ) -> None:
        super().__init__()
        input_dim = num_joints * 2
        output_dim = num_joints * 3
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding * 2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding * 2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )
        self.num_joints = num_joints

    def forward(self, keypoints_2d: Tensor) -> Tensor:
        """Predict per-frame axis-angle pose values from 2D keypoint sequences.

        Parameters
        ----------
        keypoints_2d : torch.Tensor
            2D keypoints with shape ``[batch, frames, joints, 2]``.

        Returns
        -------
        torch.Tensor
            Axis-angle pose tensor with shape ``[batch, frames, joints, 3]``.
        """

        batch_size, seq_len, _, _ = keypoints_2d.shape
        x = keypoints_2d.reshape(batch_size, seq_len, -1).transpose(1, 2)
        y = self.encoder(x).transpose(1, 2)
        return y.reshape(batch_size, seq_len, self.num_joints, 3)


@dataclass(frozen=True)
class LiftingEvaluation:
    """Aggregate evaluation metrics for a pose lifter.

    Parameters
    ----------
    fk_loss : float
        Mean forward-kinematics loss over evaluated frames.
    mpjpe : float
        Mean per-joint position error over evaluated frames.
    num_frames : int
        Number of frames evaluated.
    """

    fk_loss: float
    mpjpe: float
    num_frames: int


@dataclass(frozen=True)
class LifterTrainingState:
    """Aggregate losses for one supervised lifter training epoch.

    Parameters
    ----------
    loss : float
        Mean total training loss over frames.
    fk_loss : float
        Mean forward-kinematics loss over frames.
    pose_regularization : float
        Mean weighted pose-regularization term over frames.
    num_frames : int
        Number of frames used for training.
    """

    loss: float
    fk_loss: float
    pose_regularization: float
    num_frames: int


def _flatten_pose_and_joints(
    predicted_pose: Tensor,
    target_joints: Tensor,
    num_joints: int,
) -> tuple[Tensor, Tensor]:
    if predicted_pose.shape[-2:] != (num_joints, 3):
        raise ValueError(
            f"predicted pose must have shape [..., {num_joints}, 3], "
            f"got {tuple(predicted_pose.shape)}."
        )
    if target_joints.shape[-2:] != (num_joints, 3):
        raise ValueError(
            f"target joints must have shape [..., {num_joints}, 3], "
            f"got {tuple(target_joints.shape)}."
        )
    if predicted_pose.shape[:-2] != target_joints.shape[:-2]:
        raise ValueError(
            "predicted pose and target joints must have matching leading shapes."
        )
    return (
        predicted_pose.reshape(-1, num_joints, 3),
        target_joints.reshape(-1, num_joints, 3),
    )


def train_lifter_epoch(
    lifter: nn.Module,
    model: SkeletalModel,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    fk_loss: ForwardKinematicsLoss | None = None,
    device: torch.device | str = "cpu",
    pose_reg: float = 0.0,
) -> LifterTrainingState:
    """Train a frame or temporal pose lifter for one epoch.

    Parameters
    ----------
    lifter : torch.nn.Module
        Module that maps 2D keypoints to axis-angle pose values.
    model : SkeletalModel
        Skeleton model used to convert predicted poses to joints.
    batches : Iterable[tuple[torch.Tensor, torch.Tensor]]
        Iterable yielding ``(keypoints_2d, joints_3d)`` batches.
    optimizer : torch.optim.Optimizer
        Optimizer responsible for updating ``lifter`` parameters.
    fk_loss : ForwardKinematicsLoss, optional
        Reusable FK loss instance.
    device : torch.device or str, optional
        Training device.
    pose_reg : float, optional
        Weight for ``predicted_pose.square().mean()`` regularization.

    Returns
    -------
    LifterTrainingState
        Mean losses and frame count for the epoch.

    Raises
    ------
    ValueError
        If ``pose_reg`` is negative, no frames are observed, or tensor shapes are
        invalid.
    """

    if pose_reg < 0.0:
        raise ValueError("pose_reg must be non-negative.")

    device = torch.device(device)
    model = model.to(device)
    lifter = lifter.to(device)
    fk_loss = fk_loss or ForwardKinematicsLoss(model)
    lifter.train()

    total_loss = 0.0
    total_fk = 0.0
    total_reg = 0.0
    total_frames = 0
    for keypoints_2d, target_joints in batches:
        keypoints_2d = keypoints_2d.to(device)
        target_joints = target_joints.to(device)

        optimizer.zero_grad(set_to_none=True)
        predicted_pose = lifter(keypoints_2d)
        flat_pose, flat_joints = _flatten_pose_and_joints(
            predicted_pose,
            target_joints,
            model.NUM_JOINTS,
        )
        loss_fk = fk_loss(flat_joints, full_pose=flat_pose)
        regularization = pose_reg * predicted_pose.square().mean()
        loss = loss_fk + regularization
        loss.backward()
        optimizer.step()

        frame_count = flat_joints.shape[0]
        total_loss += float(loss.detach()) * frame_count
        total_fk += float(loss_fk.detach()) * frame_count
        total_reg += float(regularization.detach()) * frame_count
        total_frames += frame_count

    if total_frames == 0:
        raise ValueError("batches must contain at least one frame.")

    return LifterTrainingState(
        loss=total_loss / total_frames,
        fk_loss=total_fk / total_frames,
        pose_regularization=total_reg / total_frames,
        num_frames=total_frames,
    )


def evaluate_lifter(
    lifter: nn.Module,
    model: SkeletalModel,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    fk_loss: ForwardKinematicsLoss | None = None,
    device: torch.device | str = "cpu",
) -> LiftingEvaluation:
    """Evaluate a frame or temporal pose lifter.

    Parameters
    ----------
    lifter : torch.nn.Module
        Module that maps 2D keypoints to axis-angle pose values.
    model : SkeletalModel
        Skeleton model used to convert predicted poses to joints.
    batches : Iterable[tuple[torch.Tensor, torch.Tensor]]
        Iterable yielding ``(keypoints_2d, joints_3d)`` batches. Leading
        dimensions may be frame-only or sequence-shaped as long as both tensors
        end in ``[joints, channels]``.
    fk_loss : ForwardKinematicsLoss, optional
        Reusable FK loss instance.
    device : torch.device or str, optional
        Evaluation device.

    Returns
    -------
    LiftingEvaluation
        Mean FK loss, MPJPE, and evaluated frame count.

    Raises
    ------
    ValueError
        If no frames are evaluated or tensor shapes are invalid.
    """

    device = torch.device(device)
    model = model.to(device)
    lifter = lifter.to(device)
    fk_loss = fk_loss or ForwardKinematicsLoss(model)
    lifter.eval()

    total_fk = 0.0
    total_mpjpe = 0.0
    total_frames = 0

    with torch.no_grad():
        for keypoints_2d, target_joints in batches:
            keypoints_2d = keypoints_2d.to(device)
            target_joints = target_joints.to(device)
            predicted_pose = lifter(keypoints_2d)
            flat_pose, flat_joints = _flatten_pose_and_joints(
                predicted_pose,
                target_joints,
                model.NUM_JOINTS,
            )
            loss = fk_loss(flat_joints, full_pose=flat_pose)
            predicted_joints = model(full_pose=flat_pose).joints
            mpjpe = mean_per_joint_position_error(predicted_joints, flat_joints)
            frame_count = flat_joints.shape[0]
            total_fk += float(loss) * frame_count
            total_mpjpe += float(mpjpe) * frame_count
            total_frames += frame_count

    if total_frames == 0:
        raise ValueError("batches must contain at least one frame.")

    return LiftingEvaluation(
        fk_loss=total_fk / total_frames,
        mpjpe=total_mpjpe / total_frames,
        num_frames=total_frames,
    )
