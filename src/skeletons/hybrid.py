from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from .ik import InverseKinematicsResult, estimate_rotations_from_joints
from .metrics import mean_per_joint_position_error
from .model import SkeletalModel

Tensor = torch.Tensor


class JointRegressor(nn.Module):
    """Feed-forward 2D-to-3D joint regressor for hybrid IK pipelines.

    Parameters
    ----------
    num_joints : int
        Number of input and output joints.
    hidden_dim : int, optional
        Hidden layer width.
    """

    def __init__(self, num_joints: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.net = nn.Sequential(
            nn.Linear(num_joints * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3),
        )

    def forward(self, keypoints_2d: Tensor) -> Tensor:
        """Predict 3D joints from 2D keypoints.

        Parameters
        ----------
        keypoints_2d : torch.Tensor
            2D keypoints with shape ``[batch, joints, 2]``.

        Returns
        -------
        torch.Tensor
            Predicted joints with shape ``[batch, joints, 3]``.
        """

        batch_size = keypoints_2d.shape[0]
        return self.net(keypoints_2d.reshape(batch_size, -1)).reshape(
            batch_size,
            self.num_joints,
            3,
        )


@dataclass(frozen=True)
class HybridIKResult:
    """Output from analytical IK recovery over predicted joints.

    Parameters
    ----------
    predicted_joints : torch.Tensor
        Input predicted joints.
    ik : object
        Estimated local/global rotations and scales.
    reconstructed_joints : torch.Tensor
        FK reconstruction from the estimated rotations and scales.
    """

    predicted_joints: Tensor
    ik: InverseKinematicsResult
    reconstructed_joints: Tensor


@dataclass(frozen=True)
class HybridIKEvaluation:
    """Aggregate metrics for a hybrid IK regressor.

    Parameters
    ----------
    joint_mpjpe : float
        MPJPE between regressed joints and target joints.
    fk_consistency_mpjpe : float
        MPJPE between FK reconstruction and regressed joints.
    num_samples : int
        Number of samples evaluated.
    """

    joint_mpjpe: float
    fk_consistency_mpjpe: float
    num_samples: int


@dataclass(frozen=True)
class HybridRegressorTrainingState:
    """Aggregate losses for one joint-regressor training epoch.

    Parameters
    ----------
    loss : float
        Mean training loss over samples.
    joint_mse : float
        Mean squared 3D joint error over samples.
    num_samples : int
        Number of samples used for training.
    """

    loss: float
    joint_mse: float
    num_samples: int


def solve_hybrid_ik(predicted_joints: Tensor, model: SkeletalModel) -> HybridIKResult:
    """Recover rotations from predicted joints and reconstruct them with FK.

    Parameters
    ----------
    predicted_joints : torch.Tensor
        Predicted joints with shape ``[..., joints, 3]``.
    model : SkeletalModel
        Skeleton model matching ``predicted_joints``.

    Returns
    -------
    HybridIKResult
        IK estimates and FK reconstruction.

    Raises
    ------
    ValueError
        If ``predicted_joints`` does not match the model joint shape.
    """

    if predicted_joints.shape[-2:] != (model.NUM_JOINTS, 3):
        raise ValueError(
            f"predicted_joints must have shape [..., {model.NUM_JOINTS}, 3]."
        )
    ik = estimate_rotations_from_joints(predicted_joints, model)
    reconstructed = model(
        full_pose=ik.local_rotations,
        pose_repr="rotmat",
        scales=ik.scales,
    ).joints
    return HybridIKResult(
        predicted_joints=predicted_joints,
        ik=ik,
        reconstructed_joints=reconstructed,
    )


def train_joint_regressor_epoch(
    regressor: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> HybridRegressorTrainingState:
    """Train a 2D-to-3D joint regressor for one epoch.

    Parameters
    ----------
    regressor : torch.nn.Module
        Module that maps 2D keypoints to 3D joints.
    batches : Iterable[tuple[torch.Tensor, torch.Tensor]]
        Iterable yielding ``(keypoints_2d, joints_3d)`` batches.
    optimizer : torch.optim.Optimizer
        Optimizer responsible for updating ``regressor`` parameters.
    device : torch.device or str, optional
        Training device.

    Returns
    -------
    HybridRegressorTrainingState
        Mean loss and sample count for the epoch.

    Raises
    ------
    ValueError
        If no samples are observed.
    """

    device = torch.device(device)
    regressor = regressor.to(device)
    regressor.train()

    total_loss = 0.0
    total_samples = 0
    for keypoints_2d, target_joints in batches:
        keypoints_2d = keypoints_2d.to(device)
        target_joints = target_joints.to(device)
        predicted_joints = regressor(keypoints_2d)
        loss = (predicted_joints - target_joints).pow(2).sum(dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = predicted_joints.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("batches must contain at least one sample.")

    mean_loss = total_loss / total_samples
    return HybridRegressorTrainingState(
        loss=mean_loss,
        joint_mse=mean_loss,
        num_samples=total_samples,
    )


def evaluate_hybrid_ik_regressor(
    regressor: nn.Module,
    model: SkeletalModel,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str = "cpu",
) -> HybridIKEvaluation:
    """Evaluate a 2D-to-3D joint regressor with analytical IK recovery.

    Parameters
    ----------
    regressor : torch.nn.Module
        Module that maps 2D keypoints to 3D joints.
    model : SkeletalModel
        Skeleton model matching the regressor output.
    batches : Iterable[tuple[torch.Tensor, torch.Tensor]]
        Iterable yielding ``(keypoints_2d, joints_3d)`` batches.
    device : torch.device or str, optional
        Evaluation device.

    Returns
    -------
    HybridIKEvaluation
        Joint prediction and FK consistency metrics.

    Raises
    ------
    ValueError
        If no samples are evaluated.
    """

    device = torch.device(device)
    model = model.to(device)
    regressor = regressor.to(device)
    regressor.eval()

    total_joint = 0.0
    total_fk = 0.0
    total_samples = 0

    with torch.no_grad():
        for keypoints_2d, target_joints in batches:
            keypoints_2d = keypoints_2d.to(device)
            target_joints = target_joints.to(device)
            predicted_joints = regressor(keypoints_2d)
            result = solve_hybrid_ik(predicted_joints, model)
            joint_mpjpe = mean_per_joint_position_error(
                predicted_joints,
                target_joints,
            )
            fk_mpjpe = mean_per_joint_position_error(
                result.reconstructed_joints,
                predicted_joints,
            )
            batch_size = predicted_joints.shape[0]
            total_joint += float(joint_mpjpe) * batch_size
            total_fk += float(fk_mpjpe) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("batches must contain at least one sample.")

    return HybridIKEvaluation(
        joint_mpjpe=total_joint / total_samples,
        fk_consistency_mpjpe=total_fk / total_samples,
        num_samples=total_samples,
    )
