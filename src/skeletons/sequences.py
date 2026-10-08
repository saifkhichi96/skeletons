from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .fitting import (
    FittingWeights,
    JointLimitPrior,
    MotionSmoothnessPrior,
    PerspectiveCamera,
    PoseVAE,
    SequenceDataset,
    SkeletalFitter,
    TemporalPrior,
    TemporalPriorTrainer,
    TrainerState,
    WeakPerspectiveCamera,
)
from .model import SkeletalModel

Camera = PerspectiveCamera | WeakPerspectiveCamera


@dataclass(frozen=True)
class TemporalPriorTrainingResult:
    """Temporal prior and epoch metrics produced from sequence training data.

    Parameters
    ----------
    temporal_prior : TemporalPrior
        Trained autoregressive pose prior.
    history : tuple[TrainerState, ...]
        Per-epoch temporal-prior training summaries.
    """

    temporal_prior: TemporalPrior
    history: tuple[TrainerState, ...]


@dataclass(frozen=True)
class SequenceFittingResult:
    """Result returned by high-level sequence fitting helpers.

    Parameters
    ----------
    joints_3d : torch.Tensor
        Fitted 3D joints with shape ``[batch, frames, joints, 3]``.
    losses : dict[str, float]
        Final fitting loss components.
    iterations : int
        Optimizer iterations completed by the underlying fitter.
    target_joints_3d : torch.Tensor or None
        Optional 3D fitting target with shape ``[batch, frames, joints, 3]``.
    target_joints_2d : torch.Tensor or None
        Optional 2D fitting target with shape ``[batch, frames, joints, 2]``.
    projected_joints_2d : torch.Tensor or None
        Optional projection of fitted 3D joints.
    confidences : torch.Tensor or None
        Optional 2D confidence weights with shape ``[batch, frames, joints]``.
    reprojection_errors : torch.Tensor or None
        Optional per-frame mean squared reprojection error with shape
        ``[batch, frames]``.
    """

    joints_3d: Tensor
    losses: dict[str, float]
    iterations: int
    target_joints_3d: Tensor | None = None
    target_joints_2d: Tensor | None = None
    projected_joints_2d: Tensor | None = None
    confidences: Tensor | None = None
    reprojection_errors: Tensor | None = None

    @property
    def mean_reprojection_error(self) -> float:
        """Return the mean per-frame reprojection error.

        Returns
        -------
        float
            Mean of ``reprojection_errors``.

        Raises
        ------
        ValueError
            If the result was not produced from 2D observations.
        """

        if self.reprojection_errors is None:
            raise ValueError("reprojection_errors are only available for 2D fitting.")
        return float(self.reprojection_errors.mean().item())

    def to_sequence_dataset(self) -> SequenceDataset:
        """Convert fitted joints into a sequence dataset.

        Returns
        -------
        SequenceDataset
            Dataset containing fitted 3D joints plus available 2D observations
            and confidences.
        """

        return SequenceDataset(
            self.joints_3d.cpu(),
            joints_2d=self.target_joints_2d.cpu()
            if self.target_joints_2d is not None
            else None,
            confidences=self.confidences.cpu() if self.confidences is not None else None,
            expected_num_joints=self.joints_3d.shape[-2],
        )


def _require_positive_int(value: int, name: str) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be positive.")


def _validate_sequence_joints_3d(joints_3d: Tensor, model: SkeletalModel) -> None:
    if joints_3d.ndim != 4 or joints_3d.shape[-2:] != (model.NUM_JOINTS, 3):
        raise ValueError(
            f"joints_3d must have shape [N, T, {model.NUM_JOINTS}, 3]."
        )
    if joints_3d.shape[0] < 1:
        raise ValueError("joints_3d must contain at least one sequence.")
    if joints_3d.shape[1] < 2:
        raise ValueError("joints_3d sequences must contain at least two frames.")


def _validate_sequence_joints_2d(
    joints_2d: Tensor,
    confidences: Tensor | None,
    model: SkeletalModel,
) -> None:
    if joints_2d.ndim != 4 or joints_2d.shape[-2:] != (model.NUM_JOINTS, 2):
        raise ValueError(
            f"joints_2d must have shape [N, T, {model.NUM_JOINTS}, 2]."
        )
    if joints_2d.shape[0] < 1:
        raise ValueError("joints_2d must contain at least one sequence.")
    if joints_2d.shape[1] < 2:
        raise ValueError("joints_2d sequences must contain at least two frames.")
    if confidences is None:
        return
    if confidences.shape != joints_2d.shape[:-1]:
        raise ValueError(
            "confidences must have shape matching joints_2d without the final "
            f"coordinate dimension, got {tuple(confidences.shape)} and "
            f"{tuple(joints_2d.shape)}."
        )


def _per_frame_reprojection_error(
    projected: Tensor,
    target: Tensor,
    confidences: Tensor | None,
) -> Tensor:
    squared = (projected - target).pow(2).sum(dim=-1)
    if confidences is not None:
        squared = squared * confidences
    return squared.mean(dim=-1)


def train_temporal_prior(
    joints_3d: Tensor,
    *,
    model: SkeletalModel,
    batch_size: int = 16,
    epochs: int = 6,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.0,
    device: torch.device | str = "cpu",
    lr: float = 1e-3,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> TemporalPriorTrainingResult:
    """Train a temporal prior from 3D joint sequences.

    Parameters
    ----------
    joints_3d : torch.Tensor
        Training sequences with shape ``[sequences, frames, joints, 3]``.
    model : SkeletalModel
        Skeleton model whose topology defines the pose representation.
    batch_size : int, optional
        Number of sequences per training batch.
    epochs : int, optional
        Number of temporal-prior training epochs. Use ``0`` to skip training.
    hidden_dim : int, optional
        Temporal GRU hidden dimension.
    num_layers : int, optional
        Number of GRU layers.
    dropout : float, optional
        Dropout applied between GRU layers when ``num_layers`` is greater than 1.
    device : torch.device or str, optional
        Device used for training.
    lr : float, optional
        Optimizer learning rate.
    shuffle : bool, optional
        Whether to shuffle sequences while training.
    generator : torch.Generator, optional
        Generator used by the dataloader sampler for deterministic shuffling.

    Returns
    -------
    TemporalPriorTrainingResult
        Trained temporal prior and epoch metrics.

    Raises
    ------
    ValueError
        If the sequence tensor or scalar options are invalid.
    """

    _validate_sequence_joints_3d(joints_3d, model)
    _require_positive_int(batch_size, "batch_size")
    if int(epochs) < 0:
        raise ValueError("epochs must be non-negative.")
    _require_positive_int(hidden_dim, "hidden_dim")
    _require_positive_int(num_layers, "num_layers")
    if dropout < 0.0:
        raise ValueError("dropout must be non-negative.")

    dataset = SequenceDataset(
        joints_3d.detach().cpu(),
        expected_num_joints=model.NUM_JOINTS,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
    )
    temporal_prior = TemporalPrior(
        pose_dim=(model.NUM_JOINTS - 1) * 6,
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
    )
    trainer = TemporalPriorTrainer(
        temporal_prior,
        model=model,
        device=device,
        lr=lr,
    )

    history: list[TrainerState] = []
    for epoch in range(1, int(epochs) + 1):
        state = trainer.train_epoch(loader)
        history.append(
            TrainerState(epoch=epoch, loss=state.loss, metrics=dict(state.metrics))
        )
    return TemporalPriorTrainingResult(
        temporal_prior=temporal_prior,
        history=tuple(history),
    )


def denoise_joint_sequences(
    noisy_joints_3d: Tensor,
    *,
    model: SkeletalModel,
    weights: Tensor | None = None,
    pose_prior: PoseVAE | None = None,
    joint_limit_prior: JointLimitPrior | None = None,
    temporal_prior: TemporalPrior | None = None,
    smoothness_prior: MotionSmoothnessPrior | None = None,
    num_iters: int = 180,
    lr: float = 1e-2,
    optimize_scales: bool = False,
    use_pose_prior_latent: bool = False,
    fitting_weights: FittingWeights | None = None,
    device: torch.device | str = "cpu",
) -> SequenceFittingResult:
    """Denoise 3D joint sequences through temporally regularized fitting.

    Parameters
    ----------
    noisy_joints_3d : torch.Tensor
        Noisy 3D target sequences with shape ``[batch, frames, joints, 3]``.
    model : SkeletalModel
        Skeleton model to optimize.
    weights : torch.Tensor, optional
        Optional per-joint fitting weights broadcastable to ``[batch, frames,
        joints]``.
    pose_prior : PoseVAE, optional
        Optional latent pose prior used during fitting.
    joint_limit_prior : JointLimitPrior, optional
        Optional joint-limit prior used during fitting.
    temporal_prior : TemporalPrior, optional
        Optional learned temporal prior.
    smoothness_prior : MotionSmoothnessPrior, optional
        Optional explicit smoothness prior. The fitter default is used when
        omitted.
    num_iters : int, optional
        Number of optimizer iterations.
    lr : float, optional
        Optimizer learning rate.
    optimize_scales : bool, optional
        Whether body scales are optimized.
    use_pose_prior_latent : bool, optional
        Whether to use latent pose optimization if the fitter has a pose prior.
    fitting_weights : FittingWeights, optional
        Objective weights passed to ``SkeletalFitter.fit_sequence_3d``.
    device : torch.device or str, optional
        Device used for optimization.

    Returns
    -------
    SequenceFittingResult
        Denoised 3D joints and fitting diagnostics.

    Raises
    ------
    ValueError
        If the sequence tensor or scalar options are invalid.
    """

    _validate_sequence_joints_3d(noisy_joints_3d, model)
    _require_positive_int(num_iters, "num_iters")

    device = torch.device(device)
    fitter = SkeletalFitter(
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        temporal_prior=temporal_prior,
        smoothness_prior=smoothness_prior,
        device=device,
    )
    result = fitter.fit_sequence_3d(
        noisy_joints_3d.to(device),
        weights=weights.to(device) if weights is not None else None,
        num_iters=int(num_iters),
        lr=lr,
        optimize_scales=optimize_scales,
        use_pose_prior_latent=use_pose_prior_latent,
        fitting_weights=fitting_weights,
    )
    fitted = result.model_output.joints.reshape_as(noisy_joints_3d).detach().cpu()
    return SequenceFittingResult(
        joints_3d=fitted,
        losses=dict(result.losses),
        iterations=int(result.iterations),
        target_joints_3d=noisy_joints_3d.detach().cpu(),
    )


def fit_2d_joint_sequences(
    joints_2d: Tensor,
    camera: Camera,
    *,
    model: SkeletalModel,
    confidences: Tensor | None = None,
    pose_prior: PoseVAE | None = None,
    joint_limit_prior: JointLimitPrior | None = None,
    temporal_prior: TemporalPrior | None = None,
    smoothness_prior: MotionSmoothnessPrior | None = None,
    num_iters: int = 220,
    lr: float = 5e-3,
    optimize_scales: bool = False,
    use_pose_prior_latent: bool = False,
    init_depth: float = 3000.0,
    fitting_weights: FittingWeights | None = None,
    device: torch.device | str = "cpu",
) -> SequenceFittingResult:
    """Reconstruct 3D joint sequences from 2D observations.

    Parameters
    ----------
    joints_2d : torch.Tensor
        Observed 2D sequences with shape ``[batch, frames, joints, 2]``.
    camera : PerspectiveCamera or WeakPerspectiveCamera
        Camera used to project fitted 3D joints.
    model : SkeletalModel
        Skeleton model to optimize.
    confidences : torch.Tensor, optional
        Per-joint confidence weights with shape ``[batch, frames, joints]``.
    pose_prior : PoseVAE, optional
        Optional latent pose prior used during fitting.
    joint_limit_prior : JointLimitPrior, optional
        Optional joint-limit prior used during fitting.
    temporal_prior : TemporalPrior, optional
        Optional learned temporal prior.
    smoothness_prior : MotionSmoothnessPrior, optional
        Optional explicit smoothness prior. The fitter default is used when
        omitted.
    num_iters : int, optional
        Number of optimizer iterations.
    lr : float, optional
        Optimizer learning rate.
    optimize_scales : bool, optional
        Whether body scales are optimized.
    use_pose_prior_latent : bool, optional
        Whether to use latent pose optimization if the fitter has a pose prior.
    init_depth : float, optional
        Initial z-translation for perspective fitting.
    fitting_weights : FittingWeights, optional
        Objective weights passed to ``SkeletalFitter.fit_sequence_2d``.
    device : torch.device or str, optional
        Device used for optimization.

    Returns
    -------
    SequenceFittingResult
        Reconstructed 3D joints and reprojection diagnostics.

    Raises
    ------
    ValueError
        If observation shapes or scalar options are invalid.
    """

    _validate_sequence_joints_2d(joints_2d, confidences, model)
    _require_positive_int(num_iters, "num_iters")

    device = torch.device(device)
    camera = camera.to(device)
    target = joints_2d.to(device)
    confidence_values = confidences.to(device) if confidences is not None else None
    fitter = SkeletalFitter(
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        temporal_prior=temporal_prior,
        smoothness_prior=smoothness_prior,
        device=device,
    )
    result = fitter.fit_sequence_2d(
        target,
        camera,
        confidences=confidence_values,
        num_iters=int(num_iters),
        lr=lr,
        optimize_scales=optimize_scales,
        use_pose_prior_latent=use_pose_prior_latent,
        init_depth=init_depth,
        fitting_weights=fitting_weights,
    )
    joints_3d = result.model_output.joints.reshape(
        joints_2d.shape[:-1] + (3,)
    ).detach()
    projected = camera.project(joints_3d).detach()
    reprojection_errors = _per_frame_reprojection_error(
        projected,
        target,
        confidence_values,
    )
    return SequenceFittingResult(
        joints_3d=joints_3d.cpu(),
        losses=dict(result.losses),
        iterations=int(result.iterations),
        target_joints_2d=joints_2d.detach().cpu(),
        projected_joints_2d=projected.cpu(),
        confidences=confidences.detach().cpu() if confidences is not None else None,
        reprojection_errors=reprojection_errors.cpu(),
    )
