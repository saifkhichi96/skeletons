from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .fitting import (
    FittingWeights,
    FrameDataset,
    JointLimitPrior,
    JointLimitTrainer,
    PerspectiveCamera,
    PoseVAE,
    PoseVAETrainer,
    SkeletalFitter,
    TrainerState,
    WeakPerspectiveCamera,
)
from .model import SkeletalModel

Camera = PerspectiveCamera | WeakPerspectiveCamera


@dataclass(frozen=True)
class PseudoLabelPriors:
    """Priors trained for optimization-based pseudo-label generation.

    Parameters
    ----------
    pose_prior : PoseVAE
        Latent pose prior trained on bootstrap 3D joint observations.
    joint_limit_prior : JointLimitPrior
        Soft joint-limit prior estimated from bootstrap 3D joint observations.
    history : tuple[TrainerState, ...]
        Per-epoch pose-prior training summaries.
    """

    pose_prior: PoseVAE
    joint_limit_prior: JointLimitPrior
    history: tuple[TrainerState, ...]


@dataclass(frozen=True)
class PseudoLabelResult:
    """3D pseudo-labels reconstructed from 2D observations.

    Parameters
    ----------
    joints_3d : torch.Tensor
        Reconstructed joints with shape ``[*batch, joints, 3]``.
    observed_joints_2d : torch.Tensor
        Input 2D observations with shape ``[*batch, joints, 2]``.
    projected_joints_2d : torch.Tensor
        Reprojection of ``joints_3d`` with shape ``[*batch, joints, 2]``.
    confidences : torch.Tensor or None
        Optional input confidence weights with shape ``[*batch, joints]``.
    reprojection_errors : torch.Tensor
        Per-sample mean squared reprojection errors with shape ``[*batch]``.
    iterations : torch.Tensor
        Per-sample optimization-iteration counts with shape ``[*batch]``.
    fit_losses : tuple[dict[str, float], ...]
        Loss dictionaries returned by each fitted mini-batch.
    """

    joints_3d: Tensor
    observed_joints_2d: Tensor
    projected_joints_2d: Tensor
    confidences: Tensor | None
    reprojection_errors: Tensor
    iterations: Tensor
    fit_losses: tuple[dict[str, float], ...]

    @property
    def mean_reprojection_error(self) -> float:
        """Return the mean per-sample reprojection error.

        Returns
        -------
        float
            Mean of ``reprojection_errors``.
        """

        return float(self.reprojection_errors.mean().item())

    def to_frame_dataset(self) -> FrameDataset:
        """Convert pseudo-label tensors into a frame dataset.

        Returns
        -------
        FrameDataset
            Dataset containing pseudo 3D joints, observed 2D joints, and
            optional confidences.
        """

        num_joints = self.joints_3d.shape[-2]
        joints_3d = self.joints_3d.reshape(-1, num_joints, 3)
        joints_2d = self.observed_joints_2d.reshape(-1, num_joints, 2)
        confidences = (
            self.confidences.reshape(-1, num_joints)
            if self.confidences is not None
            else None
        )
        return FrameDataset(
            joints_3d.cpu(),
            joints_2d=joints_2d.cpu(),
            confidences=confidences.cpu() if confidences is not None else None,
            expected_num_joints=num_joints,
        )


def _require_positive_int(value: int, name: str) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be positive.")


def _validate_frame_joints_3d(joints_3d: Tensor, model: SkeletalModel) -> None:
    if joints_3d.ndim != 3 or joints_3d.shape[-2:] != (model.NUM_JOINTS, 3):
        raise ValueError(
            f"joints_3d must have shape [N, {model.NUM_JOINTS}, 3]."
        )
    if joints_3d.shape[0] < 1:
        raise ValueError("joints_3d must contain at least one sample.")


def _validate_observations(
    joints_2d: Tensor,
    confidences: Tensor | None,
    model: SkeletalModel,
) -> None:
    if joints_2d.ndim < 3 or joints_2d.shape[-2:] != (model.NUM_JOINTS, 2):
        raise ValueError(
            f"joints_2d must have shape [*, {model.NUM_JOINTS}, 2] with at least "
            "one batch dimension."
        )
    if joints_2d.numel() == 0:
        raise ValueError("joints_2d must contain at least one sample.")
    if confidences is None:
        return
    if confidences.shape != joints_2d.shape[:-1]:
        raise ValueError(
            "confidences must have shape matching joints_2d without the final "
            f"coordinate dimension, got {tuple(confidences.shape)} and "
            f"{tuple(joints_2d.shape)}."
        )


def _per_sample_reprojection_error(
    projected: Tensor,
    target: Tensor,
    confidences: Tensor | None,
) -> Tensor:
    squared = (projected - target).pow(2).sum(dim=-1)
    if confidences is not None:
        squared = squared * confidences
    return squared.reshape(squared.shape[0], -1).mean(dim=-1)


def _numpy_payload_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return value


def train_pseudo_label_priors(
    joints_3d: Tensor,
    *,
    model: SkeletalModel,
    batch_size: int = 32,
    epochs: int = 4,
    latent_dim: int = 16,
    hidden_dim: int = 256,
    num_hidden_layers: int = 2,
    device: torch.device | str = "cpu",
    lr: float = 1e-3,
    kl_weight: float = 1e-4,
    recon_weight: float = 1.0,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> PseudoLabelPriors:
    """Train priors used by 2D pseudo-label fitting.

    Parameters
    ----------
    joints_3d : torch.Tensor
        Bootstrap 3D joints with shape ``[samples, joints, 3]``.
    model : SkeletalModel
        Skeleton model whose topology defines the pose representation.
    batch_size : int, optional
        Number of samples per training batch.
    epochs : int, optional
        Number of pose-prior training epochs. Use ``0`` to skip VAE training.
    latent_dim : int, optional
        Pose VAE latent dimension.
    hidden_dim : int, optional
        Pose VAE hidden dimension.
    num_hidden_layers : int, optional
        Number of hidden layers in the Pose VAE encoder and decoder.
    device : torch.device or str, optional
        Device used to train the pose prior.
    lr : float, optional
        Pose VAE optimizer learning rate.
    kl_weight : float, optional
        KL term weight for VAE training.
    recon_weight : float, optional
        Reconstruction term weight for VAE training.
    shuffle : bool, optional
        Whether to shuffle bootstrap samples while training the pose prior.
    generator : torch.Generator, optional
        Generator used by the dataloader sampler for deterministic shuffling.

    Returns
    -------
    PseudoLabelPriors
        Trained pose and joint-limit priors plus epoch metrics.

    Raises
    ------
    ValueError
        If the input tensor shape or scalar options are invalid.
    """

    _validate_frame_joints_3d(joints_3d, model)
    _require_positive_int(batch_size, "batch_size")
    if int(epochs) < 0:
        raise ValueError("epochs must be non-negative.")
    _require_positive_int(latent_dim, "latent_dim")
    _require_positive_int(hidden_dim, "hidden_dim")
    _require_positive_int(num_hidden_layers, "num_hidden_layers")

    dataset = FrameDataset(
        joints_3d.detach().cpu(),
        expected_num_joints=model.NUM_JOINTS,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator,
    )
    joint_limit_prior = JointLimitTrainer(model=model).fit_from_loader(loader)
    pose_prior = PoseVAE(
        num_joints=model.NUM_JOINTS - 1,
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        num_hidden_layers=int(num_hidden_layers),
    )
    trainer = PoseVAETrainer(
        pose_prior,
        model=model,
        device=device,
        lr=lr,
        kl_weight=kl_weight,
        recon_weight=recon_weight,
    )
    history: list[TrainerState] = []
    for epoch in range(1, int(epochs) + 1):
        state = trainer.train_epoch(loader)
        history.append(
            TrainerState(epoch=epoch, loss=state.loss, metrics=dict(state.metrics))
        )

    return PseudoLabelPriors(
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        history=tuple(history),
    )


def fit_2d_pseudo_labels(
    joints_2d: Tensor,
    camera: Camera,
    *,
    model: SkeletalModel,
    confidences: Tensor | None = None,
    pose_prior: PoseVAE | None = None,
    joint_limit_prior: JointLimitPrior | None = None,
    batch_size: int = 1,
    num_iters: int = 120,
    lr: float = 5e-3,
    optimize_scales: bool = True,
    use_pose_prior_latent: bool = True,
    init_depth: float = 3000.0,
    fitting_weights: FittingWeights | None = None,
    device: torch.device | str = "cpu",
) -> PseudoLabelResult:
    """Fit 3D pseudo-labels from 2D joint observations.

    Parameters
    ----------
    joints_2d : torch.Tensor
        Observed 2D joints with shape ``[*batch, joints, 2]``.
    camera : PerspectiveCamera or WeakPerspectiveCamera
        Camera used to project fitted 3D joints.
    model : SkeletalModel
        Skeleton model to optimize.
    confidences : torch.Tensor, optional
        Per-joint confidence weights with shape ``[*batch, joints]``.
    pose_prior : PoseVAE, optional
        Optional latent pose prior used during fitting.
    joint_limit_prior : JointLimitPrior, optional
        Optional joint-limit prior used during fitting.
    batch_size : int, optional
        Number of observations optimized in each fitting call.
    num_iters : int, optional
        Number of optimizer iterations per fitting call.
    lr : float, optional
        Optimizer learning rate.
    optimize_scales : bool, optional
        Whether body scales are optimized.
    use_pose_prior_latent : bool, optional
        Whether to fit latent pose codes when ``pose_prior`` is provided.
    init_depth : float, optional
        Initial z-translation for perspective fitting.
    fitting_weights : FittingWeights, optional
        Objective weights passed to ``SkeletalFitter.fit_2d``.
    device : torch.device or str, optional
        Device used for optimization.

    Returns
    -------
    PseudoLabelResult
        Reconstructed 3D pseudo-labels and reprojection diagnostics.

    Raises
    ------
    ValueError
        If observation shapes or scalar options are invalid.
    """

    _validate_observations(joints_2d, confidences, model)
    _require_positive_int(batch_size, "batch_size")
    _require_positive_int(num_iters, "num_iters")

    device = torch.device(device)
    leading_shape = joints_2d.shape[:-2]
    num_joints = joints_2d.shape[-2]
    num_samples = math.prod(leading_shape)
    target = joints_2d.reshape(num_samples, num_joints, 2).to(device)
    confidence_values = (
        confidences.reshape(num_samples, num_joints).to(device)
        if confidences is not None
        else None
    )
    camera = camera.to(device)
    fitter = SkeletalFitter(
        model=model,
        pose_prior=pose_prior,
        joint_limit_prior=joint_limit_prior,
        device=device,
    )

    fitted_joints: list[Tensor] = []
    fitted_projections: list[Tensor] = []
    reprojection_errors: list[Tensor] = []
    iteration_counts: list[Tensor] = []
    fit_losses: list[dict[str, float]] = []
    for start in range(0, num_samples, int(batch_size)):
        end = min(start + int(batch_size), num_samples)
        target_batch = target[start:end]
        confidence_batch = (
            confidence_values[start:end] if confidence_values is not None else None
        )
        result = fitter.fit_2d(
            target_batch,
            camera,
            confidences=confidence_batch,
            num_iters=int(num_iters),
            lr=lr,
            optimize_scales=optimize_scales,
            use_pose_prior_latent=use_pose_prior_latent,
            init_depth=init_depth,
            fitting_weights=fitting_weights,
        )
        joints_batch = result.model_output.joints.detach()
        projected_batch = camera.project(joints_batch).detach()
        fitted_joints.append(joints_batch.cpu())
        fitted_projections.append(projected_batch.cpu())
        reprojection_errors.append(
            _per_sample_reprojection_error(
                projected_batch,
                target_batch,
                confidence_batch,
            ).cpu()
        )
        iteration_counts.append(
            torch.full(
                (end - start,),
                int(result.iterations),
                dtype=torch.long,
            )
        )
        fit_losses.append(dict(result.losses))

    joints_3d = torch.cat(fitted_joints, dim=0).reshape(leading_shape + (num_joints, 3))
    projected = torch.cat(fitted_projections, dim=0).reshape(
        leading_shape + (num_joints, 2)
    )
    errors = torch.cat(reprojection_errors, dim=0).reshape(leading_shape)
    iterations = torch.cat(iteration_counts, dim=0).reshape(leading_shape)
    observed = joints_2d.detach().cpu()
    confidence_output = confidences.detach().cpu() if confidences is not None else None
    return PseudoLabelResult(
        joints_3d=joints_3d,
        observed_joints_2d=observed,
        projected_joints_2d=projected,
        confidences=confidence_output,
        reprojection_errors=errors,
        iterations=iterations,
        fit_losses=tuple(fit_losses),
    )


def save_pseudo_label_npz(
    path: Path | str,
    result: PseudoLabelResult,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a pseudo-label result as a compressed ``.npz`` dataset.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination file.
    result : PseudoLabelResult
        Pseudo-label result to serialize.
    extra : dict[str, Any], optional
        Additional arrays or scalar values to include in the archive.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an extra payload key collides with a reserved dataset key.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "joints_2d": result.observed_joints_2d,
        "joints_3d": result.joints_3d,
        "projected_joints_2d": result.projected_joints_2d,
        "reprojection_errors": result.reprojection_errors,
        "iterations": result.iterations,
    }
    if result.confidences is not None:
        payload["confidences"] = result.confidences
    if extra:
        collisions = sorted(set(payload).intersection(extra))
        if collisions:
            raise ValueError(
                "extra contains reserved pseudo-label key(s): "
                + ", ".join(collisions)
            )
        payload.update(extra)
    np.savez_compressed(
        output_path,
        **{key: _numpy_payload_value(value) for key, value in payload.items()},
    )
