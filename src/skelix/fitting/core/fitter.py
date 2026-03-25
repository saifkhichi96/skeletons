from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ...ik import estimate_bone_scales_from_joints, estimate_rotations_from_joints
from ...model import SkeletalModel
from ...rotations import matrix_to_rot6d
from .camera import PerspectiveCamera, WeakPerspectiveCamera
from .priors import JointLimitPrior, MotionSmoothnessPrior, PoseVAE, TemporalPrior


@dataclass
class FittingWeights:
    """Scalar weights for the different fitting objectives."""

    reprojection: float = 1.0
    joints_3d: float = 1.0
    latent: float = 1e-3
    joint_limits: float = 1e-2
    bone_scales: float = 1e-3
    smoothness: float = 1e-2
    temporal: float = 1e-2


@dataclass
class FittingResult:
    """Outputs returned by a fitting run.

    Attributes:
        model_output: The final forward pass result from the fitted model.
        losses: Scalar loss terms from the last optimization step.
        iterations: Number of optimizer steps that were run.
    """

    model_output: object
    losses: dict[str, float]
    iterations: int


ProgressCallback = Callable[[int, int, torch.Tensor, dict[str, float]], bool | None]
SnapshotFn = Callable[[], tuple[torch.Tensor, dict[str, float]]]


class SkeletalFitter:
    """Optimize pose, translation, and optional scales for a skeletal model.

    Args:
        model: Concrete skeleton model to fit against.
        pose_prior: Optional latent pose prior operating on non-root joints.
        joint_limit_prior: Optional joint-limit regularizer.
        temporal_prior: Optional temporal pose prior for sequence fitting.
        smoothness_prior: Optional motion smoothness prior for sequence fitting.
        device: Torch device used for optimization.
    """

    def __init__(
        self,
        *,
        model: SkeletalModel,
        pose_prior: PoseVAE | None = None,
        joint_limit_prior: JointLimitPrior | None = None,
        temporal_prior: TemporalPrior | None = None,
        smoothness_prior: MotionSmoothnessPrior | None = None,
        device: torch.device | str = 'cpu',
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.pose_prior = pose_prior.to(self.device) if pose_prior is not None else None
        self.joint_limit_prior = joint_limit_prior.to(self.device) if joint_limit_prior is not None else None
        self.temporal_prior = temporal_prior.to(self.device) if temporal_prior is not None else None
        self.smoothness_prior = (smoothness_prior or MotionSmoothnessPrior()).to(self.device)

    def _default_body_pose(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(batch_shape + (self.model.num_joints - 1, 6), device=self.device)

    def _default_global_orient(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=self.device)
        return ident.expand(batch_shape + (6,)).clone()

    def _default_bone_scales(self, batch_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.ones(batch_shape + (self.model.num_joints, 3), device=self.device)

    def _match_batch_shape(
        self,
        tensor: torch.Tensor,
        batch_shape: tuple[int, ...],
        tail_shape: tuple[int, ...],
    ) -> torch.Tensor:
        tensor = tensor.to(self.device)
        expected_ndim = len(batch_shape) + len(tail_shape)
        if tensor.shape == tail_shape:
            tensor = tensor.reshape((1,) * len(batch_shape) + tail_shape)
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f'Expected tensor with shape {batch_shape + tail_shape} or {tail_shape}, got {tuple(tensor.shape)}.',
            )
        if tensor.shape[-len(tail_shape):] != tail_shape:
            raise ValueError(f'Expected tensor tail shape {tail_shape}, got {tuple(tensor.shape)}.')
        if tensor.shape[:-len(tail_shape)] == batch_shape:
            return tensor
        if tensor.shape[:-len(tail_shape)] == (1,) * len(batch_shape):
            return tensor.expand(batch_shape + tail_shape).clone()
        raise ValueError(
            f'Cannot broadcast warm-start tensor batch shape {tensor.shape[:-len(tail_shape)]} to {batch_shape}.',
        )

    def _body_pose_from_latent(
        self,
        latent: torch.Tensor | None,
        direct_pose: torch.Tensor | None,
    ) -> torch.Tensor:
        if latent is not None:
            if self.pose_prior is None:
                raise ValueError('latent optimization requested without a pose_prior.')
            return self.pose_prior.decode(latent)
        if direct_pose is None:
            raise ValueError('Either latent or direct_pose must be provided.')
        return direct_pose

    def _joint_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = pred - target
        if weights is None:
            return diff.pow(2).sum(dim=-1).mean()
        return (diff.pow(2).sum(dim=-1) * weights).mean()

    def _run_optimizer(
        self,
        closure,
        parameters: list[torch.nn.Parameter],
        *,
        lr: float,
        num_iters: int,
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 10,
        snapshot_fn: SnapshotFn | None = None,
    ) -> int:
        optimizer = torch.optim.Adam(parameters, lr=lr)
        completed_iters = 0
        interval = max(int(progress_interval), 1)
        for step in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            loss.backward()
            optimizer.step()
            completed_iters = step + 1

            if progress_callback is None or snapshot_fn is None:
                continue
            if completed_iters != num_iters and completed_iters % interval != 0:
                continue

            with torch.no_grad():
                joints, losses = snapshot_fn()
            should_continue = progress_callback(
                completed_iters,
                num_iters,
                joints.detach().cpu(),
                dict(losses),
            )
            if should_continue is False:
                break
        return completed_iters

    def fit_3d(
        self,
        target_joints_3d: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        num_iters: int = 300,
        lr: float = 1e-2,
        optimize_bone_scales: bool = False,
        use_pose_prior_latent: bool = True,
        init_from_ik: bool = True,
        fitting_weights: FittingWeights | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 10,
        init_global_orient: torch.Tensor | None = None,
        init_body_pose: torch.Tensor | None = None,
        init_bone_scales: torch.Tensor | None = None,
        init_transl: torch.Tensor | None = None,
    ) -> FittingResult:
        """Fit a single-frame 3D pose to target joints."""

        fitting_weights = fitting_weights or FittingWeights()
        target = target_joints_3d.to(self.device)
        weights = weights.to(self.device) if weights is not None else None
        batch_shape = target.shape[:-2]

        has_warm_start = any(
            value is not None
            for value in (init_global_orient, init_body_pose, init_bone_scales, init_transl)
        )

        if has_warm_start:
            if init_global_orient is None or init_body_pose is None:
                raise ValueError('Warm start requires both init_global_orient and init_body_pose.')
            global_init = self._match_batch_shape(init_global_orient, batch_shape, (6,))
            body_init = self._match_batch_shape(init_body_pose, batch_shape, (self.model.num_joints - 1, 6))
            transl_default = target[..., self.model.root_index, :].detach()
            transl_init = (
                self._match_batch_shape(init_transl, batch_shape, (3,))
                if init_transl is not None
                else transl_default
            )
            bone_scales_init = (
                self._match_batch_shape(init_bone_scales, batch_shape, (self.model.num_joints, 3))
                if init_bone_scales is not None
                else self._default_bone_scales(batch_shape)
            )
        elif init_from_ik:
            bone_scales_init = estimate_bone_scales_from_joints(target, self.model)
            ik = estimate_rotations_from_joints(target, self.model, bone_scales=bone_scales_init)
            full_rot6d = matrix_to_rot6d(ik.local_rotations)
            global_init = full_rot6d[..., self.model.root_index, :].detach()
            body_init = full_rot6d[..., list(self.model.non_root_joint_indices), :].detach()
            transl_init = target[..., self.model.root_index, :].detach()
        else:
            bone_scales_init = self._default_bone_scales(batch_shape)
            global_init = self._default_global_orient(batch_shape)
            body_init = self._default_body_pose(batch_shape)
            transl_init = target[..., self.model.root_index, :].detach()

        global_orient = torch.nn.Parameter(global_init.clone())
        transl = torch.nn.Parameter(transl_init.clone())
        bone_scales = torch.nn.Parameter(bone_scales_init.clone()) if optimize_bone_scales else bone_scales_init

        latent = None
        direct_pose = None
        if self.pose_prior is not None and use_pose_prior_latent:
            with torch.no_grad():
                mu, _ = self.pose_prior.encode(body_init)
            latent = torch.nn.Parameter(mu.clone())
            parameters = [global_orient, transl, latent]
        else:
            direct_pose = torch.nn.Parameter(body_init.clone())
            parameters = [global_orient, transl, direct_pose]
        if isinstance(bone_scales, torch.nn.Parameter):
            parameters.append(bone_scales)

        last_losses: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal last_losses
            body_pose = self._body_pose_from_latent(latent, direct_pose)
            output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                bone_scales=bone_scales,
                transl=transl,
                pose_repr='rot6d',
                return_local_rotations=True,
            )
            loss_joints = self._joint_loss(output.joints, target, weights=weights)
            loss = fitting_weights.joints_3d * loss_joints
            last_losses = {'joints_3d': float(loss_joints.detach().cpu())}

            if latent is not None and self.pose_prior is not None:
                latent_loss = self.pose_prior.latent_regularization(latent)
                loss = loss + fitting_weights.latent * latent_loss
                last_losses['latent'] = float(latent_loss.detach().cpu())

            if self.joint_limit_prior is not None:
                limit_loss = self.joint_limit_prior(
                    output.local_rotations[..., list(self.model.non_root_joint_indices), :, :],
                )
                loss = loss + fitting_weights.joint_limits * limit_loss
                last_losses['joint_limits'] = float(limit_loss.detach().cpu())

            if optimize_bone_scales:
                scale_loss = (bone_scales - 1.0).pow(2).mean()
                loss = loss + fitting_weights.bone_scales * scale_loss
                last_losses['bone_scales'] = float(scale_loss.detach().cpu())
            return loss

        def snapshot() -> tuple[torch.Tensor, dict[str, float]]:
            body_pose = self._body_pose_from_latent(latent, direct_pose)
            output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                bone_scales=bone_scales,
                transl=transl,
                pose_repr='rot6d',
            )
            return output.joints, last_losses

        completed_iters = self._run_optimizer(
            closure,
            parameters,
            lr=lr,
            num_iters=num_iters,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            snapshot_fn=snapshot,
        )

        with torch.no_grad():
            body_pose = self._body_pose_from_latent(latent, direct_pose)
            output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                bone_scales=bone_scales,
                transl=transl,
                pose_repr='rot6d',
                return_local_rotations=True,
                return_global_rotations=True,
                return_scaled_offsets=True,
            )
        return FittingResult(model_output=output, losses=last_losses, iterations=completed_iters)

    def fit_2d(
        self,
        target_joints_2d: torch.Tensor,
        camera: PerspectiveCamera | WeakPerspectiveCamera,
        *,
        confidences: torch.Tensor | None = None,
        num_iters: int = 500,
        lr: float = 5e-3,
        optimize_bone_scales: bool = False,
        use_pose_prior_latent: bool = True,
        init_depth: float = 3000.0,
        fitting_weights: FittingWeights | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 10,
        init_global_orient: torch.Tensor | None = None,
        init_body_pose: torch.Tensor | None = None,
        init_bone_scales: torch.Tensor | None = None,
        init_transl: torch.Tensor | None = None,
    ) -> FittingResult:
        """Fit a single-frame 3D pose from 2D joints and a camera model."""

        fitting_weights = fitting_weights or FittingWeights()
        target = target_joints_2d.to(self.device)
        confidences = confidences.to(self.device) if confidences is not None else None
        camera = camera.to(self.device)
        batch_shape = target.shape[:-2]
        global_init = (
            self._match_batch_shape(init_global_orient, batch_shape, (6,))
            if init_global_orient is not None
            else self._default_global_orient(batch_shape)
        )
        global_orient = torch.nn.Parameter(global_init.clone())
        if init_transl is not None:
            transl_init = self._match_batch_shape(init_transl, batch_shape, (3,))
        else:
            transl_init = torch.zeros(batch_shape + (3,), device=self.device)
            transl_init[..., 2] = init_depth
        transl = torch.nn.Parameter(transl_init)
        if optimize_bone_scales:
            bone_scales_init = (
                self._match_batch_shape(init_bone_scales, batch_shape, (self.model.num_joints, 3))
                if init_bone_scales is not None
                else self._default_bone_scales(batch_shape)
            )
            bone_scales = torch.nn.Parameter(bone_scales_init.clone())
        else:
            bone_scales = (
                self._match_batch_shape(init_bone_scales, batch_shape, (self.model.num_joints, 3))
                if init_bone_scales is not None
                else self._default_bone_scales(batch_shape)
            )

        latent = None
        direct_pose = None
        body_init = (
            self._match_batch_shape(init_body_pose, batch_shape, (self.model.num_joints - 1, 6))
            if init_body_pose is not None
            else self._default_body_pose(batch_shape)
        )
        if self.pose_prior is not None and use_pose_prior_latent:
            with torch.no_grad():
                mu, _ = self.pose_prior.encode(body_init)
            latent = torch.nn.Parameter(mu.clone())
            parameters = [global_orient, transl, latent]
        else:
            direct_pose = torch.nn.Parameter(body_init.clone())
            parameters = [global_orient, transl, direct_pose]
        if isinstance(bone_scales, torch.nn.Parameter):
            parameters.append(bone_scales)

        last_losses: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal last_losses
            body_pose = self._body_pose_from_latent(latent, direct_pose)
            output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                bone_scales=bone_scales,
                transl=transl,
                pose_repr='rot6d',
                return_local_rotations=True,
            )
            projected = camera.project(output.joints)
            loss_2d = self._joint_loss(projected, target, weights=confidences)
            loss = fitting_weights.reprojection * loss_2d
            last_losses = {'reprojection': float(loss_2d.detach().cpu())}

            if latent is not None and self.pose_prior is not None:
                latent_loss = self.pose_prior.latent_regularization(latent)
                loss = loss + fitting_weights.latent * latent_loss
                last_losses['latent'] = float(latent_loss.detach().cpu())

            if self.joint_limit_prior is not None:
                limit_loss = self.joint_limit_prior(
                    output.local_rotations[..., list(self.model.non_root_joint_indices), :, :],
                )
                loss = loss + fitting_weights.joint_limits * limit_loss
                last_losses['joint_limits'] = float(limit_loss.detach().cpu())

            if optimize_bone_scales:
                scale_loss = (bone_scales - 1.0).pow(2).mean()
                loss = loss + fitting_weights.bone_scales * scale_loss
                last_losses['bone_scales'] = float(scale_loss.detach().cpu())
            return loss

        def snapshot() -> tuple[torch.Tensor, dict[str, float]]:
            body_pose = self._body_pose_from_latent(latent, direct_pose)
            output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                bone_scales=bone_scales,
                transl=transl,
                pose_repr='rot6d',
            )
            return output.joints, last_losses

        completed_iters = self._run_optimizer(
            closure,
            parameters,
            lr=lr,
            num_iters=num_iters,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            snapshot_fn=snapshot,
        )

        with torch.no_grad():
            body_pose = self._body_pose_from_latent(latent, direct_pose)
            output = self.model(
                global_orient=global_orient,
                body_pose=body_pose,
                bone_scales=bone_scales,
                transl=transl,
                pose_repr='rot6d',
                return_local_rotations=True,
                return_global_rotations=True,
                return_scaled_offsets=True,
            )
        return FittingResult(model_output=output, losses=last_losses, iterations=completed_iters)

    def fit_sequence_3d(
        self,
        target_joints_3d: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        num_iters: int = 400,
        lr: float = 1e-2,
        optimize_bone_scales: bool = True,
        use_pose_prior_latent: bool = True,
        fitting_weights: FittingWeights | None = None,
    ) -> FittingResult:
        """Fit a 3D joint sequence with temporal regularization."""

        fitting_weights = fitting_weights or FittingWeights()
        target = target_joints_3d.to(self.device)
        weights = weights.to(self.device) if weights is not None else None
        batch_size, seq_len = target.shape[:2]

        bone_scales_init = estimate_bone_scales_from_joints(target[:, 0], self.model)
        ik = estimate_rotations_from_joints(target.reshape(-1, self.model.num_joints, 3), self.model)
        full_rot6d = matrix_to_rot6d(ik.local_rotations).reshape(batch_size, seq_len, self.model.num_joints, 6)
        global_init = full_rot6d[..., self.model.root_index, :].detach()
        body_init = full_rot6d[..., list(self.model.non_root_joint_indices), :].detach()
        transl_init = target[..., self.model.root_index, :].detach()

        global_orient = torch.nn.Parameter(global_init.clone())
        transl = torch.nn.Parameter(transl_init.clone())
        bone_scales = torch.nn.Parameter(bone_scales_init.clone()) if optimize_bone_scales else bone_scales_init

        latent = None
        direct_pose = None
        if self.pose_prior is not None and use_pose_prior_latent:
            with torch.no_grad():
                mu, _ = self.pose_prior.encode(body_init.reshape(-1, body_init.shape[-2], body_init.shape[-1]))
            latent = torch.nn.Parameter(mu.reshape(batch_size, seq_len, -1).clone())
            parameters = [global_orient, transl, latent]
        else:
            direct_pose = torch.nn.Parameter(body_init.clone())
            parameters = [global_orient, transl, direct_pose]
        if isinstance(bone_scales, torch.nn.Parameter):
            parameters.append(bone_scales)

        last_losses: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal last_losses
            if latent is not None:
                if self.pose_prior is None:
                    raise ValueError('latent optimization requested without a pose_prior.')
                body_pose = self.pose_prior.decode(latent.reshape(-1, latent.shape[-1])).reshape(
                    batch_size,
                    seq_len,
                    self.model.num_joints - 1,
                    6,
                )
            else:
                body_pose = direct_pose
            output = self.model(
                global_orient=global_orient.reshape(-1, 6),
                body_pose=body_pose.reshape(-1, self.model.num_joints - 1, 6),
                bone_scales=bone_scales.repeat_interleave(seq_len, dim=0) if bone_scales.ndim == 3 else bone_scales,
                transl=transl.reshape(-1, 3),
                pose_repr='rot6d',
                return_local_rotations=True,
            )
            pred_joints = output.joints.reshape(batch_size, seq_len, self.model.num_joints, 3)
            local_rot = output.local_rotations.reshape(batch_size, seq_len, self.model.num_joints, 3, 3)
            loss_joints = self._joint_loss(pred_joints, target, weights=weights)
            loss = fitting_weights.joints_3d * loss_joints
            last_losses = {'joints_3d': float(loss_joints.detach().cpu())}

            if latent is not None and self.pose_prior is not None:
                latent_loss = self.pose_prior.latent_regularization(latent)
                loss = loss + fitting_weights.latent * latent_loss
                last_losses['latent'] = float(latent_loss.detach().cpu())

            if self.joint_limit_prior is not None:
                limit_loss = self.joint_limit_prior(
                    local_rot[..., list(self.model.non_root_joint_indices), :, :].reshape(
                        -1,
                        self.model.num_joints - 1,
                        3,
                        3,
                    ),
                )
                loss = loss + fitting_weights.joint_limits * limit_loss
                last_losses['joint_limits'] = float(limit_loss.detach().cpu())

            if self.temporal_prior is not None:
                temporal_loss = self.temporal_prior.loss(body_pose)
                loss = loss + fitting_weights.temporal * temporal_loss
                last_losses['temporal'] = float(temporal_loss.detach().cpu())

            smooth_loss = self.smoothness_prior(body_pose, transl)
            loss = loss + fitting_weights.smoothness * smooth_loss
            last_losses['smoothness'] = float(smooth_loss.detach().cpu())

            if optimize_bone_scales:
                scale_loss = (bone_scales - 1.0).pow(2).mean()
                loss = loss + fitting_weights.bone_scales * scale_loss
                last_losses['bone_scales'] = float(scale_loss.detach().cpu())
            return loss

        self._run_optimizer(closure, parameters, lr=lr, num_iters=num_iters)

        with torch.no_grad():
            if latent is not None:
                if self.pose_prior is None:
                    raise ValueError('latent optimization requested without a pose_prior.')
                body_pose = self.pose_prior.decode(latent.reshape(-1, latent.shape[-1])).reshape(
                    batch_size,
                    seq_len,
                    self.model.num_joints - 1,
                    6,
                )
            else:
                body_pose = direct_pose
            output = self.model(
                global_orient=global_orient.reshape(-1, 6),
                body_pose=body_pose.reshape(-1, self.model.num_joints - 1, 6),
                bone_scales=bone_scales.repeat_interleave(seq_len, dim=0) if bone_scales.ndim == 3 else bone_scales,
                transl=transl.reshape(-1, 3),
                pose_repr='rot6d',
                return_local_rotations=True,
                return_global_rotations=True,
                return_scaled_offsets=True,
            )
        return FittingResult(model_output=output, losses=last_losses, iterations=num_iters)

    def fit_sequence_2d(
        self,
        target_joints_2d: torch.Tensor,
        camera: PerspectiveCamera | WeakPerspectiveCamera,
        *,
        confidences: torch.Tensor | None = None,
        num_iters: int = 500,
        lr: float = 5e-3,
        optimize_bone_scales: bool = True,
        use_pose_prior_latent: bool = True,
        init_depth: float = 3000.0,
        fitting_weights: FittingWeights | None = None,
    ) -> FittingResult:
        """Fit a 3D joint sequence from 2D observations and a camera model."""

        fitting_weights = fitting_weights or FittingWeights()
        target = target_joints_2d.to(self.device)
        confidences = confidences.to(self.device) if confidences is not None else None
        camera = camera.to(self.device)
        batch_size, seq_len = target.shape[:2]

        global_orient = torch.nn.Parameter(self._default_global_orient((batch_size, seq_len)))
        transl_init = torch.zeros(batch_size, seq_len, 3, device=self.device)
        transl_init[..., 2] = init_depth
        transl = torch.nn.Parameter(transl_init)
        if optimize_bone_scales:
            bone_scales = torch.nn.Parameter(self._default_bone_scales((batch_size,)))
        else:
            bone_scales = self._default_bone_scales((batch_size,))

        latent = None
        direct_pose = None
        if self.pose_prior is not None and use_pose_prior_latent:
            latent = torch.nn.Parameter(torch.zeros(batch_size, seq_len, self.pose_prior.latent_dim, device=self.device))
            parameters = [global_orient, transl, latent]
        else:
            direct_pose = torch.nn.Parameter(self._default_body_pose((batch_size, seq_len)))
            parameters = [global_orient, transl, direct_pose]
        if isinstance(bone_scales, torch.nn.Parameter):
            parameters.append(bone_scales)

        last_losses: dict[str, float] = {}

        def closure() -> torch.Tensor:
            nonlocal last_losses
            if latent is not None:
                if self.pose_prior is None:
                    raise ValueError('latent optimization requested without a pose_prior.')
                body_pose = self.pose_prior.decode(latent.reshape(-1, latent.shape[-1])).reshape(
                    batch_size,
                    seq_len,
                    self.model.num_joints - 1,
                    6,
                )
            else:
                body_pose = direct_pose
            output = self.model(
                global_orient=global_orient.reshape(-1, 6),
                body_pose=body_pose.reshape(-1, self.model.num_joints - 1, 6),
                bone_scales=bone_scales.repeat_interleave(seq_len, dim=0),
                transl=transl.reshape(-1, 3),
                pose_repr='rot6d',
                return_local_rotations=True,
            )
            pred_joints = output.joints.reshape(batch_size, seq_len, self.model.num_joints, 3)
            projected = camera.project(pred_joints)
            loss_2d = self._joint_loss(projected, target, weights=confidences)
            loss = fitting_weights.reprojection * loss_2d
            last_losses = {'reprojection': float(loss_2d.detach().cpu())}

            if latent is not None and self.pose_prior is not None:
                latent_loss = self.pose_prior.latent_regularization(latent)
                loss = loss + fitting_weights.latent * latent_loss
                last_losses['latent'] = float(latent_loss.detach().cpu())

            if self.joint_limit_prior is not None:
                local_rot = output.local_rotations.reshape(batch_size, seq_len, self.model.num_joints, 3, 3)
                limit_loss = self.joint_limit_prior(
                    local_rot[..., list(self.model.non_root_joint_indices), :, :].reshape(
                        -1,
                        self.model.num_joints - 1,
                        3,
                        3,
                    ),
                )
                loss = loss + fitting_weights.joint_limits * limit_loss
                last_losses['joint_limits'] = float(limit_loss.detach().cpu())

            if self.temporal_prior is not None:
                temporal_loss = self.temporal_prior.loss(body_pose)
                loss = loss + fitting_weights.temporal * temporal_loss
                last_losses['temporal'] = float(temporal_loss.detach().cpu())

            smooth_loss = self.smoothness_prior(body_pose, transl)
            loss = loss + fitting_weights.smoothness * smooth_loss
            last_losses['smoothness'] = float(smooth_loss.detach().cpu())

            if optimize_bone_scales:
                scale_loss = (bone_scales - 1.0).pow(2).mean()
                loss = loss + fitting_weights.bone_scales * scale_loss
                last_losses['bone_scales'] = float(scale_loss.detach().cpu())
            return loss

        self._run_optimizer(closure, parameters, lr=lr, num_iters=num_iters)

        with torch.no_grad():
            if latent is not None:
                if self.pose_prior is None:
                    raise ValueError('latent optimization requested without a pose_prior.')
                body_pose = self.pose_prior.decode(latent.reshape(-1, latent.shape[-1])).reshape(
                    batch_size,
                    seq_len,
                    self.model.num_joints - 1,
                    6,
                )
            else:
                body_pose = direct_pose
            output = self.model(
                global_orient=global_orient.reshape(-1, 6),
                body_pose=body_pose.reshape(-1, self.model.num_joints - 1, 6),
                bone_scales=bone_scales.repeat_interleave(seq_len, dim=0),
                transl=transl.reshape(-1, 3),
                pose_repr='rot6d',
                return_local_rotations=True,
                return_global_rotations=True,
                return_scaled_offsets=True,
            )
        return FittingResult(model_output=output, losses=last_losses, iterations=num_iters)
