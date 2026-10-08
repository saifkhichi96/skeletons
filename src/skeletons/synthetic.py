from __future__ import annotations

from dataclasses import dataclass

import torch

from .fitting import FrameDataset, PerspectiveCamera, WeakPerspectiveCamera
from .model import SkeletalModel

Tensor = torch.Tensor


@dataclass(frozen=True)
class SyntheticPoseBatch:
    """Synthetic independent pose samples.

    Parameters
    ----------
    full_pose : torch.Tensor
        Axis-angle pose tensor with shape ``[batch, joints, 3]``.
    transl : torch.Tensor
        Translation tensor with shape ``[batch, 3]``.
    scales : torch.Tensor
        Body scale tensor with shape ``[batch, joints, 3]``.
    joints : torch.Tensor
        Generated joint positions with shape ``[batch, joints, 3]``.
    """

    full_pose: Tensor
    transl: Tensor
    scales: Tensor
    joints: Tensor


@dataclass(frozen=True)
class SyntheticSequenceBatch:
    """Synthetic random-walk pose sequences.

    Parameters
    ----------
    full_pose : torch.Tensor
        Axis-angle pose tensor with shape ``[batch, frames, joints, 3]``.
    transl : torch.Tensor
        Translation tensor with shape ``[batch, frames, 3]``.
    scales : torch.Tensor
        Per-sequence scale tensor with shape ``[batch, joints, 3]``.
    joints : torch.Tensor
        Generated joint positions with shape ``[batch, frames, joints, 3]``.
    """

    full_pose: Tensor
    transl: Tensor
    scales: Tensor
    joints: Tensor


@dataclass(frozen=True)
class SyntheticLiftingDataset:
    """Synthetic paired 2D keypoints and 3D joints for lifting baselines.

    Parameters
    ----------
    keypoints_2d : torch.Tensor
        Normalized or raw 2D keypoints with shape ``[samples, joints, 2]``.
    joints_3d : torch.Tensor
        Generated 3D joints with shape ``[samples, joints, 3]``.
    full_pose : torch.Tensor
        Axis-angle pose tensor with shape ``[samples, joints, 3]``.
    """

    keypoints_2d: Tensor
    joints_3d: Tensor
    full_pose: Tensor


@dataclass(frozen=True)
class SyntheticFittingDataset:
    """Synthetic fitting dataset with paired 3D joints and 2D observations.

    Parameters
    ----------
    frame_dataset : FrameDataset
        Frame dataset containing ``joints_3d``, ``joints_2d``, confidences, and
        perspective camera tensors.
    pose_batch : SyntheticPoseBatch
        Source pose parameters and clean 3D joints.
    clean_joints_2d : torch.Tensor
        Noise-free projected 2D joints with shape ``[frames, joints, 2]``.
    noisy_joints_2d : torch.Tensor
        Noisy 2D observations with shape ``[frames, joints, 2]``.
    confidences : torch.Tensor
        Per-joint confidence weights with shape ``[frames, joints]``.
    camera : PerspectiveCamera
        Camera used to project 3D joints.
    """

    frame_dataset: FrameDataset
    pose_batch: SyntheticPoseBatch
    clean_joints_2d: Tensor
    noisy_joints_2d: Tensor
    confidences: Tensor
    camera: PerspectiveCamera


def _resolve_device(
    model: SkeletalModel,
    device: torch.device | str | None,
) -> torch.device:
    if device is None:
        return model.rest_offsets.device
    return torch.device(device)


def generate_random_pose_batch(
    model: SkeletalModel,
    *,
    batch_size: int,
    pose_std: float,
    transl_std: float = 25.0,
    depth: float = 2500.0,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> SyntheticPoseBatch:
    """Generate independent random poses and forward-kinematic joints.

    Parameters
    ----------
    model : SkeletalModel
        Skeleton model used to generate joints.
    batch_size : int
        Number of poses to generate.
    pose_std : float
        Standard deviation for sampled axis-angle pose values.
    transl_std : float, optional
        Standard deviation for sampled translations.
    depth : float, optional
        Offset added to the z translation.
    device : torch.device or str, optional
        Device used for generated tensors. Defaults to the model device.
    generator : torch.Generator, optional
        Torch random generator for deterministic sampling.

    Returns
    -------
    SyntheticPoseBatch
        Generated pose parameters and joints.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    device = _resolve_device(model, device)
    dtype = model.rest_offsets.dtype
    full_pose = (
        torch.randn(
            batch_size,
            model.NUM_JOINTS,
            3,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * pose_std
    )
    transl = (
        torch.randn(
            batch_size,
            3,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * transl_std
    )
    transl[..., 2] += depth
    scales = torch.ones(
        batch_size,
        model.NUM_JOINTS,
        3,
        dtype=dtype,
        device=device,
    )
    joints = model(full_pose=full_pose, transl=transl, scales=scales).joints.detach()
    return SyntheticPoseBatch(
        full_pose=full_pose,
        transl=transl,
        scales=scales,
        joints=joints,
    )


def generate_random_walk_sequences(
    model: SkeletalModel,
    *,
    batch_size: int,
    seq_len: int,
    pose_step_std: float,
    transl_step_std: float,
    depth: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> SyntheticSequenceBatch:
    """Generate random-walk pose sequences and joints.

    Parameters
    ----------
    model : SkeletalModel
        Skeleton model used to generate joints.
    batch_size : int
        Number of sequences to generate.
    seq_len : int
        Number of frames per sequence.
    pose_step_std : float
        Standard deviation of pose random-walk increments.
    transl_step_std : float
        Standard deviation of translation random-walk increments.
    depth : float
        Offset added to the z translation.
    device : torch.device or str, optional
        Device used for generated tensors. Defaults to the model device.
    generator : torch.Generator, optional
        Torch random generator for deterministic sampling.

    Returns
    -------
    SyntheticSequenceBatch
        Generated sequence parameters and joints.

    Raises
    ------
    ValueError
        If ``batch_size`` or ``seq_len`` is not positive.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")

    device = _resolve_device(model, device)
    dtype = model.rest_offsets.dtype
    pose_steps = (
        torch.randn(
            batch_size,
            seq_len,
            model.NUM_JOINTS,
            3,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * pose_step_std
    )
    full_pose = pose_steps.cumsum(dim=1)

    transl_steps = (
        torch.randn(
            batch_size,
            seq_len,
            3,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        * transl_step_std
    )
    transl = transl_steps.cumsum(dim=1)
    transl[..., 2] += depth
    scales = torch.ones(
        batch_size,
        model.NUM_JOINTS,
        3,
        dtype=dtype,
        device=device,
    )

    flat_output = model(
        full_pose=full_pose.reshape(-1, model.NUM_JOINTS, 3),
        transl=transl.reshape(-1, 3),
        scales=scales.repeat_interleave(seq_len, dim=0),
    )
    joints = flat_output.joints.reshape(
        batch_size,
        seq_len,
        model.NUM_JOINTS,
        3,
    ).detach()
    return SyntheticSequenceBatch(
        full_pose=full_pose,
        transl=transl,
        scales=scales,
        joints=joints,
    )


def generate_orthographic_lifting_dataset(
    model: SkeletalModel,
    *,
    num_samples: int,
    pose_std: float,
    noise_std: float = 0.0,
    root_center: bool = True,
    normalize_2d: bool = True,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> SyntheticLiftingDataset:
    """Generate paired 2D keypoints and 3D joints for lifting baselines.

    Parameters
    ----------
    model : SkeletalModel
        Skeleton model used to generate joints.
    num_samples : int
        Number of samples to generate.
    pose_std : float
        Standard deviation for sampled axis-angle pose values.
    noise_std : float, optional
        Standard deviation of additive 2D keypoint noise.
    root_center : bool, optional
        Whether to subtract the root keypoint from all 2D keypoints.
    normalize_2d : bool, optional
        Whether to normalize each sample by its largest 2D joint radius.
    device : torch.device or str, optional
        Device used for generated tensors. Defaults to the model device.
    generator : torch.Generator, optional
        Torch random generator for deterministic sampling.

    Returns
    -------
    SyntheticLiftingDataset
        Paired 2D keypoints, 3D joints, and source poses.

    Raises
    ------
    ValueError
        If ``num_samples`` is not positive.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    pose_batch = generate_random_pose_batch(
        model,
        batch_size=num_samples,
        pose_std=pose_std,
        transl_std=0.0,
        depth=0.0,
        device=device,
        generator=generator,
    )
    keypoints_2d = pose_batch.joints[..., :2].clone()
    if noise_std > 0.0:
        keypoints_2d = keypoints_2d + noise_std * torch.randn(
            keypoints_2d.shape,
            dtype=keypoints_2d.dtype,
            device=keypoints_2d.device,
            generator=generator,
        )

    if root_center:
        root = keypoints_2d[:, model.root_index : model.root_index + 1]
        keypoints_2d = keypoints_2d - root

    if normalize_2d:
        scale = (
            torch.linalg.vector_norm(keypoints_2d, dim=-1)
            .amax(dim=-1, keepdim=True)
            .clamp_min(1e-6)
        )
        keypoints_2d = keypoints_2d / scale.unsqueeze(-1)

    return SyntheticLiftingDataset(
        keypoints_2d=keypoints_2d.detach(),
        joints_3d=pose_batch.joints.detach(),
        full_pose=pose_batch.full_pose.detach(),
    )


def generate_synthetic_fitting_dataset(
    model: SkeletalModel,
    *,
    num_frames: int,
    pose_std: float,
    noise_std: float = 0.0,
    confidence_dropout: float = 0.0,
    transl_std: float = 25.0,
    depth: float = 2500.0,
    camera: PerspectiveCamera | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> SyntheticFittingDataset:
    """Generate a frame fitting dataset with 3D targets and projected 2D joints.

    Parameters
    ----------
    model : SkeletalModel
        Skeleton model used to generate joints.
    num_frames : int
        Number of independent frames to generate.
    pose_std : float
        Standard deviation for sampled axis-angle pose values.
    noise_std : float, optional
        Standard deviation of additive 2D detector noise.
    confidence_dropout : float, optional
        Probability that a generated joint confidence is set to zero.
    transl_std : float, optional
        Standard deviation for sampled translations.
    depth : float, optional
        Offset added to the z translation.
    camera : PerspectiveCamera, optional
        Camera used for projection. A default synthetic camera is created when
        omitted.
    device : torch.device or str, optional
        Device used for generated tensors. Defaults to the model device.
    generator : torch.Generator, optional
        Torch random generator for deterministic sampling.

    Returns
    -------
    SyntheticFittingDataset
        Frame dataset plus source clean/noisy observations.

    Raises
    ------
    ValueError
        If ``num_frames`` is not positive or ``confidence_dropout`` is outside
        ``[0, 1]``.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be positive.")
    if not 0.0 <= confidence_dropout <= 1.0:
        raise ValueError("confidence_dropout must be between 0 and 1.")

    device = _resolve_device(model, device)
    camera = (camera or make_perspective_camera(device)).to(device)
    pose_batch = generate_random_pose_batch(
        model,
        batch_size=num_frames,
        pose_std=pose_std,
        transl_std=transl_std,
        depth=depth,
        device=device,
        generator=generator,
    )
    clean_joints_2d = camera.project(pose_batch.joints)
    noisy_joints_2d = clean_joints_2d
    if noise_std > 0.0:
        noisy_joints_2d = noisy_joints_2d + noise_std * torch.randn(
            noisy_joints_2d.shape,
            dtype=noisy_joints_2d.dtype,
            device=noisy_joints_2d.device,
            generator=generator,
        )

    confidences = torch.ones(
        num_frames,
        model.NUM_JOINTS,
        dtype=pose_batch.joints.dtype,
        device=device,
    )
    if confidence_dropout > 0.0:
        dropout = torch.rand(
            confidences.shape,
            dtype=confidences.dtype,
            device=device,
            generator=generator,
        )
        confidences = confidences.masked_fill(dropout < confidence_dropout, 0.0)

    def camera_vector(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.reshape(()).expand(num_frames).clone()
        if value.numel() == num_frames:
            return value.reshape(num_frames).clone()
        raise ValueError(
            "camera intrinsics must be scalar or have num_frames leading values."
        )

    frame_dataset = FrameDataset(
        pose_batch.joints.detach().cpu(),
        joints_2d=noisy_joints_2d.detach().cpu(),
        confidences=confidences.detach().cpu(),
        cameras={
            "fx": camera_vector(camera.fx),
            "fy": camera_vector(camera.fy),
            "cx": camera_vector(camera.cx),
            "cy": camera_vector(camera.cy),
        },
        metadata={
            "synthetic": True,
            "pose_std": float(pose_std),
            "noise_std": float(noise_std),
            "confidence_dropout": float(confidence_dropout),
        },
        expected_num_joints=model.NUM_JOINTS,
    )
    return SyntheticFittingDataset(
        frame_dataset=frame_dataset,
        pose_batch=pose_batch,
        clean_joints_2d=clean_joints_2d.detach(),
        noisy_joints_2d=noisy_joints_2d.detach(),
        confidences=confidences.detach(),
        camera=camera,
    )


def make_perspective_camera(device: torch.device | str = "cpu") -> PerspectiveCamera:
    """Create a default perspective camera for synthetic examples.

    Parameters
    ----------
    device : torch.device or str, optional
        Device for camera tensors.

    Returns
    -------
    PerspectiveCamera
        Perspective camera with fixed focal length and principal point.
    """

    device = torch.device(device)
    return PerspectiveCamera(
        fx=torch.tensor(1000.0, device=device),
        fy=torch.tensor(1000.0, device=device),
        cx=torch.tensor(512.0, device=device),
        cy=torch.tensor(512.0, device=device),
    )


def make_weak_perspective_camera(
    device: torch.device | str = "cpu",
) -> WeakPerspectiveCamera:
    """Create a default weak-perspective camera for synthetic examples.

    Parameters
    ----------
    device : torch.device or str, optional
        Device for camera tensors.

    Returns
    -------
    WeakPerspectiveCamera
        Weak-perspective camera with fixed scale and zero image-plane offset.
    """

    device = torch.device(device)
    return WeakPerspectiveCamera(
        scale=torch.tensor(180.0, device=device),
        tx=torch.tensor(0.0, device=device),
        ty=torch.tensor(0.0, device=device),
    )
