from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..model import SkeletalModel
from ..rigs import get_schema


def robust_penalty(
    x: torch.Tensor, kind: str = "gmof", scale: float = 0.1
) -> torch.Tensor:
    """Apply a robust penalty to non-negative residual magnitudes."""
    if not scale > 0:
        raise ValueError("scale must be positive.")
    if kind == "l2":
        return x.square()
    if kind == "l1":
        return x.abs()
    if kind == "huber":
        delta = scale
        return torch.where(x <= delta, 0.5 * x.square(), delta * (x - 0.5 * delta))
    if kind == "gmof":
        rho2 = scale * scale
        return (rho2 * x.square()) / (rho2 + x.square()).clamp_min(1e-8)
    raise ValueError(f"Unknown robust penalty {kind!r}.")


def _target_frames(
    model: SkeletalModel,
    frame_names: Sequence[str] | None,
    schema: str | None,
    frame_mapping: Mapping[str, str] | None,
) -> list[str]:
    if schema is not None:
        if frame_names is not None:
            raise ValueError("Use schema or frame_names, not both.")
        names = list(get_schema(schema).joint_names)
    else:
        names = list(model.joint_names if frame_names is None else frame_names)
    if not names:
        raise ValueError("At least one target frame is required.")
    mapping = frame_mapping or {}
    frames = [mapping.get(name, name) for name in names]
    for frame in frames:
        model.frame_parent_index(frame)
    return frames


def _weighted_penalty(
    residual: torch.Tensor,
    confidence: torch.Tensor | None,
    robust: str,
    scale: float,
    reduction: str,
) -> torch.Tensor:
    penalty = robust_penalty(residual, kind=robust, scale=scale)
    if confidence is None:
        denominator = penalty.new_tensor(penalty.numel())
    else:
        if confidence.shape != penalty.shape:
            raise ValueError("Confidence must match the target batch and frame shape.")
        confidence = confidence.to(penalty)
        if not torch.isfinite(confidence).all() or (confidence < 0).any():
            raise ValueError("Confidence must be finite and nonnegative.")
        penalty = penalty * confidence
        denominator = confidence.sum()
    if reduction == "none":
        return penalty
    if reduction == "sum":
        return penalty.sum()
    if reduction == "mean":
        return penalty.sum() / denominator.clamp_min(1e-8)
    raise ValueError(f"Unknown reduction {reduction!r}.")


def keypoint_3d_loss(
    model: SkeletalModel,
    q: torch.Tensor,
    target: torch.Tensor,
    root: torch.Tensor | None = None,
    frame_names: Sequence[str] | None = None,
    schema: str | None = None,
    confidence: torch.Tensor | None = None,
    robust: str = "gmof",
    scale: float = 0.1,
    reduction: str = "mean",
    *,
    frame_mapping: Mapping[str, str] | None = None,
) -> torch.Tensor:
    """Compare joints or attached frames with 3D observations.

    Parameters
    ----------
    model : SkeletalModel
        Any registered or custom rig.
    q, target : torch.Tensor
        Generalized pose coordinates and targets in selected frame order.
    root : torch.Tensor, optional
        Root transforms.
    frame_names : sequence of str, optional
        Target frame names; defaults to all joints in rig order.
    schema : str, optional
        Target schema instead of frame_names.
    confidence : torch.Tensor, optional
        Nonnegative weights with shape matching target[..., 0].
    robust, scale, reduction : str, float, str
        Penalty kind, scale, and reduction.
    frame_mapping : mapping, optional
        Explicit target-name to model-frame mapping. Missing model frames raise.

    Returns
    -------
    torch.Tensor
        Reduced loss or per-frame penalties.

    Raises
    ------
    KeyError
        If a target frame is unavailable or ambiguous.
    ValueError
        If shapes, confidence, or selection arguments are invalid.
    """
    frames = _target_frames(model, frame_names, schema, frame_mapping)
    pred = model.frame_positions(q, root, frames)
    target = target.to(pred)
    if target.shape != pred.shape:
        raise ValueError(f"Expected targets with shape {tuple(pred.shape)}.")
    return _weighted_penalty(
        (pred - target).norm(dim=-1), confidence, robust, scale, reduction
    )


def reprojection_loss(
    model: SkeletalModel,
    q: torch.Tensor,
    cameras,
    target_2d: torch.Tensor,
    root: torch.Tensor | None = None,
    frame_names: Sequence[str] | None = None,
    confidence: torch.Tensor | None = None,
    robust: str = "huber",
    scale: float = 5.0,
    *,
    schema: str | None = None,
    frame_mapping: Mapping[str, str] | None = None,
) -> torch.Tensor:
    """Project joint or attached-frame positions and compare 2D observations.

    Parameters
    ----------
    model : SkeletalModel
        Any registered or custom rig.
    q, target_2d : torch.Tensor
        Generalized coordinates and image targets in selected frame order.
    cameras : PerspectiveCamera
        Camera providing project(points).
    root : torch.Tensor, optional
        Root transforms.
    frame_names, schema : sequence of str or str, optional
        Frame selection; defaults to the rig's joints.
    confidence : torch.Tensor, optional
        Per-target nonnegative weights.
    robust, scale : str, float
        Penalty kind and pixel-space scale.
    frame_mapping : mapping, optional
        Explicit target-name to model-frame mapping.

    Returns
    -------
    torch.Tensor
        Mean confidence-weighted penalty.

    Raises
    ------
    KeyError
        If frames are unavailable.
    ValueError
        If selection arguments, shapes, or confidence are invalid.
    """
    frames = _target_frames(model, frame_names, schema, frame_mapping)
    pred = cameras.project(model.frame_positions(q, root, frames))
    target = target_2d.to(pred)
    if pred.shape != target.shape:
        raise ValueError(f"Expected 2D targets with shape {tuple(pred.shape)}.")
    return _weighted_penalty(
        (pred - target).norm(dim=-1), confidence, robust, scale, "mean"
    )
