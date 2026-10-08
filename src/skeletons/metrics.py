from __future__ import annotations

import torch

Tensor = torch.Tensor


def _validate_joint_tensors(predicted: Tensor, target: Tensor) -> None:
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.ndim < 2 or predicted.shape[-1] != 3:
        raise ValueError("predicted and target must have shape [..., joints, 3].")
    if predicted.shape[-2] < 1:
        raise ValueError("predicted and target must contain at least one joint.")


def _canonicalize_weights(
    weights: Tensor | None,
    *,
    joint_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor | None:
    if weights is None:
        return None

    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    try:
        weights = torch.broadcast_to(weights, joint_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"weights must broadcast to joint shape {tuple(joint_shape)}."
        ) from exc

    if (weights < 0).any().item():
        raise ValueError("weights must be non-negative.")
    if (weights.sum(dim=-1) <= 0).any().item():
        raise ValueError("weights must contain positive mass for every sample.")
    return weights


def _weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean(dim=-1)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1)


def _reduce(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")


def joint_position_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Compute Euclidean joint-position errors.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted joint positions with shape ``[..., joints, 3]``.
    target : torch.Tensor
        Target joint positions with the same shape as ``predicted``.

    Returns
    -------
    torch.Tensor
        Per-joint Euclidean distances with shape ``[..., joints]``.

    Raises
    ------
    ValueError
        If ``predicted`` and ``target`` do not have matching joint shapes.
    """

    _validate_joint_tensors(predicted, target)
    return torch.linalg.vector_norm(predicted - target, dim=-1)


def mean_per_joint_position_error(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute mean per-joint position error (MPJPE).

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted joint positions with shape ``[..., joints, 3]``.
    target : torch.Tensor
        Target joint positions with the same shape as ``predicted``.
    weights : torch.Tensor, optional
        Non-negative joint weights broadcastable to ``[..., joints]``.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over leading sample dimensions after averaging over joints.

    Returns
    -------
    torch.Tensor
        Scalar error for ``"mean"`` and ``"sum"`` reductions, or per-sample
        MPJPE with shape ``[...]`` for ``"none"``.

    Raises
    ------
    ValueError
        If shapes, weights, or ``reduction`` are invalid.
    """

    errors = joint_position_error(predicted, target)
    weights = _canonicalize_weights(
        weights,
        joint_shape=errors.shape,
        dtype=errors.dtype,
        device=errors.device,
    )
    return _reduce(_weighted_mean(errors, weights), reduction)


def procrustes_align(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Align predicted joints to targets with a similarity transform.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted joint positions with shape ``[..., joints, 3]``.
    target : torch.Tensor
        Target joint positions with the same shape as ``predicted``.
    weights : torch.Tensor, optional
        Non-negative joint weights broadcastable to ``[..., joints]``.
    eps : float, optional
        Minimum denominator used for scale estimation.

    Returns
    -------
    torch.Tensor
        ``predicted`` transformed into the target coordinate frame.

    Raises
    ------
    ValueError
        If shapes or weights are invalid.
    """

    _validate_joint_tensors(predicted, target)
    weights = _canonicalize_weights(
        weights,
        joint_shape=predicted.shape[:-1],
        dtype=predicted.dtype,
        device=predicted.device,
    )
    if weights is None:
        weights = torch.ones(
            predicted.shape[:-1], dtype=predicted.dtype, device=predicted.device
        )

    normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
    predicted_mean = (predicted * normalized_weights.unsqueeze(-1)).sum(
        dim=-2, keepdim=True
    )
    target_mean = (target * normalized_weights.unsqueeze(-1)).sum(
        dim=-2, keepdim=True
    )
    predicted_centered = predicted - predicted_mean
    target_centered = target - target_mean

    covariance = predicted_centered.transpose(-2, -1) @ (
        target_centered * normalized_weights.unsqueeze(-1)
    )
    u, _, vh = torch.linalg.svd(covariance)
    det = torch.det(u @ vh)

    correction = (
        torch.eye(3, dtype=predicted.dtype, device=predicted.device)
        .expand(predicted.shape[:-2] + (3, 3))
        .clone()
    )
    correction[..., 2, 2] = torch.where(det < 0.0, -1.0, 1.0)
    rotation = u @ correction @ vh

    rotated = predicted_centered @ rotation
    numerator = (rotated * target_centered * normalized_weights.unsqueeze(-1)).sum(
        dim=(-2, -1), keepdim=True
    )
    denominator = (
        predicted_centered.square() * normalized_weights.unsqueeze(-1)
    ).sum(dim=(-2, -1), keepdim=True)
    scale = numerator / denominator.clamp_min(eps)
    return rotated * scale + target_mean


def procrustes_aligned_mpjpe(
    predicted: Tensor,
    target: Tensor,
    *,
    weights: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute MPJPE after Procrustes similarity alignment.

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted joint positions with shape ``[..., joints, 3]``.
    target : torch.Tensor
        Target joint positions with the same shape as ``predicted``.
    weights : torch.Tensor, optional
        Non-negative joint weights broadcastable to ``[..., joints]``.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over leading sample dimensions after averaging over joints.

    Returns
    -------
    torch.Tensor
        Procrustes-aligned MPJPE.

    Raises
    ------
    ValueError
        If shapes, weights, or ``reduction`` are invalid.
    """

    aligned = procrustes_align(predicted, target, weights=weights)
    return mean_per_joint_position_error(
        aligned,
        target,
        weights=weights,
        reduction=reduction,
    )


def percentage_of_correct_keypoints(
    predicted: Tensor,
    target: Tensor,
    *,
    threshold: float,
    weights: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute percentage of correct keypoints (PCK).

    Parameters
    ----------
    predicted : torch.Tensor
        Predicted joint positions with shape ``[..., joints, 3]``.
    target : torch.Tensor
        Target joint positions with the same shape as ``predicted``.
    threshold : float
        Distance threshold used to mark a joint as correct.
    weights : torch.Tensor, optional
        Non-negative joint weights broadcastable to ``[..., joints]``.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over leading sample dimensions after averaging over joints.

    Returns
    -------
    torch.Tensor
        PCK value in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``threshold`` is negative, or if shapes, weights, or ``reduction``
        are invalid.
    """

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")

    errors = joint_position_error(predicted, target)
    weights = _canonicalize_weights(
        weights,
        joint_shape=errors.shape,
        dtype=errors.dtype,
        device=errors.device,
    )
    correct = (errors <= threshold).to(errors.dtype)
    return _reduce(_weighted_mean(correct, weights), reduction)


pa_mpjpe = procrustes_aligned_mpjpe
mpjpe = mean_per_joint_position_error
