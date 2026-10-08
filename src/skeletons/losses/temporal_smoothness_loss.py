from __future__ import annotations

import torch


def temporal_smoothness_loss(q_sequence: torch.Tensor, order: int = 1) -> torch.Tensor:
    """Velocity or acceleration smoothness over a ``[T, ..., D]`` pose sequence."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    if q_sequence.ndim < 2:
        raise ValueError("q_sequence must have shape [T, ..., D].")
    if q_sequence.shape[0] <= order or q_sequence.shape[-1] == 0:
        return q_sequence.sum() * 0.0
    diff = q_sequence[1:] - q_sequence[:-1]
    if order == 2:
        diff = diff[1:] - diff[:-1]
    return diff.square().mean()
