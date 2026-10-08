from __future__ import annotations

from typing import Sequence

import torch

from ..model import SkeletalModel


def foot_sliding_loss(
    model: SkeletalModel,
    q_sequence: torch.Tensor,
    root_sequence: torch.Tensor | None = None,
    contact_names: Sequence[str] | None = None,
    contact_mask: torch.Tensor | None = None,
    *,
    up_axis: int = 1,
) -> torch.Tensor:
    """Penalize horizontal motion of active foot contacts."""
    if q_sequence.ndim != 3:
        raise ValueError("q_sequence must have shape [T, B, D].")
    contact_names = model.contact_names if contact_names is None else contact_names
    if not contact_names:
        raise ValueError("Configure contacts or provide contact_names.")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2.")
    if q_sequence.shape[0] < 2:
        return q_sequence.sum() * 0.0
    positions = []
    idx = torch.tensor(
        [model.contact_index(name) for name in contact_names], device=q_sequence.device
    )
    for t in range(q_sequence.shape[0]):
        root = None if root_sequence is None else root_sequence[t]
        fk = model.forward_kinematics(q_sequence[t], root=root)
        positions.append(fk.contact_positions.index_select(-2, idx))
    p = torch.stack(positions, dim=0)
    horizontal = [axis for axis in range(3) if axis != up_axis]
    penalty = (p[1:, ..., horizontal] - p[:-1, ..., horizontal]).square().sum(dim=-1)
    if contact_mask is not None:
        mask = contact_mask.to(device=penalty.device, dtype=penalty.dtype)
        if mask.shape != penalty.shape:
            raise ValueError("contact_mask must have shape [T - 1, B, C].")
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("contact_mask must be finite and nonnegative.")
        penalty = penalty * mask
        return penalty.sum() / mask.sum().clamp_min(1e-8)
    return penalty.mean()
