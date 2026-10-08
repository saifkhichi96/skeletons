from __future__ import annotations

import torch

from ..model import SkeletalModel


def center_of_mass_support_loss(
    model: SkeletalModel,
    q: torch.Tensor,
    root: torch.Tensor | None = None,
    support_points: torch.Tensor | None = None,
    *,
    up_axis: int = 1,
) -> torch.Tensor:
    """Penalize horizontal COM displacement from the support center."""
    com = model.center_of_mass(q, root=root)
    if support_points is None:
        support_points = model.forward_kinematics(q, root=root).contact_positions
    if support_points is None or support_points.shape[-2] == 0:
        raise ValueError("Provide support_points or configure contact frames.")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2.")
    horizontal = [axis for axis in range(3) if axis != up_axis]
    support_center = support_points[..., horizontal].mean(dim=-2)
    return (com[..., horizontal] - support_center).square().sum(dim=-1).mean()
