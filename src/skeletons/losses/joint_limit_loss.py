from __future__ import annotations

import torch

from ..model import SkeletalModel


def joint_limit_loss(
    model: SkeletalModel, q: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """Return the configured rig's joint-limit violation penalty."""
    return model.joint_limit_loss(q, margin=margin)
