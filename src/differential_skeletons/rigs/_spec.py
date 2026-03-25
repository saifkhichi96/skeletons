from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_offsets: torch.Tensor
    root_index: int
    metadata: dict[str, Any] | None = None

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)
