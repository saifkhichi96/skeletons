from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Iterator, Optional

import torch

Tensor = torch.Tensor


@dataclass
class SkeletalOutput:
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
    bone_scales: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    scaled_offsets: Optional[Tensor] = None
    local_rotations: Optional[Tensor] = None
    global_rotations: Optional[Tensor] = None
    local_transforms: Optional[Tensor] = None
    global_transforms: Optional[Tensor] = None
    joint_names: Optional[tuple[str, ...]] = None

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self) -> Iterator[str]:
        return iter(field.name for field in fields(self))

    def values(self) -> Iterator[object]:
        return iter(getattr(self, field.name) for field in fields(self))

    def items(self) -> Iterator[tuple[str, object]]:
        return iter((field.name, getattr(self, field.name)) for field in fields(self))

    def asdict(self) -> dict[str, object]:
        return asdict(self)
