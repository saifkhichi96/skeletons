from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets
from ._spec import SkeletonSpec
from ._templates import t_pose

MPII_NAMES = (
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "thorax",
    "upper_neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
)


def mpii_parents() -> tuple[int, ...]:
    return (1, 2, 6, 6, 3, 4, -1, 6, 7, 8, 11, 12, 8, 8, 13, 14)


@lru_cache(maxsize=None)
def mpii_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = mpii_parents()
    offsets = absolute_positions_to_offsets(MPII_NAMES, parents, t_pose(), dtype=dtype)
    return SkeletonSpec("mpii", MPII_NAMES, parents, offsets, root_index=6)


class MPIIModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(mpii_spec)


class MPIIModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(mpii_spec)
