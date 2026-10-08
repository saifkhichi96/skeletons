from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets
from ._spec import SkeletonSpec
from ._templates import t_pose

COCO_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def coco_parents() -> tuple[int, ...]:
    return (-1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14)


@lru_cache(maxsize=None)
def coco_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = coco_parents()
    offsets = absolute_positions_to_offsets(COCO_NAMES, parents, t_pose(), dtype=dtype)
    return SkeletonSpec("coco", COCO_NAMES, parents, offsets, root_index=0)


class CocoModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(coco_spec)


class CocoModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(coco_spec)
