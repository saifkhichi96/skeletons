from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets
from ._spec import SkeletonSpec
from ._templates import t_pose

HALPE26_NAMES = (
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
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)


def halpe26_parents() -> tuple[int, ...]:
    return (
        17,
        0,
        0,
        1,
        2,
        18,
        18,
        5,
        6,
        7,
        8,
        19,
        19,
        11,
        12,
        13,
        14,
        18,
        19,
        -1,
        15,
        16,
        20,
        21,
        15,
        16,
    )


@lru_cache(maxsize=None)
def halpe26_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = halpe26_parents()
    offsets = absolute_positions_to_offsets(
        HALPE26_NAMES, parents, t_pose(), dtype=dtype
    )
    return SkeletonSpec("halpe26", HALPE26_NAMES, parents, offsets, root_index=19)


class Halpe26Model(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(halpe26_spec)


class Halpe26ModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(halpe26_spec)
