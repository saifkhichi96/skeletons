from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets
from ._spec import SkeletonSpec
from ._templates import t_pose

SPINETRACK_NAMES = (
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
    "spine_01",
    "spine_02",
    "spine_03",
    "spine_04",
    "spine_05",
    "left_latissimus",
    "right_latissimus",
    "left_clavicle",
    "right_clavicle",
    "neck_02",
    "neck_03",
)


def spinetrack_parents() -> tuple[int, ...]:
    return (
        17,
        0,
        0,
        1,
        2,
        33,
        34,
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
        36,
        30,
        -1,
        15,
        16,
        20,
        21,
        15,
        16,
        19,
        26,
        27,
        28,
        29,
        29,
        29,
        18,
        18,
        18,
        35,
    )


@lru_cache(maxsize=None)
def spinetrack_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = spinetrack_parents()
    offsets = absolute_positions_to_offsets(
        SPINETRACK_NAMES, parents, t_pose(), dtype=dtype
    )
    return SkeletonSpec("spinetrack", SPINETRACK_NAMES, parents, offsets, root_index=19)


class SpineTrackModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(spinetrack_spec)


class SpineTrackModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(spinetrack_spec)
