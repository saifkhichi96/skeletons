from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets, merge_points
from ._spec import SkeletonSpec
from ._templates import t_pose
from .coco import COCO_NAMES
from .face68 import face68_parent_mapping, face68_points
from .hand21 import (
    LEFT_HAND_WHOLEBODY_NAMES,
    RIGHT_HAND_WHOLEBODY_NAMES,
    body_hand21_points,
)

COCO_WHOLEBODY_NAMES = (
    COCO_NAMES
    + (
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    )
    + tuple(f"face-{idx}" for idx in range(68))
    + LEFT_HAND_WHOLEBODY_NAMES
    + RIGHT_HAND_WHOLEBODY_NAMES
)


def coco_wholebody_parents() -> tuple[int, ...]:
    parents = [
        -1,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        5,
        6,
        11,
        12,
        13,
        14,
        15,
        17,
        15,
        16,
        20,
        16,
    ]

    for parent in face68_parent_mapping(root_parent=-999):
        parents.append(0 if parent == -999 else 23 + parent)

    parents.extend(
        [
            9,
            91,
            92,
            93,
            94,
            91,
            96,
            97,
            98,
            91,
            100,
            101,
            102,
            91,
            104,
            105,
            106,
            91,
            108,
            109,
            110,
        ]
    )
    parents.extend(
        [
            10,
            112,
            113,
            114,
            115,
            112,
            117,
            118,
            119,
            112,
            121,
            122,
            123,
            112,
            125,
            126,
            127,
            112,
            129,
            130,
            131,
        ]
    )
    return tuple(parents)


@lru_cache(maxsize=None)
def coco_wholebody_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = coco_wholebody_parents()
    positions = merge_points(
        t_pose(),
        face68_points(prefix="face-"),
        body_hand21_points("left"),
        body_hand21_points("right"),
    )
    offsets = absolute_positions_to_offsets(
        COCO_WHOLEBODY_NAMES, parents, positions, dtype=dtype
    )
    return SkeletonSpec(
        "coco_wholebody", COCO_WHOLEBODY_NAMES, parents, offsets, root_index=0
    )


class CocoWholeBodyModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(coco_wholebody_spec)


class CocoWholeBodyModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(coco_wholebody_spec)
