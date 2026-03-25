from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets, merge_points
from ._spec import SkeletonSpec
from ._templates import t_pose
from .face68 import face68_parent_mapping, face68_points
from .halpe26 import HALPE26_NAMES
from .hand21 import (
    LEFT_HAND_WHOLEBODY_NAMES,
    RIGHT_HAND_WHOLEBODY_NAMES,
    body_hand21_points,
)

HALPE_LEFT_HAND_NAMES = LEFT_HAND_WHOLEBODY_NAMES
HALPE_RIGHT_HAND_NAMES = RIGHT_HAND_WHOLEBODY_NAMES

HALPE_FULLBODY_NAMES = (
    HALPE26_NAMES
    + tuple(f"face-{idx}" for idx in range(68))
    + HALPE_LEFT_HAND_NAMES
    + HALPE_RIGHT_HAND_NAMES
)


def halpe_fullbody_parents() -> tuple[int, ...]:
    parents = list(
        (
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
    )

    for parent in face68_parent_mapping(root_parent=-999):
        parents.append(0 if parent == -999 else 26 + parent)

    parents.extend(
        [
            9,
            94,
            95,
            96,
            97,
            94,
            99,
            100,
            101,
            94,
            103,
            104,
            105,
            94,
            107,
            108,
            109,
            94,
            111,
            112,
            113,
        ]
    )
    parents.extend(
        [
            10,
            115,
            116,
            117,
            118,
            115,
            120,
            121,
            122,
            115,
            124,
            125,
            126,
            115,
            128,
            129,
            130,
            115,
            132,
            133,
            134,
        ]
    )
    return tuple(parents)


@lru_cache(maxsize=None)
def halpe_fullbody_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = halpe_fullbody_parents()
    positions = merge_points(
        t_pose(),
        face68_points(prefix="face-"),
        body_hand21_points("left"),
        body_hand21_points("right"),
    )
    offsets = absolute_positions_to_offsets(
        HALPE_FULLBODY_NAMES, parents, positions, dtype=dtype
    )
    return SkeletonSpec(
        "halpe_fullbody", HALPE_FULLBODY_NAMES, parents, offsets, root_index=19
    )


class HalpeFullBodyModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(halpe_fullbody_spec)


class HalpeFullBodyModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(halpe_fullbody_spec)
