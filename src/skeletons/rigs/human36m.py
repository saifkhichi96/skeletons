from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets
from ._spec import SkeletonSpec
from ._templates import t_pose

HUMAN36M_NAMES = (
    "root",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck_base",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


def human36m_parents() -> tuple[int, ...]:
    return (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


@lru_cache(maxsize=None)
def human36m_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = human36m_parents()
    offsets = absolute_positions_to_offsets(
        HUMAN36M_NAMES, parents, t_pose(), dtype=dtype
    )
    return SkeletonSpec("human36m", HUMAN36M_NAMES, parents, offsets, root_index=0)


class Human36MModel(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(human36m_spec)


class Human36MModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(human36m_spec)
