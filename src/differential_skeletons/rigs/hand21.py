from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets
from ._spec import SkeletonSpec

HAND21_NAMES = (
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb4",
    "forefinger1",
    "forefinger2",
    "forefinger3",
    "forefinger4",
    "middle_finger1",
    "middle_finger2",
    "middle_finger3",
    "middle_finger4",
    "ring_finger1",
    "ring_finger2",
    "ring_finger3",
    "ring_finger4",
    "pinky_finger1",
    "pinky_finger2",
    "pinky_finger3",
    "pinky_finger4",
)

LEFT_HAND_WHOLEBODY_NAMES = (
    "left_hand_root",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "left_thumb4",
    "left_forefinger1",
    "left_forefinger2",
    "left_forefinger3",
    "left_forefinger4",
    "left_middle_finger1",
    "left_middle_finger2",
    "left_middle_finger3",
    "left_middle_finger4",
    "left_ring_finger1",
    "left_ring_finger2",
    "left_ring_finger3",
    "left_ring_finger4",
    "left_pinky_finger1",
    "left_pinky_finger2",
    "left_pinky_finger3",
    "left_pinky_finger4",
)

RIGHT_HAND_WHOLEBODY_NAMES = (
    "right_hand_root",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "right_thumb4",
    "right_forefinger1",
    "right_forefinger2",
    "right_forefinger3",
    "right_forefinger4",
    "right_middle_finger1",
    "right_middle_finger2",
    "right_middle_finger3",
    "right_middle_finger4",
    "right_ring_finger1",
    "right_ring_finger2",
    "right_ring_finger3",
    "right_ring_finger4",
    "right_pinky_finger1",
    "right_pinky_finger2",
    "right_pinky_finger3",
    "right_pinky_finger4",
)


def generic_hand21_points() -> dict[str, tuple[float, float, float]]:
    return {
        "wrist": (0.0, 0.0, 0.0),
        "thumb1": (0.035, -0.020, -0.020),
        "thumb2": (0.075, -0.040, -0.025),
        "thumb3": (0.115, -0.055, -0.020),
        "thumb4": (0.155, -0.065, -0.015),
        "forefinger1": (0.045, 0.020, 0.0),
        "forefinger2": (0.095, 0.025, 0.0),
        "forefinger3": (0.145, 0.030, 0.0),
        "forefinger4": (0.195, 0.035, 0.0),
        "middle_finger1": (0.050, 0.005, 0.0),
        "middle_finger2": (0.105, 0.005, 0.0),
        "middle_finger3": (0.160, 0.005, 0.0),
        "middle_finger4": (0.215, 0.005, 0.0),
        "ring_finger1": (0.045, -0.010, 0.0),
        "ring_finger2": (0.095, -0.015, 0.0),
        "ring_finger3": (0.145, -0.020, 0.0),
        "ring_finger4": (0.195, -0.025, 0.0),
        "pinky_finger1": (0.040, -0.025, 0.0),
        "pinky_finger2": (0.082, -0.038, 0.0),
        "pinky_finger3": (0.124, -0.050, 0.0),
        "pinky_finger4": (0.166, -0.062, 0.0),
    }


def body_hand21_points(side: str) -> dict[str, tuple[float, float, float]]:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    wrist_x = 0.78 if side == "left" else -0.78
    direction = 1.0 if side == "left" else -1.0
    prefix = f"{side}_"

    def point(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (wrist_x + direction * x, y, z)

    return {
        f"{prefix}hand_root": (wrist_x, 0.58, 0.0),
        f"{prefix}thumb1": point(0.040, 0.565, -0.020),
        f"{prefix}thumb2": point(0.090, 0.545, -0.030),
        f"{prefix}thumb3": point(0.140, 0.525, -0.020),
        f"{prefix}thumb4": point(0.190, 0.510, -0.010),
        f"{prefix}forefinger1": point(0.055, 0.600, 0.0),
        f"{prefix}forefinger2": point(0.115, 0.610, 0.0),
        f"{prefix}forefinger3": point(0.175, 0.620, 0.0),
        f"{prefix}forefinger4": point(0.235, 0.630, 0.0),
        f"{prefix}middle_finger1": point(0.060, 0.585, 0.0),
        f"{prefix}middle_finger2": point(0.125, 0.585, 0.0),
        f"{prefix}middle_finger3": point(0.190, 0.585, 0.0),
        f"{prefix}middle_finger4": point(0.255, 0.585, 0.0),
        f"{prefix}ring_finger1": point(0.055, 0.565, 0.0),
        f"{prefix}ring_finger2": point(0.115, 0.555, 0.0),
        f"{prefix}ring_finger3": point(0.175, 0.545, 0.0),
        f"{prefix}ring_finger4": point(0.235, 0.535, 0.0),
        f"{prefix}pinky_finger1": point(0.045, 0.545, 0.0),
        f"{prefix}pinky_finger2": point(0.095, 0.525, 0.0),
        f"{prefix}pinky_finger3": point(0.145, 0.505, 0.0),
        f"{prefix}pinky_finger4": point(0.195, 0.485, 0.0),
    }


def hand21_parents(root_parent: int = -1) -> tuple[int, ...]:
    return (
        root_parent,
        0,
        1,
        2,
        3,
        0,
        5,
        6,
        7,
        0,
        9,
        10,
        11,
        0,
        13,
        14,
        15,
        0,
        17,
        18,
        19,
    )


@lru_cache(maxsize=None)
def hand21_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = hand21_parents()
    offsets = absolute_positions_to_offsets(
        HAND21_NAMES, parents, generic_hand21_points(), dtype=dtype
    )
    return SkeletonSpec("hand21", HAND21_NAMES, parents, offsets, root_index=0)


class Hand21Model(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(hand21_spec)


class Hand21ModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(hand21_spec)
