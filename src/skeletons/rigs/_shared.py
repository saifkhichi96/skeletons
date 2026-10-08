from __future__ import annotations

import math
from typing import Iterable

import torch


def ellipse_points(
    *,
    cx: float,
    cy: float,
    cz: float,
    rx: float,
    ry: float,
    num: int,
    start_deg: float,
    end_deg: float,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for idx in range(num):
        if num == 1:
            angle = math.radians(start_deg)
        else:
            angle = math.radians(start_deg + (end_deg - start_deg) * idx / (num - 1))
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle), cz))
    return points


def merge_points(
    *point_sets: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    merged: dict[str, tuple[float, float, float]] = {}
    for point_set in point_sets:
        merged.update(point_set)
    return merged


def absolute_positions_to_offsets(
    joint_names: Iterable[str],
    parents: list[int] | tuple[int, ...],
    positions: dict[str, tuple[float, float, float]],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    names = list(joint_names)
    if len(names) != len(parents):
        raise ValueError("joint_names and parents must have the same length.")

    absolute: list[tuple[float, float, float]] = []
    missing: list[str] = []
    for name in names:
        if name not in positions:
            missing.append(name)
            absolute.append((0.0, 0.0, 0.0))
        else:
            absolute.append(positions[name])

    if missing:
        formatted = ", ".join(sorted(missing))
        raise KeyError(f"Missing canonical rest-pose coordinates for: {formatted}")

    absolute_tensor = torch.tensor(absolute, dtype=dtype)
    offsets = torch.zeros_like(absolute_tensor)
    for joint_index, parent_index in enumerate(parents):
        if parent_index == -1:
            continue
        offsets[joint_index] = (
            absolute_tensor[joint_index] - absolute_tensor[parent_index]
        )
    return offsets
