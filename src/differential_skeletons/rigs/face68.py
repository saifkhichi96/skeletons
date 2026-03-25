from __future__ import annotations

from functools import lru_cache

import torch

from ._base import _SpecBackedModel, _SpecBackedModelLayer
from ._shared import absolute_positions_to_offsets, ellipse_points
from ._spec import SkeletonSpec

FACE68_NAMES = tuple(f"face_{idx}" for idx in range(68))


def face68_parent_mapping(root_parent: int) -> list[int]:
    parents = [0] * 68
    parents[27] = root_parent

    parents[0] = 27
    for idx in range(1, 17):
        parents[idx] = idx - 1

    parents[17] = 27
    for idx in range(18, 22):
        parents[idx] = idx - 1

    parents[22] = 27
    for idx in range(23, 27):
        parents[idx] = idx - 1

    for idx in range(28, 36):
        parents[idx] = idx - 1

    parents[36] = 27
    for idx in range(37, 42):
        parents[idx] = idx - 1

    parents[42] = 27
    for idx in range(43, 48):
        parents[idx] = idx - 1

    parents[48] = 27
    for idx in range(49, 60):
        parents[idx] = idx - 1

    parents[60] = 27
    for idx in range(61, 68):
        parents[idx] = idx - 1
    return parents


def face68_points(
    prefix: str = "face-", standalone: bool = False
) -> dict[str, tuple[float, float, float]]:
    def name(index: int) -> str:
        return f"{prefix}{index}"

    jaw = ellipse_points(
        cx=0.0,
        cy=0.71,
        cz=0.030,
        rx=0.115,
        ry=0.140,
        num=17,
        start_deg=200.0,
        end_deg=-20.0,
    )
    right_brow = ellipse_points(
        cx=-0.045,
        cy=0.85,
        cz=0.085,
        rx=0.050,
        ry=0.018,
        num=5,
        start_deg=200.0,
        end_deg=340.0,
    )
    left_brow = ellipse_points(
        cx=0.045,
        cy=0.85,
        cz=0.085,
        rx=0.050,
        ry=0.018,
        num=5,
        start_deg=200.0,
        end_deg=340.0,
    )
    nose_bridge = [(0.0, 0.84 - 0.035 * idx, 0.102 + 0.004 * idx) for idx in range(4)]
    nose_lower = [
        (-0.045, 0.72, 0.112),
        (-0.025, 0.70, 0.122),
        (0.0, 0.69, 0.128),
        (0.025, 0.70, 0.122),
        (0.045, 0.72, 0.112),
    ]
    right_eye = ellipse_points(
        cx=-0.045,
        cy=0.79,
        cz=0.090,
        rx=0.030,
        ry=0.015,
        num=6,
        start_deg=180.0,
        end_deg=540.0,
    )
    left_eye = ellipse_points(
        cx=0.045,
        cy=0.79,
        cz=0.090,
        rx=0.030,
        ry=0.015,
        num=6,
        start_deg=180.0,
        end_deg=540.0,
    )
    outer_mouth = ellipse_points(
        cx=0.0,
        cy=0.66,
        cz=0.112,
        rx=0.060,
        ry=0.030,
        num=12,
        start_deg=180.0,
        end_deg=540.0,
    )
    inner_mouth = ellipse_points(
        cx=0.0,
        cy=0.66,
        cz=0.117,
        rx=0.032,
        ry=0.016,
        num=8,
        start_deg=180.0,
        end_deg=540.0,
    )

    points: dict[str, tuple[float, float, float]] = {}
    for idx, point in enumerate(jaw):
        points[name(idx)] = point
    for idx, point in enumerate(right_brow, start=17):
        points[name(idx)] = point
    for idx, point in enumerate(left_brow, start=22):
        points[name(idx)] = point
    for idx, point in enumerate(nose_bridge, start=27):
        points[name(idx)] = point
    for idx, point in enumerate(nose_lower, start=31):
        points[name(idx)] = point
    for idx, point in enumerate(right_eye, start=36):
        points[name(idx)] = point
    for idx, point in enumerate(left_eye, start=42):
        points[name(idx)] = point
    for idx, point in enumerate(outer_mouth, start=48):
        points[name(idx)] = point
    for idx, point in enumerate(inner_mouth, start=60):
        points[name(idx)] = point

    if standalone:
        root = points[name(27)]
        points = {
            key: (value[0] - root[0], value[1] - root[1], value[2] - root[2])
            for key, value in points.items()
        }
    return points


@lru_cache(maxsize=None)
def face68_spec(dtype: torch.dtype = torch.float32) -> SkeletonSpec:
    parents = tuple(face68_parent_mapping(root_parent=-1))
    offsets = absolute_positions_to_offsets(
        FACE68_NAMES,
        parents,
        face68_points(prefix="face_", standalone=True),
        dtype=dtype,
    )
    return SkeletonSpec("face68", FACE68_NAMES, parents, offsets, root_index=27)


class Face68Model(_SpecBackedModel):
    SPEC_FACTORY = staticmethod(face68_spec)


class Face68ModelLayer(_SpecBackedModelLayer):
    SPEC_FACTORY = staticmethod(face68_spec)
