from __future__ import annotations

import math
from typing import Iterable

import torch


def common_body_points() -> dict[str, tuple[float, float, float]]:
    return {
        "root": (0.0, 0.0, 0.0),
        "pelvis": (0.0, 0.0, 0.0),
        "hip": (0.0, 0.0, 0.0),
        "left_hip": (0.10, -0.01, 0.0),
        "right_hip": (-0.10, -0.01, 0.0),
        "spine": (0.0, 0.18, 0.0),
        "spine_01": (0.0, 0.12, 0.0),
        "spine_02": (0.0, 0.24, 0.0),
        "spine_03": (0.0, 0.36, 0.0),
        "spine_04": (0.0, 0.48, -0.01),
        "spine_05": (0.0, 0.60, 0.0),
        "thorax": (0.0, 0.45, 0.0),
        "neck": (0.0, 0.62, 0.01),
        "upper_neck": (0.0, 0.64, 0.02),
        "neck_base": (0.0, 0.62, 0.01),
        "neck_02": (0.0, 0.69, 0.02),
        "neck_03": (0.0, 0.75, 0.03),
        "head": (0.0, 0.82, 0.04),
        "head_top": (0.0, 0.94, 0.04),
        "nose": (0.0, 0.80, 0.10),
        "left_eye": (0.035, 0.83, 0.095),
        "right_eye": (-0.035, 0.83, 0.095),
        "left_ear": (0.09, 0.82, 0.03),
        "right_ear": (-0.09, 0.82, 0.03),
        "left_clavicle": (0.08, 0.62, 0.01),
        "right_clavicle": (-0.08, 0.62, 0.01),
        "left_shoulder": (0.18, 0.62, 0.0),
        "right_shoulder": (-0.18, 0.62, 0.0),
        "left_elbow": (0.50, 0.60, 0.0),
        "right_elbow": (-0.50, 0.60, 0.0),
        "left_wrist": (0.78, 0.58, 0.0),
        "right_wrist": (-0.78, 0.58, 0.0),
        "left_latissimus": (0.14, 0.46, -0.03),
        "right_latissimus": (-0.14, 0.46, -0.03),
        "left_knee": (0.10, -0.45, 0.02),
        "right_knee": (-0.10, -0.45, 0.02),
        "left_ankle": (0.10, -0.88, 0.03),
        "right_ankle": (-0.10, -0.88, 0.03),
        "left_foot": (0.10, -0.90, 0.11),
        "right_foot": (-0.10, -0.90, 0.11),
        "left_heel": (0.09, -0.92, -0.06),
        "right_heel": (-0.09, -0.92, -0.06),
        "left_big_toe": (0.12, -0.90, 0.12),
        "right_big_toe": (-0.12, -0.90, 0.12),
        "left_small_toe": (0.06, -0.90, 0.11),
        "right_small_toe": (-0.06, -0.90, 0.11),
        "head_root": (0.0, 0.82, 0.04),
        "left_hand_root": (0.78, 0.58, 0.0),
        "right_hand_root": (-0.78, 0.58, 0.0),
    }


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

    def p(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (wrist_x + direction * x, y, z)

    return {
        f"{prefix}hand_root": (wrist_x, 0.58, 0.0),
        f"{prefix}thumb1": p(0.040, 0.565, -0.020),
        f"{prefix}thumb2": p(0.090, 0.545, -0.030),
        f"{prefix}thumb3": p(0.140, 0.525, -0.020),
        f"{prefix}thumb4": p(0.190, 0.510, -0.010),
        f"{prefix}forefinger1": p(0.055, 0.600, 0.0),
        f"{prefix}forefinger2": p(0.115, 0.610, 0.0),
        f"{prefix}forefinger3": p(0.175, 0.620, 0.0),
        f"{prefix}forefinger4": p(0.235, 0.630, 0.0),
        f"{prefix}middle_finger1": p(0.060, 0.585, 0.0),
        f"{prefix}middle_finger2": p(0.125, 0.585, 0.0),
        f"{prefix}middle_finger3": p(0.190, 0.585, 0.0),
        f"{prefix}middle_finger4": p(0.255, 0.585, 0.0),
        f"{prefix}ring_finger1": p(0.055, 0.565, 0.0),
        f"{prefix}ring_finger2": p(0.115, 0.555, 0.0),
        f"{prefix}ring_finger3": p(0.175, 0.545, 0.0),
        f"{prefix}ring_finger4": p(0.235, 0.535, 0.0),
        f"{prefix}pinky_finger1": p(0.045, 0.545, 0.0),
        f"{prefix}pinky_finger2": p(0.095, 0.525, 0.0),
        f"{prefix}pinky_finger3": p(0.145, 0.505, 0.0),
        f"{prefix}pinky_finger4": p(0.195, 0.485, 0.0),
    }


def _ellipse_points(
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
    for i in range(num):
        if num == 1:
            angle = math.radians(start_deg)
        else:
            angle = math.radians(start_deg + (end_deg - start_deg) * i / (num - 1))
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        points.append((x, y, cz))
    return points


def face68_points(
    prefix: str = "face-", standalone: bool = False
) -> dict[str, tuple[float, float, float]]:
    def name(idx: int) -> str:
        return f"{prefix}{idx}"

    points: dict[str, tuple[float, float, float]] = {}

    jaw = _ellipse_points(
        cx=0.0,
        cy=0.71,
        cz=0.030,
        rx=0.115,
        ry=0.140,
        num=17,
        start_deg=200.0,
        end_deg=-20.0,
    )
    right_brow = _ellipse_points(
        cx=-0.045,
        cy=0.85,
        cz=0.085,
        rx=0.050,
        ry=0.018,
        num=5,
        start_deg=200.0,
        end_deg=340.0,
    )
    left_brow = _ellipse_points(
        cx=0.045,
        cy=0.85,
        cz=0.085,
        rx=0.050,
        ry=0.018,
        num=5,
        start_deg=200.0,
        end_deg=340.0,
    )
    nose_bridge = [(0.0, 0.84 - 0.035 * i, 0.102 + 0.004 * i) for i in range(4)]
    nose_lower = [
        (-0.045, 0.72, 0.112),
        (-0.025, 0.70, 0.122),
        (0.0, 0.69, 0.128),
        (0.025, 0.70, 0.122),
        (0.045, 0.72, 0.112),
    ]
    right_eye = _ellipse_points(
        cx=-0.045,
        cy=0.79,
        cz=0.090,
        rx=0.030,
        ry=0.015,
        num=6,
        start_deg=180.0,
        end_deg=540.0,
    )
    left_eye = _ellipse_points(
        cx=0.045,
        cy=0.79,
        cz=0.090,
        rx=0.030,
        ry=0.015,
        num=6,
        start_deg=180.0,
        end_deg=540.0,
    )
    outer_mouth = _ellipse_points(
        cx=0.0,
        cy=0.66,
        cz=0.112,
        rx=0.060,
        ry=0.030,
        num=12,
        start_deg=180.0,
        end_deg=540.0,
    )
    inner_mouth = _ellipse_points(
        cx=0.0,
        cy=0.66,
        cz=0.117,
        rx=0.032,
        ry=0.016,
        num=8,
        start_deg=180.0,
        end_deg=540.0,
    )

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
    joint_names = list(joint_names)
    if len(joint_names) != len(parents):
        raise ValueError("joint_names and parents must have the same length.")

    absolute = []
    missing: list[str] = []
    for name in joint_names:
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
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            continue
        offsets[joint_idx] = absolute_tensor[joint_idx] - absolute_tensor[parent_idx]
    return offsets
