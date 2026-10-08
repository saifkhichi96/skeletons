from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class AnimationPreset:
    """Procedural animation preset available in the playground.

    Parameters
    ----------
    key : str
        Stable preset identifier stored in combo-box user data.
    label : str
        Human-readable preset label.
    period : float
        Natural loop period in seconds.
    matcher : callable
        Predicate deciding whether the preset supports a skeleton model.
    generator : callable
        Function producing additive pose and translation tensors.
    """

    key: str
    label: str
    period: float
    matcher: Callable[[Any], bool]
    generator: Callable[[Any, float], tuple[torch.Tensor, torch.Tensor]]


def _animation_model(source: Any) -> Any:
    return getattr(source, "model", source)


def _stable_phase_seed(text: str) -> float:
    total = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return math.radians(float(total % 360))


def _has_joints(model: Any, *joint_names: str) -> bool:
    available = set(model.joint_names)
    return all(name in available for name in joint_names)


def _has_any_prefix(model: Any, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefixes) for name in model.joint_names)


def _set_axis_angle(
    pose: torch.Tensor,
    model: Any,
    joint_name: str,
    values: tuple[float | None, float | None, float | None],
) -> None:
    if joint_name not in model.joint_name_to_index:
        return
    joint_index = model.joint_name_to_index[joint_name]
    for axis, value in enumerate(values):
        if value is not None:
            pose[joint_index, axis] = value


def _make_animation_buffers(model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = model.rest_offsets.dtype
    device = model.rest_offsets.device
    return (
        torch.zeros(model.NUM_JOINTS, 3, dtype=dtype, device=device),
        torch.zeros(3, dtype=dtype, device=device),
    )


def _generate_root_sway(
    source: Any,
    time_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _animation_model(source)
    phase = (2.0 * math.pi * time_s) / 2.8
    pose, translation = _make_animation_buffers(model)
    root = model.root_index
    pose[root, 0] = 0.07 * math.sin(phase * 0.5)
    pose[root, 1] = 0.10 * math.sin(phase)
    pose[root, 2] = 0.05 * math.sin(phase + 0.8)
    translation[0] = 0.03 * math.sin(phase)
    translation[1] = 0.03 * (1.0 - math.cos(phase * 2.0)) * 0.5
    translation[2] = 0.05 * math.cos(phase)
    return pose, translation


def _generate_walk_cycle(
    source: Any,
    time_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _animation_model(source)
    phase = (2.0 * math.pi * time_s) / 1.25
    pose, translation = _make_animation_buffers(model)

    _set_axis_angle(
        pose,
        model,
        model.joint_names[model.root_index],
        (
            0.03 * math.sin(phase * 2.0),
            0.05 * math.sin(phase),
            0.03 * math.sin(phase + math.pi / 2.0),
        ),
    )

    swing = math.sin(phase)
    support = math.sin(phase + math.pi)
    knee_left = max(0.0, math.sin(phase + math.pi / 2.0))
    knee_right = max(0.0, math.sin(phase - math.pi / 2.0))

    _set_axis_angle(pose, model, "left_hip", (-0.50 * swing, 0.0, 0.0))
    _set_axis_angle(pose, model, "right_hip", (0.50 * swing, 0.0, 0.0))
    _set_axis_angle(pose, model, "left_knee", (-0.65 * knee_left, 0.0, 0.0))
    _set_axis_angle(pose, model, "right_knee", (-0.65 * knee_right, 0.0, 0.0))
    _set_axis_angle(pose, model, "left_foot", (0.18 * support, 0.0, 0.0))
    _set_axis_angle(pose, model, "right_foot", (-0.18 * support, 0.0, 0.0))

    _set_axis_angle(pose, model, "left_shoulder", (0.28 * swing, 0.0, 0.0))
    _set_axis_angle(pose, model, "right_shoulder", (-0.28 * swing, 0.0, 0.0))
    _set_axis_angle(
        pose, model, "left_elbow", (-0.12 - 0.08 * max(0.0, support), 0.0, 0.0)
    )
    _set_axis_angle(
        pose, model, "right_elbow", (-0.12 - 0.08 * max(0.0, -support), 0.0, 0.0)
    )
    _set_axis_angle(pose, model, "spine", (0.04 * math.sin(phase + 0.6), 0.0, 0.0))
    _set_axis_angle(pose, model, "thorax", (0.05 * math.sin(phase + 0.3), 0.0, 0.0))
    _set_axis_angle(pose, model, "neck_base", (0.03 * math.sin(phase), 0.0, 0.0))

    translation[1] = 0.03 * (1.0 - math.cos(phase * 2.0)) * 0.5
    translation[2] = 0.06 * math.sin(phase)
    return pose, translation


def _generate_arm_wave(
    source: Any,
    time_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _animation_model(source)
    phase = (2.0 * math.pi * time_s) / 1.6
    pose, translation = _make_animation_buffers(model)
    _set_axis_angle(pose, model, "right_shoulder", (-0.25, 0.0, -0.95))
    _set_axis_angle(
        pose, model, "right_elbow", (-0.35 - 0.20 * math.sin(phase), 0.0, 0.0)
    )
    _set_axis_angle(
        pose, model, "right_wrist", (0.35 * math.sin(phase * 2.0), 0.0, 0.0)
    )
    _set_axis_angle(pose, model, "left_shoulder", (0.10, 0.0, 0.18))
    _set_axis_angle(pose, model, "left_elbow", (-0.10, 0.0, 0.0))
    _set_axis_angle(pose, model, "thorax", (0.03 * math.sin(phase), 0.0, -0.04))
    _set_axis_angle(pose, model, "neck_base", (0.0, 0.0, 0.04 * math.sin(phase)))
    return pose, translation


def _generate_finger_wave(
    source: Any,
    time_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _animation_model(source)
    phase = (2.0 * math.pi * time_s) / 1.8
    pose, translation = _make_animation_buffers(model)
    finger_groups = [
        ("thumb", 0.20),
        ("forefinger", 0.65),
        ("middle_finger", 1.10),
        ("ring_finger", 1.55),
        ("pinky_finger", 2.00),
    ]
    prefixes = ("", "left_", "right_")
    for side_prefix in prefixes:
        for finger_name, offset in finger_groups:
            for segment_index in range(1, 5):
                joint_name = f"{side_prefix}{finger_name}{segment_index}"
                curl = -0.30 - 0.28 * math.sin(phase + offset + segment_index * 0.18)
                spread = 0.08 * math.sin(phase * 0.5 + offset)
                _set_axis_angle(pose, model, joint_name, (curl, spread, 0.0))
    _set_axis_angle(pose, model, "wrist", (0.10 * math.sin(phase * 0.5), 0.0, 0.0))
    _set_axis_angle(
        pose, model, "left_hand_root", (0.08 * math.sin(phase * 0.5), 0.0, 0.0)
    )
    _set_axis_angle(
        pose, model, "right_hand_root", (-0.08 * math.sin(phase * 0.5), 0.0, 0.0)
    )
    return pose, translation


def _generate_rom_wander(
    source: Any,
    time_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _animation_model(source)
    pose, translation = _make_animation_buffers(model)
    rom_limits = getattr(source, "rom_limits", {})

    for joint_index, joint_name in enumerate(model.joint_names):
        limits = rom_limits.get(joint_name)
        if limits is None:
            continue

        for axis, limit in enumerate(limits):
            if not limit.enabled:
                continue
            center_deg = (limit.minimum_deg + limit.maximum_deg) * 0.5
            amplitude_deg = max(limit.maximum_deg - limit.minimum_deg, 0.0) * 0.5
            if amplitude_deg <= 1e-4:
                pose[joint_index, axis] = math.radians(center_deg)
                continue

            phase_seed = _stable_phase_seed(f"{joint_name}:{axis}")
            primary_frequency = 0.10 + 0.02 * ((joint_index + axis) % 7)
            secondary_frequency = primary_frequency * (
                1.7 + 0.15 * ((joint_index + axis) % 3)
            )
            waveform = (
                math.sin(2.0 * math.pi * primary_frequency * time_s + phase_seed)
                + 0.35
                * math.sin(
                    2.0 * math.pi * secondary_frequency * time_s + phase_seed * 0.6
                )
            ) / 1.35
            pose[joint_index, axis] = math.radians(
                center_deg + amplitude_deg * waveform
            )

    if getattr(source, "model", None) is not None:
        translation[1] = 0.02 * math.sin(2.0 * math.pi * time_s * 0.24)
        translation[2] = 0.03 * math.cos(2.0 * math.pi * time_s * 0.17)
    return pose, translation


ANIMATION_PRESETS = (
    AnimationPreset(
        key="rom_wander",
        label="ROM Wander",
        period=8.0,
        matcher=lambda model: True,
        generator=_generate_rom_wander,
    ),
    AnimationPreset(
        key="root_sway",
        label="Root Sway",
        period=2.8,
        matcher=lambda model: True,
        generator=_generate_root_sway,
    ),
    AnimationPreset(
        key="walk_cycle",
        label="Walk Cycle",
        period=1.25,
        matcher=lambda model: _has_joints(
            model, "left_hip", "left_knee", "right_hip", "right_knee"
        ),
        generator=_generate_walk_cycle,
    ),
    AnimationPreset(
        key="arm_wave",
        label="Arm Wave",
        period=1.6,
        matcher=lambda model: (
            _has_joints(model, "right_shoulder", "right_elbow")
            or _has_joints(model, "left_shoulder", "left_elbow")
        ),
        generator=_generate_arm_wave,
    ),
    AnimationPreset(
        key="finger_wave",
        label="Finger Wave",
        period=1.8,
        matcher=lambda model: _has_any_prefix(
            model,
            (
                "thumb",
                "forefinger",
                "middle_finger",
                "ring_finger",
                "pinky_finger",
                "left_thumb",
                "right_thumb",
            ),
        ),
        generator=_generate_finger_wave,
    ),
)

WALK_POSE_PARAMS = [
    {
        "label": "left_contact",
        "global_orient": [0.02, 0.0, -0.04],
        "transl": [0.0, 0.00, 0.00],
        "legs": {
            "left_hip": [-0.45, 0.0, 0.0],
            "left_knee": [-0.55, 0.0, 0.0],
            "right_hip": [0.25, 0.0, 0.0],
            "right_knee": [-0.10, 0.0, 0.0],
        },
    },
    {
        "label": "left_down",
        "global_orient": [0.01, 0.0, -0.02],
        "transl": [0.0, 0.02, 0.08],
        "legs": {
            "left_hip": [-0.25, 0.0, 0.0],
            "left_knee": [-0.30, 0.0, 0.0],
            "right_hip": [0.10, 0.0, 0.0],
            "right_knee": [-0.05, 0.0, 0.0],
        },
    },
    {
        "label": "passing",
        "global_orient": [0.0, 0.0, 0.0],
        "transl": [0.0, 0.03, 0.16],
        "legs": {
            "left_hip": [0.00, 0.0, 0.0],
            "left_knee": [-0.05, 0.0, 0.0],
            "right_hip": [0.00, 0.0, 0.0],
            "right_knee": [-0.05, 0.0, 0.0],
        },
    },
    {
        "label": "right_down",
        "global_orient": [0.01, 0.0, 0.02],
        "transl": [0.0, 0.02, 0.24],
        "legs": {
            "left_hip": [0.10, 0.0, 0.0],
            "left_knee": [-0.05, 0.0, 0.0],
            "right_hip": [-0.25, 0.0, 0.0],
            "right_knee": [-0.30, 0.0, 0.0],
        },
    },
    {
        "label": "right_contact",
        "global_orient": [0.02, 0.0, 0.04],
        "transl": [0.0, 0.00, 0.32],
        "legs": {
            "left_hip": [0.25, 0.0, 0.0],
            "left_knee": [-0.10, 0.0, 0.0],
            "right_hip": [-0.45, 0.0, 0.0],
            "right_knee": [-0.55, 0.0, 0.0],
        },
    },
]


def available_animation_presets(model: Any) -> list[AnimationPreset]:
    """Return animation presets compatible with a skeleton model.

    Parameters
    ----------
    model : object
        Skeleton model exposing ``joint_names``.

    Returns
    -------
    list[AnimationPreset]
        Presets whose matcher accepts the model.
    """

    return [preset for preset in ANIMATION_PRESETS if preset.matcher(model)]


def build_walk_tensors(
    model: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the authored walk-cycle pose tensors for a skeleton.

    Parameters
    ----------
    model : object
        Skeleton model exposing joint metadata and tensor device/dtype.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Body pose, global orientation, and translation tensors.
    """

    dtype = model.rest_offsets.dtype
    device = model.rest_offsets.device
    num_frames = len(WALK_POSE_PARAMS)

    global_orient = torch.tensor(
        [frame["global_orient"] for frame in WALK_POSE_PARAMS],
        dtype=dtype,
        device=device,
    )
    transl = torch.tensor(
        [frame["transl"] for frame in WALK_POSE_PARAMS],
        dtype=dtype,
        device=device,
    )
    body_pose = torch.zeros(
        num_frames, model.NUM_JOINTS - 1, 3, dtype=dtype, device=device
    )
    body_pose_indices = {
        model.joint_names[joint_index]: body_pose_index
        for body_pose_index, joint_index in enumerate(model.non_root_joint_indices)
    }

    for frame_index, frame in enumerate(WALK_POSE_PARAMS):
        for joint_name, axis_angle in frame["legs"].items():
            if joint_name not in body_pose_indices:
                continue
            body_pose[frame_index, body_pose_indices[joint_name]] = torch.tensor(
                axis_angle,
                dtype=dtype,
                device=device,
            )
    return body_pose, global_orient, transl
