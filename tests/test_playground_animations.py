from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from skeletons import build_layer
from skeletons.playground import (
    AxisRomLimit,
    available_animation_presets,
    build_walk_tensors,
)


def _preset_by_key(model, key: str):
    return next(
        preset for preset in available_animation_presets(model) if preset.key == key
    )


def test_available_animation_presets_match_skeleton_capabilities() -> None:
    human = build_layer("human36m")
    face = build_layer("face68")

    human_keys = {preset.key for preset in available_animation_presets(human)}
    face_keys = {preset.key for preset in available_animation_presets(face)}

    assert {"rom_wander", "root_sway", "walk_cycle", "arm_wave"}.issubset(
        human_keys
    )
    assert "walk_cycle" not in face_keys
    assert {"rom_wander", "root_sway"}.issubset(face_keys)


def test_animation_generators_return_model_shaped_buffers() -> None:
    model = build_layer("human36m")
    preset = _preset_by_key(model, "walk_cycle")

    pose, translation = preset.generator(model, 0.25)

    assert pose.shape == (model.NUM_JOINTS, 3)
    assert translation.shape == (3,)
    assert pose.dtype == model.rest_offsets.dtype
    assert translation.device == model.rest_offsets.device
    assert torch.any(pose.abs() > 0)


def test_rom_wander_uses_active_limit_ranges() -> None:
    model = build_layer("human36m")
    root_name = model.joint_names[model.root_index]
    source = SimpleNamespace(
        model=model,
        rom_limits={
            root_name: [
                AxisRomLimit(enabled=True, minimum_deg=20.0, maximum_deg=20.0),
                AxisRomLimit(),
                AxisRomLimit(),
            ]
        },
    )
    preset = _preset_by_key(model, "rom_wander")

    pose, _ = preset.generator(source, 1.0)

    assert torch.isclose(
        pose[model.root_index, 0],
        torch.tensor(math.radians(20.0), dtype=pose.dtype),
    )


def test_build_walk_tensors_respects_model_layout() -> None:
    model = build_layer("human36m")

    body_pose, global_orient, transl = build_walk_tensors(model)

    assert body_pose.shape[0] == global_orient.shape[0] == transl.shape[0]
    assert body_pose.shape[1:] == (model.NUM_JOINTS - 1, 3)
    assert global_orient.shape[1:] == (3,)
    assert transl.shape[1:] == (3,)
