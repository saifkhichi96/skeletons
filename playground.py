"""Interactive playground app for visualizing skeletons and testing animation presets."""

from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import numpy as np
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    import torch
    from PySide6 import QtCore, QtGui, QtWidgets
except (ImportError, ModuleNotFoundError) as exc:
    raise SystemExit(
        "Missing app dependencies. Install them with `pip install 'PyOpenGL>=3.1' 'pyqtgraph>=0.13' 'PySide6>=6.7'`, "
        "then run `python playground.py`."
    ) from exc

from differential_skeletons import (
    CocoModel,
    CocoWholeBodyModel,
    Face68Model,
    Halpe26Model,
    HalpeFullBodyModel,
    Hand21Model,
    Human36MModel,
    MPIIModel,
    SpineTrackModel,
)
from differential_skeletons.fitting import (
    FrameDataset,
    JointLimitPrior,
    JointLimitStatistics,
    PerspectiveCamera,
    PoseVAE,
    SkeletalFitter,
)
from differential_skeletons.rotations import matrix_to_axis_angle


def softlight_shader() -> gl.shaders.ShaderProgram:
    shader = gl.shaders.ShaderProgram.names.get("softlight")
    if shader is not None:
        return shader

    vertex_shader = """
        uniform mat4 u_mvp;
        uniform mat3 u_normal;
        attribute vec4 a_position;
        attribute vec3 a_normal;
        attribute vec4 a_color;
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            v_normal = normalize(u_normal * a_normal);
            v_color = a_color;
            gl_Position = u_mvp * a_position;
        }
    """
    fragment_shader = """
        #ifdef GL_ES
        precision mediump float;
        #endif
        varying vec4 v_color;
        varying vec3 v_normal;
        void main() {
            vec3 n = normalize(v_normal);
            vec3 view_dir = vec3(0.0, 0.0, 1.0);
            vec3 key_dir = normalize(vec3(0.42, -0.35, 0.84));
            vec3 fill_dir = normalize(vec3(-0.58, 0.18, 0.55));
            vec3 bounce_dir = normalize(vec3(0.12, 0.96, 0.24));

            float key = max(dot(n, key_dir), 0.0);
            float fill = max(dot(n, fill_dir), 0.0);
            float bounce = max(dot(n, bounce_dir), 0.0);
            float hemi = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);
            float rim = pow(1.0 - max(dot(n, view_dir), 0.0), 2.2);
            float spec = pow(max(dot(n, normalize(key_dir + view_dir)), 0.0), 28.0);

            vec3 base = v_color.rgb;
            vec3 lit = base * vec3(0.16, 0.17, 0.19);
            lit += base * vec3(1.00, 0.98, 0.95) * (0.70 * key);
            lit += base * vec3(0.82, 0.89, 1.00) * (0.24 * fill);
            lit += base * vec3(0.92, 0.96, 1.00) * (0.18 * bounce);
            lit += base * (0.14 * hemi);
            lit += vec3(1.0, 0.98, 0.95) * (0.10 * spec);
            lit += base * (0.12 * rim);

            lit = clamp(lit, 0.0, 1.0);
            gl_FragColor = vec4(pow(lit, vec3(0.95)), v_color.a);
        }
    """
    return gl.shaders.ShaderProgram(
        "softlight",
        [
            gl.shaders.VertexShader(vertex_shader),
            gl.shaders.FragmentShader(fragment_shader),
        ],
    )


SKELETON_FACTORIES = OrderedDict(
    [
        ("COCO", CocoModel),
        ("MPII", MPIIModel),
        ("Human3.6M", Human36MModel),
        ("HALPE26", Halpe26Model),
        ("Hand21", Hand21Model),
        ("Face68", Face68Model),
        ("HALPE FullBody", HalpeFullBodyModel),
        ("COCO WholeBody", CocoWholeBodyModel),
        ("SpineTrack", SpineTrackModel),
    ]
)

BONE_COLOR = (0.77, 0.81, 0.88, 1.0)
JOINT_COLOR = (0.00, 0.48, 1.00, 1.0)
ROOT_COLOR = (0.10, 0.63, 0.95, 1.0)
SELECTED_COLOR = (1.00, 0.62, 0.20, 1.0)
TARGET_BONE_COLOR = (0.24, 0.68, 0.97, 0.40)
TARGET_JOINT_COLOR = (0.47, 0.77, 1.00, 0.64)
PREVIEW_TARGET_COLOR = QtGui.QColor(0, 122, 255)
PREVIEW_PREDICTION_COLOR = QtGui.QColor(255, 149, 0)
FLOOR_LIGHT_COLOR = (0.99, 0.99, 1.00, 1.0)
FLOOR_DARK_COLOR = (0.90, 0.92, 0.95, 1.0)
AXIS_COLORS = {
    "X": QtGui.QColor(255, 95, 87),
    "Y": QtGui.QColor(52, 199, 89),
    "Z": QtGui.QColor(10, 132, 255),
}
SECONDARY_TEXT_STYLE = "color: #6e7582;"
SCENE_TO_VIEW = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _normalize_skeleton_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _infer_pose_prior_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    linear_keys = sorted(
        key
        for key, value in state_dict.items()
        if key.startswith("encoder.") and key.endswith(".weight") and value.ndim == 2
    )
    if not linear_keys:
        raise ValueError("Could not infer PoseVAE architecture from the checkpoint.")

    input_dim = int(state_dict[linear_keys[0]].shape[1])
    hidden_dim = int(state_dict[linear_keys[0]].shape[0])
    latent_dim = int(state_dict["encoder_mu.weight"].shape[0])
    if input_dim % 6 != 0:
        raise ValueError(
            f"PoseVAE input dimension must be divisible by 6, got {input_dim}."
        )
    return {
        "num_joints": input_dim // 6,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": len(linear_keys),
    }


def _load_priors(
    path: Path,
    *,
    skeleton: str,
) -> tuple[PoseVAE | None, JointLimitPrior | None]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected a dict checkpoint in {path}, got {type(checkpoint).__name__}."
        )

    checkpoint_skeleton = checkpoint.get("skeleton")
    if checkpoint_skeleton is not None and _normalize_skeleton_name(
        str(checkpoint_skeleton)
    ) != _normalize_skeleton_name(skeleton):
        raise ValueError(
            f"Prior checkpoint skeleton {checkpoint_skeleton!r} does not match requested skeleton {skeleton!r}.",
        )

    pose_prior = None
    pose_state = checkpoint.get("pose_prior")
    if pose_state is not None:
        pose_config = checkpoint.get("pose_prior_config")
        if pose_config is None:
            pose_config = _infer_pose_prior_config(pose_state)
        pose_prior = PoseVAE(**pose_config)
        pose_prior.load_state_dict(pose_state)
        pose_prior.eval()

    joint_limit_prior = None
    joint_limit_state = checkpoint.get("joint_limit_prior")
    if joint_limit_state is not None:
        joint_limit_config = checkpoint.get("joint_limit_prior_config", {})
        joint_limit_prior = JointLimitPrior(
            JointLimitStatistics(
                mean=joint_limit_state["mean"],
                std=joint_limit_state["std"],
                lower=joint_limit_state["lower"],
                upper=joint_limit_state["upper"],
            ),
            barrier_scale=float(joint_limit_config.get("barrier_scale", 10.0)),
        )
        joint_limit_prior.load_state_dict(joint_limit_state)
        joint_limit_prior.eval()

    if pose_prior is None and joint_limit_prior is None:
        raise ValueError(f"No supported priors were found in {path}.")
    return pose_prior, joint_limit_prior


def vec3_to_numpy(value) -> np.ndarray:
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return np.array([value.x(), value.y(), value.z()], dtype=float)
    return np.asarray(value, dtype=float)


def scene_to_view(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    return array @ SCENE_TO_VIEW.T


def create_bone_pyramid_mesh() -> gl.MeshData:
    # A square pyramid reads more clearly as a directional "bone" than a cylinder.
    vertexes = np.array(
        [
            [-0.70, -0.70, 0.0],
            [0.70, -0.70, 0.0],
            [0.70, 0.70, 0.0],
            [-0.70, 0.70, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.int32,
    )
    return gl.MeshData(vertexes=vertexes, faces=faces)


def create_checkerboard_mesh(
    *,
    size: float,
    tile_size: float,
    z: float = -0.002,
) -> gl.MeshData:
    half_size = size * 0.5
    num_tiles = max(int(round(size / tile_size)), 1)
    vertexes: list[list[float]] = []
    faces: list[list[int]] = []
    face_colors: list[tuple[float, float, float, float]] = []

    for ix in range(num_tiles):
        x0 = -half_size + ix * tile_size
        x1 = min(x0 + tile_size, half_size)
        for iy in range(num_tiles):
            y0 = -half_size + iy * tile_size
            y1 = min(y0 + tile_size, half_size)
            base_index = len(vertexes)
            vertexes.extend(
                [
                    [x0, y0, z],
                    [x1, y0, z],
                    [x1, y1, z],
                    [x0, y1, z],
                ]
            )
            faces.extend(
                [
                    [base_index + 0, base_index + 1, base_index + 2],
                    [base_index + 0, base_index + 2, base_index + 3],
                ]
            )
            color = FLOOR_LIGHT_COLOR if (ix + iy) % 2 == 0 else FLOOR_DARK_COLOR
            face_colors.extend([color, color])

    return gl.MeshData(
        vertexes=np.asarray(vertexes, dtype=float),
        faces=np.asarray(faces, dtype=np.int32),
        faceColors=np.asarray(face_colors, dtype=float),
    )


@dataclass(frozen=True)
class AnimationPreset:
    key: str
    label: str
    period: float
    matcher: Callable[[object], bool]
    generator: Callable[[object, float], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class AxisRomLimit:
    enabled: bool = False
    minimum_deg: float = -180.0
    maximum_deg: float = 180.0


AXIS_NAMES = ("X", "Y", "Z")
AXIS_FILE_KEYS = ("x", "y", "z")


def _animation_model(source) -> object:
    return getattr(source, "model", source)


def _stable_phase_seed(text: str) -> float:
    total = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return math.radians(float(total % 360))


def _has_joints(model, *joint_names: str) -> bool:
    available = set(model.joint_names)
    return all(name in available for name in joint_names)


def _has_any_prefix(model, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefixes) for name in model.joint_names)


def _set_axis_angle(
    pose: torch.Tensor,
    model,
    joint_name: str,
    values: tuple[float | None, float | None, float | None],
) -> None:
    if joint_name not in model.joint_name_to_index:
        return
    joint_index = model.joint_name_to_index[joint_name]
    for axis, value in enumerate(values):
        if value is not None:
            pose[joint_index, axis] = value


def _make_animation_buffers(model) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = model.rest_offsets.dtype
    device = model.rest_offsets.device
    return (
        torch.zeros(model.num_joints, 3, dtype=dtype, device=device),
        torch.zeros(3, dtype=dtype, device=device),
    )


def generate_root_sway(source, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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


def generate_walk_cycle(source, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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


def generate_arm_wave(source, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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


def generate_finger_wave(source, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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


def generate_rom_wander(source, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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
        generator=generate_rom_wander,
    ),
    AnimationPreset(
        key="root_sway",
        label="Root Sway",
        period=2.8,
        matcher=lambda model: True,
        generator=generate_root_sway,
    ),
    AnimationPreset(
        key="walk_cycle",
        label="Walk Cycle",
        period=1.25,
        matcher=lambda model: _has_joints(
            model, "left_hip", "left_knee", "right_hip", "right_knee"
        ),
        generator=generate_walk_cycle,
    ),
    AnimationPreset(
        key="arm_wave",
        label="Arm Wave",
        period=1.6,
        matcher=lambda model: (
            _has_joints(model, "right_shoulder", "right_elbow")
            or _has_joints(model, "left_shoulder", "left_elbow")
        ),
        generator=generate_arm_wave,
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
        generator=generate_finger_wave,
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


def build_walk_tensors(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        num_frames, model.num_joints - 1, 3, dtype=dtype, device=device
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


class FloatSlider(QtWidgets.QWidget):
    value_changed = QtCore.Signal(float)

    def __init__(
        self,
        title: str,
        *,
        minimum: float,
        maximum: float,
        factor: int,
        decimals: int,
        suffix: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.factor = factor
        self.decimals = decimals
        self.suffix = suffix

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = QtWidgets.QLabel(title)
        self.value_label = QtWidgets.QLabel()
        self.value_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum * factor)), int(round(maximum * factor)))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self.set_value(0.0)

    def value(self) -> float:
        return self.slider.value() / self.factor

    def set_value(self, value: float, *, emit: bool = False) -> None:
        clamped = min(max(value, self.minimum), self.maximum)
        raw = int(round(clamped * self.factor))
        if emit:
            self.slider.setValue(raw)
            return
        was_blocked = self.slider.blockSignals(True)
        self.slider.setValue(raw)
        self.slider.blockSignals(was_blocked)
        self.value_label.setText(self._format(clamped))

    def set_range(self, minimum: float, maximum: float) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.slider.setRange(
            int(round(minimum * self.factor)), int(round(maximum * self.factor))
        )
        self.set_value(self.value(), emit=False)

    def _format(self, value: float) -> str:
        return f"{value:.{self.decimals}f}{self.suffix}"

    def _on_slider_changed(self, raw_value: int) -> None:
        value = raw_value / self.factor
        self.value_label.setText(self._format(value))
        self.value_changed.emit(value)


class AxisGizmoOverlay(QtWidgets.QWidget):
    def __init__(self, view: "SkeletonViewport") -> None:
        super().__init__(view)
        self.view = view
        self.setFixedSize(140, 140)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.axis_vectors = {
            "X": scene_to_view(np.array([1.0, 0.0, 0.0], dtype=float)),
            "Y": scene_to_view(np.array([0.0, 1.0, 0.0], dtype=float)),
            "Z": scene_to_view(np.array([0.0, 0.0, 1.0], dtype=float)),
        }

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        frame = self.rect().adjusted(8, 8, -8, -8)
        bg = QtGui.QColor(255, 255, 255, 226)
        outline = QtGui.QColor(210, 214, 222, 220)
        painter.setPen(QtGui.QPen(outline, 1.5))
        painter.setBrush(bg)
        painter.drawRoundedRect(frame, 18, 18)

        center = QtCore.QPointF(frame.center())
        radius = min(frame.width(), frame.height()) * 0.28
        basis = self._screen_basis()

        axis_draw_data: list[tuple[float, str, np.ndarray, QtGui.QColor]] = []
        for label, axis_vector in self.axis_vectors.items():
            direction_2d = np.array(
                [
                    float(np.dot(axis_vector, basis["right"])),
                    float(np.dot(axis_vector, basis["up"])),
                ],
                dtype=float,
            )
            depth = float(np.dot(axis_vector, basis["forward"]))
            axis_draw_data.append((depth, label, direction_2d, AXIS_COLORS[label]))

        for depth, label, direction_2d, color in sorted(
            axis_draw_data, key=lambda item: item[0]
        ):
            alpha = 255 if depth >= 0.0 else 150
            draw_color = QtGui.QColor(color)
            draw_color.setAlpha(alpha)
            self._draw_arrow(painter, center, radius, direction_2d, draw_color, label)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(245, 247, 250))
        painter.drawEllipse(center, 4.5, 4.5)

    def _screen_basis(self) -> dict[str, np.ndarray]:
        camera = vec3_to_numpy(self.view.cameraPosition())
        center = vec3_to_numpy(self.view.opts["center"])

        forward = center - camera
        norm = float(np.linalg.norm(forward))
        if norm < 1e-8:
            forward = np.array([0.0, 1.0, -1.0], dtype=float)
            norm = float(np.linalg.norm(forward))
        forward /= norm

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= right_norm

        screen_up = np.cross(right, forward)
        screen_up /= max(float(np.linalg.norm(screen_up)), 1e-8)
        return {"forward": forward, "right": right, "up": screen_up}

    def _draw_arrow(
        self,
        painter: QtGui.QPainter,
        origin: QtCore.QPointF,
        radius: float,
        direction_2d: np.ndarray,
        color: QtGui.QColor,
        label: str,
    ) -> None:
        length = float(np.linalg.norm(direction_2d))
        if length < 1e-8:
            return

        direction = direction_2d / length
        tip = QtCore.QPointF(
            origin.x() + direction[0] * radius,
            origin.y() - direction[1] * radius,
        )
        shaft = QtCore.QPointF(
            origin.x() + direction[0] * radius * 0.72,
            origin.y() - direction[1] * radius * 0.72,
        )
        normal = np.array([-direction[1], direction[0]], dtype=float)

        pen = QtGui.QPen(
            color, 6.0, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap
        )
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(origin, shaft)

        arrow_size = radius * 0.15
        left = QtCore.QPointF(
            tip.x() - direction[0] * arrow_size + normal[0] * arrow_size * 0.7,
            tip.y() + direction[1] * arrow_size - normal[1] * arrow_size * 0.7,
        )
        right = QtCore.QPointF(
            tip.x() - direction[0] * arrow_size - normal[0] * arrow_size * 0.7,
            tip.y() + direction[1] * arrow_size + normal[1] * arrow_size * 0.7,
        )
        arrow_head = QtGui.QPolygonF([tip, left, right])
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(arrow_head)

        label_point = QtCore.QPointF(
            tip.x() + direction[0] * 13.0,
            tip.y() - direction[1] * 13.0,
        )
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(color, 1.5))
        painter.drawText(label_point, label)


class ProjectionPreview(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.parents: tuple[int, ...] = ()
        self.target_points: np.ndarray | None = None
        self.predicted_points: np.ndarray | None = None

    def set_topology(self, parents: tuple[int, ...]) -> None:
        self.parents = tuple(parents)
        self.update()

    def set_target_points(self, points: np.ndarray | None) -> None:
        self.target_points = None if points is None else np.asarray(points, dtype=float)
        self.update()

    def set_predicted_points(self, points: np.ndarray | None) -> None:
        self.predicted_points = (
            None if points is None else np.asarray(points, dtype=float)
        )
        self.update()

    def clear(self) -> None:
        self.target_points = None
        self.predicted_points = None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        frame = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QtGui.QPen(QtGui.QColor(222, 226, 234), 1.2))
        painter.setBrush(QtGui.QColor(255, 255, 255))
        painter.drawRoundedRect(frame, 12, 12)

        content = frame.adjusted(14, 34, -14, -14)
        self._draw_legend(painter, frame)

        point_sets = [
            points
            for points in (self.target_points, self.predicted_points)
            if points is not None and len(points) > 0
        ]
        if not point_sets:
            painter.setPen(QtGui.QColor(110, 117, 130))
            painter.drawText(
                content,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "Load a dataset sample to preview 2D fitting.",
            )
            return

        stacked = np.concatenate(point_sets, axis=0)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        extent = np.maximum(maxs - mins, 1.0)
        scale = min(
            content.width() / float(extent[0]), content.height() / float(extent[1])
        )
        center = (mins + maxs) * 0.5
        canvas_center = np.array(
            [content.center().x(), content.center().y()], dtype=float
        )

        def map_point(point: np.ndarray) -> QtCore.QPointF:
            mapped = (point - center) * scale + canvas_center
            return QtCore.QPointF(float(mapped[0]), float(mapped[1]))

        if self.target_points is not None:
            self._draw_skeleton_2d(
                painter,
                self.target_points,
                map_point,
                bone_color=PREVIEW_TARGET_COLOR,
                joint_color=PREVIEW_TARGET_COLOR,
                dashed=True,
            )
        if self.predicted_points is not None:
            self._draw_skeleton_2d(
                painter,
                self.predicted_points,
                map_point,
                bone_color=PREVIEW_PREDICTION_COLOR,
                joint_color=PREVIEW_PREDICTION_COLOR,
                dashed=False,
            )

    def _draw_legend(self, painter: QtGui.QPainter, frame: QtCore.QRect) -> None:
        legend_rect = frame.adjusted(12, 8, -12, -frame.height() + 28)
        painter.setPen(QtGui.QColor(29, 29, 31))
        painter.drawText(
            legend_rect,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            "Projection Preview",
        )

        x = legend_rect.right() - 140
        self._draw_legend_entry(
            painter,
            x,
            legend_rect.center().y(),
            PREVIEW_TARGET_COLOR,
            "Target",
            dashed=True,
        )
        self._draw_legend_entry(
            painter,
            x + 72,
            legend_rect.center().y(),
            PREVIEW_PREDICTION_COLOR,
            "Prediction",
            dashed=False,
        )

    def _draw_legend_entry(
        self,
        painter: QtGui.QPainter,
        x: int,
        y: int,
        color: QtGui.QColor,
        label: str,
        *,
        dashed: bool,
    ) -> None:
        pen = QtGui.QPen(
            color,
            2.0,
            QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine,
        )
        painter.setPen(pen)
        painter.drawLine(x, y, x + 16, y)
        painter.setPen(QtGui.QColor(110, 117, 130))
        painter.drawText(x + 22, y + 5, label)

    def _draw_skeleton_2d(
        self,
        painter: QtGui.QPainter,
        points: np.ndarray,
        map_point: Callable[[np.ndarray], QtCore.QPointF],
        *,
        bone_color: QtGui.QColor,
        joint_color: QtGui.QColor,
        dashed: bool,
    ) -> None:
        bone_pen = QtGui.QPen(
            bone_color,
            1.8,
            QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine,
        )
        bone_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(bone_pen)
        for joint_index, parent_index in enumerate(self.parents):
            if parent_index < 0:
                continue
            painter.drawLine(
                map_point(points[parent_index]), map_point(points[joint_index])
            )

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(joint_color)
        radius = 3.5
        for point in points:
            mapped = map_point(point)
            painter.drawEllipse(mapped, radius, radius)


class FittingWorker(QtCore.QObject):
    progress = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(
        self,
        *,
        model_factory: type,
        sequence: dict[str, torch.Tensor],
        priors_path: Path | None,
        mode: str,
        num_iters: int,
        lr: float,
        optimize_bone_scales: bool,
        use_pose_prior_latent: bool,
        init_from_ik: bool,
        progress_interval: int,
    ) -> None:
        super().__init__()
        self.model_factory = model_factory
        self.sequence = sequence
        self.priors_path = priors_path
        self.mode = mode
        self.num_iters = int(num_iters)
        self.lr = float(lr)
        self.optimize_bone_scales = optimize_bone_scales
        self.use_pose_prior_latent = use_pose_prior_latent
        self.init_from_ik = init_from_ik
        self.progress_interval = max(int(progress_interval), 1)
        self._cancel_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            model = self.model_factory(
                create_global_orient=False,
                create_body_pose=False,
                create_bone_scales=False,
                create_transl=False,
            )
            pose_prior = None
            joint_limit_prior = None
            if self.priors_path is not None:
                pose_prior, joint_limit_prior = _load_priors(
                    self.priors_path, skeleton=model.spec.name
                )
            fitter = SkeletalFitter(
                model=model,
                pose_prior=pose_prior,
                joint_limit_prior=joint_limit_prior,
            )
            total_frames = int(self.sequence["joints_3d"].shape[0])
            frame_joints: list[np.ndarray | None] = [None] * total_frames
            frame_full_pose: list[torch.Tensor] = []
            frame_transl: list[torch.Tensor] = []
            frame_bone_scales: list[torch.Tensor] = []
            prev_state: dict[str, torch.Tensor] | None = None
            last_losses: dict[str, float] = {}
            total_completed_iters = 0

            for frame_index in range(total_frames):
                if self._cancel_requested:
                    self.canceled.emit()
                    return

                sample = self._slice_frame(frame_index)

                def on_progress(
                    step: int,
                    total: int,
                    joints: torch.Tensor,
                    losses: dict[str, float],
                ) -> bool:
                    joints_np = joints.squeeze(0).detach().cpu().numpy()
                    frame_joints[frame_index] = joints_np
                    self.progress.emit(
                        {
                            "frame_index": frame_index,
                            "frame_count": total_frames,
                            "iter": step,
                            "iters_per_frame": total,
                            "total_iter": frame_index * self.num_iters + step,
                            "total_iters": total_frames * self.num_iters,
                            "joints": joints_np,
                            "losses": dict(losses),
                        }
                    )
                    return not self._cancel_requested

                fit_kwargs = {
                    "num_iters": self.num_iters,
                    "lr": self.lr,
                    "optimize_bone_scales": self.optimize_bone_scales,
                    "use_pose_prior_latent": self.use_pose_prior_latent,
                    "progress_callback": on_progress,
                    "progress_interval": self.progress_interval,
                }
                if prev_state is not None:
                    fit_kwargs.update(prev_state)

                if self.mode == "3d":
                    result = fitter.fit_3d(
                        sample["joints_3d"].unsqueeze(0),
                        init_from_ik=self.init_from_ik and prev_state is None,
                        **fit_kwargs,
                    )
                elif self.mode == "2d":
                    if "joints_2d" not in sample:
                        raise ValueError(
                            "The selected range does not contain joints_2d."
                        )
                    camera = PerspectiveCamera(
                        fx=sample.get("fx", torch.tensor(1000.0)).reshape(()),
                        fy=sample.get("fy", torch.tensor(1000.0)).reshape(()),
                        cx=sample.get("cx", torch.tensor(512.0)).reshape(()),
                        cy=sample.get("cy", torch.tensor(512.0)).reshape(()),
                    )
                    confidences = sample.get("confidences")
                    result = fitter.fit_2d(
                        sample["joints_2d"].unsqueeze(0),
                        camera,
                        confidences=None
                        if confidences is None
                        else confidences.unsqueeze(0),
                        **fit_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported fitting mode: {self.mode!r}")

                if self._cancel_requested:
                    self.canceled.emit()
                    return

                output = result.model_output
                frame_full_pose.append(
                    matrix_to_axis_angle(
                        output.local_rotations.squeeze(0).detach().cpu()
                    )
                )
                frame_transl.append(output.transl.squeeze(0).detach().cpu())
                frame_bone_scales.append(output.bone_scales.squeeze(0).detach().cpu())
                frame_joints[frame_index] = (
                    output.joints.squeeze(0).detach().cpu().numpy()
                )
                prev_state = {
                    "init_global_orient": output.global_orient.squeeze(0)
                    .detach()
                    .cpu(),
                    "init_body_pose": output.body_pose.squeeze(0).detach().cpu(),
                    "init_bone_scales": output.bone_scales.squeeze(0).detach().cpu(),
                    "init_transl": output.transl.squeeze(0).detach().cpu(),
                }
                last_losses = dict(result.losses)
                total_completed_iters += int(result.iterations)

            payload = {
                "full_pose": torch.stack(frame_full_pose, dim=0),
                "transl": torch.stack(frame_transl, dim=0),
                "bone_scales": torch.stack(frame_bone_scales, dim=0),
                "joints": np.stack(
                    [value for value in frame_joints if value is not None], axis=0
                ),
                "losses": last_losses,
                "iterations": total_completed_iters,
                "iters_per_frame": self.num_iters,
                "mode": self.mode,
                "frame_count": total_frames,
            }
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def _slice_frame(self, frame_index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        total_frames = int(self.sequence["joints_3d"].shape[0])
        for key, value in self.sequence.items():
            if not isinstance(value, torch.Tensor):
                continue
            sample[key] = (
                value[frame_index] if value.shape[0] == total_frames else value
            )
        return sample


class SkeletonViewport(gl.GLViewWidget):
    CAMERA_ELEVATION = 26.0
    CAMERA_DISTANCE = 4.4
    CAMERA_AZIMUTH = -36.0
    GIZMO_MARGIN = 18
    FLOOR_SIZE = 8.0
    FLOOR_TILE_SIZE = 0.5

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.parents: tuple[int, ...] = ()
        self.bone_items: list[gl.GLMeshItem | None] = []
        self.joint_items: list[gl.GLMeshItem] = []
        self.target_bone_items: list[gl.GLMeshItem | None] = []
        self.target_joint_items: list[gl.GLMeshItem] = []
        self._last_render_joints: np.ndarray | None = None
        self._target_render_joints: np.ndarray | None = None
        self.floor_visible = True

        self.setBackgroundColor((244, 246, 250))
        self.setCameraPosition(
            pos=QtGui.QVector3D(0.0, 0.0, 0.0),
            distance=self.CAMERA_DISTANCE,
            elevation=self.CAMERA_ELEVATION,
            azimuth=self.CAMERA_AZIMUTH,
        )

        self.floor_mesh = create_checkerboard_mesh(
            size=self.FLOOR_SIZE,
            tile_size=self.FLOOR_TILE_SIZE,
        )
        self.floor_item = gl.GLMeshItem(
            meshdata=self.floor_mesh,
            smooth=False,
            drawFaces=True,
            drawEdges=False,
            shader=softlight_shader(),
        )
        self.floor_item.setGLOptions("opaque")
        self.addItem(self.floor_item)

        self.bone_mesh = create_bone_pyramid_mesh()
        self.joint_mesh = gl.MeshData.sphere(rows=10, cols=20, radius=1.0)
        self.gizmo = AxisGizmoOverlay(self)
        self.gizmo.show()
        self.gizmo.raise_()
        self._place_gizmo()

    def set_floor_visible(self, visible: bool) -> None:
        self.floor_visible = bool(visible)
        if self.floor_visible:
            self.floor_item.show()
        else:
            self.floor_item.hide()

    def set_topology(self, parents: tuple[int, ...]) -> None:
        for item in self.joint_items:
            self.removeItem(item)
        for item in self.bone_items:
            if item is not None:
                self.removeItem(item)
        for item in self.target_joint_items:
            self.removeItem(item)
        for item in self.target_bone_items:
            if item is not None:
                self.removeItem(item)

        self.parents = tuple(parents)
        self.joint_items = []
        self.bone_items = [None] * len(self.parents)
        self.target_joint_items = []
        self.target_bone_items = [None] * len(self.parents)

        for joint_index, parent_index in enumerate(self.parents):
            joint_item = self._make_joint_item(JOINT_COLOR, opaque=True)
            self.addItem(joint_item)
            self.joint_items.append(joint_item)

            if parent_index < 0:
                target_joint_item = self._make_joint_item(
                    TARGET_JOINT_COLOR, opaque=False
                )
                target_joint_item.hide()
                self.addItem(target_joint_item)
                self.target_joint_items.append(target_joint_item)
                continue

            bone_item = self._make_bone_item(BONE_COLOR, opaque=True)
            self.addItem(bone_item)
            self.bone_items[joint_index] = bone_item

            target_joint_item = self._make_joint_item(TARGET_JOINT_COLOR, opaque=False)
            target_joint_item.hide()
            self.addItem(target_joint_item)
            self.target_joint_items.append(target_joint_item)

            target_bone_item = self._make_bone_item(TARGET_BONE_COLOR, opaque=False)
            target_bone_item.hide()
            self.addItem(target_bone_item)
            self.target_bone_items[joint_index] = target_bone_item

        if len(self.target_joint_items) < len(self.parents):
            missing = len(self.parents) - len(self.target_joint_items)
            for _ in range(missing):
                target_joint_item = self._make_joint_item(
                    TARGET_JOINT_COLOR, opaque=False
                )
                target_joint_item.hide()
                self.addItem(target_joint_item)
                self.target_joint_items.append(target_joint_item)

    def update_skeleton(
        self, joints: np.ndarray, *, selected_joint: int | None = None
    ) -> None:
        if not self.parents or len(self.parents) != len(joints):
            raise ValueError(
                "Topology must be initialized before updating the skeleton."
            )

        render_joints = scene_to_view(joints)
        self._last_render_joints = render_joints
        self._render_layer(
            render_joints,
            self.joint_items,
            self.bone_items,
            selected_joint=selected_joint,
            bone_color=BONE_COLOR,
            joint_color=JOINT_COLOR,
            root_color=ROOT_COLOR,
            highlight_color=SELECTED_COLOR,
            allow_selection=True,
        )
        self.gizmo.update()

    def set_target_overlay(self, joints: np.ndarray | None) -> None:
        if joints is None:
            self._target_render_joints = None
            self._hide_layer(self.target_joint_items, self.target_bone_items)
            self.gizmo.update()
            return
        if not self.parents or len(self.parents) != len(joints):
            raise ValueError(
                "Topology must be initialized before updating the target overlay."
            )

        render_joints = scene_to_view(joints)
        self._target_render_joints = render_joints
        self._render_layer(
            render_joints,
            self.target_joint_items,
            self.target_bone_items,
            selected_joint=None,
            bone_color=TARGET_BONE_COLOR,
            joint_color=TARGET_JOINT_COLOR,
            root_color=TARGET_JOINT_COLOR,
            highlight_color=TARGET_JOINT_COLOR,
            allow_selection=False,
        )
        self.gizmo.update()

    def fit_camera_to_joints(self, joints: np.ndarray) -> None:
        render_joints = scene_to_view(joints)
        mins = render_joints.min(axis=0)
        maxs = render_joints.max(axis=0)
        extent = np.maximum(maxs - mins, 0.2)
        center = np.array(
            [
                float((mins[0] + maxs[0]) * 0.5),
                float((mins[1] + maxs[1]) * 0.5),
                float(mins[2] + extent[2] * 0.62),
            ],
            dtype=float,
        )
        distance = max(float(np.linalg.norm(extent) * 1.85), 1.75)
        self.setCameraPosition(
            pos=QtGui.QVector3D(*map(float, center)),
            distance=distance,
            elevation=self.CAMERA_ELEVATION,
            azimuth=self.opts["azimuth"],
        )
        self.gizmo.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        lpos = ev.position() if hasattr(ev, "position") else ev.localPos()
        self.mousePos = lpos
        ev.accept()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        lpos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if not hasattr(self, "mousePos"):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.orbit(-diff.x(), 0.0)
        elif ev.buttons() in (
            QtCore.Qt.MouseButton.MiddleButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            self.pan(diff.x(), diff.y(), 0.0, relative="view-upright")
        self.gizmo.update()
        ev.accept()

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        super().wheelEvent(ev)
        self.setCameraPosition(elevation=self.CAMERA_ELEVATION)
        self.gizmo.update()
        ev.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_gizmo()

    def _place_gizmo(self) -> None:
        self.gizmo.move(
            self.width() - self.gizmo.width() - self.GIZMO_MARGIN,
            self.height() - self.gizmo.height() - self.GIZMO_MARGIN,
        )
        self.gizmo.raise_()

    def _place_bone(
        self,
        item: gl.GLMeshItem,
        start: np.ndarray,
        end: np.ndarray,
        *,
        bone_radius: float,
        color: tuple[float, float, float, float],
    ) -> None:
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-8:
            item.hide()
            return

        item.show()
        item.setColor(color)
        item.resetTransform()
        item.scale(bone_radius * 1.55, bone_radius * 1.55, length)

        direction = segment / length
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        axis = np.cross(z_axis, direction)
        axis_norm = float(np.linalg.norm(axis))
        dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        if axis_norm > 1e-8:
            angle = math.degrees(math.atan2(axis_norm, dot))
            item.rotate(angle, *(axis / axis_norm), local=False)
        elif dot < 0.0:
            item.rotate(180.0, 1.0, 0.0, 0.0, local=False)

        item.translate(*map(float, start), local=False)

    def _make_joint_item(
        self, color: tuple[float, float, float, float], *, opaque: bool
    ) -> gl.GLMeshItem:
        item = gl.GLMeshItem(
            meshdata=self.joint_mesh,
            smooth=True,
            drawFaces=True,
            drawEdges=False,
            shader=softlight_shader(),
            color=color,
        )
        item.setGLOptions("opaque" if opaque else "translucent")
        return item

    def _make_bone_item(
        self, color: tuple[float, float, float, float], *, opaque: bool
    ) -> gl.GLMeshItem:
        item = gl.GLMeshItem(
            meshdata=self.bone_mesh,
            smooth=False,
            drawFaces=True,
            drawEdges=True,
            shader=softlight_shader(),
            color=color,
        )
        item.setGLOptions("opaque" if opaque else "translucent")
        return item

    def _compute_radii(
        self, render_joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mins = render_joints.min(axis=0)
        maxs = render_joints.max(axis=0)
        extent = float(np.max(maxs - mins))
        extent = max(extent, 0.35)
        min_bone_radius = max(extent * 0.006, 0.0025)
        max_bone_radius = max(extent * 0.028, min_bone_radius * 1.8)
        min_joint_radius = max(extent * 0.0055, 0.0022)

        bone_radii = np.full(len(self.parents), min_bone_radius, dtype=float)
        joint_radii = np.full(len(self.parents), min_joint_radius, dtype=float)
        for joint_index, parent_index in enumerate(self.parents):
            if parent_index < 0:
                continue
            segment_length = float(
                np.linalg.norm(render_joints[joint_index] - render_joints[parent_index])
            )
            segment_radius = float(
                np.clip(segment_length * 0.12, min_bone_radius, max_bone_radius)
            )
            bone_radii[joint_index] = segment_radius
            joint_radius = max(segment_radius * 0.90, min_joint_radius)
            joint_radii[joint_index] = max(joint_radii[joint_index], joint_radius)
            joint_radii[parent_index] = max(joint_radii[parent_index], joint_radius)
        return bone_radii, joint_radii

    def _render_layer(
        self,
        render_joints: np.ndarray,
        joint_items: list[gl.GLMeshItem],
        bone_items: list[gl.GLMeshItem | None],
        *,
        selected_joint: int | None,
        bone_color: tuple[float, float, float, float],
        joint_color: tuple[float, float, float, float],
        root_color: tuple[float, float, float, float],
        highlight_color: tuple[float, float, float, float],
        allow_selection: bool,
    ) -> None:
        bone_radii, joint_radii = self._compute_radii(render_joints)

        for joint_index, item in enumerate(joint_items):
            item.show()
            item.resetTransform()
            radius = float(joint_radii[joint_index])
            item.scale(radius, radius, radius)
            item.translate(*map(float, render_joints[joint_index]), local=False)

            if allow_selection and joint_index == selected_joint:
                item.setColor(highlight_color)
            elif self.parents[joint_index] < 0:
                item.setColor(root_color)
            else:
                item.setColor(joint_color)

        for joint_index, parent_index in enumerate(self.parents):
            if parent_index < 0:
                continue
            item = bone_items[joint_index]
            if item is None:
                continue
            highlight = allow_selection and joint_index == selected_joint
            self._place_bone(
                item,
                render_joints[parent_index],
                render_joints[joint_index],
                bone_radius=float(bone_radii[joint_index]),
                color=highlight_color if highlight else bone_color,
            )

    def _hide_layer(
        self,
        joint_items: list[gl.GLMeshItem],
        bone_items: list[gl.GLMeshItem | None],
    ) -> None:
        for item in joint_items:
            item.hide()
        for item in bone_items:
            if item is not None:
                item.hide()


class SkelixPlayground(QtWidgets.QMainWindow):
    ANIMATION_DT = 1.0 / 60.0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DifferentialSkeletons Playground")
        self.resize(1500, 920)

        self.model = None
        self.model_factory = None
        self.available_animations: list[AnimationPreset] = []
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.full_pose = torch.zeros(1, 3)
        self.translation = torch.zeros(3)
        self.bone_scales = torch.ones(1, 3)
        self.global_bone_scale = 1.0
        self.rom_limits: dict[str, list[AxisRomLimit]] = {}
        self.scale_joint_indices: list[int] = []
        self.loaded_dataset: FrameDataset | None = None
        self.dataset_path: Path | None = None
        self.range_start_index = 0
        self.range_end_index = 0
        self.current_sample_index = 0
        self.range_sequence: dict[str, torch.Tensor] | None = None
        self.sequence_fit_payload: dict[str, object] | None = None
        self.current_sample: dict[str, torch.Tensor] | None = None
        self.target_joints_3d: np.ndarray | None = None
        self.target_joints_2d: np.ndarray | None = None
        self.current_camera: PerspectiveCamera | None = None
        self.priors_path: Path | None = None
        self.preview_override_joints: np.ndarray | None = None
        self.fit_thread: QtCore.QThread | None = None
        self.fit_worker: FittingWorker | None = None
        self.example_body_pose: torch.Tensor | None = None
        self.example_global_orient: torch.Tensor | None = None
        self.example_transl: torch.Tensor | None = None
        self.example_labels: list[str] = []
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setInterval(int(round(self.ANIMATION_DT * 1000)))
        self.animation_timer.timeout.connect(self._advance_animation)
        self.example_timer = QtCore.QTimer(self)
        self.example_timer.setInterval(240)
        self.example_timer.timeout.connect(self._advance_example_frame)
        self.sequence_timer = QtCore.QTimer(self)
        self.sequence_timer.setInterval(120)
        self.sequence_timer.timeout.connect(self._advance_sequence_frame)

        pg.setConfigOptions(antialias=True)
        self._apply_styles()
        self._build_ui()
        self._load_skeleton(self.skeleton_combo.currentText())

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget#PlaygroundRoot, QWidget#SidebarBody, QScrollArea {
                background: #f5f5f7;
            }
            QWidget {
                color: #1d1d1f;
                font-size: 13px;
                font-family: "SF Pro Display", "Helvetica Neue", "Segoe UI", sans-serif;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e3e7ee;
                border-radius: 18px;
                margin-top: 14px;
                padding: 14px 14px 14px 14px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #5f6773;
            }
            QComboBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #d8dde6;
                border-radius: 12px;
                padding: 7px 10px;
                min-height: 20px;
                selection-background-color: #007aff;
            }
            QPushButton:hover, QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #b9c3d3;
                background: #fcfcfe;
            }
            QPushButton:pressed {
                background: #eef3fb;
            }
            QPushButton[prominent="true"] {
                background: #007aff;
                color: white;
                border: none;
                font-weight: 600;
            }
            QPushButton[prominent="true"]:hover {
                background: #2488ff;
            }
            QPushButton[prominent="true"]:pressed {
                background: #0063d1;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QScrollArea {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #d9dde5;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -6px 0;
                background: #ffffff;
                border: 1px solid #c6cfdb;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                border-color: #97a4b7;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 6px;
                border: 1px solid #cfd5df;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #007aff;
                border-color: #007aff;
            }
            QCheckBox::indicator:unchecked:hover, QCheckBox::indicator:checked:hover {
                border-color: #8fa0b8;
            }
            QProgressBar {
                background: #edf1f6;
                border: 1px solid #d7dde7;
                border-radius: 10px;
                text-align: center;
                color: #4f5b6b;
            }
            QProgressBar::chunk {
                background: #007aff;
                border-radius: 9px;
            }
            QSplitter::handle {
                background: transparent;
                width: 10px;
            }
            QStatusBar {
                background: #f5f5f7;
                border-top: 1px solid #e4e8ef;
                color: #5f6773;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setObjectName("PlaygroundRoot")
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        def create_sidebar(
            *, minimum_width: int, maximum_width: int
        ) -> tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
            sidebar = QtWidgets.QScrollArea()
            sidebar.setWidgetResizable(True)
            sidebar.setMinimumWidth(minimum_width)
            sidebar.setMaximumWidth(maximum_width)

            sidebar_body = QtWidgets.QWidget()
            sidebar_body.setObjectName("SidebarBody")
            sidebar.setWidget(sidebar_body)
            sidebar_layout = QtWidgets.QVBoxLayout(sidebar_body)
            sidebar_layout.setContentsMargins(6, 6, 6, 6)
            sidebar_layout.setSpacing(12)
            return sidebar, sidebar_layout

        left_sidebar, left_sidebar_layout = create_sidebar(
            minimum_width=340, maximum_width=430
        )
        splitter.addWidget(left_sidebar)

        self.viewport = SkeletonViewport()
        splitter.addWidget(self.viewport)

        right_sidebar, right_sidebar_layout = create_sidebar(
            minimum_width=340, maximum_width=430
        )
        splitter.addWidget(right_sidebar)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([380, 920, 380])

        skeleton_group = QtWidgets.QGroupBox("Skeleton")
        skeleton_layout = QtWidgets.QVBoxLayout(skeleton_group)
        self.skeleton_combo = QtWidgets.QComboBox()
        self.skeleton_combo.addItems(list(SKELETON_FACTORIES.keys()))
        self.skeleton_combo.setCurrentText("Human3.6M")
        self.skeleton_combo.currentTextChanged.connect(self._load_skeleton)
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        skeleton_layout.addWidget(self.skeleton_combo)
        skeleton_layout.addWidget(self.info_label)
        left_sidebar_layout.addWidget(skeleton_group)

        animation_group = QtWidgets.QGroupBox("Animation")
        animation_layout = QtWidgets.QVBoxLayout(animation_group)
        animation_note = QtWidgets.QLabel(
            "Preset motion is added on top of the current slider pose. "
            "ROM Wander uses the active joint limits as a plausible motion envelope."
        )
        animation_note.setWordWrap(True)
        animation_note.setStyleSheet(SECONDARY_TEXT_STYLE)
        animation_layout.addWidget(animation_note)
        self.animation_combo = QtWidgets.QComboBox()
        self.animation_combo.currentIndexChanged.connect(self._on_animation_changed)
        animation_layout.addWidget(self.animation_combo)

        animation_buttons = QtWidgets.QHBoxLayout()
        self.animation_toggle_button = QtWidgets.QPushButton("Play")
        self.animation_toggle_button.setProperty("prominent", True)
        self.animation_toggle_button.clicked.connect(self._toggle_animation)
        self.animation_restart_button = QtWidgets.QPushButton("Restart")
        self.animation_restart_button.clicked.connect(self._restart_animation)
        animation_buttons.addWidget(self.animation_toggle_button)
        animation_buttons.addWidget(self.animation_restart_button)
        animation_layout.addLayout(animation_buttons)

        self.animation_speed_slider = FloatSlider(
            "Speed",
            minimum=0.25,
            maximum=2.50,
            factor=100,
            decimals=2,
            suffix="x",
        )
        self.animation_speed_slider.value_changed.connect(
            self._on_animation_speed_changed
        )
        animation_layout.addWidget(self.animation_speed_slider)

        self.animation_phase_slider = FloatSlider(
            "Phase",
            minimum=0.0,
            maximum=1.0,
            factor=1000,
            decimals=3,
            suffix=" turn",
        )
        self.animation_phase_slider.value_changed.connect(
            self._on_animation_phase_changed
        )
        animation_layout.addWidget(self.animation_phase_slider)
        left_sidebar_layout.addWidget(animation_group)

        example_group = QtWidgets.QGroupBox("Examples")
        example_layout = QtWidgets.QVBoxLayout(example_group)
        example_note = QtWidgets.QLabel(
            "Replay pose-parameter examples inside the editor. The Human3.6M walk keyframes come from examples/visualize_pose_params.py."
        )
        example_note.setWordWrap(True)
        example_note.setStyleSheet(SECONDARY_TEXT_STYLE)
        example_layout.addWidget(example_note)
        self.example_combo = QtWidgets.QComboBox()
        self.example_combo.currentIndexChanged.connect(self._on_example_changed)
        example_layout.addWidget(self.example_combo)

        self.example_frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.example_frame_slider.setRange(0, 0)
        self.example_frame_slider.valueChanged.connect(self._on_example_frame_changed)
        example_layout.addWidget(self.example_frame_slider)

        self.example_frame_label = QtWidgets.QLabel("No example loaded.")
        self.example_frame_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        example_layout.addWidget(self.example_frame_label)

        example_buttons = QtWidgets.QHBoxLayout()
        self.example_toggle_button = QtWidgets.QPushButton("Play")
        self.example_toggle_button.setProperty("prominent", True)
        self.example_toggle_button.clicked.connect(self._toggle_example_playback)
        self.example_restart_button = QtWidgets.QPushButton("Restart")
        self.example_restart_button.clicked.connect(self._restart_example)
        example_buttons.addWidget(self.example_toggle_button)
        example_buttons.addWidget(self.example_restart_button)
        example_layout.addLayout(example_buttons)
        left_sidebar_layout.addWidget(example_group)

        fitting_group = QtWidgets.QGroupBox("Fitting")
        fitting_layout = QtWidgets.QVBoxLayout(fitting_group)
        fitting_note = QtWidgets.QLabel(
            "Load shared .npz fitting data, run the same single-frame fitting flow as examples/fit_demo.py, and watch optimization update the viewport."
        )
        fitting_note.setWordWrap(True)
        fitting_note.setStyleSheet(SECONDARY_TEXT_STYLE)
        fitting_layout.addWidget(fitting_note)

        dataset_row = QtWidgets.QHBoxLayout()
        self.dataset_path_edit = QtWidgets.QLineEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setPlaceholderText("No dataset loaded")
        self.load_dataset_button = QtWidgets.QPushButton("Dataset…")
        self.load_dataset_button.clicked.connect(self._load_dataset_from_file)
        self.clear_dataset_button = QtWidgets.QPushButton("Clear")
        self.clear_dataset_button.clicked.connect(self._clear_dataset)
        dataset_row.addWidget(self.dataset_path_edit, 1)
        dataset_row.addWidget(self.load_dataset_button)
        dataset_row.addWidget(self.clear_dataset_button)
        fitting_layout.addLayout(dataset_row)

        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(QtWidgets.QLabel("Range"))
        self.range_start_spin = QtWidgets.QSpinBox()
        self.range_start_spin.setRange(0, 0)
        self.range_start_spin.setEnabled(False)
        self.range_start_spin.valueChanged.connect(self._on_range_start_changed)
        sample_row.addWidget(self.range_start_spin)
        sample_row.addWidget(QtWidgets.QLabel("to"))
        self.range_end_spin = QtWidgets.QSpinBox()
        self.range_end_spin.setRange(0, 0)
        self.range_end_spin.setEnabled(False)
        self.range_end_spin.valueChanged.connect(self._on_range_end_changed)
        sample_row.addWidget(self.range_end_spin)
        fitting_layout.addLayout(sample_row)

        frame_row = QtWidgets.QHBoxLayout()
        frame_row.addWidget(QtWidgets.QLabel("Frame"))
        self.sample_spin = QtWidgets.QSpinBox()
        self.sample_spin.setRange(0, 0)
        self.sample_spin.setEnabled(False)
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)
        frame_row.addWidget(self.sample_spin)
        self.sequence_play_button = QtWidgets.QPushButton("Play Range")
        self.sequence_play_button.setProperty("prominent", True)
        self.sequence_play_button.clicked.connect(self._toggle_sequence_playback)
        frame_row.addWidget(self.sequence_play_button)
        frame_row.addStretch(1)
        fitting_layout.addLayout(frame_row)

        self.sequence_frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sequence_frame_slider.setRange(0, 0)
        self.sequence_frame_slider.setEnabled(False)
        self.sequence_frame_slider.valueChanged.connect(
            self._on_sequence_frame_slider_changed
        )
        fitting_layout.addWidget(self.sequence_frame_slider)

        self.sequence_frame_label = QtWidgets.QLabel("No active range.")
        self.sequence_frame_label.setWordWrap(True)
        self.sequence_frame_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        fitting_layout.addWidget(self.sequence_frame_label)

        self.dataset_info_label = QtWidgets.QLabel("No fitting dataset loaded.")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        fitting_layout.addWidget(self.dataset_info_label)

        priors_row = QtWidgets.QHBoxLayout()
        self.priors_path_edit = QtWidgets.QLineEdit()
        self.priors_path_edit.setReadOnly(True)
        self.priors_path_edit.setPlaceholderText("Optional priors checkpoint")
        self.load_priors_button = QtWidgets.QPushButton("Priors…")
        self.load_priors_button.clicked.connect(self._select_priors_checkpoint)
        self.clear_priors_button = QtWidgets.QPushButton("Clear")
        self.clear_priors_button.clicked.connect(self._clear_priors_checkpoint)
        priors_row.addWidget(self.priors_path_edit, 1)
        priors_row.addWidget(self.load_priors_button)
        priors_row.addWidget(self.clear_priors_button)
        fitting_layout.addLayout(priors_row)

        fit_mode_row = QtWidgets.QHBoxLayout()
        fit_mode_row.addWidget(QtWidgets.QLabel("Mode"))
        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItem("3D joints", userData="3d")
        self.fit_mode_combo.addItem("2D reprojection", userData="2d")
        self.fit_mode_combo.currentIndexChanged.connect(self._update_fit_controls)
        fit_mode_row.addWidget(self.fit_mode_combo)
        fit_mode_row.addStretch(1)
        fitting_layout.addLayout(fit_mode_row)

        self.fit_optimize_scale_check = QtWidgets.QCheckBox("Optimize body scales")
        self.fit_optimize_scale_check.setChecked(True)
        fitting_layout.addWidget(self.fit_optimize_scale_check)

        self.fit_pose_prior_check = QtWidgets.QCheckBox(
            "Use pose prior latent when available"
        )
        self.fit_pose_prior_check.setChecked(True)
        fitting_layout.addWidget(self.fit_pose_prior_check)

        self.fit_init_from_ik_check = QtWidgets.QCheckBox(
            "Initialize 3D fit from inverse kinematics"
        )
        self.fit_init_from_ik_check.setChecked(True)
        fitting_layout.addWidget(self.fit_init_from_ik_check)

        fit_hparams_row = QtWidgets.QHBoxLayout()
        fit_hparams_row.addWidget(QtWidgets.QLabel("Iters"))
        self.fit_iters_spin = QtWidgets.QSpinBox()
        self.fit_iters_spin.setRange(10, 2000)
        self.fit_iters_spin.setValue(220)
        fit_hparams_row.addWidget(self.fit_iters_spin)
        fit_hparams_row.addWidget(QtWidgets.QLabel("LR"))
        self.fit_lr_spin = QtWidgets.QDoubleSpinBox()
        self.fit_lr_spin.setDecimals(4)
        self.fit_lr_spin.setRange(0.0001, 0.1000)
        self.fit_lr_spin.setSingleStep(0.0010)
        self.fit_lr_spin.setValue(0.0100)
        fit_hparams_row.addWidget(self.fit_lr_spin)
        fitting_layout.addLayout(fit_hparams_row)

        self.fit_progress_bar = QtWidgets.QProgressBar()
        self.fit_progress_bar.setRange(0, 1)
        self.fit_progress_bar.setValue(0)
        fitting_layout.addWidget(self.fit_progress_bar)

        self.fit_status_label = QtWidgets.QLabel("Idle.")
        self.fit_status_label.setWordWrap(True)
        self.fit_status_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        fitting_layout.addWidget(self.fit_status_label)

        fit_buttons = QtWidgets.QHBoxLayout()
        self.run_fit_button = QtWidgets.QPushButton("Run Range Fit")
        self.run_fit_button.setProperty("prominent", True)
        self.run_fit_button.clicked.connect(self._start_fit)
        self.cancel_fit_button = QtWidgets.QPushButton("Cancel")
        self.cancel_fit_button.clicked.connect(self._cancel_fit)
        fit_buttons.addWidget(self.run_fit_button)
        fit_buttons.addWidget(self.cancel_fit_button)
        fitting_layout.addLayout(fit_buttons)

        self.projection_preview = ProjectionPreview()
        fitting_layout.addWidget(self.projection_preview)
        left_sidebar_layout.addWidget(fitting_group)

        pose_group = QtWidgets.QGroupBox("Joint Pose")
        pose_layout = QtWidgets.QVBoxLayout(pose_group)
        pose_note = QtWidgets.QLabel(
            "Axis-angle in degrees. Select the root joint to edit global orientation."
        )
        pose_note.setWordWrap(True)
        pose_note.setStyleSheet(SECONDARY_TEXT_STYLE)
        pose_layout.addWidget(pose_note)
        self.pose_joint_combo = QtWidgets.QComboBox()
        self.pose_joint_combo.currentIndexChanged.connect(self._sync_pose_sliders)
        self.pose_joint_combo.currentIndexChanged.connect(self._sync_rom_controls)
        self.pose_joint_combo.currentIndexChanged.connect(self._refresh_view)
        pose_layout.addWidget(self.pose_joint_combo)
        self.pose_sliders: list[FloatSlider] = []
        for axis_name in AXIS_NAMES:
            slider = FloatSlider(
                f"{axis_name} rotation",
                minimum=-180.0,
                maximum=180.0,
                factor=10,
                decimals=1,
                suffix=" deg",
            )
            slider.value_changed.connect(
                self._make_pose_callback(len(self.pose_sliders))
            )
            pose_layout.addWidget(slider)
            self.pose_sliders.append(slider)
        self.zero_joint_button = QtWidgets.QPushButton("Zero Selected Joint")
        self.zero_joint_button.clicked.connect(self._zero_selected_joint)
        pose_layout.addWidget(self.zero_joint_button)
        right_sidebar_layout.addWidget(pose_group)

        rom_group = QtWidgets.QGroupBox("ROM Limits")
        rom_layout = QtWidgets.QVBoxLayout(rom_group)
        rom_note = QtWidgets.QLabel(
            "Enable per-axis biomechanical limits for the selected joint. "
            "The pose sliders and animations are clamped to these ranges."
        )
        rom_note.setWordWrap(True)
        rom_note.setStyleSheet(SECONDARY_TEXT_STYLE)
        rom_layout.addWidget(rom_note)

        rom_grid = QtWidgets.QGridLayout()
        rom_grid.setHorizontalSpacing(8)
        rom_grid.setVerticalSpacing(6)
        rom_grid.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        rom_grid.addWidget(QtWidgets.QLabel("Lock"), 0, 1)
        rom_grid.addWidget(QtWidgets.QLabel("Min"), 0, 2)
        rom_grid.addWidget(QtWidgets.QLabel("Max"), 0, 3)

        self.rom_enable_checks: list[QtWidgets.QCheckBox] = []
        self.rom_min_spins: list[QtWidgets.QDoubleSpinBox] = []
        self.rom_max_spins: list[QtWidgets.QDoubleSpinBox] = []
        for axis, axis_name in enumerate(AXIS_NAMES):
            axis_label = QtWidgets.QLabel(axis_name)
            rom_grid.addWidget(axis_label, axis + 1, 0)

            enable_check = QtWidgets.QCheckBox()
            enable_check.toggled.connect(self._on_rom_limit_changed)
            rom_grid.addWidget(
                enable_check, axis + 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
            )
            self.rom_enable_checks.append(enable_check)

            min_spin = QtWidgets.QDoubleSpinBox()
            min_spin.setRange(-180.0, 180.0)
            min_spin.setDecimals(1)
            min_spin.setSingleStep(1.0)
            min_spin.setSuffix(" deg")
            min_spin.valueChanged.connect(self._on_rom_limit_changed)
            rom_grid.addWidget(min_spin, axis + 1, 2)
            self.rom_min_spins.append(min_spin)

            max_spin = QtWidgets.QDoubleSpinBox()
            max_spin.setRange(-180.0, 180.0)
            max_spin.setDecimals(1)
            max_spin.setSingleStep(1.0)
            max_spin.setSuffix(" deg")
            max_spin.valueChanged.connect(self._on_rom_limit_changed)
            rom_grid.addWidget(max_spin, axis + 1, 3)
            self.rom_max_spins.append(max_spin)
        rom_layout.addLayout(rom_grid)

        rom_button_row = QtWidgets.QHBoxLayout()
        self.reset_joint_rom_button = QtWidgets.QPushButton("Reset Joint ROM")
        self.reset_joint_rom_button.clicked.connect(self._reset_selected_joint_rom)
        self.clear_all_rom_button = QtWidgets.QPushButton("Clear All ROM")
        self.clear_all_rom_button.clicked.connect(self._clear_all_rom)
        rom_button_row.addWidget(self.reset_joint_rom_button)
        rom_button_row.addWidget(self.clear_all_rom_button)
        rom_layout.addLayout(rom_button_row)

        rom_io_row = QtWidgets.QHBoxLayout()
        self.load_rom_button = QtWidgets.QPushButton("Load ROM…")
        self.load_rom_button.clicked.connect(self._load_rom_limits_from_file)
        self.save_rom_button = QtWidgets.QPushButton("Save ROM…")
        self.save_rom_button.clicked.connect(self._save_rom_limits_to_file)
        rom_io_row.addWidget(self.load_rom_button)
        rom_io_row.addWidget(self.save_rom_button)
        rom_layout.addLayout(rom_io_row)
        right_sidebar_layout.addWidget(rom_group)

        translation_group = QtWidgets.QGroupBox("Translation")
        translation_layout = QtWidgets.QVBoxLayout(translation_group)
        self.translation_sliders: list[FloatSlider] = []
        for axis_name in AXIS_NAMES:
            slider = FloatSlider(
                f"{axis_name} offset",
                minimum=-2.0,
                maximum=2.0,
                factor=100,
                decimals=2,
                suffix=" m",
            )
            slider.value_changed.connect(
                self._make_translation_callback(len(self.translation_sliders))
            )
            translation_layout.addWidget(slider)
            self.translation_sliders.append(slider)
        right_sidebar_layout.addWidget(translation_group)

        scale_group = QtWidgets.QGroupBox("Body Scale")
        scale_layout = QtWidgets.QVBoxLayout(scale_group)
        scale_note = QtWidgets.QLabel(
            "Global scale multiplies every body uniformly. "
            "Local sliders edit the selected body's OpenSim-style X/Y/Z scale factors."
        )
        scale_note.setWordWrap(True)
        scale_note.setStyleSheet(SECONDARY_TEXT_STYLE)
        scale_layout.addWidget(scale_note)
        self.global_scale_slider = FloatSlider(
            "Global scale",
            minimum=0.25,
            maximum=2.50,
            factor=100,
            decimals=2,
            suffix="x",
        )
        self.global_scale_slider.value_changed.connect(self._on_global_scale_changed)
        scale_layout.addWidget(self.global_scale_slider)
        self.scale_joint_combo = QtWidgets.QComboBox()
        self.scale_joint_combo.currentIndexChanged.connect(self._sync_scale_sliders)
        scale_layout.addWidget(self.scale_joint_combo)
        self.scale_sliders: list[FloatSlider] = []
        for axis_name in AXIS_NAMES:
            slider = FloatSlider(
                f"{axis_name} scale",
                minimum=0.25,
                maximum=2.50,
                factor=100,
                decimals=2,
                suffix="x",
            )
            slider.value_changed.connect(
                self._make_scale_callback(len(self.scale_sliders))
            )
            scale_layout.addWidget(slider)
            self.scale_sliders.append(slider)
        self.reset_scale_button = QtWidgets.QPushButton("Reset Selected Body")
        self.reset_scale_button.clicked.connect(self._reset_selected_bone)
        scale_layout.addWidget(self.reset_scale_button)
        right_sidebar_layout.addWidget(scale_group)

        button_row = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton("Fit Camera")
        self.fit_button.clicked.connect(self._fit_camera)
        self.floor_toggle_button = QtWidgets.QPushButton()
        self.floor_toggle_button.clicked.connect(self._toggle_floor_visibility)
        self.reset_button = QtWidgets.QPushButton("Reset All")
        self.reset_button.clicked.connect(self._reset_all)
        button_row.addWidget(self.fit_button)
        button_row.addWidget(self.floor_toggle_button)
        button_row.addWidget(self.reset_button)
        right_sidebar_layout.addLayout(button_row)
        left_sidebar_layout.addStretch(1)
        right_sidebar_layout.addStretch(1)
        self._update_floor_toggle_button()

    def _default_rom_limits(self) -> dict[str, list[AxisRomLimit]]:
        return {
            joint_name: [AxisRomLimit() for _ in range(3)]
            for joint_name in self.model.joint_names
        }

    def _rom_limits_for_joint(self, joint_index: int) -> list[AxisRomLimit]:
        if joint_index < 0:
            return [AxisRomLimit() for _ in range(3)]
        return self.rom_limits[self.model.joint_names[joint_index]]

    def _clamp_angle_to_rom_limit(
        self, joint_index: int, axis: int, value_rad: float
    ) -> float:
        limit = self._rom_limits_for_joint(joint_index)[axis]
        if not limit.enabled:
            return value_rad
        minimum = math.radians(limit.minimum_deg)
        maximum = math.radians(limit.maximum_deg)
        return min(max(value_rad, minimum), maximum)

    def _clamp_pose_to_rom_limits(self, pose: torch.Tensor) -> torch.Tensor:
        clamped = pose.clone()
        for joint_index, joint_name in enumerate(self.model.joint_names):
            limits = self.rom_limits.get(joint_name)
            if limits is None:
                continue
            for axis, limit in enumerate(limits):
                if not limit.enabled:
                    continue
                clamped[joint_index, axis] = self._clamp_angle_to_rom_limit(
                    joint_index,
                    axis,
                    float(clamped[joint_index, axis]),
                )
        return clamped

    def _clamp_full_pose_in_place(self) -> None:
        self.full_pose = self._clamp_pose_to_rom_limits(self.full_pose)

    def _update_pose_slider_ranges(self) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return

        self._clamp_full_pose_in_place()
        limits = self._rom_limits_for_joint(joint_index)
        for axis, slider in enumerate(self.pose_sliders):
            limit = limits[axis]
            if limit.enabled:
                slider.set_range(limit.minimum_deg, limit.maximum_deg)
            else:
                slider.set_range(-180.0, 180.0)

    def _sync_rom_controls(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self._update_pose_slider_ranges()
        limits = self._rom_limits_for_joint(joint_index)
        for axis, limit in enumerate(limits):
            widgets = (
                self.rom_enable_checks[axis],
                self.rom_min_spins[axis],
                self.rom_max_spins[axis],
            )
            for widget in widgets:
                was_blocked = widget.blockSignals(True)
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(limit.enabled)
                elif widget is self.rom_min_spins[axis]:
                    widget.setValue(limit.minimum_deg)
                else:
                    widget.setValue(limit.maximum_deg)
                widget.blockSignals(was_blocked)
            self.rom_min_spins[axis].setEnabled(limit.enabled)
            self.rom_max_spins[axis].setEnabled(limit.enabled)

    def _status_message(self, message: str, timeout_ms: int = 4000) -> None:
        self.statusBar().showMessage(message, timeout_ms)

    def _format_losses(self, losses: dict[str, float]) -> str:
        if not losses:
            return "No loss terms yet."
        return ", ".join(f"{name}={value:.4f}" for name, value in losses.items())

    def _sample_camera(self) -> PerspectiveCamera | None:
        if self.current_sample is None or "joints_2d" not in self.current_sample:
            return None
        return PerspectiveCamera(
            fx=self.current_sample.get("fx", torch.tensor(1000.0)).reshape(()),
            fy=self.current_sample.get("fy", torch.tensor(1000.0)).reshape(()),
            cx=self.current_sample.get("cx", torch.tensor(512.0)).reshape(()),
            cy=self.current_sample.get("cy", torch.tensor(512.0)).reshape(()),
        )

    def _project_joints(self, joints: np.ndarray | None) -> np.ndarray | None:
        camera = self.current_camera
        if joints is None or camera is None:
            return None
        with torch.no_grad():
            projected = camera.project(
                torch.from_numpy(np.asarray(joints, dtype=np.float32)).unsqueeze(0)
            )
        return projected.squeeze(0).detach().cpu().numpy()

    def _update_projection_preview(self, joints: np.ndarray | None) -> None:
        self.projection_preview.set_target_points(self.target_joints_2d)
        self.projection_preview.set_predicted_points(self._project_joints(joints))

    def _clear_fit_preview(self) -> None:
        self.preview_override_joints = None

    def _clear_sequence_fit(self) -> None:
        self.sequence_fit_payload = None
        self._clear_fit_preview()

    def _set_fit_idle(self, message: str = "Idle.") -> None:
        self.fit_progress_bar.setRange(0, 1)
        self.fit_progress_bar.setValue(0)
        self.fit_status_label.setText(message)
        self._update_fit_controls()

    def _update_floor_toggle_button(self) -> None:
        self.floor_toggle_button.setText(
            "Hide Floor" if self.viewport.floor_visible else "Show Floor"
        )

    def _toggle_floor_visibility(self, *_args) -> None:
        self.viewport.set_floor_visible(not self.viewport.floor_visible)
        self._update_floor_toggle_button()

    def _selected_range_length(self) -> int:
        return max(self.range_end_index - self.range_start_index + 1, 0)

    def _update_sequence_frame_controls(self) -> None:
        dataset_ready = self.loaded_dataset is not None and len(self.loaded_dataset) > 0
        if not dataset_ready:
            self.sequence_frame_label.setText("No active range.")
            return

        self.sample_spin.blockSignals(True)
        self.sample_spin.setRange(self.range_start_index, self.range_end_index)
        self.sample_spin.setValue(self.current_sample_index)
        self.sample_spin.blockSignals(False)

        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setRange(
            self.range_start_index, self.range_end_index
        )
        self.sequence_frame_slider.setValue(self.current_sample_index)
        self.sequence_frame_slider.blockSignals(False)

        frame_offset = self.current_sample_index - self.range_start_index + 1
        range_length = self._selected_range_length()
        fit_state = (
            "fitted sequence"
            if self.sequence_fit_payload is not None
            else "target range"
        )
        self.sequence_frame_label.setText(
            f"Frame {frame_offset}/{range_length} in selected range "
            f"({self.current_sample_index} absolute, {fit_state})."
        )

    def _stop_sequence_playback(self) -> None:
        self.sequence_timer.stop()

    def _toggle_sequence_playback(self, *_args) -> None:
        if self.loaded_dataset is None or self._selected_range_length() <= 1:
            return
        if self.sequence_timer.isActive():
            self.sequence_timer.stop()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()
                self._update_animation_controls()
            if self.example_timer.isActive():
                self.example_timer.stop()
                self._update_example_controls()
            self.sequence_timer.start()
        self._update_fit_controls()

    def _advance_sequence_frame(self) -> None:
        if self.loaded_dataset is None or self._selected_range_length() <= 1:
            self.sequence_timer.stop()
            self._update_fit_controls()
            return
        next_index = self.current_sample_index + 1
        if next_index > self.range_end_index:
            next_index = self.range_start_index
        self._apply_sample_index(next_index)

    def _build_range_sequence(self) -> dict[str, torch.Tensor] | None:
        if self.loaded_dataset is None or len(self.loaded_dataset) == 0:
            return None
        indices = range(self.range_start_index, self.range_end_index + 1)
        samples = [self.loaded_dataset[index] for index in indices]
        if not samples:
            return None

        sequence: dict[str, torch.Tensor] = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            sequence[key] = torch.stack(values, dim=0)
        return sequence

    def _populate_example_selectors(self) -> None:
        current_key = (
            self.example_combo.currentData()
            if hasattr(self, "example_combo")
            else "none"
        )
        self.example_combo.blockSignals(True)
        self.example_combo.clear()
        self.example_combo.addItem("None", userData="none")
        if _normalize_skeleton_name(self.model.spec.name) == "human36m":
            self.example_combo.addItem("Walk Keyframes", userData="walk_keyframes")

        restore_index = 0
        for index in range(self.example_combo.count()):
            if self.example_combo.itemData(index) == current_key:
                restore_index = index
                break
        self.example_combo.setCurrentIndex(restore_index)
        self.example_combo.blockSignals(False)
        self._on_example_changed()

    def _current_example_key(self) -> str:
        return self.example_combo.currentData() or "none"

    def _on_example_changed(self, *_args) -> None:
        self.example_timer.stop()
        self.example_body_pose = None
        self.example_global_orient = None
        self.example_transl = None
        self.example_labels = []

        if self._current_example_key() == "walk_keyframes":
            body_pose, global_orient, transl = build_walk_tensors(self.model)
            self.example_body_pose = body_pose.detach().cpu()
            self.example_global_orient = global_orient.detach().cpu()
            self.example_transl = transl.detach().cpu()
            self.example_labels = [frame["label"] for frame in WALK_POSE_PARAMS]
            self.example_frame_slider.blockSignals(True)
            self.example_frame_slider.setRange(0, len(self.example_labels) - 1)
            self.example_frame_slider.setValue(0)
            self.example_frame_slider.blockSignals(False)
            self._apply_example_frame(0)
        else:
            self.example_frame_slider.blockSignals(True)
            self.example_frame_slider.setRange(0, 0)
            self.example_frame_slider.setValue(0)
            self.example_frame_slider.blockSignals(False)
            self.example_frame_label.setText("No example loaded.")
        self._update_example_controls()

    def _apply_example_frame(self, frame_index: int) -> None:
        if (
            self.example_body_pose is None
            or self.example_global_orient is None
            or self.example_transl is None
        ):
            return
        frame_index = int(np.clip(frame_index, 0, len(self.example_labels) - 1))
        self.full_pose.zero_()
        self.full_pose[self.model.root_index] = self.example_global_orient[frame_index]
        self.full_pose[list(self.model.non_root_joint_indices)] = (
            self.example_body_pose[frame_index]
        )
        self.translation = self.example_transl[frame_index].clone()
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self.example_frame_label.setText(
            f"Frame {frame_index + 1}/{len(self.example_labels)}: {self.example_labels[frame_index]}"
        )
        self._refresh_view()

    def _on_example_frame_changed(self, value: int) -> None:
        self._apply_example_frame(value)

    def _toggle_example_playback(self, *_args) -> None:
        if self.example_body_pose is None:
            return
        if self.example_timer.isActive():
            self.example_timer.stop()
        else:
            if self.animation_timer.isActive():
                self.animation_timer.stop()
                self._update_animation_controls()
            if self.sequence_timer.isActive():
                self.sequence_timer.stop()
                self._update_fit_controls()
            self.example_timer.start()
        self._update_example_controls()

    def _restart_example(self, *_args) -> None:
        if self.example_body_pose is None:
            return
        self.example_frame_slider.setValue(0)
        self._update_example_controls()

    def _advance_example_frame(self) -> None:
        if self.example_body_pose is None:
            self.example_timer.stop()
            self._update_example_controls()
            return
        next_index = (self.example_frame_slider.value() + 1) % len(self.example_labels)
        self.example_frame_slider.setValue(next_index)
        self._update_example_controls()

    def _update_example_controls(self) -> None:
        active = self.example_body_pose is not None
        playing = self.example_timer.isActive()
        self.example_frame_slider.setEnabled(active)
        self.example_toggle_button.setEnabled(active)
        self.example_restart_button.setEnabled(active)
        self.example_toggle_button.setText("Pause" if playing else "Play")

    def _load_dataset_from_file(self, *_args) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Fitting Dataset",
            str(ROOT),
            "NumPy Archives (*.npz)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            dataset = FrameDataset.from_npz(
                path, expected_num_joints=self.model.num_joints
            )
        except Exception as exc:  # pragma: no cover - GUI error reporting
            QtWidgets.QMessageBox.warning(self, "Load Dataset", str(exc))
            return

        self.loaded_dataset = dataset
        self.dataset_path = path
        self.range_start_index = 0
        self.range_end_index = max(len(dataset) - 1, 0)
        self.current_sample_index = 0
        self.range_sequence = self._build_range_sequence()
        self._clear_sequence_fit()
        self.dataset_path_edit.setText(str(path))
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)
        self.range_start_spin.setRange(0, max(len(dataset) - 1, 0))
        self.range_end_spin.setRange(0, max(len(dataset) - 1, 0))
        self.range_start_spin.setValue(self.range_start_index)
        self.range_end_spin.setValue(self.range_end_index)
        self.range_start_spin.setEnabled(len(dataset) > 0)
        self.range_end_spin.setEnabled(len(dataset) > 0)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)
        self.sample_spin.blockSignals(True)
        self.sample_spin.setRange(self.range_start_index, self.range_end_index)
        self.sample_spin.setEnabled(len(dataset) > 0)
        self.sample_spin.setValue(self.current_sample_index)
        self.sample_spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setRange(
            self.range_start_index, self.range_end_index
        )
        self.sequence_frame_slider.setEnabled(len(dataset) > 0)
        self.sequence_frame_slider.setValue(self.current_sample_index)
        self.sequence_frame_slider.blockSignals(False)
        if len(dataset) > 0:
            self._apply_sample_index(0)
        self._status_message(f"Loaded fitting dataset {path.name}")
        self._update_fit_controls()

    def _clear_dataset(self, *_args) -> None:
        self.loaded_dataset = None
        self.dataset_path = None
        self.range_start_index = 0
        self.range_end_index = 0
        self.current_sample_index = 0
        self.range_sequence = None
        self._clear_sequence_fit()
        self._stop_sequence_playback()
        self.current_sample = None
        self.target_joints_3d = None
        self.target_joints_2d = None
        self.current_camera = None
        self.dataset_path_edit.clear()
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)
        self.range_start_spin.setRange(0, 0)
        self.range_end_spin.setRange(0, 0)
        self.range_start_spin.setValue(0)
        self.range_end_spin.setValue(0)
        self.range_start_spin.setEnabled(False)
        self.range_end_spin.setEnabled(False)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)
        self.sample_spin.blockSignals(True)
        self.sample_spin.setRange(0, 0)
        self.sample_spin.setValue(0)
        self.sample_spin.setEnabled(False)
        self.sample_spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setRange(0, 0)
        self.sequence_frame_slider.setValue(0)
        self.sequence_frame_slider.setEnabled(False)
        self.sequence_frame_slider.blockSignals(False)
        self.dataset_info_label.setText("No fitting dataset loaded.")
        self.sequence_frame_label.setText("No active range.")
        self.viewport.set_target_overlay(None)
        self.projection_preview.clear()
        self._clear_fit_preview()
        self._refresh_view()
        self._update_fit_controls()

    def _on_range_start_changed(self, value: int) -> None:
        if self.loaded_dataset is None:
            return
        value = int(np.clip(value, 0, max(len(self.loaded_dataset) - 1, 0)))
        if value > self.range_end_index:
            self.range_end_index = value
            self.range_end_spin.blockSignals(True)
            self.range_end_spin.setValue(value)
            self.range_end_spin.blockSignals(False)
        self.range_start_index = value
        self.range_sequence = self._build_range_sequence()
        self._clear_sequence_fit()
        if self.current_sample_index < self.range_start_index:
            self.current_sample_index = self.range_start_index
        self._apply_sample_index(self.current_sample_index)

    def _on_range_end_changed(self, value: int) -> None:
        if self.loaded_dataset is None:
            return
        value = int(np.clip(value, 0, max(len(self.loaded_dataset) - 1, 0)))
        if value < self.range_start_index:
            self.range_start_index = value
            self.range_start_spin.blockSignals(True)
            self.range_start_spin.setValue(value)
            self.range_start_spin.blockSignals(False)
        self.range_end_index = value
        self.range_sequence = self._build_range_sequence()
        self._clear_sequence_fit()
        if self.current_sample_index > self.range_end_index:
            self.current_sample_index = self.range_end_index
        self._apply_sample_index(self.current_sample_index)

    def _on_sample_index_changed(self, value: int) -> None:
        self._apply_sample_index(value)

    def _on_sequence_frame_slider_changed(self, value: int) -> None:
        self._apply_sample_index(value)

    def _apply_sample_index(self, index: int) -> None:
        if self.loaded_dataset is None or len(self.loaded_dataset) == 0:
            return
        index = int(np.clip(index, self.range_start_index, self.range_end_index))
        self.current_sample_index = index
        sample = self.loaded_dataset[index]
        self.current_sample = {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in sample.items()
        }
        self.target_joints_3d = sample["joints_3d"].detach().cpu().numpy()
        self.target_joints_2d = (
            sample["joints_2d"].detach().cpu().numpy()
            if "joints_2d" in sample
            else None
        )
        self.current_camera = self._sample_camera()
        if self.sequence_fit_payload is not None:
            relative_index = self.current_sample_index - self.range_start_index
            self.full_pose = self.sequence_fit_payload["full_pose"][
                relative_index
            ].clone()
            self.translation = self.sequence_fit_payload["transl"][
                relative_index
            ].clone()
            self.bone_scales = self.sequence_fit_payload["bone_scales"][
                relative_index
            ].clone()
            self._sync_pose_sliders()
            self._sync_translation_sliders()
            self._sync_scale_sliders()

        has_2d = "joints_2d" in sample
        camera_note = (
            "camera defaults"
            if has_2d and self.current_camera is not None
            else "no camera"
        )
        self.dataset_info_label.setText(
            f"Frames: {len(self.loaded_dataset)} | Range: {self.range_start_index}-{self.range_end_index}\n"
            f"Current frame: {index} | 3D joints: yes | 2D joints: {'yes' if has_2d else 'no'} | {camera_note}"
        )
        self._update_sequence_frame_controls()
        self._refresh_view(fit_camera=True)
        self._update_fit_controls()

    def _select_priors_checkpoint(self, *_args) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Priors Checkpoint",
            str(ROOT),
            "PyTorch Checkpoints (*.pt *.pth *.ckpt);;All Files (*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            _load_priors(path, skeleton=self.model.spec.name)
        except Exception as exc:  # pragma: no cover - GUI error reporting
            QtWidgets.QMessageBox.warning(self, "Load Priors", str(exc))
            return
        self.priors_path = path
        self.priors_path_edit.setText(str(path))
        self._status_message(f"Loaded priors checkpoint {path.name}")
        self._update_fit_controls()

    def _clear_priors_checkpoint(self, *_args) -> None:
        self.priors_path = None
        self.priors_path_edit.clear()
        self._update_fit_controls()

    def _update_fit_controls(self, *_args) -> None:
        dataset_ready = (
            self.loaded_dataset is not None and self.current_sample is not None
        )
        sample_has_2d = dataset_ready and "joints_2d" in self.current_sample
        fitting_running = self.fit_thread is not None
        fit_mode = (
            self.fit_mode_combo.currentData()
            if hasattr(self, "fit_mode_combo")
            else "3d"
        )
        range_length = self._selected_range_length()

        if fit_mode == "2d" and not sample_has_2d:
            self.fit_mode_combo.blockSignals(True)
            self.fit_mode_combo.setCurrentIndex(0)
            self.fit_mode_combo.blockSignals(False)
            fit_mode = "3d"

        self.fit_mode_combo.setEnabled(dataset_ready and not fitting_running)
        self.fit_optimize_scale_check.setEnabled(not fitting_running)
        self.fit_pose_prior_check.setEnabled(not fitting_running)
        self.fit_init_from_ik_check.setEnabled(fit_mode == "3d" and not fitting_running)
        self.fit_iters_spin.setEnabled(not fitting_running)
        self.fit_lr_spin.setEnabled(not fitting_running)
        self.run_fit_button.setEnabled(dataset_ready and not fitting_running)
        self.cancel_fit_button.setEnabled(fitting_running)
        self.load_dataset_button.setEnabled(not fitting_running)
        self.clear_dataset_button.setEnabled(not fitting_running and dataset_ready)
        self.load_priors_button.setEnabled(not fitting_running)
        self.clear_priors_button.setEnabled(
            not fitting_running and self.priors_path is not None
        )
        self.range_start_spin.setEnabled(dataset_ready and not fitting_running)
        self.range_end_spin.setEnabled(dataset_ready and not fitting_running)
        self.sample_spin.setEnabled(dataset_ready and not fitting_running)
        self.sequence_frame_slider.setEnabled(dataset_ready and not fitting_running)
        self.sequence_play_button.setEnabled(
            dataset_ready and range_length > 1 and not fitting_running
        )
        self.sequence_play_button.setText(
            "Pause Range" if self.sequence_timer.isActive() else "Play Range"
        )
        self.skeleton_combo.setEnabled(not fitting_running)

    def _start_fit(self, *_args) -> None:
        if (
            self.current_sample is None
            or self.model_factory is None
            or self.range_sequence is None
        ):
            return
        if self.fit_thread is not None:
            return

        sequence = {
            key: value.detach().cpu().clone()
            for key, value in self.range_sequence.items()
        }
        fit_mode = self.fit_mode_combo.currentData()
        if fit_mode == "2d" and "joints_2d" not in sequence:
            QtWidgets.QMessageBox.warning(
                self, "Run Fit", "The selected range does not contain joints_2d."
            )
            return

        self._stop_sequence_playback()
        self._clear_sequence_fit()
        progress_interval = max(1, min(10, self.fit_iters_spin.value() // 25 or 1))
        self.fit_thread = QtCore.QThread(self)
        self.fit_worker = FittingWorker(
            model_factory=self.model_factory,
            sequence=sequence,
            priors_path=self.priors_path,
            mode=fit_mode,
            num_iters=self.fit_iters_spin.value(),
            lr=self.fit_lr_spin.value(),
            optimize_bone_scales=self.fit_optimize_scale_check.isChecked(),
            use_pose_prior_latent=self.fit_pose_prior_check.isChecked(),
            init_from_ik=self.fit_init_from_ik_check.isChecked(),
            progress_interval=progress_interval,
        )
        self.fit_worker.moveToThread(self.fit_thread)
        self.fit_thread.started.connect(self.fit_worker.run)
        self.fit_worker.progress.connect(self._on_fit_progress)
        self.fit_worker.finished.connect(self._on_fit_finished)
        self.fit_worker.failed.connect(self._on_fit_failed)
        self.fit_worker.canceled.connect(self._on_fit_canceled)
        self.fit_worker.finished.connect(self.fit_thread.quit)
        self.fit_worker.failed.connect(self.fit_thread.quit)
        self.fit_worker.canceled.connect(self.fit_thread.quit)
        self.fit_worker.finished.connect(self.fit_worker.deleteLater)
        self.fit_worker.failed.connect(self.fit_worker.deleteLater)
        self.fit_worker.canceled.connect(self.fit_worker.deleteLater)
        self.fit_thread.finished.connect(self._cleanup_fit_worker)
        self.fit_thread.finished.connect(self.fit_thread.deleteLater)

        self.fit_progress_bar.setRange(
            0, self.fit_iters_spin.value() * self._selected_range_length()
        )
        self.fit_progress_bar.setValue(0)
        self.fit_status_label.setText(
            f"Running {fit_mode.upper()} fit over frames {self.range_start_index}-{self.range_end_index}..."
        )
        self._update_fit_controls()
        self.fit_thread.start()

    def _cancel_fit(self, *_args) -> None:
        if self.fit_worker is None:
            return
        self.fit_worker.cancel()
        self.fit_status_label.setText("Cancel requested...")

    def _on_fit_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        joints_np = np.asarray(payload["joints"], dtype=float)
        loss_dict = dict(payload.get("losses", {}))
        frame_index = int(payload["frame_index"])
        absolute_index = self.range_start_index + frame_index
        self.preview_override_joints = joints_np
        self.fit_progress_bar.setRange(0, int(payload["total_iters"]))
        self.fit_progress_bar.setValue(int(payload["total_iter"]))
        self.current_sample_index = absolute_index
        if self.loaded_dataset is not None and 0 <= absolute_index < len(
            self.loaded_dataset
        ):
            sample = self.loaded_dataset[absolute_index]
            self.current_sample = {
                key: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in sample.items()
            }
            self.target_joints_3d = sample["joints_3d"].detach().cpu().numpy()
            self.target_joints_2d = (
                sample["joints_2d"].detach().cpu().numpy()
                if "joints_2d" in sample
                else None
            )
            self.current_camera = self._sample_camera()
        self._update_sequence_frame_controls()
        self.sample_spin.blockSignals(True)
        self.sample_spin.setValue(absolute_index)
        self.sample_spin.blockSignals(False)
        self.sequence_frame_slider.blockSignals(True)
        self.sequence_frame_slider.setValue(absolute_index)
        self.sequence_frame_slider.blockSignals(False)
        self.fit_status_label.setText(
            f"Frame {frame_index + 1}/{int(payload['frame_count'])}, "
            f"iter {int(payload['iter'])}/{int(payload['iters_per_frame'])}: {self._format_losses(loss_dict)}"
        )
        self._refresh_view()

    def _on_fit_finished(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._on_fit_failed("Unexpected fitting payload.")
            return
        self.sequence_fit_payload = {
            "full_pose": payload["full_pose"].clone(),
            "transl": payload["transl"].clone(),
            "bone_scales": payload["bone_scales"].clone(),
            "joints": np.asarray(payload["joints"], dtype=float).copy(),
        }
        self.current_sample_index = self.range_start_index
        self.full_pose = self.sequence_fit_payload["full_pose"][0].clone()
        self.translation = self.sequence_fit_payload["transl"][0].clone()
        self.bone_scales = self.sequence_fit_payload["bone_scales"][0].clone()
        self.global_bone_scale = 1.0
        self.preview_override_joints = None
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._sync_scale_sliders()
        self._sync_global_scale_slider()
        self._update_sequence_frame_controls()
        self.fit_progress_bar.setRange(0, max(int(payload["iterations"]), 1))
        self.fit_progress_bar.setValue(int(payload["iterations"]))
        self.fit_status_label.setText(
            f"Completed {str(payload['mode']).upper()} sequence fit across {int(payload['frame_count'])} frames "
            f"in {int(payload['iterations'])} steps: "
            f"{self._format_losses(payload['losses'])}"
        )
        self._apply_sample_index(self.range_start_index)
        self._status_message("Applied fitted sequence to the editor state.")

    def _on_fit_failed(self, message: str) -> None:
        self._clear_fit_preview()
        self.fit_status_label.setText(f"Fit failed: {message}")
        self._refresh_view()
        QtWidgets.QMessageBox.warning(self, "Fitting Failed", message)

    def _on_fit_canceled(self) -> None:
        self._clear_fit_preview()
        self.fit_status_label.setText("Fit canceled.")
        self._refresh_view()
        self._status_message("Canceled fitting run.")

    def _cleanup_fit_worker(self) -> None:
        self.fit_thread = None
        self.fit_worker = None
        self._update_fit_controls()

    def _default_rom_path(self) -> Path:
        return (
            Path(__file__).resolve().parent
            / "rom_limits"
            / f"{self.model.spec.name}.json"
        )

    def _serialize_rom_limits(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for joint_name, limits in self.rom_limits.items():
            joint_payload: dict[str, object] = {}
            for axis_key, limit in zip(AXIS_FILE_KEYS, limits):
                if not limit.enabled:
                    continue
                joint_payload[axis_key] = {
                    "enabled": True,
                    "min_deg": limit.minimum_deg,
                    "max_deg": limit.maximum_deg,
                }
            if joint_payload:
                payload[joint_name] = joint_payload
        return {
            "format_version": 1,
            "skeleton": self.model.spec.name,
            "joint_limits": payload,
        }

    def _apply_rom_payload(self, payload: dict[str, object]) -> None:
        skeleton_name = payload.get("skeleton")
        if skeleton_name is not None and _normalize_skeleton_name(
            str(skeleton_name)
        ) != _normalize_skeleton_name(self.model.spec.name):
            raise ValueError(
                f"ROM file skeleton {skeleton_name!r} does not match current skeleton {self.model.spec.name!r}.",
            )

        rom_limits = self._default_rom_limits()
        joint_payload = payload.get("joint_limits", {})
        if not isinstance(joint_payload, dict):
            raise ValueError("ROM file must contain a 'joint_limits' object.")

        for joint_name, axis_payload in joint_payload.items():
            if joint_name not in rom_limits or not isinstance(axis_payload, dict):
                continue
            for axis, axis_key in enumerate(AXIS_FILE_KEYS):
                if axis_key not in axis_payload:
                    continue
                limit_payload = axis_payload[axis_key]
                if not isinstance(limit_payload, dict):
                    continue
                minimum_deg = float(
                    limit_payload.get(
                        "min_deg", limit_payload.get("minimum_deg", -180.0)
                    )
                )
                maximum_deg = float(
                    limit_payload.get(
                        "max_deg", limit_payload.get("maximum_deg", 180.0)
                    )
                )
                if minimum_deg > maximum_deg:
                    minimum_deg, maximum_deg = maximum_deg, minimum_deg
                rom_limits[joint_name][axis] = AxisRomLimit(
                    enabled=bool(limit_payload.get("enabled", True)),
                    minimum_deg=minimum_deg,
                    maximum_deg=maximum_deg,
                )
        self.rom_limits = rom_limits
        self._clamp_full_pose_in_place()
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()

    def _save_rom_limits_to_file(self, *_args) -> None:
        default_path = self._default_rom_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save ROM Limits",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not file_path:
            return
        path = Path(file_path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._serialize_rom_limits(), indent=2), encoding="utf-8"
        )
        self._status_message(f"Saved ROM limits to {path.name}")

    def _load_rom_limits_from_file(self, *_args) -> None:
        default_path = self._default_rom_path()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load ROM Limits",
            str(default_path),
            "JSON Files (*.json)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("ROM file must contain a JSON object.")
            self._apply_rom_payload(payload)
        except Exception as exc:  # pragma: no cover - GUI error reporting
            QtWidgets.QMessageBox.warning(self, "Load ROM Limits", str(exc))
            return
        self._status_message(f"Loaded ROM limits from {path.name}")

    def _load_skeleton(self, display_name: str) -> None:
        model_cls = SKELETON_FACTORIES[display_name]
        self.model_factory = model_cls
        self.model = model_cls(
            create_global_orient=False,
            create_body_pose=False,
            create_bone_scales=False,
            create_transl=False,
        )
        dtype = self.model.rest_offsets.dtype
        self.full_pose = torch.zeros(self.model.num_joints, 3, dtype=dtype)
        self.translation = torch.zeros(3, dtype=dtype)
        self.bone_scales = torch.ones(self.model.num_joints, 3, dtype=dtype)
        self.global_bone_scale = 1.0
        self.rom_limits = self._default_rom_limits()
        self.scale_joint_indices = list(range(self.model.num_joints))
        self.animation_time = 0.0
        self.animation_timer.stop()
        self.example_timer.stop()
        self._clear_fit_preview()

        self.viewport.set_topology(self.model.parents)
        self.projection_preview.set_topology(self.model.parents)
        self._clear_dataset()
        self._clear_priors_checkpoint()
        self.info_label.setText(
            f"Spec: {self.model.spec.name}\n"
            f"Root joint: {self.model.joint_names[self.model.root_index]}\n"
            f"Joints: {self.model.num_joints}"
        )

        self._populate_animation_selectors()
        self._populate_example_selectors()
        self._populate_joint_selectors()
        self._sync_pose_sliders()
        self._sync_rom_controls()
        self._sync_translation_sliders()
        self._sync_scale_sliders()
        self._sync_global_scale_slider()
        self._sync_animation_phase_slider()
        self._update_animation_controls()
        self._update_example_controls()
        self._set_fit_idle()
        self._refresh_view(fit_camera=True)

    def _populate_animation_selectors(self) -> None:
        current_key = (
            self._active_animation().key
            if self._active_animation() is not None
            else "none"
        )
        self.available_animations = [
            preset for preset in ANIMATION_PRESETS if preset.matcher(self.model)
        ]

        self.animation_combo.blockSignals(True)
        self.animation_combo.clear()
        self.animation_combo.addItem("None", userData="none")
        for preset in self.available_animations:
            self.animation_combo.addItem(preset.label, userData=preset.key)

        restore_index = 0
        for index in range(self.animation_combo.count()):
            if self.animation_combo.itemData(index) == current_key:
                restore_index = index
                break
        self.animation_combo.setCurrentIndex(restore_index)
        self.animation_combo.blockSignals(False)
        self.animation_speed_slider.set_value(self.animation_speed, emit=False)

    def _populate_joint_selectors(self) -> None:
        pose_joint = self.model.root_index
        scale_joint = self.model.root_index

        self.pose_joint_combo.blockSignals(True)
        self.pose_joint_combo.clear()
        self.pose_joint_combo.addItems(self.model.joint_names)
        self.pose_joint_combo.setCurrentIndex(pose_joint)
        self.pose_joint_combo.blockSignals(False)

        self.scale_joint_combo.blockSignals(True)
        self.scale_joint_combo.clear()
        self.scale_joint_combo.addItems(
            [self.model.joint_names[idx] for idx in self.scale_joint_indices]
        )
        self.scale_joint_combo.setCurrentText(self.model.joint_names[scale_joint])
        self.scale_joint_combo.blockSignals(False)

    def _make_pose_callback(self, axis: int):
        def callback(value: float) -> None:
            joint_index = self.pose_joint_combo.currentIndex()
            self.full_pose[joint_index, axis] = self._clamp_angle_to_rom_limit(
                joint_index,
                axis,
                math.radians(value),
            )
            self._sync_pose_sliders()
            self._refresh_view()

        return callback

    def _make_translation_callback(self, axis: int):
        def callback(value: float) -> None:
            self.translation[axis] = value
            self._refresh_view()

        return callback

    def _make_scale_callback(self, axis: int):
        def callback(value: float) -> None:
            joint_index = self._selected_scale_joint()
            self.bone_scales[joint_index, axis] = value
            self._refresh_view()

        return callback

    def _active_animation(self) -> AnimationPreset | None:
        if not hasattr(self, "animation_combo"):
            return None
        key = self.animation_combo.currentData()
        if key in (None, "none"):
            return None
        for preset in self.available_animations:
            if preset.key == key:
                return preset
        return None

    def _on_animation_changed(self, *_args) -> None:
        self.animation_time = 0.0
        if self._active_animation() is None:
            self.animation_timer.stop()
        self._sync_animation_phase_slider()
        self._update_animation_controls()
        self._refresh_view()

    def _toggle_animation(self, *_args) -> None:
        if self._active_animation() is None:
            return
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        else:
            if self.example_timer.isActive():
                self.example_timer.stop()
                self._update_example_controls()
            if self.sequence_timer.isActive():
                self.sequence_timer.stop()
                self._update_fit_controls()
            self.animation_timer.start()
        self._update_animation_controls()

    def _restart_animation(self, *_args) -> None:
        self.animation_time = 0.0
        self._sync_animation_phase_slider()
        self._refresh_view()

    def _on_animation_speed_changed(self, value: float) -> None:
        self.animation_speed = value

    def _on_animation_phase_changed(self, value: float) -> None:
        preset = self._active_animation()
        if preset is None:
            return
        self.animation_time = float(value) * preset.period
        self._refresh_view()

    def _sync_animation_phase_slider(self) -> None:
        preset = self._active_animation()
        phase = 0.0
        if preset is not None and preset.period > 0.0:
            phase = (self.animation_time / preset.period) % 1.0
        self.animation_phase_slider.set_value(phase, emit=False)

    def _update_animation_controls(self) -> None:
        active = self._active_animation() is not None
        playing = self.animation_timer.isActive()
        self.animation_toggle_button.setEnabled(active)
        self.animation_restart_button.setEnabled(active)
        self.animation_phase_slider.setEnabled(active)
        self.animation_speed_slider.setEnabled(active)
        self.animation_toggle_button.setText("Pause" if playing else "Play")

    def _advance_animation(self) -> None:
        preset = self._active_animation()
        if preset is None:
            self.animation_timer.stop()
            self._update_animation_controls()
            return
        self.animation_time = (
            self.animation_time + self.ANIMATION_DT * self.animation_speed
        ) % preset.period
        self._sync_animation_phase_slider()
        self._refresh_view()

    def _sync_pose_sliders(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        self._update_pose_slider_ranges()
        values_deg = [math.degrees(float(v)) for v in self.full_pose[joint_index]]
        for axis, slider in enumerate(self.pose_sliders):
            slider.set_value(values_deg[axis], emit=False)

    def _sync_translation_sliders(self) -> None:
        for axis, slider in enumerate(self.translation_sliders):
            slider.set_value(float(self.translation[axis]), emit=False)

    def _sync_scale_sliders(self, *_args) -> None:
        selection = self.scale_joint_combo.currentIndex()
        if selection < 0 or not self.scale_joint_indices:
            return
        joint_index = self.scale_joint_indices[selection]
        for axis, slider in enumerate(self.scale_sliders):
            slider.set_value(float(self.bone_scales[joint_index, axis]), emit=False)

    def _sync_global_scale_slider(self) -> None:
        self.global_scale_slider.set_value(self.global_bone_scale, emit=False)

    def _selected_pose_joint(self) -> int:
        return self.pose_joint_combo.currentIndex()

    def _selected_scale_joint(self) -> int:
        selection = self.scale_joint_combo.currentIndex()
        if selection < 0:
            return self.model.root_index
        return self.scale_joint_indices[selection]

    def _zero_selected_joint(self, *_args) -> None:
        joint_index = self._selected_pose_joint()
        if joint_index < 0:
            return
        self.full_pose[joint_index].zero_()
        self._clamp_full_pose_in_place()
        self._sync_pose_sliders()
        self._refresh_view()

    def _on_rom_limit_changed(self, *_args) -> None:
        joint_index = self._selected_pose_joint()
        if joint_index < 0:
            return

        joint_name = self.model.joint_names[joint_index]
        for axis in range(3):
            enabled = self.rom_enable_checks[axis].isChecked()
            minimum_deg = float(self.rom_min_spins[axis].value())
            maximum_deg = float(self.rom_max_spins[axis].value())
            if minimum_deg > maximum_deg:
                if self.sender() is self.rom_min_spins[axis]:
                    maximum_deg = minimum_deg
                else:
                    minimum_deg = maximum_deg
            self.rom_limits[joint_name][axis] = AxisRomLimit(
                enabled=enabled,
                minimum_deg=minimum_deg,
                maximum_deg=maximum_deg,
            )
            self.rom_min_spins[axis].setEnabled(enabled)
            self.rom_max_spins[axis].setEnabled(enabled)

        self._clamp_full_pose_in_place()
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()

    def _reset_selected_joint_rom(self, *_args) -> None:
        joint_index = self._selected_pose_joint()
        if joint_index < 0:
            return
        self.rom_limits[self.model.joint_names[joint_index]] = [
            AxisRomLimit() for _ in range(3)
        ]
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()
        self._status_message(
            f"Cleared ROM limits for {self.model.joint_names[joint_index]}"
        )

    def _clear_all_rom(self, *_args) -> None:
        self.rom_limits = self._default_rom_limits()
        self._sync_rom_controls()
        self._sync_pose_sliders()
        self._refresh_view()
        self._status_message("Cleared all ROM limits")

    def _reset_selected_bone(self, *_args) -> None:
        joint_index = self._selected_scale_joint()
        self.bone_scales[joint_index].fill_(1.0)
        self._sync_scale_sliders()
        self._refresh_view()

    def _reset_all(self, *_args) -> None:
        self.full_pose.zero_()
        self.translation.zero_()
        self.bone_scales.fill_(1.0)
        self.global_bone_scale = 1.0
        self.animation_time = 0.0
        self.example_timer.stop()
        self.sequence_timer.stop()
        self._clear_sequence_fit()
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._sync_scale_sliders()
        self._sync_global_scale_slider()
        self._sync_animation_phase_slider()
        self._update_fit_controls()
        self._update_example_controls()
        self._refresh_view(fit_camera=True)

    def _on_global_scale_changed(self, value: float) -> None:
        self.global_bone_scale = value
        self._refresh_view()

    def _fit_camera(self, *_args) -> None:
        joints = self._display_joints()
        if self.target_joints_3d is not None:
            joints = np.concatenate([joints, self.target_joints_3d], axis=0)
        self.viewport.fit_camera_to_joints(joints)

    def _current_pose_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.full_pose.clone()
        translation = self.translation.clone()
        preset = self._active_animation()
        if preset is None:
            return self._clamp_pose_to_rom_limits(pose), translation

        pose_delta, translation_delta = preset.generator(self, self.animation_time)
        pose = self._clamp_pose_to_rom_limits(pose + pose_delta)
        translation = translation + translation_delta
        return pose, translation

    def _current_joints(self) -> np.ndarray:
        pose, translation = self._current_pose_state()
        with torch.no_grad():
            output = self.model(
                full_pose=pose,
                bone_scales=self.bone_scales * self.global_bone_scale,
                transl=translation,
            )
        return output.joints.detach().cpu().numpy()

    def _display_joints(self) -> np.ndarray:
        if self.preview_override_joints is not None:
            return self.preview_override_joints
        return self._current_joints()

    def _refresh_view(self, *_args, fit_camera: bool = False) -> None:
        joints = self._display_joints()
        self.viewport.update_skeleton(
            joints, selected_joint=self._selected_pose_joint()
        )
        self.viewport.set_target_overlay(self.target_joints_3d)
        self._update_projection_preview(joints)
        if fit_camera:
            self._fit_camera()


def main() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("DifferentialSkeletons Playground")

    window = SkelixPlayground()
    app._skelix_window = window
    window.show()
    if not owns_app:
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
