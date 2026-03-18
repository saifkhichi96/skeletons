"""Interactive playground app for visualizing skeletons and testing animation presets."""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
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

from skelix import (
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

BONE_COLOR = (0.93, 0.91, 0.82, 1.0)
JOINT_COLOR = (0.65, 0.69, 0.74, 1.0)
ROOT_COLOR = (0.61, 0.73, 0.60, 1.0)
SELECTED_COLOR = (0.89, 0.77, 0.49, 1.0)
AXIS_COLORS = {
    "X": QtGui.QColor(255, 99, 99),
    "Y": QtGui.QColor(110, 227, 140),
    "Z": QtGui.QColor(110, 170, 255),
}
SCENE_TO_VIEW = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


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


@dataclass(frozen=True)
class AnimationPreset:
    key: str
    label: str
    period: float
    matcher: Callable[[object], bool]
    generator: Callable[[object, float], tuple[torch.Tensor, torch.Tensor]]


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


def generate_root_sway(model, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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


def generate_walk_cycle(model, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    phase = (2.0 * math.pi * time_s) / 1.25
    pose, translation = _make_animation_buffers(model)

    _set_axis_angle(pose, model, model.joint_names[model.root_index], (0.03 * math.sin(phase * 2.0), 0.05 * math.sin(phase), 0.03 * math.sin(phase + math.pi / 2.0)))

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
    _set_axis_angle(pose, model, "left_elbow", (-0.12 - 0.08 * max(0.0, support), 0.0, 0.0))
    _set_axis_angle(pose, model, "right_elbow", (-0.12 - 0.08 * max(0.0, -support), 0.0, 0.0))
    _set_axis_angle(pose, model, "spine", (0.04 * math.sin(phase + 0.6), 0.0, 0.0))
    _set_axis_angle(pose, model, "thorax", (0.05 * math.sin(phase + 0.3), 0.0, 0.0))
    _set_axis_angle(pose, model, "neck_base", (0.03 * math.sin(phase), 0.0, 0.0))

    translation[1] = 0.03 * (1.0 - math.cos(phase * 2.0)) * 0.5
    translation[2] = 0.06 * math.sin(phase)
    return pose, translation


def generate_arm_wave(model, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
    phase = (2.0 * math.pi * time_s) / 1.6
    pose, translation = _make_animation_buffers(model)
    _set_axis_angle(pose, model, "right_shoulder", (-0.25, 0.0, -0.95))
    _set_axis_angle(pose, model, "right_elbow", (-0.35 - 0.20 * math.sin(phase), 0.0, 0.0))
    _set_axis_angle(pose, model, "right_wrist", (0.35 * math.sin(phase * 2.0), 0.0, 0.0))
    _set_axis_angle(pose, model, "left_shoulder", (0.10, 0.0, 0.18))
    _set_axis_angle(pose, model, "left_elbow", (-0.10, 0.0, 0.0))
    _set_axis_angle(pose, model, "thorax", (0.03 * math.sin(phase), 0.0, -0.04))
    _set_axis_angle(pose, model, "neck_base", (0.0, 0.0, 0.04 * math.sin(phase)))
    return pose, translation


def generate_finger_wave(model, time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
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
    _set_axis_angle(pose, model, "left_hand_root", (0.08 * math.sin(phase * 0.5), 0.0, 0.0))
    _set_axis_angle(pose, model, "right_hand_root", (-0.08 * math.sin(phase * 0.5), 0.0, 0.0))
    return pose, translation


ANIMATION_PRESETS = (
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
        matcher=lambda model: _has_joints(model, "left_hip", "left_knee", "right_hip", "right_knee"),
        generator=generate_walk_cycle,
    ),
    AnimationPreset(
        key="arm_wave",
        label="Arm Wave",
        period=1.6,
        matcher=lambda model: _has_joints(model, "right_shoulder", "right_elbow") or _has_joints(model, "left_shoulder", "left_elbow"),
        generator=generate_arm_wave,
    ),
    AnimationPreset(
        key="finger_wave",
        label="Finger Wave",
        period=1.8,
        matcher=lambda model: _has_any_prefix(model, ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger", "left_thumb", "right_thumb")),
        generator=generate_finger_wave,
    ),
)


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
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
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
        bg = QtGui.QColor(8, 11, 17, 188)
        outline = QtGui.QColor(84, 98, 118, 180)
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

        for depth, label, direction_2d, color in sorted(axis_draw_data, key=lambda item: item[0]):
            alpha = 255 if depth >= 0.0 else 150
            draw_color = QtGui.QColor(color)
            draw_color.setAlpha(alpha)
            self._draw_arrow(painter, center, radius, direction_2d, draw_color, label)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(235, 240, 246))
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

        pen = QtGui.QPen(color, 6.0, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
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


class SkeletonViewport(gl.GLViewWidget):
    CAMERA_ELEVATION = 26.0
    CAMERA_DISTANCE = 4.4
    CAMERA_AZIMUTH = -36.0
    GIZMO_MARGIN = 18

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.parents: tuple[int, ...] = ()
        self.bone_items: list[gl.GLMeshItem | None] = []
        self.joint_items: list[gl.GLMeshItem] = []
        self._last_render_joints: np.ndarray | None = None

        self.setBackgroundColor((14, 18, 24))
        self.setCameraPosition(
            pos=QtGui.QVector3D(0.0, 0.0, 0.0),
            distance=self.CAMERA_DISTANCE,
            elevation=self.CAMERA_ELEVATION,
            azimuth=self.CAMERA_AZIMUTH,
        )

        self.grid = gl.GLGridItem()
        self.grid.setSize(2.0, 2.0, 1.0)
        self.grid.setSpacing(0.2, 0.2, 0.2)
        self.addItem(self.grid)

        self.bone_mesh = create_bone_pyramid_mesh()
        self.joint_mesh = gl.MeshData.sphere(rows=10, cols=20, radius=1.0)
        self.gizmo = AxisGizmoOverlay(self)
        self.gizmo.show()
        self.gizmo.raise_()
        self._place_gizmo()

    def set_topology(self, parents: tuple[int, ...]) -> None:
        for item in self.joint_items:
            self.removeItem(item)
        for item in self.bone_items:
            if item is not None:
                self.removeItem(item)

        self.parents = tuple(parents)
        self.joint_items = []
        self.bone_items = [None] * len(self.parents)

        for joint_index, parent_index in enumerate(self.parents):
            joint_item = gl.GLMeshItem(
                meshdata=self.joint_mesh,
                smooth=True,
                drawFaces=True,
                drawEdges=False,
                shader="shaded",
                color=JOINT_COLOR,
            )
            joint_item.setGLOptions("opaque")
            self.addItem(joint_item)
            self.joint_items.append(joint_item)

            if parent_index < 0:
                continue

            bone_item = gl.GLMeshItem(
                meshdata=self.bone_mesh,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                shader="shaded",
                color=BONE_COLOR,
            )
            bone_item.setGLOptions("opaque")
            self.addItem(bone_item)
            self.bone_items[joint_index] = bone_item

    def update_skeleton(self, joints: np.ndarray, *, selected_joint: int | None = None) -> None:
        if not self.parents or len(self.parents) != len(joints):
            raise ValueError("Topology must be initialized before updating the skeleton.")

        render_joints = scene_to_view(joints)
        self._last_render_joints = render_joints

        mins = render_joints.min(axis=0)
        maxs = render_joints.max(axis=0)
        extent = float(np.max(maxs - mins))
        extent = max(extent, 0.35)
        bone_radius = max(extent * 0.015, 0.003)
        joint_radius = max(extent * 0.018, 0.004)
        grid_size = max(extent * 2.4, 1.2)
        grid_spacing = max(grid_size / 12.0, 0.05)
        ground_height = float(mins[2])
        horizontal_center = (mins[:2] + maxs[:2]) * 0.5

        self.grid.resetTransform()
        self.grid.setSize(grid_size, grid_size, 1.0)
        self.grid.setSpacing(grid_spacing, grid_spacing, grid_spacing)
        self.grid.translate(
            float(horizontal_center[0]),
            float(horizontal_center[1]),
            ground_height,
            local=False,
        )

        for joint_index, item in enumerate(self.joint_items):
            item.resetTransform()
            item.scale(joint_radius, joint_radius, joint_radius)
            item.translate(*map(float, render_joints[joint_index]), local=False)

            if joint_index == selected_joint:
                item.setColor(SELECTED_COLOR)
            elif self.parents[joint_index] < 0:
                item.setColor(ROOT_COLOR)
            else:
                item.setColor(JOINT_COLOR)

        for joint_index, parent_index in enumerate(self.parents):
            if parent_index < 0:
                continue

            item = self.bone_items[joint_index]
            if item is None:
                continue

            highlight = joint_index == selected_joint
            self._place_bone(
                item,
                render_joints[parent_index],
                render_joints[joint_index],
                bone_radius=bone_radius,
                color=SELECTED_COLOR if highlight else BONE_COLOR,
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


class SkelixPlayground(QtWidgets.QMainWindow):
    ANIMATION_DT = 1.0 / 60.0

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("skelix Playground")
        self.resize(1500, 920)

        self.model = None
        self.available_animations: list[AnimationPreset] = []
        self.animation_time = 0.0
        self.animation_speed = 1.0
        self.full_pose = torch.zeros(1, 3)
        self.translation = torch.zeros(3)
        self.bone_scales = torch.ones(1)
        self.scale_joint_indices: list[int] = []
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setInterval(int(round(self.ANIMATION_DT * 1000)))
        self.animation_timer.timeout.connect(self._advance_animation)

        pg.setConfigOptions(antialias=True)
        self._apply_styles()
        self._build_ui()
        self._load_skeleton(self.skeleton_combo.currentText())

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #11161d;
                color: #e6edf3;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #2b3441;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 12px 12px 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QComboBox, QPushButton {
                background: #0b1016;
                border: 1px solid #2b3441;
                border-radius: 7px;
                padding: 6px 8px;
            }
            QPushButton:hover, QComboBox:hover {
                border-color: #4d94ff;
            }
            QScrollArea {
                border: none;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #26303c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                background: #58a6ff;
                border-radius: 8px;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        self.viewport = SkeletonViewport()
        splitter.addWidget(self.viewport)

        sidebar = QtWidgets.QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setMinimumWidth(360)
        sidebar.setMaximumWidth(440)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        sidebar_body = QtWidgets.QWidget()
        sidebar.setWidget(sidebar_body)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_body)
        sidebar_layout.setContentsMargins(6, 6, 6, 6)
        sidebar_layout.setSpacing(12)

        skeleton_group = QtWidgets.QGroupBox("Skeleton")
        skeleton_layout = QtWidgets.QVBoxLayout(skeleton_group)
        self.skeleton_combo = QtWidgets.QComboBox()
        self.skeleton_combo.addItems(list(SKELETON_FACTORIES.keys()))
        self.skeleton_combo.setCurrentText("Human3.6M")
        self.skeleton_combo.currentTextChanged.connect(self._load_skeleton)
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #9fb0c0;")
        skeleton_layout.addWidget(self.skeleton_combo)
        skeleton_layout.addWidget(self.info_label)
        sidebar_layout.addWidget(skeleton_group)

        animation_group = QtWidgets.QGroupBox("Animation")
        animation_layout = QtWidgets.QVBoxLayout(animation_group)
        animation_note = QtWidgets.QLabel("Preset motion is added on top of the current slider pose.")
        animation_note.setWordWrap(True)
        animation_note.setStyleSheet("color: #9fb0c0;")
        animation_layout.addWidget(animation_note)
        self.animation_combo = QtWidgets.QComboBox()
        self.animation_combo.currentIndexChanged.connect(self._on_animation_changed)
        animation_layout.addWidget(self.animation_combo)

        animation_buttons = QtWidgets.QHBoxLayout()
        self.animation_toggle_button = QtWidgets.QPushButton("Play")
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
        self.animation_speed_slider.value_changed.connect(self._on_animation_speed_changed)
        animation_layout.addWidget(self.animation_speed_slider)

        self.animation_phase_slider = FloatSlider(
            "Phase",
            minimum=0.0,
            maximum=1.0,
            factor=1000,
            decimals=3,
            suffix=" turn",
        )
        self.animation_phase_slider.value_changed.connect(self._on_animation_phase_changed)
        animation_layout.addWidget(self.animation_phase_slider)
        sidebar_layout.addWidget(animation_group)

        pose_group = QtWidgets.QGroupBox("Joint Pose")
        pose_layout = QtWidgets.QVBoxLayout(pose_group)
        pose_note = QtWidgets.QLabel("Axis-angle in degrees. Select the root joint to edit global orientation.")
        pose_note.setWordWrap(True)
        pose_note.setStyleSheet("color: #9fb0c0;")
        pose_layout.addWidget(pose_note)
        self.pose_joint_combo = QtWidgets.QComboBox()
        self.pose_joint_combo.currentIndexChanged.connect(self._sync_pose_sliders)
        self.pose_joint_combo.currentIndexChanged.connect(self._refresh_view)
        pose_layout.addWidget(self.pose_joint_combo)
        self.pose_sliders: list[FloatSlider] = []
        for axis_name in ("X", "Y", "Z"):
            slider = FloatSlider(
                f"{axis_name} rotation",
                minimum=-180.0,
                maximum=180.0,
                factor=10,
                decimals=1,
                suffix=" deg",
            )
            slider.value_changed.connect(self._make_pose_callback(len(self.pose_sliders)))
            pose_layout.addWidget(slider)
            self.pose_sliders.append(slider)
        self.zero_joint_button = QtWidgets.QPushButton("Zero Selected Joint")
        self.zero_joint_button.clicked.connect(self._zero_selected_joint)
        pose_layout.addWidget(self.zero_joint_button)
        sidebar_layout.addWidget(pose_group)

        translation_group = QtWidgets.QGroupBox("Translation")
        translation_layout = QtWidgets.QVBoxLayout(translation_group)
        self.translation_sliders: list[FloatSlider] = []
        for axis_name in ("X", "Y", "Z"):
            slider = FloatSlider(
                f"{axis_name} offset",
                minimum=-2.0,
                maximum=2.0,
                factor=100,
                decimals=2,
                suffix=" m",
            )
            slider.value_changed.connect(self._make_translation_callback(len(self.translation_sliders)))
            translation_layout.addWidget(slider)
            self.translation_sliders.append(slider)
        sidebar_layout.addWidget(translation_group)

        scale_group = QtWidgets.QGroupBox("Bone Scale")
        scale_layout = QtWidgets.QVBoxLayout(scale_group)
        scale_note = QtWidgets.QLabel("Scale the segment ending at the selected joint.")
        scale_note.setWordWrap(True)
        scale_note.setStyleSheet("color: #9fb0c0;")
        scale_layout.addWidget(scale_note)
        self.scale_joint_combo = QtWidgets.QComboBox()
        self.scale_joint_combo.currentIndexChanged.connect(self._sync_scale_slider)
        scale_layout.addWidget(self.scale_joint_combo)
        self.scale_slider = FloatSlider(
            "Length scale",
            minimum=0.25,
            maximum=2.50,
            factor=100,
            decimals=2,
            suffix="x",
        )
        self.scale_slider.value_changed.connect(self._on_scale_changed)
        scale_layout.addWidget(self.scale_slider)
        self.reset_scale_button = QtWidgets.QPushButton("Reset Selected Bone")
        self.reset_scale_button.clicked.connect(self._reset_selected_bone)
        scale_layout.addWidget(self.reset_scale_button)
        sidebar_layout.addWidget(scale_group)

        button_row = QtWidgets.QHBoxLayout()
        self.fit_button = QtWidgets.QPushButton("Fit Camera")
        self.fit_button.clicked.connect(self._fit_camera)
        self.reset_button = QtWidgets.QPushButton("Reset All")
        self.reset_button.clicked.connect(self._reset_all)
        button_row.addWidget(self.fit_button)
        button_row.addWidget(self.reset_button)
        sidebar_layout.addLayout(button_row)
        sidebar_layout.addStretch(1)

    def _load_skeleton(self, display_name: str) -> None:
        model_cls = SKELETON_FACTORIES[display_name]
        self.model = model_cls(
            create_global_orient=False,
            create_body_pose=False,
            create_bone_scales=False,
            create_transl=False,
        )
        dtype = self.model.rest_offsets.dtype
        self.full_pose = torch.zeros(self.model.num_joints, 3, dtype=dtype)
        self.translation = torch.zeros(3, dtype=dtype)
        self.bone_scales = torch.ones(self.model.num_joints, dtype=dtype)
        self.scale_joint_indices = list(self.model.non_root_joint_indices)
        self.animation_time = 0.0
        self.animation_timer.stop()

        self.viewport.set_topology(self.model.parents)
        self.info_label.setText(
            f"Spec: {self.model.spec.name}\n"
            f"Root joint: {self.model.joint_names[self.model.root_index]}\n"
            f"Joints: {self.model.num_joints}"
        )

        self._populate_animation_selectors()
        self._populate_joint_selectors()
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._sync_scale_slider()
        self._sync_animation_phase_slider()
        self._update_animation_controls()
        self._refresh_view(fit_camera=True)

    def _populate_animation_selectors(self) -> None:
        current_key = self._active_animation().key if self._active_animation() is not None else "none"
        self.available_animations = [preset for preset in ANIMATION_PRESETS if preset.matcher(self.model)]

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
        scale_joint = self.scale_joint_indices[0] if self.scale_joint_indices else 0

        self.pose_joint_combo.blockSignals(True)
        self.pose_joint_combo.clear()
        self.pose_joint_combo.addItems(self.model.joint_names)
        self.pose_joint_combo.setCurrentIndex(pose_joint)
        self.pose_joint_combo.blockSignals(False)

        self.scale_joint_combo.blockSignals(True)
        self.scale_joint_combo.clear()
        self.scale_joint_combo.addItems([self.model.joint_names[idx] for idx in self.scale_joint_indices])
        self.scale_joint_combo.setCurrentText(self.model.joint_names[scale_joint])
        self.scale_joint_combo.blockSignals(False)

    def _make_pose_callback(self, axis: int):
        def callback(value: float) -> None:
            joint_index = self.pose_joint_combo.currentIndex()
            self.full_pose[joint_index, axis] = math.radians(value)
            self._refresh_view()

        return callback

    def _make_translation_callback(self, axis: int):
        def callback(value: float) -> None:
            self.translation[axis] = value
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
        self.animation_time = (self.animation_time + self.ANIMATION_DT * self.animation_speed) % preset.period
        self._sync_animation_phase_slider()
        self._refresh_view()

    def _sync_pose_sliders(self, *_args) -> None:
        joint_index = self.pose_joint_combo.currentIndex()
        if joint_index < 0:
            return
        values_deg = [math.degrees(float(v)) for v in self.full_pose[joint_index]]
        for axis, slider in enumerate(self.pose_sliders):
            slider.set_value(values_deg[axis], emit=False)

    def _sync_translation_sliders(self) -> None:
        for axis, slider in enumerate(self.translation_sliders):
            slider.set_value(float(self.translation[axis]), emit=False)

    def _sync_scale_slider(self, *_args) -> None:
        selection = self.scale_joint_combo.currentIndex()
        if selection < 0 or not self.scale_joint_indices:
            return
        joint_index = self.scale_joint_indices[selection]
        self.scale_slider.set_value(float(self.bone_scales[joint_index]), emit=False)

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
        self._sync_pose_sliders()
        self._refresh_view()

    def _reset_selected_bone(self, *_args) -> None:
        joint_index = self._selected_scale_joint()
        self.bone_scales[joint_index] = 1.0
        self._sync_scale_slider()
        self._refresh_view()

    def _reset_all(self, *_args) -> None:
        self.full_pose.zero_()
        self.translation.zero_()
        self.bone_scales.fill_(1.0)
        self.animation_time = 0.0
        self._sync_pose_sliders()
        self._sync_translation_sliders()
        self._sync_scale_slider()
        self._sync_animation_phase_slider()
        self._refresh_view(fit_camera=True)

    def _on_scale_changed(self, value: float) -> None:
        joint_index = self._selected_scale_joint()
        self.bone_scales[joint_index] = value
        self._refresh_view()

    def _fit_camera(self, *_args) -> None:
        joints = self._current_joints()
        self.viewport.fit_camera_to_joints(joints)

    def _current_pose_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.full_pose.clone()
        translation = self.translation.clone()
        preset = self._active_animation()
        if preset is None:
            return pose, translation

        pose_delta, translation_delta = preset.generator(self.model, self.animation_time)
        pose = pose + pose_delta
        translation = translation + translation_delta
        return pose, translation

    def _current_joints(self) -> np.ndarray:
        pose, translation = self._current_pose_state()
        with torch.no_grad():
            output = self.model(
                full_pose=pose,
                bone_scales=self.bone_scales,
                transl=translation,
            )
        return output.joints.detach().cpu().numpy()

    def _refresh_view(self, *_args, fit_camera: bool = False) -> None:
        joints = self._current_joints()
        self.viewport.update_skeleton(joints, selected_joint=self._selected_pose_joint())
        if fit_camera:
            self.viewport.fit_camera_to_joints(joints)


def main() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("skelix Playground")

    window = SkelixPlayground()
    app._skelix_window = window
    window.show()
    if not owns_app:
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
