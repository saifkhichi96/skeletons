from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph.opengl as gl
from OpenGL import GL
from PySide6 import QtCore, QtGui, QtWidgets

from .theme import SCENE_COLORS

SCENE_TO_VIEW = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
FLOOR_LIGHT_COLOR = (0.99, 0.99, 1.00, 1.0)
FLOOR_DARK_COLOR = (0.90, 0.92, 0.95, 1.0)
AXIS_COLORS = {
    "X": QtGui.QColor(255, 95, 87),
    "Y": QtGui.QColor(52, 199, 89),
    "Z": QtGui.QColor(10, 132, 255),
}


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


def vec3_to_numpy(value) -> np.ndarray:
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return np.array([value.x(), value.y(), value.z()], dtype=float)
    return np.asarray(value, dtype=float)


def scene_to_view(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ SCENE_TO_VIEW.T


def create_bone_pyramid_mesh() -> gl.MeshData:
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
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2], [0, 2, 3]],
        dtype=np.int32,
    )
    return gl.MeshData(vertexes=vertexes, faces=faces)


def create_checkerboard_mesh(
    *, size: float, tile_size: float, z: float = -0.002
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
            vertexes.extend([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
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


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


class AxisGizmoOverlay(QtWidgets.QWidget):
    def __init__(self, view: "SceneViewport") -> None:
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
        painter.setPen(QtGui.QPen(QtGui.QColor(210, 214, 222, 220), 1.5))
        painter.setBrush(QtGui.QColor(255, 255, 255, 226))
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
            draw_color = QtGui.QColor(color)
            draw_color.setAlpha(255 if depth >= 0.0 else 150)
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
            origin.x() + direction[0] * radius, origin.y() - direction[1] * radius
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
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QtGui.QPolygonF([tip, left, right]))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(color, 1.5))
        label_point = QtCore.QPointF(
            tip.x() + direction[0] * 13.0, tip.y() - direction[1] * 13.0
        )
        painter.drawText(label_point, label)


@dataclass
class SkeletonLayer:
    view: "SceneViewport"
    opaque: bool = True

    def __post_init__(self) -> None:
        self.parents: tuple[int, ...] = ()
        self.bone_mesh = create_bone_pyramid_mesh()
        self.joint_mesh = gl.MeshData.sphere(rows=10, cols=20, radius=1.0)
        self.joint_items: list[gl.GLMeshItem] = []
        self.bone_items: list[gl.GLMeshItem | None] = []

    def clear(self) -> None:
        for item in self.joint_items:
            self.view.removeItem(item)
        for item in self.bone_items:
            if item is not None:
                self.view.removeItem(item)
        self.joint_items = []
        self.bone_items = []
        self.parents = ()

    def set_topology(self, parents: tuple[int, ...]) -> None:
        self.clear()
        self.parents = tuple(parents)
        for joint_index, parent_index in enumerate(self.parents):
            joint_item = gl.GLMeshItem(
                meshdata=self.joint_mesh,
                smooth=True,
                drawFaces=True,
                drawEdges=False,
                shader=softlight_shader(),
                color=SCENE_COLORS["joint"],
            )
            joint_item.setGLOptions("opaque" if self.opaque else "translucent")
            self.view.addItem(joint_item)
            self.joint_items.append(joint_item)
            if parent_index < 0:
                self.bone_items.append(None)
                continue
            bone_item = gl.GLMeshItem(
                meshdata=self.bone_mesh,
                smooth=False,
                drawFaces=True,
                drawEdges=True,
                shader=softlight_shader(),
                color=SCENE_COLORS["bone"],
            )
            bone_item.setGLOptions("opaque" if self.opaque else "translucent")
            self.view.addItem(bone_item)
            self.bone_items.append(bone_item)

    def hide(self) -> None:
        for item in self.joint_items:
            item.hide()
        for item in self.bone_items:
            if item is not None:
                item.hide()

    def update(
        self,
        joints: np.ndarray,
        *,
        selected_joint: int | None = None,
        joint_color: tuple[float, float, float, float] | None = None,
        root_color: tuple[float, float, float, float] | None = None,
        bone_color: tuple[float, float, float, float] | None = None,
        highlight_color: tuple[float, float, float, float] | None = None,
    ) -> None:
        if not self.parents or len(self.parents) != len(joints):
            raise ValueError(
                "Topology must be initialized before updating the skeleton layer."
            )
        render_joints = scene_to_view(joints)
        mins = render_joints.min(axis=0)
        maxs = render_joints.max(axis=0)
        extent = max(float(np.max(maxs - mins)), 0.35)
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
        joint_color = joint_color or SCENE_COLORS["joint"]
        root_color = root_color or SCENE_COLORS["root"]
        bone_color = bone_color or SCENE_COLORS["bone"]
        highlight_color = highlight_color or SCENE_COLORS["selected"]
        for joint_index, item in enumerate(self.joint_items):
            item.show()
            item.resetTransform()
            radius = float(joint_radii[joint_index])
            item.scale(radius, radius, radius)
            item.translate(*map(float, render_joints[joint_index]), local=False)
            if selected_joint is not None and joint_index == selected_joint:
                item.setColor(highlight_color)
            elif self.parents[joint_index] < 0:
                item.setColor(root_color)
            else:
                item.setColor(joint_color)
        for joint_index, parent_index in enumerate(self.parents):
            if parent_index < 0:
                continue
            item = self.bone_items[joint_index]
            if item is None:
                continue
            start = render_joints[parent_index]
            end = render_joints[joint_index]
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length < 1e-8:
                item.hide()
                continue
            item.show()
            item.resetTransform()
            item.scale(
                float(bone_radii[joint_index]) * 1.55,
                float(bone_radii[joint_index]) * 1.55,
                length,
            )
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
            item.setColor(
                highlight_color
                if selected_joint is not None and joint_index == selected_joint
                else bone_color
            )


class SceneViewport(gl.GLViewWidget):
    point_placed = QtCore.Signal(object)
    placement_cancelled = QtCore.Signal()

    CAMERA_ELEVATION = 26.0
    CAMERA_DISTANCE = 4.4
    CAMERA_AZIMUTH = -36.0
    GIZMO_MARGIN = 18
    FLOOR_SIZE = 8.0
    FLOOR_TILE_SIZE = 0.5

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._placement_height: float | None = None
        self._placement_press = False
        self.setBackgroundColor((244, 246, 250))
        self.setCameraPosition(
            pos=QtGui.QVector3D(0.0, 0.0, 0.0),
            distance=self.CAMERA_DISTANCE,
            elevation=self.CAMERA_ELEVATION,
            azimuth=self.CAMERA_AZIMUTH,
        )
        self.floor_visible = True
        self._place_floor_on_next_pose = True
        self.floor_mesh = create_checkerboard_mesh(
            size=self.FLOOR_SIZE, tile_size=self.FLOOR_TILE_SIZE
        )
        self.floor_item = gl.GLMeshItem(
            meshdata=self.floor_mesh,
            smooth=False,
            drawFaces=True,
            drawEdges=False,
        )
        # Draw the reference plane first. Disabling depth testing also prevents
        # depth writes, so it cannot hide scene geometry from any camera angle.
        self.floor_item.setDepthValue(-1000)
        self.floor_item.setGLOptions(
            {GL.GL_DEPTH_TEST: False, GL.GL_BLEND: False, GL.GL_CULL_FACE: False}
        )
        self.addItem(self.floor_item)
        self.primary_layer = SkeletonLayer(self, opaque=True)
        self.overlay_layer = SkeletonLayer(self, opaque=False)
        self.scatter_layers: dict[str, gl.GLScatterPlotItem] = {}
        self.line_layers: dict[str, gl.GLLinePlotItem] = {}
        self.gizmo = AxisGizmoOverlay(self)
        self.gizmo.show()
        self.gizmo.raise_()
        self._place_gizmo()

    def set_floor_visible(self, visible: bool) -> None:
        self.floor_visible = bool(visible)
        self.floor_item.setVisible(self.floor_visible)

    def set_topology(self, parents: tuple[int, ...]) -> None:
        self._place_floor_on_next_pose = True
        self.primary_layer.set_topology(parents)
        self.overlay_layer.set_topology(parents)
        self.overlay_layer.hide()

    def update_skeleton(
        self, joints: np.ndarray, *, selected_joint: int | None = None
    ) -> None:
        if self._place_floor_on_next_pose:
            points = np.asarray(joints, dtype=float).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                clearance = max(float(np.ptp(points, axis=0).max()) * 0.03, 0.02)
                self.floor_item.resetTransform()
                self.floor_item.translate(0, 0, float(points[:, 1].min()) - clearance)
                self._place_floor_on_next_pose = False
        self.primary_layer.update(joints, selected_joint=selected_joint)
        self.gizmo.update()

    def set_target_overlay(self, joints: np.ndarray | None) -> None:
        if joints is None:
            self.overlay_layer.hide()
            self.gizmo.update()
            return
        self.overlay_layer.update(
            joints,
            selected_joint=None,
            joint_color=SCENE_COLORS["overlay_joint"],
            root_color=SCENE_COLORS["overlay_joint"],
            bone_color=SCENE_COLORS["overlay_bone"],
            highlight_color=SCENE_COLORS["overlay_joint"],
        )
        self.gizmo.update()

    def set_scatter_layer(
        self,
        key: str,
        positions: np.ndarray | None,
        *,
        size: float = 0.08,
        color: tuple[float, float, float, float] | None = None,
        colors: np.ndarray | None = None,
        px_mode: bool = False,
    ) -> None:
        item = self.scatter_layers.get(key)
        if positions is None:
            if item is not None:
                item.hide()
            return
        positions = scene_to_view(np.asarray(positions, dtype=float))
        if item is None:
            item = gl.GLScatterPlotItem(
                pos=positions,
                size=size,
                color=colors if colors is not None else color,
                pxMode=px_mode,
            )
            self.addItem(item)
            self.scatter_layers[key] = item
        else:
            item.show()
            item.setData(
                pos=positions,
                size=size,
                color=colors if colors is not None else color,
                pxMode=px_mode,
            )

    def set_line_layer(
        self,
        key: str,
        positions: np.ndarray | None,
        *,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        width: float = 2.0,
    ) -> None:
        item = self.line_layers.get(key)
        if positions is None:
            if item is not None:
                item.hide()
            return
        positions = scene_to_view(np.asarray(positions, dtype=float))
        if item is None:
            item = gl.GLLinePlotItem(
                pos=positions, color=color, width=width, antialias=True
            )
            self.addItem(item)
            self.line_layers[key] = item
        else:
            item.show()
            item.setData(pos=positions, color=color, width=width, antialias=True)

    def clear_overlay_layers(self) -> None:
        self.overlay_layer.hide()
        for item in self.scatter_layers.values():
            item.hide()
        for item in self.line_layers.values():
            item.hide()

    def fit_camera_to_points(self, points: np.ndarray) -> None:
        render_points = scene_to_view(np.asarray(points, dtype=float))
        mins = render_points.min(axis=0)
        maxs = render_points.max(axis=0)
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

    def set_placement_height(self, height: float | None) -> None:
        """Pick scene points on a horizontal plane at the given height."""
        self._placement_height = height
        if height is None:
            self.unsetCursor()
        else:
            self.setFocus()
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._placement_height is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self._placement_press = True
            width, height = self.width(), self.height()
            if width > 0 and height > 0:
                viewport = (0, 0, width, height)
                inverse, valid = (
                    self.projectionMatrix(viewport, viewport) * self.viewMatrix()
                ).inverted()
                if valid:
                    x = 2 * event.position().x() / width - 1
                    y = 1 - 2 * event.position().y() / height
                    points = []
                    for depth in (-1, 1):
                        point = inverse.map(QtGui.QVector4D(x, y, depth, 1))
                        points.append(
                            SCENE_TO_VIEW.T
                            @ np.array([point.x(), point.y(), point.z()])
                            / point.w()
                        )
                    direction = points[1] - points[0]
                    if abs(direction[1]) > 1e-9:
                        fraction = (self._placement_height - points[0][1]) / direction[
                            1
                        ]
                        if 0 <= fraction <= 1:
                            self.set_placement_height(None)
                            self.point_placed.emit(points[0] + fraction * direction)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._placement_press:
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._placement_press and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._placement_press = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if (
            self._placement_height is not None
            and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            self.set_placement_height(None)
            self.placement_cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_gizmo()

    def _place_gizmo(self) -> None:
        self.gizmo.move(
            self.width() - self.gizmo.width() - self.GIZMO_MARGIN,
            self.height() - self.gizmo.height() - self.GIZMO_MARGIN,
        )
        self.gizmo.raise_()
