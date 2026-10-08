from __future__ import annotations

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .models import SceneNode
from .theme import SECONDARY_TEXT_STYLE


class _FlowLayout(QtWidgets.QLayout):
    """Wrap sidebar action rows to the width supplied by their parent."""

    def __init__(self) -> None:
        super().__init__()
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(8)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def addWidget(self, widget: QtWidgets.QWidget, stretch: int = 0) -> None:
        del stretch
        super().addWidget(widget)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def minimumSize(self) -> QtCore.QSize:
        return QtCore.QSize(
            0, max((i.minimumSize().height() for i in self._items), default=0)
        )

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def _arrange(self, rect: QtCore.QRect, *, apply: bool) -> int:
        x, y, row_height = rect.x(), rect.y(), 0
        for item in self._items:
            size = item.sizeHint()
            width = min(size.width(), max(rect.width(), 0))
            if x > rect.x() and x + width > rect.x() + rect.width():
                x = rect.x()
                y += row_height + self.spacing()
                row_height = 0
            height = (
                item.heightForWidth(width)
                if item.hasHeightForWidth()
                else size.height()
            )
            if apply:
                item.setGeometry(QtCore.QRect(x, y, width, height))
            x += width + self.spacing()
            row_height = max(row_height, height)
        return y + row_height - rect.y()


class _SidebarScrollArea(QtWidgets.QScrollArea):
    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        result = super().viewportEvent(event)
        if event.type() == QtCore.QEvent.Type.Resize:
            content = self.widget()
            if content is not None:
                for combo in content.findChildren(QtWidgets.QComboBox):
                    combo.setMinimumContentsLength(8)
                    combo.setSizeAdjustPolicy(
                        QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                    )
                width = self.viewport().width()
                if content.minimumWidth() != width or content.maximumWidth() != width:
                    content.setFixedWidth(width)
        return result


def _scroll_panel(content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    content.setObjectName("SidebarBody")
    area = _SidebarScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setSizeAdjustPolicy(
        QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
    )
    area.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored
    )
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    area.viewport().setObjectName("SidebarViewport")
    area.setWidget(content)
    return area


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
        compact: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.factor = factor
        self.decimals = decimals
        self.suffix = suffix

        layout = QtWidgets.QHBoxLayout(self) if compact else QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.title_label = QtWidgets.QLabel(title)
        self.value_label = QtWidgets.QLabel()
        self.value_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        if compact:
            layout.addWidget(self.title_label)
        else:
            header = QtWidgets.QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.addWidget(self.title_label)
            header.addStretch(1)
            header.addWidget(self.value_label)
            layout.addLayout(header)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum * factor)), int(round(maximum * factor)))
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, 1 if compact else 0)
        if compact:
            layout.addWidget(self.value_label)
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


class EventLogWidget(QtWidgets.QPlainTextEdit):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMaximumBlockCount(500)
        self.setPlaceholderText("Events and workflow messages appear here.")

    def append_event(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.appendPlainText(f"[{timestamp}] {message}")


class MetricPlotWidget(pg.PlotWidget):
    def __init__(
        self, title: str = "Metric History", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent=parent)
        self.setBackground("w")
        self.setTitle(title)
        self.showGrid(x=True, y=True, alpha=0.18)
        self.addLegend(offset=(12, 12))
        self._series: dict[str, tuple[list[float], list[float], pg.PlotDataItem]] = {}

    def clear_series(self) -> None:
        for _, _, item in self._series.values():
            self.removeItem(item)
        self._series.clear()

    def add_point(
        self, series: str, x: float, y: float, *, pen: str | None = None
    ) -> None:
        if series not in self._series:
            item = self.plot(name=series, pen=pen or None)
            self._series[series] = ([], [], item)
        xs, ys, item = self._series[series]
        xs.append(float(x))
        ys.append(float(y))
        item.setData(xs, ys)

    def set_series(
        self, series: str, xs: list[float], ys: list[float], *, pen: str | None = None
    ) -> None:
        if series not in self._series:
            item = self.plot(name=series, pen=pen or None)
            self._series[series] = ([], [], item)
        _, _, item = self._series[series]
        self._series[series] = (list(xs), list(ys), item)
        item.setData(xs, ys)


class SceneExplorer(QtWidgets.QTreeWidget):
    node_selected = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["Scene", "Kind"])
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setAlternatingRowColors(False)
        self.currentItemChanged.connect(self._on_current_item_changed)

    def set_nodes(self, nodes: list[SceneNode]) -> None:
        current = self.currentItem()
        selected = current.data(0, QtCore.Qt.ItemDataRole.UserRole) if current else None
        selected_key = (
            (selected.group, selected.kind, selected.label)
            if isinstance(selected, SceneNode)
            else None
        )
        restored = False
        was_blocked = self.blockSignals(True)
        try:
            self.clear()
            groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
            for node in nodes:
                group_item = groups.get(node.group)
                if group_item is None:
                    group_item = QtWidgets.QTreeWidgetItem([node.group, ""])
                    group_item.setFlags(
                        group_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable
                    )
                    self.addTopLevelItem(group_item)
                    groups[node.group] = group_item
                item = QtWidgets.QTreeWidgetItem([node.label, node.kind])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node)
                if node.description:
                    item.setToolTip(0, node.description)
                group_item.addChild(item)
                if (node.group, node.kind, node.label) == selected_key:
                    self.setCurrentItem(item)
                    restored = True
            self.expandAll()
        finally:
            self.blockSignals(was_blocked)
        if selected_key is not None and not restored:
            self.node_selected.emit(None)

    def _on_current_item_changed(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        del previous
        if current is None:
            self.node_selected.emit(None)
            return
        node = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        self.node_selected.emit(node)


class InspectorPanel(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QtWidgets.QLabel("Inspector")
        self.title_label.setStyleSheet("font-weight: 600;")
        self.subtitle_label = QtWidgets.QLabel("Select a scene element to inspect it.")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(SECONDARY_TEXT_STYLE)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["Property", "Value"])
        self.table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.table.setWordWrap(True)
        self.table.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(False)
        layout.addWidget(self.table, 1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setPlaceholderText("Additional notes or serialized metadata.")
        self.notes.setMaximumHeight(120)
        layout.addWidget(self.notes)

    def clear(self) -> None:
        self.title_label.setText("Inspector")
        self.subtitle_label.setText("Select a scene element to inspect it.")
        self.table.clear()
        self.notes.clear()

    def set_content(
        self,
        *,
        title: str,
        subtitle: str = "",
        properties: dict[str, object] | None = None,
        notes: str = "",
    ) -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.table.clear()
        for key, value in (properties or {}).items():
            item = QtWidgets.QTreeWidgetItem([str(key), self._format_value(value)])
            self.table.addTopLevelItem(item)
        self.table.expandAll()
        self.notes.setPlainText(notes)

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=4, suppress_small=True)
        return str(value)


class ProjectionPreview(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.parents: tuple[int, ...] = ()
        self.target_points: np.ndarray | None = None
        self.predicted_points: np.ndarray | None = None
        self.reference_points: np.ndarray | None = None
        self.reference_color = QtGui.QColor(92, 102, 118)
        self.target_color = QtGui.QColor(0, 122, 255)
        self.prediction_color = QtGui.QColor(255, 149, 0)

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

    def set_reference_points(self, points: np.ndarray | None) -> None:
        self.reference_points = (
            None if points is None else np.asarray(points, dtype=float)
        )
        self.update()

    def clear(self) -> None:
        self.reference_points = None
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
            for points in (
                self.reference_points,
                self.target_points,
                self.predicted_points,
            )
            if points is not None and len(points) > 0
        ]
        if not point_sets:
            painter.setPen(QtGui.QColor(110, 117, 130))
            painter.drawText(
                content,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "Load or generate data to preview 2D projections.",
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

        if self.reference_points is not None:
            self._draw_skeleton_2d(
                painter,
                self.reference_points,
                map_point,
                bone_color=self.reference_color,
                joint_color=self.reference_color,
                dashed=True,
            )
        if self.target_points is not None:
            self._draw_skeleton_2d(
                painter,
                self.target_points,
                map_point,
                bone_color=self.target_color,
                joint_color=self.target_color,
                dashed=True,
            )
        if self.predicted_points is not None:
            self._draw_skeleton_2d(
                painter,
                self.predicted_points,
                map_point,
                bone_color=self.prediction_color,
                joint_color=self.prediction_color,
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
        x = legend_rect.right() - 208
        self._draw_legend_entry(
            painter,
            x,
            legend_rect.center().y(),
            self.reference_color,
            "Reference",
            dashed=True,
        )
        self._draw_legend_entry(
            painter,
            x + 78,
            legend_rect.center().y(),
            self.target_color,
            "Target",
            dashed=True,
        )
        self._draw_legend_entry(
            painter,
            x + 138,
            legend_rect.center().y(),
            self.prediction_color,
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
        for point in points:
            mapped = map_point(point)
            painter.drawEllipse(mapped, 3.5, 3.5)
