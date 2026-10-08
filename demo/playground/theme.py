from __future__ import annotations

from PySide6 import QtGui

APP_TITLE = "Skeletons Playground"
SECONDARY_TEXT_STYLE = "color: #6e7582;"

SCENE_COLORS = {
    "bone": (0.77, 0.81, 0.88, 1.0),
    "joint": (0.00, 0.48, 1.00, 1.0),
    "root": (0.10, 0.63, 0.95, 1.0),
    "selected": (1.00, 0.62, 0.20, 1.0),
    "overlay_bone": (0.24, 0.68, 0.97, 0.40),
    "overlay_joint": (0.47, 0.77, 1.00, 0.64),
    "marker": (1.00, 0.80, 0.20, 1.0),
    "contact": (0.82, 0.24, 0.24, 0.95),
    "link": (0.64, 0.66, 0.72, 0.95),
    "target_a": (0.95, 0.45, 0.15, 1.0),
    "target_b": (0.15, 0.80, 0.95, 1.0),
    "trail_a": (0.95, 0.45, 0.15, 0.55),
    "trail_b": (0.15, 0.80, 0.95, 0.55),
}

STYLE_SHEET = """
QMainWindow, QWidget#PlaygroundRoot, QWidget#SidebarBody, QWidget#SidebarViewport, QScrollArea {
    background: #f5f5f7;
}
QWidget {
    color: #1d1d1f;
    font-size: 13px;
    font-family: "SF Pro Display", "Helvetica Neue", "Segoe UI", sans-serif;
}
QMenuBar, QToolBar, QMenu {
    background: #f5f5f7;
    color: #1d1d1f;
    border: none;
}
QToolBar {
    spacing: 8px;
    padding: 6px;
    border-bottom: 1px solid #e3e7ee;
}
QMenuBar::item, QMenu::item {
    background: transparent;
    padding: 6px 12px;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: #e7effc;
    color: #1d1d1f;
}
QMenu::separator {
    height: 1px;
    background: #e3e7ee;
    margin: 4px 8px;
}
QHeaderView, QHeaderView::section, QTableCornerButton::section {
    background: #edf1f6;
    color: #5f6773;
    border: none;
    padding: 5px 8px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d8dde6;
    selection-background-color: #e7effc;
    selection-color: #1d1d1f;
    outline: none;
}
QToolTip {
    background: #ffffff;
    color: #1d1d1f;
    border: 1px solid #d8dde6;
    padding: 5px;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e3e7ee;
    border-radius: 18px;
    margin-top: 14px;
    padding: 14px 14px 14px 14px;
    font-weight: 600;
}
QGroupBox[compact="true"] {
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: #5f6773;
}
QComboBox, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QListWidget, QTreeWidget {
    background: #ffffff;
    border: 1px solid #d8dde6;
    border-radius: 12px;
    padding: 7px 10px;
    min-height: 20px;
    selection-background-color: #007aff;
}
QPushButton:hover, QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QTreeWidget:hover {
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
QScrollArea {
    border: none;
}
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 2px 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 0 2px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #cbd2db;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    min-height: 28px;
}
QScrollBar::handle:horizontal {
    min-width: 28px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #98a5b6;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
    background: transparent;
    border: none;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
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
QTabWidget::pane {
    border: none;
    background: transparent;
}
QTabBar::tab {
    background: #edf1f6;
    border: 1px solid #d5dbe7;
    border-bottom: none;
    padding: 8px 14px;
    margin-right: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    color: #5f6773;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1d1d1f;
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
    height: 10px;
}
QStatusBar {
    background: #f5f5f7;
    border-top: 1px solid #e4e8ef;
    color: #5f6773;
}
"""


def _light_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    colors = {
        roles.Window: "#f5f5f7",
        roles.WindowText: "#1d1d1f",
        roles.Base: "#ffffff",
        roles.AlternateBase: "#edf1f6",
        roles.Text: "#1d1d1f",
        roles.Button: "#ffffff",
        roles.ButtonText: "#1d1d1f",
        roles.ToolTipBase: "#ffffff",
        roles.ToolTipText: "#1d1d1f",
        roles.Highlight: "#007aff",
        roles.HighlightedText: "#ffffff",
        roles.PlaceholderText: "#6e7582",
        roles.Light: "#ffffff",
        roles.Midlight: "#edf1f6",
        roles.Mid: "#d8dde6",
        roles.Dark: "#b9c3d3",
        roles.Shadow: "#6e7582",
    }
    for role, color in colors.items():
        palette.setColor(role, QtGui.QColor(color))
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled, role, QtGui.QColor("#929ba8")
        )
    return palette
