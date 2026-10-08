from __future__ import annotations

from PySide6 import QtGui, QtWidgets

from .theme import APP_TITLE, STYLE_SHEET, _light_palette
from .workbenches import SkeletonWorkbench


class DifferentialSkeletonsPlaygroundWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setPalette(_light_palette())
        self.setStyleSheet(STYLE_SHEET)

        self.skeleton_workbench = SkeletonWorkbench()
        self.setCentralWidget(self.skeleton_workbench)
        self.skeleton_workbench.status_message.connect(self._show_status_message)

        self._build_menus()
        self.statusBar().showMessage("Ready.", 3000)
        self._resize_to_screen()

    def _resize_to_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            self.resize(1280, 800)
            return
        available = screen.availableGeometry()
        self.resize(
            min(1440, int(available.width() * 0.90)),
            min(900, int(available.height() * 0.85)),
        )
        self.move(
            available.center().x() - self.width() // 2,
            available.center().y() - self.height() // 2,
        )

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        quit_action = QtGui.QAction("Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu_bar.addMenu("&View")
        fit_camera_action = QtGui.QAction("Fit Camera", self)
        fit_camera_action.setShortcut(QtGui.QKeySequence("F"))
        fit_camera_action.triggered.connect(self._fit_active_camera)
        view_menu.addAction(fit_camera_action)

        toggle_floor_action = QtGui.QAction("Toggle Floor", self)
        toggle_floor_action.setShortcut(QtGui.QKeySequence("G"))
        toggle_floor_action.triggered.connect(self._toggle_active_floor)
        view_menu.addAction(toggle_floor_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QtGui.QAction("About Playground", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _active_workbench(self):
        return self.skeleton_workbench

    def _fit_active_camera(self) -> None:
        workbench = self._active_workbench()
        if getattr(workbench, "model", None) is not None:
            workbench._fit_camera()  # noqa: SLF001 - deliberate internal app API

    def _toggle_active_floor(self) -> None:
        workbench = self._active_workbench()
        viewport = getattr(workbench, "viewport", None)
        if viewport is None or not hasattr(viewport, "set_floor_visible"):
            return
        visible = bool(getattr(viewport, "floor_visible", True))
        workbench.floor_check.setChecked(not visible)
        self._show_status_message(
            f"Floor {'shown' if not visible else 'hidden'}.",
            2500,
        )

    def _show_about_dialog(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About Skeletons Playground",
            "Pose, animate, fit, and solve inverse kinematics on one shared rig.\n"
            "Use Scene, Data, Move, Fit, and IK on the left; display controls,\n"
            "metrics, and logs on the right.",
        )

    def _show_status_message(self, message: str, timeout_ms: int = 4000) -> None:
        self.statusBar().showMessage(message, timeout_ms)
