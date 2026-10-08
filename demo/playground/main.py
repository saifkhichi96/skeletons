from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .theme import APP_TITLE, STYLE_SHEET, _light_palette
from .window import DifferentialSkeletonsPlaygroundWindow


def main(argv: list[str] | None = None) -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(argv or sys.argv)
        app.setApplicationName(APP_TITLE)
        app.setStyle("Fusion")
        app.setPalette(_light_palette())
        app.setStyleSheet(STYLE_SHEET)

    window = DifferentialSkeletonsPlaygroundWindow()
    app._skeletons_playground = window  # type: ignore[attr-defined]
    window.show()
    if not owns_app:
        return 0
    return app.exec()
