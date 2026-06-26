"""Qt Designer support widgets for the P1 UI.

These classes are imported by the generated UI module. They intentionally live
outside the main entrypoint so source imports and standalone packaging do not
depend on simulate_hydrogen_diffusion.py importing itself again.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel


class ClickableLabel(QLabel):
    # Qt Designer instantiates this class directly, so it must stay in a small
    # import-safe support module rather than the main application entrypoint.
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class CustomLabel(QLabel):
    # The UI stores some computed display labels in settings, so label text
    # changes need a signal even though QLabel does not provide one by default.
    textChanged = pyqtSignal(str)

    def setText(self, text):
        if text != self.text():
            super().setText(text)
            self.textChanged.emit(text)
