"""Shared startup-ready signaling for launcher-managed WeldCraft modules."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt5 import QtCore


READY_FILE_ENV_VAR = "WELDCRAFT_STARTUP_READY_FILE"


def get_ready_file_path(environment=None):
    env = os.environ if environment is None else environment
    ready_file_path = env.get(READY_FILE_ENV_VAR, "").strip()
    return ready_file_path or None


def mark_startup_ready(environment=None):
    ready_file_path = get_ready_file_path(environment)
    if not ready_file_path:
        return False

    try:
        Path(ready_file_path).touch()
    except OSError:
        return False
    return True


class StartupReadySignal(QtCore.QObject):
    """Mark a child app as ready once its main window is visible."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.ready_emitted = False
        self.window.installEventFilter(self)
        QtCore.QTimer.singleShot(0, self.schedule_ready_signal)

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() in (
            QtCore.QEvent.Show,
            QtCore.QEvent.ShowToParent,
            QtCore.QEvent.WindowStateChange,
        ):
            self.schedule_ready_signal()
        return super().eventFilter(watched, event)

    def schedule_ready_signal(self):
        if self.ready_emitted or not self.window.isVisible():
            return
        QtCore.QTimer.singleShot(0, self.emit_ready_if_visible)

    def emit_ready_if_visible(self):
        if self.ready_emitted or not self.window.isVisible():
            return
        if mark_startup_ready():
            self.ready_emitted = True
            self.window.removeEventFilter(self)
