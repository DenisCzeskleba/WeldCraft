import ctypes
import os
import subprocess
import sys
import tempfile
import time

from ctypes import wintypes
from functools import partial
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QFontDatabase, QPixmap
from PyQt5.QtWidgets import QDesktopWidget, QMainWindow, QPushButton, QTextEdit


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
RESOURCES_DIR = os.path.join(REPO_ROOT, "Resources")
if RESOURCES_DIR not in sys.path:
    sys.path.insert(0, RESOURCES_DIR)

from Common.launch_ready import READY_FILE_ENV_VAR


SLOT_IDS = [f"pushButton_slot_{index}" for index in range(1, 8)]
MAIN_PAGE = "main"
ANALYSIS_PAGE = "analysis"
SPLASH_IMAGE_PATH = os.path.join(REPO_ROOT, "Resources", "Images", "WeldCraft Long.png")
FALLBACK_SPLASH_IMAGE_PATH = os.path.join(REPO_ROOT, "Resources", "Images", "WeldCraft.png")
SPLASH_MAX_WIDTH = 460
WINDOWS = sys.platform.startswith("win")
WINDOW_ENUM_OWNER = 4
MAX_LAUNCH_SPLASH_SECONDS = 15.0
SPLASH_TITLE_BASE = "Firing Up The Forges"
SPLASH_TITLE_DOTS = [".", "..", "..."]
SPLASH_STATUS_LAUNCHER = "WeldCraft Launcher"

if WINDOWS:
    USER32 = ctypes.windll.user32
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class HoverButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hovered_text = ""

    def enterEvent(self, event):
        main_window = self.window()
        if hasattr(main_window, "update_info_text"):
            main_window.update_info_text(self.hovered_text)
        super().enterEvent(event)

    def leaveEvent(self, event):
        main_window = self.window()
        if hasattr(main_window, "update_info_text"):
            main_window.update_info_text(main_window.default_text)
        super().leaveEvent(event)


def choose_font(preferred_families, point_size, *, weight=QFont.Normal, letter_spacing=0.0):
    font_database = QFontDatabase()
    for family in preferred_families:
        if family in font_database.families():
            font = QFont(family, point_size, weight)
            font.setLetterSpacing(QFont.AbsoluteSpacing, letter_spacing)
            return font

    font = QFont()
    font.setPointSize(point_size)
    font.setWeight(weight)
    font.setLetterSpacing(QFont.AbsoluteSpacing, letter_spacing)
    return font


class WeldCraftSplashScreen(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        splash_path = SPLASH_IMAGE_PATH if os.path.exists(SPLASH_IMAGE_PATH) else FALLBACK_SPLASH_IMAGE_PATH
        base_pixmap = QPixmap(splash_path)
        if not base_pixmap.isNull() and base_pixmap.width() > SPLASH_MAX_WIDTH:
            base_pixmap = base_pixmap.scaledToWidth(SPLASH_MAX_WIDTH, Qt.SmoothTransformation)
        self.base_pixmap = base_pixmap
        self.current_pixmap = QPixmap(self.base_pixmap)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFixedSize(self.base_pixmap.size())

        self.title_font = (
            choose_font(["Palatino Linotype", "Georgia", "Times New Roman"], 15, weight=QFont.Bold, letter_spacing=1.8)
        )
        self.status_font = (
            choose_font(["Segoe UI Semibold", "Segoe UI", "Arial"], 9, weight=QFont.DemiBold, letter_spacing=0.8)
        )

        self.dot_index = 0
        self.dot_timer = QTimer(self)
        self.dot_timer.setInterval(600)
        self.dot_timer.timeout.connect(self.advance_title_dots)
        self.dot_timer.start()

        self.status_text = SPLASH_STATUS_LAUNCHER
        self.update_overlay_text()

    def center_on_screen(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return

        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def advance_title_dots(self):
        self.dot_index = (self.dot_index + 1) % len(SPLASH_TITLE_DOTS)
        self.update_overlay_text()

    def update_overlay_text(self):
        self.title_text = f"{SPLASH_TITLE_BASE}{SPLASH_TITLE_DOTS[self.dot_index]}"
        self.current_pixmap = self.compose_pixmap()
        self.update()

    def set_status_text(self, status_text):
        self.status_text = status_text
        self.update_overlay_text()

    def compose_pixmap(self):
        composed_pixmap = QPixmap(self.base_pixmap)
        painter = QtGui.QPainter(composed_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        width = composed_pixmap.width()
        height = composed_pixmap.height()
        title_y = height - 60
        status_y = height - 30

        title_metrics = QtGui.QFontMetrics(self.title_font)
        status_metrics = QtGui.QFontMetrics(self.status_font)
        base_text_width = title_metrics.horizontalAdvance(SPLASH_TITLE_BASE)
        dot_text = SPLASH_TITLE_DOTS[self.dot_index]
        title_left = int((width - base_text_width) / 2)
        title_base_right = title_left + base_text_width

        painter.setFont(self.title_font)
        painter.setPen(QtGui.QColor(0, 0, 0, 210))
        painter.drawText(title_left, title_y + 2, SPLASH_TITLE_BASE)
        painter.drawText(title_base_right + 8, title_y + 2, dot_text)
        painter.setPen(QtGui.QColor("#E7EDF6"))
        painter.drawText(title_left, title_y, SPLASH_TITLE_BASE)
        painter.drawText(title_base_right + 8, title_y, dot_text)

        painter.setFont(self.status_font)
        status_rect = QtCore.QRect(36, status_y - status_metrics.ascent(), width - 72, status_metrics.height() + 8)
        painter.setPen(QtGui.QColor(0, 0, 0, 180))
        painter.drawText(status_rect.translated(0, 1), Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextSingleLine, self.status_text)
        painter.setPen(QtGui.QColor("#94A0B2"))
        painter.drawText(status_rect, Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextSingleLine, self.status_text)
        painter.end()
        return composed_pixmap

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self.current_pixmap)
        painter.end()
        super().paintEvent(event)

    def show_splash(self, status_text=None):
        if status_text is not None:
            self.set_status_text(status_text)
        if not self.isVisible():
            self.center_on_screen()
            self.show()
        else:
            self.raise_()
        QtWidgets.QApplication.processEvents()


class Launcher(QMainWindow):
    def __init__(self, splash_screen):
        super().__init__()

        self.splash_screen = splash_screen
        self.startup_splash_active = True

        uic.loadUi(os.path.join(BASE_DIR, "Launcher_UI_Window.ui"), self)

        self.setWindowTitle("WeldCraft Launcher")
        self.setWindowFlag(Qt.Window)
        self.setFixedSize(1000, 330)
        self.center()

        ico_path = os.path.join(REPO_ROOT, "Resources", "Images", "WeldCraft.ico")
        qt_app = QtWidgets.QApplication.instance()
        self.setWindowIcon(QtGui.QIcon(ico_path))
        if qt_app is not None:
            qt_app.setWindowIcon(QtGui.QIcon(ico_path))

        self.main_default_text = (
            "Hover a module button to see a short description. Some modules are still code-only and not launchable "
            "from the GUI yet."
        )
        self.analysis_default_text = "Choose an analysis tool placeholder or return to the main launcher page."
        self.default_text = self.main_default_text
        self.p2_hydrogen_info = (
            "P2 Hydrogen Diffusion During Welding is code-only for now. Use the scripts directly; a GUI is planned "
            "later, but not confirmed yet."
        )

        self.launcher_info_text = self.findChild(QTextEdit, "textEdit_info_text")
        self.launcher_info_text.setPlainText(self.default_text)

        self.path_simulate_hydrogen_diffusion = os.path.join(
            REPO_ROOT,
            "P1_Simulate_Hydrogen_Diffusion",
            "simulate_hydrogen_diffusion.py",
        )

        self.active_processes = {}
        self.pending_launches = {}
        self.slot_buttons = {}
        self.slot_actions = {}
        self.auto_minimized_for_launch = False
        self.page_default_texts = {
            MAIN_PAGE: self.main_default_text,
            ANALYSIS_PAGE: self.analysis_default_text,
        }
        self.current_page = MAIN_PAGE

        self.compile_p1_ui()
        self.replace_buttons_with_hover_buttons()
        self.cache_slot_buttons()
        self.connect_slot_buttons()
        self.page_definitions = self.build_page_definitions()
        self.show_page(MAIN_PAGE)

    def compile_p1_ui(self):
        ui_file = os.path.join(
            REPO_ROOT,
            "P1_Simulate_Hydrogen_Diffusion",
            "App_Files",
            "ui_simulate_hydrogen_diffusion.ui",
        )
        py_file = os.path.join(
            REPO_ROOT,
            "P1_Simulate_Hydrogen_Diffusion",
            "App_Files",
            "ui_simulate_hydrogen_diffusion.py",
        )

        if not os.path.exists(py_file) or os.path.getmtime(ui_file) > os.path.getmtime(py_file):
            subprocess.run([sys.executable, "-m", "PyQt5.uic.pyuic", "-x", ui_file, "-o", py_file], check=True)

    def replace_buttons_with_hover_buttons(self):
        for button_name in SLOT_IDS:
            button = self.findChild(QPushButton, button_name)
            if not button:
                continue

            hover_button = HoverButton(button.text(), self)
            hover_button.setObjectName(button.objectName())
            hover_button.setSizePolicy(button.sizePolicy())
            hover_button.setMinimumSize(button.minimumSize())
            hover_button.setMaximumSize(button.maximumSize())
            hover_button.setFont(button.font())
            hover_button.setStyleSheet(button.styleSheet())
            hover_button.setToolTip(button.toolTip())
            hover_button.setFocusPolicy(button.focusPolicy())

            parent_layout = button.parentWidget().layout()
            parent_layout.replaceWidget(button, hover_button)
            button.deleteLater()

    def cache_slot_buttons(self):
        for slot_id in SLOT_IDS:
            button = self.findChild(HoverButton, slot_id)
            if button:
                self.slot_buttons[slot_id] = button

    def connect_slot_buttons(self):
        for slot_id, button in self.slot_buttons.items():
            button.clicked.connect(partial(self.handle_slot_clicked, slot_id))

    def build_page_entry(self, text, hover_text, action, enabled=True, visible=True):
        return {
            "text": text,
            "hover_text": hover_text,
            "action": action,
            "enabled": enabled,
            "visible": visible,
        }

    def build_page_definitions(self):
        return {
            MAIN_PAGE: [
                self.build_page_entry(
                    "Simulate Hydrogen Diffusion",
                    (
                        "WeldCraft - Simulate Hydrogen Diffusion: visual 1D/2D hydrogen diffusion and heat transport "
                        "simulation with numerical accuracy and animation tools."
                    ),
                    self.start_simulate_hydrogen_diffusion,
                ),
                self.build_page_entry(
                    "Hydrogen Diffusion During Welding",
                    self.p2_hydrogen_info,
                    self.start_hydrogen_during_welding,
                ),
                self.build_page_entry(
                    "Placeholder 1",
                    "Placeholder 1 (maybe Heat Map Simulation?): Write this later!",
                    partial(
                        self.show_placeholder_message,
                        "Placeholder 1 (maybe Heat Map Simulation?): Write this later!",
                    ),
                ),
                self.build_page_entry(
                    "Placeholder 2",
                    "Placeholder 2 (maybe Diffusion Visualization / Brownian Motion?): Write this later!",
                    partial(
                        self.show_placeholder_message,
                        "Placeholder 2 (maybe Diffusion Visualization / Brownian Motion?): Write this later!",
                    ),
                ),
                self.build_page_entry(
                    "Placeholder 3",
                    "Placeholder 3 (maybe Bead-On-Plate-Weld / ISO3690?): Write this later!",
                    partial(
                        self.show_placeholder_message,
                        "Placeholder 3 (maybe Bead-On-Plate-Weld / ISO3690?): Write this later!",
                    ),
                ),
                self.build_page_entry(
                    "Placeholder 4",
                    "Placeholder 4 (maybe 1D Diffusion / 1111 Rule?): Write this later!",
                    partial(
                        self.show_placeholder_message,
                        "Placeholder 4 (maybe 1D Diffusion / 1111 Rule?): Write this later!",
                    ),
                ),
                self.build_page_entry(
                    "Analysis Tools",
                    "Open the analysis tools launcher page.",
                    partial(self.show_page, ANALYSIS_PAGE),
                ),
            ],
            ANALYSIS_PAGE: [
                self.build_page_entry(
                    "Placeholder 1",
                    "Analysis Tools - Placeholder 1: Write this later!",
                    partial(self.show_placeholder_message, "Analysis Tools - Placeholder 1: Write this later!"),
                ),
                self.build_page_entry(
                    "Placeholder 2",
                    "Analysis Tools - Placeholder 2: Write this later!",
                    partial(self.show_placeholder_message, "Analysis Tools - Placeholder 2: Write this later!"),
                ),
                self.build_page_entry(
                    "Placeholder 3",
                    "Analysis Tools - Placeholder 3: Write this later!",
                    partial(self.show_placeholder_message, "Analysis Tools - Placeholder 3: Write this later!"),
                ),
                self.build_page_entry(
                    "Placeholder 4",
                    "Analysis Tools - Placeholder 4: Write this later!",
                    partial(self.show_placeholder_message, "Analysis Tools - Placeholder 4: Write this later!"),
                ),
                self.build_page_entry(
                    "Placeholder 5",
                    "Analysis Tools - Placeholder 5: Write this later!",
                    partial(self.show_placeholder_message, "Analysis Tools - Placeholder 5: Write this later!"),
                ),
                self.build_page_entry(
                    "Placeholder 6",
                    "Analysis Tools - Placeholder 6: Write this later!",
                    partial(self.show_placeholder_message, "Analysis Tools - Placeholder 6: Write this later!"),
                ),
                self.build_page_entry(
                    "Back",
                    "Return to the main launcher page.",
                    partial(self.show_page, MAIN_PAGE),
                ),
            ],
        }

    def handle_slot_clicked(self, slot_id):
        action = self.slot_actions.get(slot_id)
        if action:
            action()

    def show_page(self, page_name):
        self.current_page = page_name
        self.default_text = self.page_default_texts.get(page_name, self.main_default_text)
        self.render_current_page(reset_info_text=True)

    def render_current_page(self, reset_info_text=False):
        page_entries = self.page_definitions[self.current_page]

        for slot_index, slot_id in enumerate(SLOT_IDS):
            button = self.slot_buttons.get(slot_id)
            if not button:
                continue

            entry = page_entries[slot_index]
            button.setText(entry["text"])
            button.hovered_text = entry["hover_text"]
            button.setToolTip(entry["hover_text"])
            button.setVisible(entry.get("visible", True))
            button.setEnabled(entry.get("enabled", True) and slot_id not in self.active_processes)
            self.slot_actions[slot_id] = entry["action"] if entry.get("enabled", True) else None

        if reset_info_text:
            self.update_info_text(self.default_text)

    def update_info_text(self, text):
        self.launcher_info_text.setPlainText(text)

    def show_placeholder_message(self, message):
        self.update_info_text(message)
        print(message)

    def create_ready_file_path(self, slot_id):
        ready_file_name = f"weldcraft_ready_{os.getpid()}_{slot_id}_{time.monotonic_ns()}.flag"
        ready_file_path = Path(tempfile.gettempdir()) / ready_file_name
        if ready_file_path.exists():
            ready_file_path.unlink()
        return str(ready_file_path)

    def refresh_splash_visibility(self):
        if self.startup_splash_active or self.pending_launches:
            if self.pending_launches:
                pending_labels = [launch_info["button_text"] for launch_info in self.pending_launches.values()]
                if len(pending_labels) == 1:
                    status_text = pending_labels[0]
                else:
                    status_text = f"{len(pending_labels)} Modules"
            else:
                status_text = SPLASH_STATUS_LAUNCHER

            self.splash_screen.show_splash(status_text)
        else:
            self.splash_screen.hide()

    def finish_startup_splash(self):
        self.startup_splash_active = False
        self.refresh_splash_visibility()

    def minimize_for_child_launch(self):
        if self.isMinimized():
            return

        self.auto_minimized_for_launch = True
        self.showMinimized()

    def restore_after_child_exit(self):
        should_restore = self.auto_minimized_for_launch and not self.active_processes
        self.auto_minimized_for_launch = False if not self.active_processes else self.auto_minimized_for_launch

        if should_restore:
            self.showNormal()
            self.raise_()
            self.activateWindow()

    def start_program(self, slot_id, program_args):
        button = self.slot_buttons.get(slot_id)
        if not button or slot_id in self.active_processes:
            return

        button.setEnabled(False)

        ready_file_path = self.create_ready_file_path(slot_id)
        self.pending_launches[slot_id] = {
            "button_text": button.text(),
            "ready_file_path": ready_file_path,
            "started_at": time.monotonic(),
        }
        self.refresh_splash_visibility()

        environment = os.environ.copy()
        environment[READY_FILE_ENV_VAR] = ready_file_path

        try:
            command = [sys.executable] + program_args
            working_directory = os.path.dirname(os.path.abspath(program_args[0])) if program_args else BASE_DIR
            process = subprocess.Popen(command, cwd=working_directory, env=environment)
        except Exception as exc:
            self.finish_launch_feedback(slot_id)
            button.setEnabled(True)
            self.update_info_text(f"Failed to start {button.text()}: {exc}")
            return

        self.active_processes[slot_id] = process
        self.pending_launches[slot_id]["process"] = process
        self.minimize_for_child_launch()
        QTimer.singleShot(150, partial(self.monitor_program, slot_id, process))

    def finish_launch_feedback(self, slot_id):
        launch_info = self.pending_launches.pop(slot_id, None)
        if launch_info is None:
            return

        ready_file_path = launch_info.get("ready_file_path")
        if ready_file_path:
            ready_file = Path(ready_file_path)
            if ready_file.exists():
                ready_file.unlink()

        self.refresh_splash_visibility()

    def process_has_visible_window(self, pid):
        if not WINDOWS:
            return False

        visible_window_found = False

        @EnumWindowsProc
        def enum_windows(hwnd, l_param):
            nonlocal visible_window_found
            if not USER32.IsWindowVisible(hwnd):
                return True
            if USER32.GetWindow(hwnd, WINDOW_ENUM_OWNER):
                return True

            class_name = ctypes.create_unicode_buffer(256)
            USER32.GetClassNameW(hwnd, class_name, len(class_name))
            if class_name.value == "ConsoleWindowClass":
                return True

            process_id = wintypes.DWORD()
            USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            if process_id.value != pid:
                return True

            visible_window_found = True
            return False

        USER32.EnumWindows(enum_windows, 0)
        return visible_window_found

    def update_launch_feedback(self, slot_id, process):
        launch_info = self.pending_launches.get(slot_id)
        if launch_info is None:
            return

        ready_file_path = launch_info["ready_file_path"]
        if os.path.exists(ready_file_path):
            self.finish_launch_feedback(slot_id)
            return

        if self.process_has_visible_window(process.pid):
            self.finish_launch_feedback(slot_id)
            return

        elapsed = time.monotonic() - launch_info["started_at"]
        if elapsed >= MAX_LAUNCH_SPLASH_SECONDS:
            self.finish_launch_feedback(slot_id)

    def monitor_program(self, slot_id, process):
        if process.poll() is None:
            self.update_launch_feedback(slot_id, process)
            QTimer.singleShot(250, partial(self.monitor_program, slot_id, process))
            return

        self.finish_launch_feedback(slot_id)

        if slot_id in self.active_processes:
            del self.active_processes[slot_id]

        self.render_current_page(reset_info_text=False)
        self.restore_after_child_exit()

    def start_simulate_hydrogen_diffusion(self):
        self.start_program("pushButton_slot_1", [self.path_simulate_hydrogen_diffusion])

    def start_hydrogen_during_welding(self):
        self.update_info_text(self.p2_hydrogen_info)
        print(self.p2_hydrogen_info)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    splash_screen = WeldCraftSplashScreen()
    splash_screen.show_splash()

    try:
        launcher = Launcher(splash_screen)
    except Exception:
        splash_screen.hide()
        raise

    launcher.show()
    QTimer.singleShot(0, launcher.finish_startup_splash)

    sys.exit(app.exec_())
