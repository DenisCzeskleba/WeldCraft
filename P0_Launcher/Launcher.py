import sys
import os
import subprocess

from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QMainWindow, QWidget, QDesktopWidget, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
VENV_PYTHON = os.path.normpath(r"F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe")


class HoverButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hovered_text = ""  # Text to display in the QTextEdit on hover

    def enterEvent(self, event):
        """Handles mouse entering the button."""
        main_window = self.window()  # Use self.window() to get the main window (Launcher)
        if hasattr(main_window, 'update_info_text'):
            main_window.update_info_text(self.hovered_text)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handles mouse leaving the button."""
        main_window = self.window()  # Use self.window() to get the main window (Launcher)
        if hasattr(main_window, 'update_info_text'):
            # Reset to the default text when the mouse leaves
            main_window.update_info_text(main_window.default_text)
        super().leaveEvent(event)


# SplashScreen class
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setFixedSize(400, 400)
        self.center()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Load splash image
        splash_image_path = os.path.join(REPO_ROOT, "Resources", "Images", "WeldCraft.png")
        splash_image = QPixmap(splash_image_path)

        label = QLabel()
        label.setPixmap(splash_image)
        layout.addWidget(label)
        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


# Launcher class
class Launcher(QMainWindow):
    splash_closed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        # Load UI
        uic.loadUi(os.path.join(BASE_DIR, "Launcher_UI_Window.ui"), self)

        # General Window Settings
        self.setWindowTitle("WeldCraft Launcher")
        self.setWindowFlag(Qt.Window)
        self.setFixedSize(1000, 330)
        self.center()

        # Set taskbar icon
        ico_path = os.path.join(REPO_ROOT, "Resources", "Images", "WeldCraft.ico")
        app.setWindowIcon(QtGui.QIcon(ico_path))

        # Default text for the QTextEdit
        self.default_text = (
            "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore "
            "et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. "
            "Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit "
            "amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam "
            "erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, "
            "no sea takimata sanctus est Lorem ipsum dolor sit amet."
        )

        # TextEdit for launcher info
        self.launcher_info_text = self.findChild(QTextEdit, 'textEdit_info_text')
        self.launcher_info_text.setPlainText(self.default_text)  # Set default text on load

        # Replace QPushButton with HoverButton instances
        self.replace_buttons_with_hover_buttons()

        # Connect buttons and set hover texts
        self.connect_buttons()
        self.set_hover_text()

        # Set program directories
        self.path_diffusion_overview = os.path.join(REPO_ROOT, "P1_Diffusion_Overview", "diffusion_overview.py")
        # self.path_heat_map = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_hydrogen_during_welding = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_brownian_motion = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        self.path_iso3690 = os.path.join(REPO_ROOT, "P5_BoP_ISO3690", "bead_on_plate.py")
        # self.path_1d_diffusion_1111_rule = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_data_visualization = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")

        # Store the active process handles
        self.active_processes = {}

        # Recompile that one UI that you didn't want to completely rewrite
        self.compile_overview_ui()

        # Connect signal to close splash screen
        self.splash_closed.connect(self.close_splash)

    def compile_overview_ui(self):
        # Recompiles that one UI
        ui_file = os.path.join(REPO_ROOT, "P1_Diffusion_Overview", "settings", "ui_diffusion_overview.ui")
        py_file = os.path.join(REPO_ROOT, "P1_Diffusion_Overview", "settings", "ui_diffusion_overview.py")

        # Check if the .py file is outdated compared to the .ui file
        if not os.path.exists(py_file) or os.path.getmtime(ui_file) > os.path.getmtime(py_file):
            subprocess.run([VENV_PYTHON, '-m', 'PyQt5.uic.pyuic', '-x', ui_file, '-o', py_file], check=True)

    def replace_buttons_with_hover_buttons(self):
        """Replaces QPushButton with HoverButton after loading the UI."""
        button_names = [
            'pushButton_diffusion_overview', 'pushButton_heat_map', 'pushButton_hydrogen_during_welding',
            'pushButton_brownian_motion', 'pushButton_iso3690', 'pushButton_1d_diffusion_1111_rule',
            'pushButton_data_visualization'
        ]

        for btn_name in button_names:
            btn = self.findChild(QPushButton, btn_name)
            if btn:
                hover_btn = HoverButton(btn.text(), self)  # Create HoverButton with the same text
                hover_btn.setObjectName(btn.objectName())  # Keep the same object name
                hover_btn.setGeometry(btn.geometry())  # Copy the geometry to match original button position and size
                hover_btn.setStyleSheet(btn.styleSheet())  # Copy the stylesheet (if any)
                hover_btn.clicked.connect(btn.clicked)  # Transfer the clicked signal
                btn.parentWidget().layout().replaceWidget(btn, hover_btn)  # Replace the original button with hover button
                btn.deleteLater()  # Delete the old QPushButton

    def connect_buttons(self):
        """Connect each button to its respective function."""
        buttons = {
            'pushButton_diffusion_overview': self.start_diffusion_overview,
            'pushButton_heat_map': self.start_heat_map,
            'pushButton_hydrogen_during_welding': self.start_hydrogen_during_welding,
            'pushButton_brownian_motion': self.start_brownian_motion,
            'pushButton_iso3690': self.start_iso3690,
            'pushButton_1d_diffusion_1111_rule': self.start_diffusion_1111_rule,
            'pushButton_data_visualization': self.start_data_visualization
        }

        for btn_name, func in buttons.items():
            btn = self.findChild(HoverButton, btn_name)
            if btn:
                btn.clicked.connect(func)

    def set_hover_text(self):
        """Assign hover text to each button."""
        hover_texts = {
            'pushButton_diffusion_overview': "Diffusion Overview: Write this later!",
            'pushButton_heat_map': "Simulate Heat Map: Write this later!",
            'pushButton_hydrogen_during_welding': "Hydrogen During Welding: Combined heat and hydrogen diffusion simulation. Write this later!",
            'pushButton_brownian_motion': "Simulate Brownian Motion: Visualize random motion of particles. Write this later!",
            'pushButton_iso3690': "Bead on plate (ISO 3690): Write this later!",
            'pushButton_1d_diffusion_1111_rule': "1D Diffusion (Rule 1111): Explore one-dimensional diffusion. Write this later!",
            'pushButton_data_visualization': "Visualize Data: Might not be necessary, people can make their own. Write this later!"
        }

        for btn_name, hover_text in hover_texts.items():
            btn = self.findChild(HoverButton, btn_name)
            if btn:
                btn.hovered_text = hover_text

    def update_info_text(self, text):
        """Updates the QTextEdit based on the hovered button."""
        self.launcher_info_text.setPlainText(text)

    # Program starter methods with process tracking
    def start_program(self, button_name, program_args):
        """General function to start a program."""
        btn = self.findChild(HoverButton, button_name)
        if btn and button_name not in self.active_processes:
            # Disable the button to prevent launching multiple instances
            btn.setEnabled(False)

            # Start the process and track it
            command = [VENV_PYTHON] + program_args
            working_directory = os.path.dirname(os.path.abspath(program_args[0])) if program_args else BASE_DIR
            process = subprocess.Popen(command, cwd=working_directory)
            self.active_processes[button_name] = process

            # Monitor the process to reactivate the button once it closes
            QTimer.singleShot(500, lambda: self.monitor_program(button_name, process))

    def monitor_program(self, button_name, process):
        """Check if the program is still running and re-enable the button when it finishes."""
        if process.poll() is None:
            # If the process is still running, check again after some time
            QTimer.singleShot(1000, lambda: self.monitor_program(button_name, process))
        else:
            # Process finished, re-enable the button
            btn = self.findChild(HoverButton, button_name)
            if btn:
                btn.setEnabled(True)
            # Remove the process from active list
            if button_name in self.active_processes:
                del self.active_processes[button_name]
            # Bring launcher to the front
            self.raise_()
            self.activateWindow()

    # Program starter methods
    def start_diffusion_overview(self):
        self.start_program('pushButton_diffusion_overview', [self.path_diffusion_overview])

    def start_heat_map(self):
        # self.start_program('pushButton_heat_map', ['python', 'path_to_heat_map.py'])
        print("start_heat_map")

    def start_hydrogen_during_welding(self):
        # self.start_program('pushButton_hydrogen_during_welding', ['python', 'path_to_hydrogen_during_welding.py'])
        print("start_hydrogen_during_welding")

    def start_brownian_motion(self):
        # self.start_program('pushButton_brownian_motion', ['python', 'path_to_brownian_motion.py'])
        print("start_brownian_motion")

    def start_iso3690(self):
        # self.start_program('pushButton_iso3690', ['python', self.path_iso3690])
        print("start_iso3690")

    def start_diffusion_1111_rule(self):
        # self.start_program('pushButton_1d_diffusion_1111_rule', ['python', 'path_to_diffusion_1111_rule.py'])
        print("start_diffusion_1111_rule")

    def start_data_visualization(self):
        # self.start_program('pushButton_data_visualization', ['python', 'path_to_data_visualization.py'])
        print("start_data_visualization")

    def close_splash(self):
        splash.close()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    splash = SplashScreen()
    splash.show()

    launcher = Launcher()
    launcher.show()

    # Close splash after 8 seconds
    QTimer.singleShot(6000, launcher.splash_closed.emit)

    sys.exit(app.exec_())
