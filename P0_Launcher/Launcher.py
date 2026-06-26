import sys
import os
import subprocess

from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QMainWindow, QWidget, QDesktopWidget, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))


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
            "Hover a module button to see a short description. Some modules are still code-only and not launchable "
            "from the GUI yet."
        )
        self.p2_hydrogen_info = (
            "P2 Hydrogen Diffusion During Welding is code-only for now. Use the scripts directly; a GUI is planned "
            "later, but not confirmed yet."
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
        self.path_simulate_hydrogen_diffusion = os.path.join(
            REPO_ROOT,
            "P1_Simulate_Hydrogen_Diffusion",
            "simulate_hydrogen_diffusion.py",
        )
        # self.path_heat_map = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_hydrogen_during_welding = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_brownian_motion = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_1d_diffusion_1111_rule = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")
        # self.path_data_visualization = os.path.join(current_dir, "..", "Resources", "Images", "WeldCraft.ico")

        # Store the active process handles
        self.active_processes = {}

        # Recompile that one UI that you didn't want to completely rewrite
        self.compile_p1_ui()

        # Connect signal to close splash screen
        self.splash_closed.connect(self.close_splash)

    def compile_p1_ui(self):
        # Recompile the P1 UI if the source changed.
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

        # Check if the .py file is outdated compared to the .ui file
        if not os.path.exists(py_file) or os.path.getmtime(ui_file) > os.path.getmtime(py_file):
            subprocess.run([sys.executable, '-m', 'PyQt5.uic.pyuic', '-x', ui_file, '-o', py_file], check=True)

    def replace_buttons_with_hover_buttons(self):
        """Replaces QPushButton with HoverButton after loading the UI."""
        button_names = [
            'pushButton_simulate_hydrogen_diffusion', 'pushButton_heat_map', 'pushButton_hydrogen_during_welding',
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
            'pushButton_simulate_hydrogen_diffusion': self.start_simulate_hydrogen_diffusion,
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
            'pushButton_simulate_hydrogen_diffusion': (
                "WeldCraft - Simulate Hydrogen Diffusion: visual 1D/2D hydrogen diffusion and heat transport "
                "simulation with numerical accuracy and animation tools."
            ),
            'pushButton_hydrogen_during_welding': self.p2_hydrogen_info,
            'pushButton_heat_map': "Placeholder 3 (maybe Heat Map Simulation?): Write this later!",
            'pushButton_brownian_motion': "Placeholder 4 (maybe Diffusion Visualization / Brownian Motion?): Write this later!",
            'pushButton_iso3690': "Placeholder 5 (maybe Bead-On-Plate-Weld / ISO3690?): Write this later!",
            'pushButton_1d_diffusion_1111_rule': "Placeholder 6 (maybe 1D Diffusion / 1111 Rule?): Write this later!",
            'pushButton_data_visualization': "Placeholder 7 (maybe Data Visualization?): Write this later!"
        }

        for btn_name, hover_text in hover_texts.items():
            btn = self.findChild(HoverButton, btn_name)
            if btn:
                btn.hovered_text = hover_text
                btn.setToolTip(hover_text)

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
            command = [sys.executable] + program_args
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
    def start_simulate_hydrogen_diffusion(self):
        self.start_program('pushButton_simulate_hydrogen_diffusion', [self.path_simulate_hydrogen_diffusion])

    def start_heat_map(self):
        print("Placeholder 3 (maybe Heat Map Simulation?): Write this later!")

    def start_hydrogen_during_welding(self):
        self.update_info_text(self.p2_hydrogen_info)
        print(self.p2_hydrogen_info)

    def start_brownian_motion(self):
        print("Placeholder 4 (maybe Diffusion Visualization / Brownian Motion?): Write this later!")

    def start_iso3690(self):
        print("Placeholder 5 (maybe Bead-On-Plate-Weld / ISO3690?): Write this later!")

    def start_diffusion_1111_rule(self):
        print("Placeholder 6 (maybe 1D Diffusion / 1111 Rule?): Write this later!")

    def start_data_visualization(self):
        print("Placeholder 7 (maybe Data Visualization?): Write this later!")

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
