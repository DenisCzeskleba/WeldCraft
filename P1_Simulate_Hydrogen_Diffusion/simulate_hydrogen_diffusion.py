import json
import os
import re
import shutil
import sys
import subprocess
import time
import traceback
from datetime import datetime

import h5py
import matplotlib.colors as mcolors
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QObject, QEvent, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QTableWidgetItem, QHBoxLayout, QMessageBox, QLabel, QSizePolicy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from App_Files.calculate_stuff_for_gui import analytical_solution
from App_Files.calculate_stuff_for_gui import calculate_1d, simulate_2d, create_hydrogen_animation, calculate_boundary_flux
from App_Files.custom_widgets import ClickableLabel, CustomLabel
from App_Files import ui_simulate_hydrogen_diffusion

APP_NAME = "WeldCraft - Simulate Hydrogen Diffusion"
IS_FROZEN = getattr(sys, "frozen", False)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = getattr(sys, "_MEIPASS", MODULE_DIR)
REPO_ROOT = os.path.normpath(os.path.join(MODULE_DIR, ".."))
APP_FILES_DIR = os.path.join(MODULE_DIR, "App_Files")
SOURCE_IMAGES_DIR = os.path.join(REPO_ROOT, "Resources", "Images")
RUNTIME_ROOT = os.path.dirname(sys.executable) if IS_FROZEN else MODULE_DIR
RUNTIME_SETTINGS_DIR = os.path.join(RUNTIME_ROOT, "settings")
SOURCE_SETTINGS_PATH = os.path.join(APP_FILES_DIR, "settings.json")
RESULTS_DIR = os.path.join(RUNTIME_ROOT, "Results")


def resolve_image_path(file_name):
    bundled_path = os.path.join(BUNDLE_ROOT, "Resources", "Images", file_name)
    if IS_FROZEN and os.path.exists(bundled_path):
        return bundled_path
    return os.path.join(SOURCE_IMAGES_DIR, file_name)


def ensure_runtime_directories():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if IS_FROZEN:
        os.makedirs(RUNTIME_SETTINGS_DIR, exist_ok=True)


def ensure_frozen_settings_seeded():
    ensure_runtime_directories()
    settings_path = os.path.join(RUNTIME_SETTINGS_DIR, "settings.json")
    if not os.path.exists(settings_path):
        template_path = os.path.join(BUNDLE_ROOT, "settings", "default_settings.json")
        if os.path.exists(template_path):
            shutil.copyfile(template_path, settings_path)
        else:
            with open(settings_path, "w", encoding="utf-8") as handle:
                json.dump({}, handle, indent=4)
    return settings_path


def resolve_simulation_file_path(file_name):
    normalized_path = os.path.normpath(file_name)
    if os.path.isabs(normalized_path):
        return normalized_path

    legacy_results_prefixes = [
        os.path.normpath("Results"),
        os.path.normpath(os.path.join(".", "Results")),
        os.path.normpath(os.path.join("..", "Results")),
    ]
    for prefix in legacy_results_prefixes:
        if normalized_path == prefix or normalized_path.startswith(prefix + os.sep):
            relative_suffix = os.path.relpath(normalized_path, prefix)
            if relative_suffix == ".":
                return RESULTS_DIR
            return os.path.join(RESULTS_DIR, relative_suffix)

    return os.path.join(RUNTIME_ROOT, normalized_path)


def resolve_animation_output_path(simulation_file_path):
    root, extension = os.path.splitext(simulation_file_path)
    if extension.lower() == ".h5":
        return root + ".mp4"
    return simulation_file_path + ".mp4"


class ErrorHandler(QObject):
    show_error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.show_error_signal.connect(self.show_error_dialog)

    def show_error_dialog(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText("An unexpected error occurred!")
        error_dialog.setInformativeText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec_()


error_handler = ErrorHandler()


def global_exception_handler(exctype, value, tb):
    error_message = ''.join(traceback.format_exception(exctype, value, tb))
    error_handler.show_error_signal.emit(error_message)


sys.excepthook = global_exception_handler


class PlotPointsValidator(QtGui.QValidator):
    def validate(self, text, pos):
        if text == "":
            return QtGui.QValidator.Intermediate, text, pos
        if text.isdigit() and int(text) > 0:
            return QtGui.QValidator.Acceptable, text, pos
        return QtGui.QValidator.Invalid, text, pos


def set_combo_box_margins(parent):
    for combo_box in parent.findChildren(QtWidgets.QGroupBox):
        layout = combo_box.layout()
        if layout:
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)


class EmittingStream(QObject):
    textWritten = pyqtSignal(str)  # Signal to communicate text written
    stepCompleted = pyqtSignal(str)  # Signal to indicate a step is completed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_step = None
        self.last_progress_message = ""

    def write(self, text):
        self.textWritten.emit(str(text))
        self.last_progress_message = str(text)  # Store the last progress message

    def flush(self):
        pass  # Not needed here but should be present

    def set_current_step(self, step):
        self.current_step = step

    def get_last_progress_message(self):
        return self.last_progress_message


class MplCanvas(FigureCanvas):
    mouse_moved = QtCore.pyqtSignal(float, float, float)  # x, y, value

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(object)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, length, alpha, u0, uL, dt, t, init_conc, dx_values):
        super().__init__()
        self.length = length
        self.alpha = alpha
        self.u0 = u0
        self.uL = uL
        self.dt = dt
        self.t = t
        self.init_conc = init_conc
        self.dx_values = dx_values
        self.stop_requested = False

    def run(self):
        try:
            result = calculate_1d(self.length, self.alpha, self.u0, self.uL, self.dt, self.t, self.init_conc, self.dx_values, self)
            self.result.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.stop_requested = True


class Worker2D(QtCore.QThread):
    progress = pyqtSignal(str)
    result = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, dx, dim_x, dim_y, sim_time, init_hydrogen_conc, border_hydrogen, border_bc, diffusion_coefficient, warm_up_time, k_value, save_every_x_s, file_name):
        super().__init__()
        self.dx = dx
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sim_time = sim_time
        self.init_hydrogen_conc = init_hydrogen_conc
        self.border_hydrogen = border_hydrogen
        self.border_bc = border_bc
        self.diffusion_coefficient = diffusion_coefficient
        self.save_every_x_s = save_every_x_s
        self.file_name = file_name
        self.stop_requested = False
        self.warm_up_time = warm_up_time
        self.k_value = k_value

    def run(self):
        try:
            result = simulate_2d(self.dx, self.dim_x, self.dim_y, self.sim_time, self.init_hydrogen_conc,
                                 self.border_hydrogen, self.border_bc, self.diffusion_coefficient, self.warm_up_time, self.k_value, self.save_every_x_s,
                                 self.file_name, self)
            self.result.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.stop_requested = True


class DataLoadingWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(object, object, object, object, object)

    def __init__(self, file_name):
        super().__init__()
        self.file_name = file_name

    def run(self):
        loaded_u_arrays = []
        loaded_t_values = []
        u_sums = []  # List to store the sums of each u array
        try:
            print("Currently Loading data:")  # keep track of progress in GUI
            start_time = time.time()
            QtCore.QCoreApplication.processEvents()  # Process the events to update the GUI
            with h5py.File(self.file_name, 'r') as hf:
                keys = sorted([key for key in hf.keys() if key.startswith('h_snapshot_') or key.startswith('t_snapshot_')])
                total_keys = len(keys)
                for i, key in enumerate(keys):
                    if key.startswith('h_snapshot_'):
                        u_array = hf[key][:]
                        loaded_u_arrays.append(u_array)
                        u_sums.append(np.sum(u_array))  # Sum the elements and store the result
                    elif key.startswith('t_snapshot_'):
                        loaded_t_values.append(hf[key][()])
                    if i % 50 == 0:
                        percentage = ((i + 1) / total_keys) * 100
                        self.progress.emit(f"Loading data: {percentage:.0f} %")
                        QtCore.QCoreApplication.processEvents()  # Process the events to update the GUI

            # Calculate the rate of change of u_sums
            u_sums = np.array(u_sums)
            loaded_t_values = np.array(loaded_t_values)
            rate_of_change = np.zeros_like(u_sums)
            if len(u_sums) > 1:
                dt = np.diff(loaded_t_values)
                rate_of_change[1:-1] = (u_sums[2:] - u_sums[:-2]) / (loaded_t_values[2:] - loaded_t_values[:-2])
                rate_of_change[0] = (u_sums[1] - u_sums[0]) / dt[0]
                rate_of_change[-1] = (u_sums[-1] - u_sums[-2]) / dt[-1]

                # Normalize the rate_of_change and scale it to percentage
                rate_of_change = np.abs(rate_of_change)
                rate_of_change = rate_of_change / np.max(rate_of_change) * 100

            # Calculate saturation
            max_value = np.max(u_sums)
            if max_value == 0:
                u_saturation = np.zeros_like(u_sums)
            else:
                u_saturation = (u_sums / max_value) * 100

            time_taken = time.time() - start_time
            minutes, seconds = divmod(time_taken, 60)
            formatted_time = f"{int(minutes):02d}:{int(seconds):02d}"
            it_per_sec = total_keys / time_taken
            print(f"STEP_COMPLETED: Loading data Completed: {formatted_time}, {it_per_sec:.2f}it/s")  # keep track of progress in GUI
            self.result.emit(loaded_u_arrays, loaded_t_values, u_sums, rate_of_change, u_saturation)
        except Exception as e:
            self.result.emit([], [], [], [], [])
            self.progress.emit(f"An error occurred in load_data: {e}")


class CreateAnimationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, loaded_u_arrays, loaded_t_values, output_file):
        super().__init__()
        self.loaded_u_arrays = loaded_u_arrays
        self.loaded_t_values = loaded_t_values
        self.output_file = output_file

    def run(self):
        create_hydrogen_animation(self.loaded_u_arrays, self.loaded_t_values, output_file=self.output_file)
        self.finished.emit()


class InvalidInputError(Exception):
    pass


class VerticalNavigationToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setOrientation(QtCore.Qt.Vertical)

    # disable showing mouse position in toolbar
    def set_message(self, s):
        pass


class OverlayWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel(self)
        self.label.setStyleSheet("background-color: rgba(255, 255, 255, 0.8);"
                                 "border: 1px solid black;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.hide()

    def update_position(self, x, y, text):
        self.label.setText(text)
        self.label.adjustSize()
        self.resize(self.label.size())
        self.move(int(x + 2), int(y - 17))  # Adjust offset here
        self.show()

    def hide_overlay(self):
        self.hide()


class MainWindow(QtWidgets.QMainWindow, ui_simulate_hydrogen_diffusion.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.apply_static_assets()
        self.setup_settings()
        self.load_settings()

        # Initialize worker attributes
        self.worker = None
        self.worker2d = None

        # Create the custom stream objects
        self.stdout_stream = EmittingStream()
        sys.stdout = self.stdout_stream
        sys.stderr = self.stdout_stream

        # Connect the custom stream signals to the text edit update slot
        self.stdout_stream.textWritten.connect(self.update_console_output)
        self.stdout_stream.stepCompleted.connect(self.handle_step_completed)

        # Initialize storage for completed steps
        self.completed_steps = []

        # Output stuffs
        self.kept_display_message = ""
        self.kept_display_message_with_current = ""
        self.last_update_msg = ""

        # Data storage
        self.loaded_u_arrays = []
        self.loaded_t_values = []
        self.loaded_u_sums = []
        self.u_saturation = []
        self.rate_of_change = []
        self.canvas_1_plot_line = []
        self.canvas_1_plot_marker = []
        self.canvas_2_plot_line = []
        self.canvas_2_plot_marker = []
        self.canvas_1_data = []
        self.canvas_2_data = []
        self.current_frame = 0
        self.time_unit = "[s]"
        self.last_update_time = 0
        self.update_interval = 1 / 10  # 5 frames per second
        self.force_update = False  # Reset the forced update flag
        self.current_alpha_option = self.horizontalSlider_fixed_variable_alpha.value()

        # Create QLabel for the tooltip
        self.tooltip_label = QLabel(self)
        self.tooltip_label.setStyleSheet("""
            QLabel {
                background-color: yellow;
                color: black;
                border: 1px solid black;
                padding: 2px;
                border-radius: 3px;
            }
        """)
        self.tooltip_label.hide()

        # Initialize sliders
        self.init_sliders()

        # Install the event filter to update toolip dynamically
        self.info_label.installEventFilter(self)

        # ---- Add the View in Tab1 (Mesh) ----
        # Initialize Matplotlib canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.graphicsView_tab1.setLayout(layout)

        # Add the navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # ---- Add tab5 (only analytical) ----
        # Initialize Matplotlib canvas
        self.canvas5 = MplCanvas(self, width=5, height=4, dpi=100)
        layout5 = QVBoxLayout()
        layout5.addWidget(self.canvas5)
        self.graphicsView_anal_anal.setLayout(layout5)

        # Add the navigation toolbar
        self.toolbar5 = NavigationToolbar(self.canvas5, self)
        layout5.addWidget(self.toolbar5)

        # ---- Add the View in Tab3 (Analytical vs Numerical) ----
        # Initialize Matplotlib canvas
        self.canvas3 = MplCanvas(self, width=5, height=4, dpi=100)
        layout3 = QVBoxLayout()
        layout3.addWidget(self.canvas3)
        self.graphicsView_tab3.setLayout(layout3)

        # ---- Add the View in Tab4 (Table) ----
        self.tableWidget_tab4.setColumnCount(4)
        self.tableWidget_tab4.setHorizontalHeaderLabels(["X Values", "Analytical Solution", "Numerical Solution", "Difference"])
        header = self.tableWidget_tab4.horizontalHeader()
        for column in range(4):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Stretch)

        # Add the navigation toolbar
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        layout3.addWidget(self.toolbar3)

        # ------- Set up the View for the Animation (Tab2) ------------
        # Add the Matplotlib canvas for heatmap
        self.canvas_heatmap = MplCanvas(self.widget_animation, width=5, height=4, dpi=100)
        self.widget_animation.layout().addWidget(self.canvas_heatmap)

        # Add the Matplotlib canvas for time series plot
        self.canvas_1 = MplCanvas(self.widget_diagram_1, width=5, height=2.5, dpi=100)
        self.widget_diagram_1.layout().addWidget(self.canvas_1)

        # Add the Matplotlib canvas for line plot
        self.canvas_2 = MplCanvas(self.widget_diagram_2, width=5, height=2.5, dpi=100)
        self.widget_diagram_2.layout().addWidget(self.canvas_2)

        # Add the custom vertical navigation toolbar for heatmap
        self.toolbar_heatmap = VerticalNavigationToolbar(self.canvas_heatmap, self)
        self.widget_animation_toolbar.layout().addWidget(self.toolbar_heatmap)

        # Add the navigation toolbar for the time series plot
        self.toolbar_time_series = NavigationToolbar(self.canvas_1, self)
        self.widget_diagram_1_toolbar.layout().addWidget(self.toolbar_time_series)

        # Add the navigation toolbar for the line plot
        self.toolbar_line_plot = NavigationToolbar(self.canvas_2, self)
        self.widget_diagram_1_toolbar_2.layout().addWidget(self.toolbar_line_plot)

        # Initialize the overlay widget
        self.overlay_widget = OverlayWidget(self.canvas_heatmap)

        # Connect slider to the update display function
        self.horizontalSlider_video.valueChanged.connect(self.slider_moved)

        # Connect combo boxes to the update function
        self.comboBox_diagram_1.currentIndexChanged.connect(self.update_combobox1)
        self.comboBox_diagram_2.currentIndexChanged.connect(self.update_combobox2)

        # Connect line edits to update functions
        self.lineEdit_canvas_1_point.editingFinished.connect(self.on_editing_canvas_1_input)
        self.lineEdit_canvas_1_point.returnPressed.connect(self.on_editing_canvas_1_input)

        self.lineEdit_canvas_1_line.editingFinished.connect(self.on_editing_canvas_1_input)
        self.lineEdit_canvas_1_line.returnPressed.connect(self.on_editing_canvas_1_input)

        self.lineEdit_canvas_2_point.editingFinished.connect(self.on_editing_canvas_2_input)
        self.lineEdit_canvas_2_point.returnPressed.connect(self.on_editing_canvas_2_input)

        self.lineEdit_canvas_2_line.editingFinished.connect(self.on_editing_canvas_2_input)
        self.lineEdit_canvas_2_line.returnPressed.connect(self.on_editing_canvas_2_input)

        # Connect the canvas signal to update the overlay widget
        self.canvas_heatmap.mpl_connect('motion_notify_event', self.on_canvas_mouse_move)
        self.canvas_heatmap.mpl_connect('figure_leave_event', self.on_canvas_mouse_leave)

        # set the initial diagram types and set visibility
        # Diagram 1:
        self.label_canvas_1_point_or_line.setVisible(True)
        self.comboBox_diagram_1.setCurrentIndex(0)
        self.comboBox_diagram_1.setEnabled(False)
        self.lineEdit_canvas_1_point.setVisible(True)
        self.lineEdit_canvas_1_point.setEnabled(False)
        self.lineEdit_canvas_1_line.setVisible(False)
        self.lineEdit_canvas_1_line.setEnabled(False)

        # Diagram 2:
        self.label_canvas_2_point_or_line.setVisible(True)
        self.comboBox_diagram_2.setCurrentIndex(1)
        self.comboBox_diagram_2.setEnabled(False)
        self.lineEdit_canvas_2_point.setVisible(False)
        self.lineEdit_canvas_2_point.setEnabled(False)
        self.lineEdit_canvas_2_line.setVisible(True)
        self.lineEdit_canvas_2_line.setEnabled(False)

        # Connect signals for all widgets to update settings
        self.connect_signals_to_update_settings(self)

        # Connect input fields to specific functions
        # ---- Sample Dimensions ----
        self.lineEdit_sample_width.textChanged.connect(self.on_sample_width_changed)
        self.lineEdit_sample_height.textChanged.connect(self.on_sample_height_changed)

        # ---- Hydrogen Parameters ----
        self.lineEdit_diff_coeff.textChanged.connect(self.on_lineedit_diff_coeff_changed)
        self.lineEdit_alpha_ramp_start.textChanged.connect(self.on_lineedit_diff_coeff_changed)
        self.lineEdit_alpha_ramp_end_alpha.textChanged.connect(self.on_lineedit_diff_coeff_changed)
        self.horizontalSlider_fixed_variable_alpha.valueChanged.connect(self.update_alpha_option)

        self.pushButton_show_ramp.clicked.connect(self.on_pushbutton_show_ramp_clicked)

        # -- allow clicks on labels --
        self.label_alpha_fixed.clicked.connect(self.on_label_fixed_clicked)
        self.label_alpha_ramp.clicked.connect(self.on_label_ramp_clicked)
        # -- Set Visibility --
        self.update_alpha_option()

        # ---- Discretization ----
        self.lineEdit_discret_space.textChanged.connect(self.on_discret_space_changed)
        self.lineEdit_discret_time.textChanged.connect(self.on_lineedit_discret_time_changed)
        self.checkBox_discret_time.stateChanged.connect(self.on_checkbox_discret_time_changed)

        # ---- Animation ----
        self.checkBox_incl_animation.stateChanged.connect(self.on_checkbox_incl_animation_changed)

        # ---- Start/Stop Simulation ----
        self.pushButton_start_all_sims.clicked.connect(self.on_start_all_sims_button_clicked)
        self.pushButton_stop_all_sims.clicked.connect(self.on_stop_all_sims_button_clicked)
        self.pushButton_start_all_sims.setEnabled(True)
        self.pushButton_stop_all_sims.setEnabled(False)
        self.pushButton_stop_all_sims.setVisible(True)
        self.pushButton_stop_all_sims.setVisible(False)
        self.checkBox_save_animation.setEnabled(True)  # Reenable Animation if it crashes

        # ---- Mode Tab Change ----
        self.tabWidget_Settings.currentChanged.connect(self.on_tabwidget_settings_changed)

        # ---- Display Time ----
        self.radioButton_s.toggled.connect(self.on_radio_button_toggled)
        self.radioButton_min.toggled.connect(self.on_radio_button_toggled)
        self.radioButton_d.toggled.connect(self.on_radio_button_toggled)
        self.radioButton_y.toggled.connect(self.on_radio_button_toggled)

        # ---- Simple Tab ----
        self.checkBox_equilibrium.stateChanged.connect(self.on_checkBox_equilibrium_changed)
        self.checkBox_anal_number_of_terms.stateChanged.connect(self.on_checkBox_anal_number_of_terms_changed)
        self.checkBox_show_flux.stateChanged.connect(self.on_checkBox_flux_changed)
        self.checkBox_show_min_value.stateChanged.connect(self.on_checkBox_show_min_value_changed)
        self.comboBox_flux_units.currentIndexChanged.connect(self.on_flux_units_changed)
        self.comboBox_mode_switch.currentIndexChanged.connect(self.calc_and_plot_to_tab_5)
        self.checkBox_plot_points_auto.stateChanged.connect(self.on_checkBox_plot_points_auto_changed)
        self.lineEdit_density.editingFinished.connect(self.calc_and_plot_to_tab_5)
        self.lineEdit_heat_capa.editingFinished.connect(self.calc_and_plot_to_tab_5)
        self.lineEdit_plot_points.editingFinished.connect(self.on_lineEdit_plot_points_changed)
        self.lineEdit_plot_points.setValidator(PlotPointsValidator())

        # Set the initial view
        self.tabWidget_Settings.setCurrentIndex(0)  # Show Mesh Tab
        self.tabWidget.setCurrentIndex(0)  # Show Mesh Tab
        self.toggle_tab_visibility()  # Set visible Tabs accordingly
        self.on_checkBox_anal_number_of_terms_changed()  # Set the Terms lineEdit accordingly
        self.on_checkBox_plot_points_auto_changed()
        self.update_flux_input_labels()

        # Draw the initial shape
        self.update_drawing()

        # # Draw the simple Graph (Set slider values)
        self.update_time_simple()
        self.update_width_simple()
        self.update_alpha_simple()
        self.update_init_hydro_simple()
        self.update_left_boundary_simple()
        self.update_right_boundary_simple()

        # # Draw the simple Graph
        self.calc_and_plot_to_tab_5()

        # Initialize the CreateAnimationWorker
        self.animation_worker = None
        self.current_simulation_file_path = None

        # Adjust the Layouts
        set_combo_box_margins(self)

        # Show maximized window
        self.showMaximized()

    def on_canvas_mouse_move(self, event):
        if event.inaxes:
            x, y = event.x, event.y
            ax = event.inaxes
            inv = ax.transData.inverted()
            xdata, ydata = inv.transform((x, y))
            value = self.get_value_at(xdata, ydata)
            self.overlay_widget.update_position(event.guiEvent.pos().x(), event.guiEvent.pos().y(),
                                                f'x: {xdata:.2f}, y: {ydata:.2f}, value: {value:.2f}')

    def on_canvas_mouse_leave(self, event):
        self.overlay_widget.hide()

    def show_tooltip(self, widget, message):
        self.tooltip_label.setText(message)
        self.tooltip_label.adjustSize()
        pos = widget.mapToGlobal(QtCore.QPoint(0, 0))
        self.tooltip_label.move(pos.x(), pos.y() - self.tooltip_label.height() - 5)
        self.tooltip_label.show()

    def get_value_at(self, x, y):
        try:
            frame = self.horizontalSlider_video.value()
            data = self.loaded_u_arrays[frame]
            ax = self.canvas_heatmap.axes
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            i = int((x - x0) / (x1 - x0) * (data.shape[1] - 1))
            j = int((y - y0) / (y1 - y0) * (data.shape[0] - 1))
            i = min(max(i, 0), data.shape[1] - 1)
            j = min(max(j, 0), data.shape[0] - 1)
            return data[j, i]
        except IndexError:
            return float('nan')

    def handle_worker_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.on_stop_all_sims_button_clicked()

    @QtCore.pyqtSlot(str)
    def handle_step_completed(self, step):
        last_progress_message = self.stdout_stream.get_last_progress_message()
        if "|" in last_progress_message:
            # Extract elapsed time and rate from the last progress message
            parts = last_progress_message.split(" ")
            rate_fmt = parts[-1].replace(",", "")
            elapsed = parts[-2].split("<")[-1]

            completed_message = f"{step}: {elapsed}, {rate_fmt}"
            self.completed_steps.append(completed_message)
            self.textEdit_console_output.clear()
            for step_message in self.completed_steps:
                self.textEdit_console_output.append(step_message)

    @QtCore.pyqtSlot(str)
    def update_console_output(self, text):
        try:
            # Remove carriage returns and whitespaces
            text = text.replace('\x1b[A', '').replace('\r', '').replace('\n', '').strip()

            if text == "":
                pass  # just to get out quick for debug

            elif "Starting" in text and "Matplotlib" not in text:  # We know it wants us to us agg, but its too slow
                self.textEdit_console_output.clear()
                curent_time_formatted = datetime.now().strftime("%d.%m.%Y %H:%M")
                self.kept_display_message = f"---- Starting 2D Simulation ({curent_time_formatted})----"
                self.textEdit_console_output.setText(self.kept_display_message)
                self.kept_display_message_with_current = self.kept_display_message

            elif "Currently" in text:
                self.kept_display_message_with_current += f"\n{text}"
                self.textEdit_console_output.setText(self.kept_display_message)

            elif "|" in text:
                text_to_put_in = self.kept_display_message_with_current + f"\n{text}"
                self.last_update_msg = text
                self.textEdit_console_output.setText(text_to_put_in)
            elif "Loading data: " in text:
                text_to_put_in = self.kept_display_message_with_current + f"\n{text}"
                self.textEdit_console_output.setText(text_to_put_in)

            elif "STEP_COMPLETED" in text:
                if "Loading data Completed:" in text:
                    completed_message = text.replace("STEP_COMPLETED: ", "")
                    self.kept_display_message += "\n" + completed_message
                    self.textEdit_console_output.setText(self.kept_display_message)
                    self.kept_display_message_with_current = self.kept_display_message
                elif "|" in self.last_update_msg:
                    # Extract elapsed time and rate from the last progress message
                    parts = self.last_update_msg.split(" ")
                    rate_fmt = parts[-1].replace("]", "")
                    elapsed = parts[-2].split("<")[-2]
                    elapsed = elapsed.replace("[", "")
                    step = text.replace("STEP_COMPLETED: ", "")
                    completed_message = f"\n{step} Completed: {elapsed}, {rate_fmt}"
                    self.kept_display_message += completed_message
                    self.textEdit_console_output.setText(self.kept_display_message)
                    self.kept_display_message_with_current = self.kept_display_message

            else:  # This will handle all other unexpected output (errors and such)
                if "Matplotlib" not in text:
                    self.kept_display_message = text + "\n\n" + self.kept_display_message
                    self.kept_display_message_with_current = self.kept_display_message

        except IndexError as print_error:
            print(f"Error: {print_error}")
        except Exception as print_error:
            print(f"An error occurred: {print_error}")

    def eventFilter(self, obj, event):
        if obj == self.info_label and event.type() == QEvent.Enter:
            # Update tooltip text dynamically
            new_tooltip_text = self.generate_dynamic_tooltip()
            self.info_label.setToolTip(new_tooltip_text)
        return super().eventFilter(obj, event)

    def init_sliders(self):
        # Set ranges for sliders
        self.horizontalSlider_time_simple.setRange(1, 100)
        self.horizontalSlider_width_simple.setRange(1, 100)
        self.horizontalSlider_alpha_simple.setRange(1, 100)
        self.horizontalSlider_init_hydro_simple.setRange(1, 100)
        self.horizontalSlider_left_boundary_simple.setRange(1, 100)
        self.horizontalSlider_right_boundary_simple.setRange(1, 100)

        # Connect sliders to methods
        self.horizontalSlider_time_simple.valueChanged.connect(self.update_time_simple)
        self.horizontalSlider_width_simple.valueChanged.connect(self.update_width_simple)
        self.horizontalSlider_alpha_simple.valueChanged.connect(self.update_alpha_simple)
        self.horizontalSlider_init_hydro_simple.valueChanged.connect(self.update_init_hydro_simple)
        self.horizontalSlider_left_boundary_simple.valueChanged.connect(self.update_left_boundary_simple)
        self.horizontalSlider_right_boundary_simple.valueChanged.connect(self.update_right_boundary_simple)

    def apply_static_assets(self):
        app_icon = QtGui.QIcon(resolve_image_path("WeldCraft.ico"))
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon)
        qt_app = QtWidgets.QApplication.instance()
        if qt_app is not None:
            qt_app.setApplicationDisplayName(APP_NAME)
            qt_app.setWindowIcon(app_icon)

        logo_path = resolve_image_path("BAM Logo.png")
        if os.path.exists(logo_path):
            self.label_2.setPixmap(QtGui.QPixmap(logo_path))
        else:
            self.label_2.clear()

    def calc_time_and_units(self):
        # Set the display time stuff here already
        if len(self.loaded_t_values) > 0:
            self.time_unit = "[s]"
            if max(self.loaded_t_values) >= 31557600:
                self.loaded_t_values = np.array(self.loaded_t_values) / 31557600
                self.time_unit = "[y]"
            elif max(self.loaded_t_values) >= 864000:
                self.loaded_t_values = np.array(self.loaded_t_values) / 86400
                self.time_unit = "[d]"
            elif max(self.loaded_t_values) >= 7200:
                self.loaded_t_values = np.array(self.loaded_t_values) / 3600
                self.time_unit = "[h]"
            elif max(self.loaded_t_values) >= 600:
                self.loaded_t_values = np.array(self.loaded_t_values) / 60
                self.time_unit = "[min]"

    def load_data(self):
        self.horizontalSlider_video.setEnabled(False)

        file_name = resolve_simulation_file_path(self.lineEdit_file_name.text())
        self.current_simulation_file_path = file_name
        self.data_loading_worker = DataLoadingWorker(file_name)
        self.data_loading_worker.progress.connect(self.update_console_output)
        self.data_loading_worker.result.connect(self.on_data_loaded)
        self.data_loading_worker.start()

    @QtCore.pyqtSlot(object, object, object, object, object)
    def on_data_loaded(self, loaded_u_arrays, loaded_t_values, u_sums, rate_of_change, u_saturation):

        self.loaded_u_arrays = loaded_u_arrays
        self.loaded_t_values = loaded_t_values
        self.loaded_u_sums = u_sums
        self.rate_of_change = rate_of_change
        self.u_saturation = u_saturation

        if not self.loaded_u_arrays:
            self.statusBar().showMessage("Failed to load data.")
            return

        self.calc_time_and_units()
        self.initialize_heatmap()
        self.initialize_plots()
        self.horizontalSlider_video.setMaximum(len(self.loaded_u_arrays) - 1)
        self.horizontalSlider_video.setValue(self.horizontalSlider_video.minimum() + 1)
        self.update_display(1)
        self.horizontalSlider_video.setEnabled(True)
        self.update_combobox1()
        self.update_combobox2()
        self.comboBox_diagram_1.setEnabled(True)
        self.comboBox_diagram_2.setEnabled(True)

        if self.checkBox_save_animation.isChecked():
            # Disable the checkbox and start the worker
            self.checkBox_save_animation.setEnabled(False)  # disable the checkboxes
            self.checkBox_save_animation.setChecked(False)
            self.pushButton_start_all_sims.setEnabled(False)  # disable the option to make a new animation during saving
            animation_output_path = resolve_animation_output_path(self.current_simulation_file_path)
            self.animation_worker = CreateAnimationWorker(loaded_u_arrays, loaded_t_values, animation_output_path)
            self.animation_worker.progress.connect(self.update_console_output)
            self.animation_worker.finished.connect(self.on_animation_worker_finished)
            self.animation_worker.start()

    @QtCore.pyqtSlot()
    def on_animation_worker_finished(self):
        self.checkBox_save_animation.setEnabled(True)  # reenable stuff
        self.checkBox_save_animation.setChecked(True)  # reenable stuff
        self.pushButton_start_all_sims.setEnabled(True)  # reenable stuff

    def calculate_highest_value(self):
        high_left = float(self.lineEdit_boundary_left.text())
        high_right = float(self.lineEdit_boundary_right.text())
        high_top = float(self.lineEdit_boundary_top.text())
        high_bottom = float(self.lineEdit_boundary_bottom.text())
        high_init_conc = float(self.lineEdit_hydro_conc_init.text())
        return max(high_left, high_right, high_top, high_bottom, high_init_conc)

    def initialize_heatmap(self):
        # Clear the current figure to avoid duplicating colorbars
        self.canvas_heatmap.fig.clear()

        # get the dx values
        dx_values = self.get_dx_values()  # Get the list of dx values
        dx = max(dx_values)
        dy = dx

        # Recreate the axes
        self.canvas_heatmap.axes = self.canvas_heatmap.fig.add_subplot(111)

        # Set up the initial heatmap and colorbar
        cmap = plt.get_cmap('viridis')
        if len(self.loaded_u_arrays) > 0:
            self.norm = mcolors.Normalize(vmin=np.min(self.loaded_u_arrays), vmax=np.max(self.loaded_u_arrays))
            data = self.loaded_u_arrays[0]
        else:
            self.norm = mcolors.Normalize(vmin=0, vmax=100)
            data = np.zeros((100, 100))  # Dummy data

        self.im_heatmap = self.canvas_heatmap.axes.imshow(data, cmap=cmap, norm=self.norm)
        self.canvas_heatmap.axes.set_title('Hydrogen Concentration')
        self.canvas_heatmap.axes.set_xlabel(f'X Position [{dx}mm]')
        self.canvas_heatmap.axes.set_ylabel(f'Y Position [{dy}mm]')

        # Add the colorbar
        self.canvas_heatmap.fig.colorbar(self.im_heatmap, ax=self.canvas_heatmap.axes, orientation='vertical')

        # Adjust the axes
        self.canvas_heatmap.axes.set_xlim(-2 * dx, data.shape[1] - 1)
        self.canvas_heatmap.axes.set_ylim(-2 * dy, data.shape[0] - 1)

        # Apply tight layout to adjust space
        self.canvas_heatmap.fig.tight_layout()

        # Initial draw to cache the background
        self.canvas_heatmap.draw_idle()
        self.canvas_heatmap.flush_events()
        self.im_heatmap.set_animated(True)
        self.heatmap_background = self.canvas_heatmap.copy_from_bbox(self.canvas_heatmap.figure.bbox)

    def initialize_plots(self):
        # Loop through the 2 canvases with their corresponding combo boxes and initialize them
        canvases = [
            (self.canvas_1, self.comboBox_diagram_1, self.lineEdit_canvas_1_point, self.lineEdit_canvas_1_line),
            (self.canvas_2, self.comboBox_diagram_2, self.lineEdit_canvas_2_point, self.lineEdit_canvas_2_line)
        ]

        for canvas, combo_box, line_edit_point, line_edit_line in canvases:
            canvas_diagram_option = combo_box.currentIndex()  # 0: Point, 1: Line, 2: Total_Conc, 3: Flux
            self.initialize_plot(canvas, canvas_diagram_option, line_edit_point, line_edit_line)

    def initialize_plot(self, canvas, option, line_edit_point, line_edit_line):
        if not self.loaded_u_arrays:
            return

        # Clear previous lines to avoid multiple lines issue
        canvas.axes.cla()
        # Highest value for axis settings
        highest_value = self.calculate_highest_value()

        # Get the dx values
        dx_values = self.get_dx_values()  # Get the list of dx values
        dx = max(dx_values)

        # get the user input if necessary:
        point_input = line_edit_point.text()
        line_input = line_edit_line.text()

        # 0: Point, 1: Line, 2: Total_Conc, 3: Flux
        if option == 0:  # Point
            x, y = self.is_valid_point_input(point_input)
            if canvas == self.canvas_1:
                self.canvas_1_data = [u[y, x] for u in self.loaded_u_arrays]  # Note: y, x

                self.canvas_1_plot_line, = canvas.axes.plot(self.loaded_t_values, self.canvas_1_data, animated=False)
                self.canvas_1_plot_marker, = canvas.axes.plot([], [], 'ko', animated=False)
            else:
                self.canvas_2_data = [u[y, x] for u in self.loaded_u_arrays]  # Note: y, x

                self.canvas_2_plot_line, = canvas.axes.plot(self.loaded_t_values, self.canvas_2_data, animated=False)
                self.canvas_2_plot_marker, = canvas.axes.plot([], [], 'ko', animated=False)

            canvas.axes.set_xlim(0, max(self.loaded_t_values))
            canvas.axes.set_ylim(0, highest_value + highest_value / 100)
            canvas.axes.set_xlabel(f'Time {self.time_unit}')
            canvas.axes.set_ylabel('Hydrogen Concentration')

        elif option == 1:  # Line
            # Precompute data for line plot
            y_pos = self.is_valid_line_input(line_input)
            if canvas == self.canvas_1:
                self.canvas_1_data = np.array([u[y_pos, :] for u in self.loaded_u_arrays])

                self.canvas_1_plot_line, = canvas.axes.plot([], [], animated=False)
                self.canvas_1_plot_marker = []
            else:
                self.canvas_2_data = np.array([u[y_pos, :] for u in self.loaded_u_arrays])

                self.canvas_2_plot_line, = canvas.axes.plot([], [], animated=False)
                self.canvas_2_plot_marker = []

            canvas.axes.set_xlim(0, self.loaded_u_arrays[0].shape[1])
            canvas.axes.set_ylim(0, highest_value + highest_value / 100)
            canvas.axes.set_xlabel(f'X Position [{dx}mm]')
            canvas.axes.set_ylabel('Hydrogen Concentration')

        elif option == 2:  # Total Concentration
            # Precompute data for total concentration
            if canvas == self.canvas_1:
                self.canvas_1_data = self.u_saturation  # No need for calculations, it is done
                self.canvas_1_plot_line, = canvas.axes.plot(self.loaded_t_values, self.canvas_1_data, animated=False)
                self.canvas_1_plot_marker, = canvas.axes.plot([], [], 'ko', animated=False)
            else:
                self.canvas_2_data = self.u_saturation
                self.canvas_2_plot_line, = canvas.axes.plot(self.loaded_t_values, self.canvas_2_data, animated=False)
                self.canvas_2_plot_marker, = canvas.axes.plot([], [], 'ko', animated=False)

            canvas.axes.set_xlim(0, max(self.loaded_t_values))
            canvas.axes.set_ylim(0, max(self.u_saturation) + max(self.u_saturation) / 100)
            canvas.axes.set_xlabel(f'Time {self.time_unit}')
            canvas.axes.set_ylabel('Total Hydrogen Concentration [%]')

        elif option == 3:  # Flux
            # Precompute data for flux (rate_of_change)
            if canvas == self.canvas_1:
                self.canvas_1_data = self.rate_of_change  # No need for calculations, it is done
                self.canvas_1_plot_line, = canvas.axes.plot(self.loaded_t_values, self.canvas_1_data, animated=False)
                self.canvas_1_plot_marker, = canvas.axes.plot([], [], 'ko', animated=False)
            else:
                self.canvas_2_data = self.rate_of_change
                self.canvas_2_plot_line, = canvas.axes.plot(self.loaded_t_values, self.canvas_2_data, animated=False)
                self.canvas_2_plot_marker, = canvas.axes.plot([], [], 'ko', animated=False)

            canvas.axes.set_xlim(0, max(self.loaded_t_values))
            canvas.axes.set_ylim(0, max(self.rate_of_change) + max(self.rate_of_change) / 100)
            canvas.axes.set_xlabel(f'Time {self.time_unit}')
            canvas.axes.set_ylabel('Hydrogen Flux in [% of max]')

        else:  # Dummy stuff
            canvas.axes.set_xlim(-100, 100)
            canvas.axes.set_ylim(-100, 100)

        # Set the layout here.
        canvas.fig.tight_layout()
        canvas.fig.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)

    @QtCore.pyqtSlot()
    def slider_moved(self):
        self.force_update = True
        self.update_display(self.horizontalSlider_video.value())

    def update_display(self, frame):
        current_time = time.time()
        # Check if enough time has passed since the last update
        if current_time - self.last_update_time < self.update_interval and frame != self.current_frame:
            return

        self.current_frame = frame
        if not self.loaded_u_arrays:
            return

        # Update the heatmap, this worked fine, so we will leave it as is
        self.im_heatmap.set_data(self.loaded_u_arrays[frame])
        self.canvas_heatmap.restore_region(self.heatmap_background)
        self.canvas_heatmap.axes.draw_artist(self.im_heatmap)
        self.canvas_heatmap.blit(self.canvas_heatmap.figure.bbox)

        # Set the canvas list again:
        canvases = [
            (self.canvas_1, self.comboBox_diagram_1),
            (self.canvas_2, self.comboBox_diagram_2)
        ]

        # Loop over the canvases and combobox options and update the data and titles etc.
        for canvas, combo_box in canvases:
            option = combo_box.currentIndex()
            self.update_plot(canvas, option, frame)

        # Actually draw the stuff.
        for canvas, combo_box in canvases:
            canvas.draw()

        # Update the last update time
        self.last_update_time = current_time
        self.force_update = False  # Reset the forced update flag

    def update_plot(self, canvas, option, frame):
        time_in_units = self.loaded_t_values[frame]  # Update time text

        # 0: Point, 1: Line, 2: Total_Conc, 3: Flux
        if option == 0:  # Point
            # We only really need to re-plot the marker. the line stays the same. And update the title
            if canvas == self.canvas_1:
                point = self.lineEdit_canvas_1_point.text()
                self.canvas_1_plot_marker.set_data([self.loaded_t_values[frame]], [self.canvas_1_data[frame]])
                canvas.draw()  # _artist(self.canvas_1_plot_marker)
            else:
                point = self.lineEdit_canvas_2_point.text()
                self.canvas_2_plot_marker.set_data([self.loaded_t_values[frame]], [self.canvas_2_data[frame]])
                canvas.draw()  # _artist(self.canvas_2_plot_marker)

            # Update Titles
            canvas.axes.set_title(
                f'Hydrogen Concentration at Point {point}, Time: {time_in_units:.1f} {self.time_unit}')

        elif option == 1:  # Line
            # We need to redraw the line, there is no marker here. And update the title
            if canvas == self.canvas_1:
                y_pos = self.lineEdit_canvas_1_line.text()
                self.canvas_1_plot_line.set_data(np.arange(self.loaded_u_arrays[0].shape[1]), self.canvas_1_data[frame])
                canvas.draw()  # _artist(self.canvas_1_plot_line)
            else:
                y_pos = self.lineEdit_canvas_2_line.text()
                self.canvas_2_plot_line.set_data(np.arange(self.loaded_u_arrays[0].shape[1]), self.canvas_2_data[frame])
                canvas.draw()  # _artist(self.canvas_2_plot_line)

            # Update Titles
            canvas.axes.set_title(
                f'Hydrogen Concentration along Y={y_pos}, Time: {time_in_units:.1f} {self.time_unit}')

        elif option == 2:  # Total Concentration
            # We only really need to re-plot the marker. the line stays the same. And update the title
            if canvas == self.canvas_1:
                self.canvas_1_plot_marker.set_data([self.loaded_t_values[frame]], [self.canvas_1_data[frame]])
                canvas.draw()  # _artist(self.canvas_1_plot_marker)
                # Update Titles
                canvas.axes.set_title(
                    f'Total Saturation {self.canvas_1_data[frame]:.1f}%, Time: {time_in_units:.1f} {self.time_unit}')
            else:
                self.canvas_2_plot_marker.set_data([self.loaded_t_values[frame]], [self.canvas_2_data[frame]])
                canvas.draw()  # _artist(self.canvas_2_plot_marker)
                # Update Titles
                canvas.axes.set_title(
                    f'Total Saturation {self.canvas_2_data[frame]:.1f}%, Time: {time_in_units:.1f} {self.time_unit}')

        elif option == 3:  # Flux
            # We only really need to re-plot the marker. the line stays the same. And update the title
            if canvas == self.canvas_1:
                self.canvas_1_plot_marker.set_data([self.loaded_t_values[frame]], [self.canvas_1_data[frame]])
                canvas.draw()  # _artist(self.canvas_1_plot_marker)
            else:
                self.canvas_2_plot_marker.set_data([self.loaded_t_values[frame]], [self.canvas_2_data[frame]])
                canvas.draw()  # _artist(self.canvas_2_plot_marker)

            # Update Titles
            canvas.axes.set_title(
                f'Rate of Change of Hydrogen Concentration (flux), Time: {time_in_units:.1f} {self.time_unit}')

    def update_combobox1(self):
        # get new value
        new_combobox_1_value = self.comboBox_diagram_1.currentIndex()
        # for relevant options, show the line edit and add a valid input if necessary
        if new_combobox_1_value == 0:  # Point
            self.lineEdit_canvas_1_point.setVisible(True)  # Make Point Input Visible
            self.lineEdit_canvas_1_point.setEnabled(True)  # Make Point Input Enabled
            self.lineEdit_canvas_1_line.setVisible(False)  # Make Line Input Invisible
            self.lineEdit_canvas_1_line.setEnabled(False)  # Make Line Input Disabled

            self.label_canvas_1_point_or_line.setVisible(True)  # Make Label Visible
            self.label_canvas_1_point_or_line.setText("Plot for Point:")  # also change the label

            cleaned_input = self.is_valid_point_input(self.lineEdit_canvas_1_point.text())
            if cleaned_input is None:  # act if cleaned input returned false
                try:
                    points = self.loaded_u_arrays[0].shape[1] // 2, self.loaded_u_arrays[0].shape[0] // 2
                    self.lineEdit_canvas_1_point.setText(f"{points[0]},{points[1]}")
                except IndexError:  # Catch specific error
                    self.lineEdit_canvas_1_point.setText("1,1")

        elif new_combobox_1_value == 1:  # Line
            self.lineEdit_canvas_1_point.setVisible(False)  # Make Point Input Invisible
            self.lineEdit_canvas_1_point.setEnabled(False)  # Make Point Input Disabled
            self.lineEdit_canvas_1_line.setVisible(True)  # Make Line Input Visible
            self.lineEdit_canvas_1_line.setEnabled(True)  # Make Line Input Enabled

            self.label_canvas_1_point_or_line.setVisible(True)  # Make Label Visible
            self.label_canvas_1_point_or_line.setText("Plot line at Y-Value:")  # also change the label

            cleaned_input = self.is_valid_line_input(self.lineEdit_canvas_1_line.text())
            if cleaned_input is None:  # act if cleaned input returned None
                try:
                    y_pos = self.loaded_u_arrays[0].shape[0] // 2
                    self.lineEdit_canvas_1_line.setText(str(y_pos))
                except IndexError:  # Catch specific error
                    self.lineEdit_canvas_1_line.setText("1")

        else:  # The other two "fluxy" options
            self.lineEdit_canvas_1_point.setVisible(False)  # Make Everything go away
            self.lineEdit_canvas_1_point.setEnabled(False)  # Make Everything go away
            self.lineEdit_canvas_1_line.setVisible(False)  # Make Everything go away
            self.lineEdit_canvas_1_line.setEnabled(False)  # Make Everything go away
            self.label_canvas_1_point_or_line.setVisible(False)  # Make Everything go away

        if self.loaded_u_arrays:  # During init, this is empty. Don't initialize... duhhh
            self.initialize_plots()
            self.update_display(self.horizontalSlider_video.value())  # update the display

    def update_combobox2(self):
        # get new value
        new_combobox_2_value = self.comboBox_diagram_2.currentIndex()
        # for relevant options, show the line edit and add a valid input if necessary
        if new_combobox_2_value == 0:  # Point
            self.lineEdit_canvas_2_point.setVisible(True)  # Make Point Input Visible
            self.lineEdit_canvas_2_point.setEnabled(True)  # Make Point Input Enabled
            self.lineEdit_canvas_2_line.setVisible(False)  # Make Line Input Invisible
            self.lineEdit_canvas_2_line.setEnabled(False)  # Make Line Input Disabled

            self.label_canvas_2_point_or_line.setVisible(True)  # Make Label Visible
            self.label_canvas_2_point_or_line.setText("Plot for Point:")  # also change the label

            cleaned_input = self.is_valid_point_input(self.lineEdit_canvas_2_point.text())
            if cleaned_input is None:  # act if cleaned input returned None
                try:
                    points = self.loaded_u_arrays[0].shape[1] // 2, self.loaded_u_arrays[0].shape[0] // 2
                    self.lineEdit_canvas_2_point.setText(f"{points[0]},{points[1]}")
                except IndexError:  # Catch specific error
                    self.lineEdit_canvas_2_point.setText("1,1")

        elif new_combobox_2_value == 1:  # Line
            self.lineEdit_canvas_2_point.setVisible(False)  # Make Point Input Invisible
            self.lineEdit_canvas_2_point.setEnabled(False)  # Make Point Input Disabled
            self.lineEdit_canvas_2_line.setVisible(True)  # Make Line Input Visible
            self.lineEdit_canvas_2_line.setEnabled(True)  # Make Line Input Enabled

            self.label_canvas_2_point_or_line.setVisible(True)  # Make Label Visible
            self.label_canvas_2_point_or_line.setText("Plot line at Y-Value:")  # also change the label

            cleaned_input = self.is_valid_line_input(self.lineEdit_canvas_2_line.text())
            if cleaned_input is None:  # act if cleaned input returned false
                try:
                    y_pos = self.loaded_u_arrays[0].shape[0] // 2
                    self.lineEdit_canvas_2_line.setText(str(y_pos))
                except IndexError:  # Catch specific error
                    self.lineEdit_canvas_2_line.setText("1")

        else:  # The other two "fluxy" options
            self.lineEdit_canvas_2_point.setVisible(False)  # Make Everything go away
            self.lineEdit_canvas_2_point.setEnabled(False)  # Make Everything go away
            self.lineEdit_canvas_2_line.setVisible(False)  # Make Everything go away
            self.lineEdit_canvas_2_line.setEnabled(False)  # Make Everything go away
            self.label_canvas_2_point_or_line.setVisible(False)  # Make Everything go away

        if self.loaded_u_arrays:  # During init, this is empty. Don't initialize... duhhh
            self.initialize_plots()
            self.update_display(self.horizontalSlider_video.value())  # update the display

    def on_editing_canvas_1_input(self):
        combobox_1_value = self.comboBox_diagram_1.currentIndex()
        if combobox_1_value == 0:  # Point
            cleaned_input = self.is_valid_point_input(self.lineEdit_canvas_1_point.text())  # check if ok
            if cleaned_input is None:  # act if cleaned input returned None
                QMessageBox.warning(None, "Warning", "Invalid input! Please enter a valid value.")
            else:
                self.initialize_plots()
                self.update_display(self.horizontalSlider_video.value())  # update the display
        else:
            cleaned_input = self.is_valid_line_input(self.lineEdit_canvas_1_line.text())  # check if ok
            if cleaned_input is None:  # act if cleaned input returned None
                QMessageBox.warning(None, "Warning", "Invalid input! Please enter a valid value.")
            else:
                self.initialize_plots()
                self.update_display(self.horizontalSlider_video.value())  # update the display

    def on_editing_canvas_2_input(self):
        combobox_2_value = self.comboBox_diagram_2.currentIndex()
        if combobox_2_value == 0:  # Point
            cleaned_input = self.is_valid_point_input(self.lineEdit_canvas_2_point.text())  # check if ok
            if cleaned_input is None:  # act if cleaned input returned None
                QMessageBox.warning(None, "Warning", "Invalid input! Please enter a valid value.")
            else:
                self.initialize_plots()
                self.update_display(self.horizontalSlider_video.value())  # update the display
        else:
            cleaned_input = self.is_valid_line_input(self.lineEdit_canvas_2_line.text())  # check if ok
            if cleaned_input is None:  # act if cleaned input returned None
                QMessageBox.warning(None, "Warning", "Invalid input! Please enter a valid value.")
            else:
                self.initialize_plots()
                self.update_display(self.horizontalSlider_video.value())  # update the display

    def is_valid_point_input(self, input_string):
        # Define the regular expression pattern
        pattern = r'^\s*\d+\s*,\s*\d+\s*$'

        # Use the re.match function to check if the input_string matches the pattern
        if re.match(pattern, input_string):
            # Remove all whitespace characters
            cleaned_input = input_string.replace(" ", "")
            x_retr, y_retr = cleaned_input.split(',')
            x_retr = int(x_retr)
            y_retr = int(y_retr)

            if not self.loaded_u_arrays:  # During init, this is empty, but the entries can't be wrong!
                return x_retr, y_retr
            # Check if the input is within the bounds of the matrix - if no u loaded?? happens during init
            if 0 <= x_retr < self.loaded_u_arrays[0].shape[1] and 0 <= y_retr < self.loaded_u_arrays[0].shape[0]:
                return x_retr, y_retr
            else:  # if its out of bounds, change it to the mid point
                x_mid = int(self.loaded_u_arrays[0].shape[1] / 2)
                y_mid = int(self.loaded_u_arrays[0].shape[0] / 2)
                self.lineEdit_canvas_1_point.setText(f"{x_mid},{y_mid}")
                self.lineEdit_canvas_2_point.setText(f"{x_mid},{y_mid}")

                return x_mid, y_mid
        return None  # return None cuz 0 is a valid point to check, 0 would then "if not 0" be evaluated as True

    def is_valid_line_input(self, input_string):
        # Define the regular expression pattern for a single integer
        pattern = r'^\s*\d+\s*$'

        # Use the re.match function to check if the input_string matches the pattern
        if re.match(pattern, input_string):
            # Remove all whitespace characters
            cleaned_input = input_string.replace(" ", "")
            value = int(cleaned_input)

            if not self.loaded_u_arrays:  # During init, this is empty, but the entries can't be wrong!
                return value
            # Check if the input is within the bounds
            if 0 <= value < self.loaded_u_arrays[0].shape[0]:  # Example bound check
                return value
            else:
                line_mid_location = int(self.loaded_u_arrays[0].shape[0] / 2)
                self.lineEdit_canvas_1_line.setText(f"{line_mid_location}")
                self.lineEdit_canvas_2_line.setText(f"{line_mid_location}")

                return line_mid_location
        return None  # Explicitly return None for invalid input

    def on_radio_button_toggled(self):
        self.update_time_simple()

    def on_label_fixed_clicked(self):
        self.horizontalSlider_fixed_variable_alpha.setValue(0)  # Changed to 0 for Fixed

    def on_label_ramp_clicked(self):
        self.horizontalSlider_fixed_variable_alpha.setValue(1)  # Changed to 1 for Ramp

    def update_alpha_option(self):
        self.current_alpha_option = self.horizontalSlider_fixed_variable_alpha.value()

        if self.current_alpha_option == 0:  # Fixed
            # Labels for slider
            self.label_alpha_fixed.setStyleSheet("color: black;")
            self.label_alpha_ramp.setStyleSheet("color: gray;")

            # Show ramp options
            self.lineEdit_warm_up.setVisible(False)
            self.label_warm_up.setVisible(False)
            self.lineEdit_k_value.setVisible(False)
            self.label_k_value.setVisible(False)
            self.pushButton_show_ramp.setVisible(False)

            # Show the relevant option (fixed)
            self.lineEdit_diff_coeff.setVisible(True)
            self.label_diff_coeff.setVisible(True)

            # Hide the other options (ramp)
            self.lineEdit_alpha_ramp_start.setVisible(False)
            self.label_alpha_ramp_start.setVisible(False)
            self.lineEdit_alpha_ramp_end_time.setVisible(False)
            self.label_alpha_ramp_end_time.setVisible(False)
            self.lineEdit_alpha_ramp_end_alpha.setVisible(False)
            self.label_alpha_ramp_end_alpha.setVisible(False)

        elif self.current_alpha_option == 1:  # Ramp
            # Labels for slider
            self.label_alpha_fixed.setStyleSheet("color: gray;")
            self.label_alpha_ramp.setStyleSheet("color: black;")

            # Show ramp options
            self.lineEdit_warm_up.setVisible(True)
            self.label_warm_up.setVisible(True)
            self.lineEdit_k_value.setVisible(True)
            self.label_k_value.setVisible(True)
            self.pushButton_show_ramp.setVisible(True)

            # Show the relevant option (ramp)
            self.lineEdit_alpha_ramp_start.setVisible(True)
            self.label_alpha_ramp_start.setVisible(True)
            self.lineEdit_alpha_ramp_end_time.setVisible(True)
            self.label_alpha_ramp_end_time.setVisible(True)
            self.lineEdit_alpha_ramp_end_alpha.setVisible(True)
            self.label_alpha_ramp_end_alpha.setVisible(True)

            # Hide the other options (fixed)
            self.lineEdit_diff_coeff.setVisible(False)
            self.label_diff_coeff.setVisible(False)

        # Only recalculate dt if the use auto_dt_checkbox is checked
        if self.checkBox_discret_time.isChecked():
            self.calculate_dt_for_ui()

        self.check_if_dt_is_ok()

    def on_pushbutton_show_ramp_clicked(self):

        def sigmoid(t, alpha_0, alpha_1, t_0, k):
            return alpha_0 + (alpha_1 - alpha_0) / (1 + np.exp(-k * (t - t_0)))

        # Define parameters for the sigmoid function
        alpha_0 = float(self.lineEdit_alpha_ramp_start.text())  # Initial alpha
        alpha_1 = float(self.lineEdit_alpha_ramp_end_alpha.text())  # Final alpha
        alpha_end_time = float(self.safe_eval(self.lineEdit_alpha_ramp_end_time.text()))  # End time for alpha change
        warmup_period = float(self.safe_eval(self.lineEdit_warm_up.text()))  # Initial warm-up period in seconds
        sim_time = float(self.safe_eval(self.lineEdit_sim_time.text()))  # Total simulation time in seconds
        t_0 = (alpha_end_time + warmup_period) / 2  # Midpoint for the sigmoid function
        k = float(self.lineEdit_k_value.text())  # Sigmoid steepness factor

        # Generate time values
        time_values = np.linspace(0, sim_time + warmup_period, num=10000)

        # Calculate alpha values using the sigmoid function
        alpha_values = []
        for t in time_values:
            if t <= warmup_period:
                alpha_values.append(alpha_0)  # Warm-up period: alpha_0
            elif t <= alpha_end_time + warmup_period:
                alpha_values.append(sigmoid(t - warmup_period, alpha_0, alpha_1, t_0, k))
            else:
                alpha_values.append(alpha_1)  # Period after reaching alpha_1

        # Plot the alpha values over time
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, alpha_values, label='Alpha (Diffusion Coefficient)')
        plt.xlabel('Time (s)')
        plt.ylabel('Alpha')
        plt.title('Alpha (Diffusion Coefficient) over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def on_checkBox_equilibrium_changed(self):
        self.calc_and_plot_to_tab_5()

    def on_checkBox_flux_changed(self):
        self.calc_and_plot_to_tab_5()

    def on_checkBox_show_min_value_changed(self):
        self.calc_and_plot_to_tab_5()

    def on_flux_units_changed(self):
        self.update_flux_input_labels()
        self.calc_and_plot_to_tab_5()

    def on_checkBox_plot_points_auto_changed(self):
        use_auto = self.checkBox_plot_points_auto.isChecked()
        self.lineEdit_plot_points.setEnabled(not use_auto)
        self.calc_and_plot_to_tab_5()

    def on_lineEdit_plot_points_changed(self):
        if self.checkBox_plot_points_auto.isChecked():
            return
        text = self.lineEdit_plot_points.text().strip()
        if text and text.isdigit() and int(text) > 0:
            self.calc_and_plot_to_tab_5()

    def get_plot_point_count(self):
        if self.checkBox_plot_points_auto.isChecked():
            return 200
        text = self.lineEdit_plot_points.text().strip()
        if not text.isdigit():
            raise InvalidInputError("Plot points must be a positive integer.")
        value = int(text)
        if value <= 0:
            raise InvalidInputError("Plot points must be larger than zero.")
        return value

    def get_flux_selection(self):
        return self.comboBox_flux_units.currentText()

    def get_selected_flux_mode(self):
        if "Fick" in self.get_flux_selection():
            return "fick"
        return "fourier"

    def is_relative_flux_mode(self):
        return "Relative" in self.get_flux_selection()

    @staticmethod
    def format_flux_value(value, is_relative=False):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NaN"
        if is_relative:
            if abs(value) < 0.005:
                return "0.00"
            return f"{value:.2f}"
        if abs(value) < 1e-21:
            return "<1.0e-21"
        return f"{value:.1e}"

    def update_flux_input_labels(self):
        if self.is_relative_flux_mode():
            tooltip = "Not used in relative mode. Relative flux is reported as a percent of the steady-state flux."
            self.label_density.setText("Not used in relative mode")
            self.label_density.setToolTip(tooltip)
            self.label_density_2.setText("Not used in relative mode")
            self.label_density_2.setToolTip(tooltip)
            self.lineEdit_density.setToolTip(tooltip)
            self.lineEdit_heat_capa.setToolTip(tooltip)
            self.lineEdit_density.setEnabled(False)
            self.lineEdit_heat_capa.setEnabled(False)
        elif self.get_selected_flux_mode() == "fick":
            tooltip_1 = ("Use this only for absolute hydrogen flux. It converts the plotted hydrogen value "
                         "into a molar concentration for Fick's law: C = u × factor.")
            tooltip_2 = ("Use this only for absolute hydrogen flux. It sets the diffusion coefficient used "
                         "in Fick's law relative to the simple-tab alpha: D = alpha × ratio.")
            self.label_density.setText("Molar concentration conversion factor [mol/mm³ per plotted unit]")
            self.label_density.setToolTip(tooltip_1)
            self.label_density_2.setText("Diffusion coefficient ratio D/alpha [-]")
            self.label_density_2.setToolTip(tooltip_2)
            self.lineEdit_density.setToolTip(tooltip_1)
            self.lineEdit_heat_capa.setToolTip(tooltip_2)
            self.lineEdit_density.setEnabled(True)
            self.lineEdit_heat_capa.setEnabled(True)
        else:
            tooltip_1 = "Mass density used with alpha and cp to compute thermal conductivity for Fourier's law."
            tooltip_2 = "Specific heat capacity used with alpha and density to compute thermal conductivity for Fourier's law."
            self.label_density.setText("Mass density ρ [kg/m³]")
            self.label_density.setToolTip(tooltip_1)
            self.label_density_2.setText("Specific heat capacity cₚ [J/(kg·K)]")
            self.label_density_2.setToolTip(tooltip_2)
            self.lineEdit_density.setToolTip(tooltip_1)
            self.lineEdit_heat_capa.setToolTip(tooltip_2)
            self.lineEdit_density.setEnabled(True)
            self.lineEdit_heat_capa.setEnabled(True)

    def get_simple_mode_axis_label(self):
        if "Heat" in self.comboBox_mode_switch.currentText():
            return "Temperature [°C]"
        return "Hydrogen Concentration"

    def _legacy_update_flux_input_labels_notes(self):
        # Legacy prototype kept for reference:
        # it used a generic concentration conversion factor and a D/alpha ratio.
        if self.is_relative_flux_mode():
            tooltip = "Not used in relative mode. Relative flux is reported as a percent of the steady-state flux."
            self.label_density.setText("Not used in relative mode")
            self.label_density.setToolTip(tooltip)
            self.label_density_2.setText("Not used in relative mode")
            self.label_density_2.setToolTip(tooltip)
            self.lineEdit_density.setToolTip(tooltip)
            self.lineEdit_heat_capa.setToolTip(tooltip)
            self.lineEdit_density.setEnabled(False)
            self.lineEdit_heat_capa.setEnabled(False)
            self.lineEdit_heat_capa.setVisible(False)
            self.label_density_2.setVisible(False)
        elif self.get_selected_flux_mode() == "fick":
            tooltip_1 = ("EXPERIMENTAL FEATURE. Use this only for left-to-right permeation-style interpretation "
                         "of the simple plot. Enter the left-side subsurface hydrogen concentration that corresponds "
                         "to the \"fully filled\" normalized state u = 100, i.e. left boundary value 1.0. "
                         "The code maps the normalized field to concentration with C = (u/100) x C_ref. "
                         "Default: C_ref = 3.5 x 10^-8 mol/mm^3, based on 5 ml/100 g Fe ≈ 4.5 wt ppm ≈ 3.5 x 10^-8 mol/mm^3.")
            tooltip_2 = "Not used for hydrogen absolute mode. Alpha is treated as the hydrogen diffusivity D_H."
            tooltip_1 = ("EXPERIMENTAL FEATURE. Use this only for left-to-right permeation-style interpretation "
                         "of the simple plot. Enter the left-side subsurface hydrogen concentration that corresponds "
                         "to the fully filled normalized state u = 100, i.e. left boundary value 1.0. "
                         "The code maps the normalized field to concentration with C = (u/100) x C_ref. "
                         "Default: C_ref = 3.5 x 10^-8 mol/mm³, based on 5 ml/100 g Fe ≈ 4.5 wt ppm ≈ 3.5 x 10^-8 mol/mm³.")
            self.label_density.setText("Left Side Subsurface Concentration [mol/mm³]")
            self.label_density.setToolTip(tooltip_1)
            self.label_density_2.setText("Not used in hydrogen absolute mode")
            self.label_density_2.setToolTip(tooltip_2)
            self.lineEdit_density.setToolTip(tooltip_1)
            self.lineEdit_heat_capa.setToolTip(tooltip_2)
            if self.lineEdit_density.text().strip() in {"", "100"}:
                self.lineEdit_density.setText("3.5e-08")
            self.lineEdit_density.setEnabled(True)
            self.lineEdit_heat_capa.setEnabled(False)
            self.lineEdit_heat_capa.setVisible(False)
            self.label_density_2.setVisible(False)
        else:
            tooltip_1 = "Mass density used with alpha and c_p to compute thermal conductivity for Fourier's law."
            tooltip_2 = "Specific heat capacity used with alpha and density to compute thermal conductivity for Fourier's law."
            self.label_density.setText("Mass density rho [kg/m³]")
            self.label_density.setToolTip(tooltip_1)
            self.label_density_2.setText("Specific heat capacity c_p [J/(kg·K)]")
            self.label_density_2.setToolTip(tooltip_2)
            self.lineEdit_density.setToolTip(tooltip_1)
            self.lineEdit_heat_capa.setToolTip(tooltip_2)
            self.lineEdit_density.setEnabled(True)
            self.lineEdit_heat_capa.setEnabled(True)
            self.lineEdit_heat_capa.setVisible(True)
            self.label_density_2.setVisible(True)

    def update_flux_input_labels(self):
        if self.is_relative_flux_mode():
            tooltip = "Not used in relative mode. Relative flux is reported as a percent of the steady-state flux."
            self.label_density.setText("Not used in relative mode")
            self.label_density.setToolTip(tooltip)
            self.label_density_2.setText("Not used in relative mode")
            self.label_density_2.setToolTip(tooltip)
            self.lineEdit_density.setToolTip(tooltip)
            self.lineEdit_heat_capa.setToolTip(tooltip)
            self.lineEdit_density.setEnabled(False)
            self.lineEdit_heat_capa.setEnabled(False)
            self.lineEdit_heat_capa.setVisible(False)
            self.label_density_2.setVisible(False)
        elif self.get_selected_flux_mode() == "fick":
            tooltip_1 = (
                "EXPERIMENTAL FEATURE. Use this only for left-to-right permeation-style interpretation "
                "of the simple plot. Enter the left-side subsurface hydrogen concentration that corresponds "
                "to the fully filled normalized state u = 100, i.e. left boundary value 1.0. "
                "The code maps the normalized field to concentration with C = (u/100) x C_ref. "
                "Default: C_ref = 3.5 x 10^-8 mol/mm³, based on 5 ml/100 g Fe ≈ 4.5 wt ppm ≈ 3.5 x 10^-8 mol/mm³."
            )
            tooltip_2 = "Not used for hydrogen absolute mode. Alpha is treated as the hydrogen diffusivity D_H."
            self.label_density.setText("Left Side Subsurface Concentration [mol/mm³]")
            self.label_density.setToolTip(tooltip_1)
            self.label_density_2.setText("Not used in hydrogen absolute mode")
            self.label_density_2.setToolTip(tooltip_2)
            self.lineEdit_density.setToolTip(tooltip_1)
            self.lineEdit_heat_capa.setToolTip(tooltip_2)
            if self.lineEdit_density.text().strip() in {"", "100"}:
                self.lineEdit_density.setText("3.5e-08")
            self.lineEdit_density.setEnabled(True)
            self.lineEdit_heat_capa.setEnabled(False)
            self.lineEdit_heat_capa.setVisible(False)
            self.label_density_2.setVisible(False)
        else:
            tooltip_1 = "Mass density used with alpha and c_p to compute thermal conductivity for Fourier's law."
            tooltip_2 = "Specific heat capacity used with alpha and density to compute thermal conductivity for Fourier's law."
            self.label_density.setText("Mass density ρ [kg/m³]")
            self.label_density.setToolTip(tooltip_1)
            self.label_density_2.setText("Specific heat capacity cₚ [J/(kg·K)]")
            self.label_density_2.setToolTip(tooltip_2)
            self.lineEdit_density.setToolTip(tooltip_1)
            self.lineEdit_heat_capa.setToolTip(tooltip_2)
            self.lineEdit_density.setEnabled(True)
            self.lineEdit_heat_capa.setEnabled(True)
            self.lineEdit_heat_capa.setVisible(True)
            self.label_density_2.setVisible(True)

    def get_simple_mode_axis_label(self):
        if "Heat" in self.comboBox_mode_switch.currentText():
            return "Temperature [°C]"
        return "Hydrogen Concentration"

    def on_checkBox_anal_number_of_terms_changed(self):
        if self.checkBox_anal_number_of_terms.isChecked():
            self.lineEdit_anal_number_of_terms.setText("Calculate until Error < 1e-12")
            self.lineEdit_anal_number_of_terms.setEnabled(False)
        else:
            self.lineEdit_anal_number_of_terms.setText("100")
            self.lineEdit_anal_number_of_terms.setEnabled(True)

    def get_nth_part(self, input_string, n):
        try:
            parts = input_string.split(" ")
            return parts[n-1]
        except IndexError:
            return "Error"

    def calc_and_plot_to_tab_5(self):
        try:
            # Retrieve and validate all necessary values (remove the html formatting)
            rod_width = float(self.get_nth_part(
                self.label_width_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 2))
            alpha = float(self.get_nth_part(
                self.label_alpha_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 2))
            u0 = float(self.get_nth_part(
                self.label_left_boundary_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 3))
            uL = float(self.get_nth_part(
                self.label_right_boundary_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 3))
            time_to_show = float(self.get_nth_part(
                self.label_time_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 3))
            unit_to_show = str(self.get_nth_part(
                self.label_time_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 4))
            init_conc = float(self.get_nth_part(
                self.label_init_hydro_simple.text().replace('<html><head/><body><p align="center">', '').replace(
                    '</p></body></html>', ''), 3))
            plot_points = self.get_plot_point_count()

            show_flux = self.checkBox_show_flux.isChecked()
            show_min_value = self.checkBox_show_min_value.isChecked()
            flux_mode = self.get_selected_flux_mode()
            use_relative_flux = self.is_relative_flux_mode()
            if show_flux and not use_relative_flux:
                flux_factor_1 = float(self.lineEdit_density.text())
                if flux_mode == "fick":
                    flux_factor_1 = flux_factor_1 / 100.0
                    flux_factor_2 = 1.0
                else:
                    flux_factor_2 = float(self.lineEdit_heat_capa.text())
            else:
                flux_factor_1 = 1.0
                flux_factor_2 = 1.0

            display_time_on_graph = time_to_show

            if unit_to_show == "[s]":
                time_to_show = time_to_show
            elif unit_to_show == "[min]":
                time_to_show = time_to_show * 60
            elif unit_to_show == "[h]":
                time_to_show = time_to_show * 60 * 60
            elif unit_to_show == "[d]":
                time_to_show = time_to_show * 60 * 60 * 24
            elif unit_to_show == "[y]":
                time_to_show = time_to_show * 60 * 60 * 24 * 365.25
            values = [rod_width, alpha, u0, uL, time_to_show, init_conc, plot_points, False, flux_factor_1, flux_factor_2]

            # Get the number of terms to calculate
            if self.checkBox_anal_number_of_terms.isChecked():
                num_terms = "auto"  # use the default version of "auto" to achieve an error <= e-12
            else:
                try:
                    num_terms = int(self.safe_eval(self.lineEdit_anal_number_of_terms.text()))
                except ValueError:
                    raise InvalidInputError("Invalid input for number of Fourier series terms!")

            # If all values are valid, proceed to calculate and plot
            x, u, u_stable, error_estimate, _ = analytical_solution(*values, num_terms)

            # Clear the canvas and plot the new solution

            self.canvas5.axes.clear()
            self.canvas5.axes.plot(
                x, u,
                label=f'Time = {display_time_on_graph} {unit_to_show}',
                zorder=4,
                clip_on=False,
            )
            if self.checkBox_equilibrium.isChecked():
                self.canvas5.axes.plot(x, u_stable, label='Steady-State Solution', color='purple', zorder=3)
            self.canvas5.axes.legend()
            # Set the title
            title = f"1D analytical solution"
            self.canvas5.axes.set_title(title)
            y_upper = max(uL, u0, init_conc) * 1.05
            self.canvas5.axes.set_xlim(0, rod_width)
            self.canvas5.axes.set_ylim(0, y_upper)
            self.canvas5.axes.set_xlabel("Length L [mm]")
            self.canvas5.axes.set_ylabel(self.get_simple_mode_axis_label())
            self.canvas5.axes.grid(True)
            # Get the axis limits for the optional annotations below.
            ylim_min, ylim_max = self.canvas5.axes.get_ylim()
            xlim_min, xlim_max = self.canvas5.axes.get_xlim()

            if show_min_value:
                min_value = u.min()
                min_index = u.argmin()
                min_x = x[min_index]

                if (min_value - ylim_min) > (ylim_max - min_value):
                    arrow_pos_y = min_value - (ylim_max - ylim_min) * 0.15
                else:
                    arrow_pos_y = min_value + (ylim_max - ylim_min) * 0.15

                self.canvas5.axes.annotate(f"{min_value:.2f}",
                            xy=(min_x, min_value), xycoords='data',
                            xytext=(min_x, arrow_pos_y), textcoords='data',
                            arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=5, shrinkB=5,
                                            connectionstyle='arc3,rad=0.2'),
                            ha='center', va='bottom', fontsize=10)

            if show_flux:
                flux, flux_units, flux_name = calculate_boundary_flux(
                    x, u, alpha, flux_mode, flux_factor_1, flux_factor_2
                )
                if use_relative_flux:
                    steady_flux, _, _ = calculate_boundary_flux(
                        x, u_stable, alpha, flux_mode, flux_factor_1, flux_factor_2
                    )
                    if steady_flux is None or steady_flux == 0:
                        flux = np.nan
                    else:
                        flux = (flux / steady_flux) * 100
                    flux_units = "%"
                    if flux_mode == "fick":
                        flux_name = "Rel. Hydrogen Flux"
                    else:
                        flux_name = "Rel. Heat Flux"
            else:
                flux = None
                flux_units = None
                flux_name = None

            # Add the flux arrow
            if flux is not None:
                arrow_start_x = x[-1]
                arrow_start_y = u[-1]
                delta_u = u[-1] - u[-2]
                delta_x = x[-1] - x[-2]

                # Scaling factors to normalize the axes
                x_scale = (xlim_max - xlim_min)
                y_scale = (ylim_max - ylim_min)

                # Normalize delta_x and delta_u to the same scale
                norm_delta_x = delta_x / x_scale
                norm_delta_u = delta_u / y_scale

                # Desired arrow length
                arrow_length = 0.04  # Adjust arrow length

                # Calculate the length of the vector (delta_x, delta_u)
                vector_length = (norm_delta_x ** 2 + norm_delta_u ** 2) ** 0.5

                # Normalize the vector to unit length
                if vector_length != 0:  # Avoid division by zero
                    unit_delta_x = (delta_x * arrow_length) / vector_length
                    unit_delta_y = (delta_u * arrow_length) / vector_length
                else:
                    unit_delta_x = 0
                    unit_delta_y = arrow_length  # Arbitrary direction if vector_length is zero

                # Calculate end points for the arrow
                arrow_end_x = arrow_start_x + unit_delta_x
                arrow_end_y = arrow_start_y + unit_delta_y

                if flux < 0:
                    arrow_tail_x, arrow_tail_y = arrow_end_x, arrow_end_y
                    arrow_head_x, arrow_head_y = arrow_start_x, arrow_start_y
                else:
                    arrow_tail_x, arrow_tail_y = arrow_start_x, arrow_start_y
                    arrow_head_x, arrow_head_y = arrow_end_x, arrow_end_y
                label_x = xlim_max - (xlim_max - xlim_min) * -0.06
                label_y = ylim_min + (ylim_max - ylim_min) * -0.06
                formatted_flux = self.format_flux_value(flux, use_relative_flux)
                if formatted_flux == "NaN":
                    flux_annotation = f"{flux_name}: NaN"
                elif use_relative_flux:
                    flux_annotation = f"{flux_name} {formatted_flux}%"
                else:
                    flux_annotation = f"{flux_name}: {formatted_flux} {flux_units}"
                self.canvas5.axes.annotate(
                    "",
                    xy=(arrow_head_x, arrow_head_y), xycoords='data',
                    xytext=(arrow_tail_x, arrow_tail_y), textcoords='data',
                    arrowprops=dict(color='red', arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0),
                    annotation_clip=False,
                )
                self.canvas5.axes.text(
                    label_x, label_y, flux_annotation,
                    ha='right', va='bottom', fontsize=10, color='red'
                )
                self.canvas5.draw()
                return

                self.canvas5.axes.annotate(f"Flux: {flux:.1e} W/m²",
                                           xy=(arrow_start_x, arrow_start_y), xycoords='data',
                                           xytext=(arrow_end_x + text_offset_x, arrow_end_y + text_offset_y),
                                           textcoords='data',
                                           arrowprops=dict(facecolor='red', arrowstyle=arrowstyle, linewidth=2, shrinkA=0,
                                                           shrinkB=0),
                                           ha='center', va='bottom', fontsize=10, color='red')

            self.canvas5.draw()

        except (InvalidInputError, ValueError, TypeError) as e:
            self.textEdit_console_output.clear()
            if str(e):
                print(e)
            else:
                print("Invalid input")
            self.canvas5.axes.clear()
            self.canvas5.axes.text(0.5, 0.5, 'Invalid input', horizontalalignment='center', verticalalignment='center')
            self.canvas5.draw()

    def update_time_simple(self):
        try:
            min_time_simple = self.safe_eval(self.lineEdit_time_simple_min.text())
            max_time_simple = self.safe_eval(self.lineEdit_time_simple_max.text())
            slider_value = self.horizontalSlider_time_simple.value()
            value = np.interp(slider_value, [1, 100], [min_time_simple, max_time_simple])

            if self.radioButton_s.isChecked():
                unit = "[s]"
            elif self.radioButton_min.isChecked():
                unit = "[min]"
            elif self.radioButton_h.isChecked():
                unit = "[h]"
            elif self.radioButton_d.isChecked():
                unit = "[d]"
            elif self.radioButton_y.isChecked():
                unit = "[y]"

            if value == 0:
                value = f"{value:.0f}"
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                value = f"{value:.3e}"
            else:
                value = f"{value:.2f}"

            self.label_time_simple.setText(f"Time Displayed: {value} {unit}")
            self.label_time_simple.setAlignment(QtCore.Qt.AlignCenter)

            self.calc_and_plot_to_tab_5()

        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid Display Time Step input. Check Min/Max Values")
            self.label_time_simple.setText("Invalid input")
            self.label_time_simple.setAlignment(QtCore.Qt.AlignCenter)
        except TypeError:
            self.textEdit_console_output.clear()
            print("Type error encountered. Ensure input is numerical.")
            self.label_time_simple.setText("Invalid input")
            self.label_time_simple.setAlignment(QtCore.Qt.AlignCenter)

    def update_width_simple(self):
        try:
            min_width_simple = self.safe_eval(self.lineEdit_width_simple_min.text())
            max_width_simple = self.safe_eval(self.lineEdit_width_simple_max.text())
            slider_value = self.horizontalSlider_width_simple.value()
            value = np.interp(slider_value, [1, 100], [min_width_simple, max_width_simple])

            if value == 0:
                value = f"{value:.0f}"
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                value = f"{value:.3e}"
            else:
                value = f"{value:.2f}"

            self.label_width_simple.setText(f"Width: {value}")
            self.label_width_simple.setAlignment(QtCore.Qt.AlignCenter)

            self.calc_and_plot_to_tab_5()

        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid Width input. Check Min/Max Values")
            self.label_width_simple.setText("Invalid input")
            self.label_width_simple.setAlignment(QtCore.Qt.AlignCenter)
        except TypeError:
            self.textEdit_console_output.clear()
            print("Type error encountered. Ensure input is numerical.")
            self.label_width_simple.setText("Invalid input")
            self.label_width_simple.setAlignment(QtCore.Qt.AlignCenter)

    def update_alpha_simple(self):
        try:
            min_alpha_simple = self.safe_eval(self.lineEdit_alpha_simple_min.text())
            max_alpha_simple = self.safe_eval(self.lineEdit_alpha_simple_max.text())
            slider_value = self.horizontalSlider_alpha_simple.value()
            value = np.interp(slider_value, [1, 100], [min_alpha_simple, max_alpha_simple])

            if value == 0:
                value = f"{value:.0f}"
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                value = f"{value:.3e}"
            else:
                value = f"{value:.2f}"

            self.label_alpha_simple.setText(f"Alpha: {value} [mm²/s]")
            self.label_alpha_simple.setAlignment(QtCore.Qt.AlignCenter)

            self.calc_and_plot_to_tab_5()

        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid Alpha input. Check Min/Max Values")
            self.label_alpha_simple.setText("Invalid input")
            self.label_alpha_simple.setAlignment(QtCore.Qt.AlignCenter)
        except TypeError:
            self.textEdit_console_output.clear()
            print("Type error encountered. Ensure input is numerical.")
            self.label_alpha_simple.setText("Invalid input")
            self.label_alpha_simple.setAlignment(QtCore.Qt.AlignCenter)

    def update_init_hydro_simple(self):
        try:
            min_init_hydro_simple = self.safe_eval(self.lineEdit_init_hydro_simple_min.text())
            max_init_hydro_simple = self.safe_eval(self.lineEdit_init_hydro_simple_max.text())
            slider_value = self.horizontalSlider_init_hydro_simple.value()
            value = np.interp(slider_value, [1, 100], [min_init_hydro_simple, max_init_hydro_simple])

            if value == 0:
                value = f"{value:.0f}"
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                value = f"{value:.3e}"
            else:
                value = f"{value:.2f}"

            self.label_init_hydro_simple.setText(f"Initial Hydro: {value}")
            self.label_init_hydro_simple.setAlignment(QtCore.Qt.AlignCenter)

            self.calc_and_plot_to_tab_5()

        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid Initial Hydro input. Check Min/Max Values")
            self.label_init_hydro_simple.setText("Invalid input")
            self.label_init_hydro_simple.setAlignment(QtCore.Qt.AlignCenter)
        except TypeError:
            self.textEdit_console_output.clear()
            print("Type error encountered. Ensure input is numerical.")
            self.label_init_hydro_simple.setText("Invalid input")
            self.label_init_hydro_simple.setAlignment(QtCore.Qt.AlignCenter)

    def update_left_boundary_simple(self):
        try:
            min_left_boundary_simple = self.safe_eval(self.lineEdit_left_boundary_simple_min.text())
            max_left_boundary_simple = self.safe_eval(self.lineEdit_left_boundary_simple_max.text())
            slider_value = self.horizontalSlider_left_boundary_simple.value()
            value = np.interp(slider_value, [1, 100], [min_left_boundary_simple, max_left_boundary_simple])

            if value == 0:
                value = f"{value:.0f}"
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                value = f"{value:.3e}"
            else:
                value = f"{value:.2f}"

            self.label_left_boundary_simple.setText(f"Left Boundary: {value}")
            self.label_left_boundary_simple.setAlignment(QtCore.Qt.AlignCenter)

            self.calc_and_plot_to_tab_5()

        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid Left Boundary input. Check Min/Max Values")
            self.label_left_boundary_simple.setText("Invalid input")
            self.label_left_boundary_simple.setAlignment(QtCore.Qt.AlignCenter)
        except TypeError:
            self.textEdit_console_output.clear()
            print("Type error encountered. Ensure input is numerical.")
            self.label_left_boundary_simple.setText("Invalid input")
            self.label_left_boundary_simple.setAlignment(QtCore.Qt.AlignCenter)

    def update_right_boundary_simple(self):
        try:
            min_right_boundary_simple = self.safe_eval(self.lineEdit_right_boundary_simple_min.text())
            max_right_boundary_simple = self.safe_eval(self.lineEdit_right_boundary_simple_max.text())
            slider_value = self.horizontalSlider_right_boundary_simple.value()
            value = np.interp(slider_value, [1, 100], [min_right_boundary_simple, max_right_boundary_simple])

            if value == 0:
                value = f"{value:.0f}"
            elif abs(value) < 1e-3 or abs(value) > 1e3:
                value = f"{value:.3e}"
            else:
                value = f"{value:.2f}"

            self.label_right_boundary_simple.setText(f"Right Boundary: {value}")
            self.label_right_boundary_simple.setAlignment(QtCore.Qt.AlignCenter)

            self.calc_and_plot_to_tab_5()

        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid Right Boundary input. Check Min/Max Values")
            self.label_right_boundary_simple.setText("Invalid input")
            self.label_right_boundary_simple.setAlignment(QtCore.Qt.AlignCenter)
        except TypeError:
            self.textEdit_console_output.clear()
            print("Type error encountered. Ensure input is numerical.")
            self.label_right_boundary_simple.setText("Invalid input")
            self.label_right_boundary_simple.setAlignment(QtCore.Qt.AlignCenter)

    def generate_dynamic_tooltip(self):
        # Logic to generate the dynamic tooltip text
        high_left = float(self.lineEdit_boundary_left.text())
        high_right = float(self.lineEdit_boundary_right.text())
        width = float(self.lineEdit_sample_width.text())
        dx = str(self.lineEdit_discret_space.text())
        dt = float(self.lineEdit_discret_time.text())
        return f"This assumes a plate with width = {width} and does a 1D calculation of the hydrogen distribution. " \
               f"BC are left = {high_left} and right = {high_right}. This tab assumes infinite height so Neumann BC " \
               f"for top and bottom. \nAll dx = {dx} are used and dt = {dt}.\nFor almost all shapes and dimensions " \
               f"1D is fine. e.i. plates, cubes, spheres, torruses etc."

    def connect_signals_to_update_settings(self, parent):
        self.sliders = [
            self.horizontalSlider_fixed_variable_alpha,
            self.horizontalSlider_time_simple,
            self.horizontalSlider_width_simple,
            self.horizontalSlider_alpha_simple,
            self.horizontalSlider_init_hydro_simple,
            self.horizontalSlider_left_boundary_simple,
            self.horizontalSlider_right_boundary_simple
        ]

        for widget in parent.findChildren(QtWidgets.QWidget):
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.textChanged.connect(self.update_settings)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.stateChanged.connect(self.update_settings)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self.update_settings)
            elif isinstance(widget, QtWidgets.QRadioButton):
                widget.toggled.connect(self.update_settings)
            elif isinstance(widget, CustomLabel):
                widget.textChanged.connect(self.update_settings)
            elif isinstance(widget, QtWidgets.QSlider) and widget in self.sliders:
                widget.valueChanged.connect(self.update_settings)

    def get_settings_path(self):
        if IS_FROZEN:
            return ensure_frozen_settings_seeded()
        return SOURCE_SETTINGS_PATH

    def setup_settings(self):
        self.settings_path = self.get_settings_path()
        if not os.path.exists(self.settings_path):
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)  # Format JSON with indentation

    def load_settings(self):
        with open(self.settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        self.sliders = [
            self.horizontalSlider_fixed_variable_alpha,
            self.horizontalSlider_time_simple,
            self.horizontalSlider_width_simple,
            self.horizontalSlider_alpha_simple,
            self.horizontalSlider_init_hydro_simple,
            self.horizontalSlider_left_boundary_simple,
            self.horizontalSlider_right_boundary_simple
        ]

        for widget in self.findChildren(QtWidgets.QWidget):
            name = widget.objectName()
            if name in settings:
                if isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(settings[name]["value"])
                    widget.setEnabled(settings[name]["enabled"])
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(settings[name]["value"])
                    widget.setEnabled(settings[name]["enabled"])
                elif isinstance(widget, QtWidgets.QComboBox):
                    widget.setCurrentIndex(settings[name]["value"])
                    widget.setEnabled(settings[name]["enabled"])
                elif isinstance(widget, QtWidgets.QRadioButton):
                    widget.setChecked(settings[name]["value"])
                    widget.setEnabled(settings[name]["enabled"])
                elif isinstance(widget, CustomLabel):
                    widget.setText(settings[name]["value"])
                    widget.setEnabled(settings[name]["enabled"])
                elif isinstance(widget, QtWidgets.QSlider) and widget in self.sliders:
                    widget.setValue(settings[name]["value"])
                    widget.setEnabled(settings[name]["enabled"])

        # Ensure the state of lineEdit_discret_time is correct on load
        self.toggle_discret_time()
        # Ensure the state of tab visibility is correct on load
        self.toggle_tab_visibility()
        # Check if you need to mark the dt in red
        self.check_if_dt_is_ok()

    def update_settings(self):
        settings = {}

        self.sliders = [
            self.horizontalSlider_fixed_variable_alpha,
            self.horizontalSlider_time_simple,
            self.horizontalSlider_width_simple,
            self.horizontalSlider_alpha_simple,
            self.horizontalSlider_init_hydro_simple,
            self.horizontalSlider_left_boundary_simple,
            self.horizontalSlider_right_boundary_simple
        ]

        for widget in self.findChildren(QtWidgets.QWidget):
            name = widget.objectName()
            if isinstance(widget, QtWidgets.QLineEdit):
                settings[name] = {"value": widget.text(), "enabled": widget.isEnabled()}
            elif isinstance(widget, QtWidgets.QCheckBox):
                settings[name] = {"value": widget.isChecked(), "enabled": widget.isEnabled()}
            elif isinstance(widget, QtWidgets.QComboBox):
                settings[name] = {"value": widget.currentIndex(), "enabled": widget.isEnabled()}
            elif isinstance(widget, QtWidgets.QRadioButton):
                settings[name] = {"value": widget.isChecked(), "enabled": widget.isEnabled()}
            elif isinstance(widget, CustomLabel):
                settings[name] = {"value": widget.text(), "enabled": widget.isEnabled()}
            elif isinstance(widget, QtWidgets.QSlider) and widget in self.sliders:
                settings[name] = {"value": widget.value(), "enabled": widget.isEnabled()}
            # Add more widget types as needed

        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)  # Format JSON with indentation

    def on_sample_width_changed(self):
        self.update_drawing()

    def on_sample_height_changed(self):
        self.update_drawing()

    def on_lineedit_diff_coeff_changed(self):
        # Only recalculate dt if the use auto_dt_checkbox is checked
        if self.checkBox_discret_time.isChecked():
            self.calculate_dt_for_ui()

        self.check_if_dt_is_ok()

    def on_discret_space_changed(self):
        self.update_drawing()

        # Only recalculate dt if the use auto_dt_checkbox is checked
        if self.checkBox_discret_time.isChecked():
            self.calculate_dt_for_ui()

        self.check_if_dt_is_ok()

    def on_lineedit_discret_time_changed(self):
        self.check_if_dt_is_ok()

    def on_checkbox_discret_time_changed(self):
        # Calculate the biggest stable time step
        self.calculate_dt_for_ui()
        # Check if its ok (should change background back to white)
        self.check_if_dt_is_ok()
        # Enable/Disable the manual dt input
        self.toggle_discret_time()  # Enable or disable lineEdit_discret_time based on checkbox state

    def on_checkbox_incl_animation_changed(self):
        self.toggle_tab_visibility()

    def on_start_all_sims_button_clicked(self):
        print("Starting 2D Simulation")

        # Manipulate the Buttons so you can stop the simulation
        self.pushButton_start_all_sims.setEnabled(False)
        self.pushButton_start_all_sims.setVisible(False)
        self.pushButton_stop_all_sims.setEnabled(True)
        self.pushButton_stop_all_sims.setVisible(True)

        # 1D Stuff
        height = float(self.lineEdit_sample_height.text())
        width = float(self.lineEdit_sample_width.text())

        # Use the alpha given or the higher alpha
        # maybe lower instead? Not sure but dont vary for 1D numerical accuracy calc
        if self.current_alpha_option == 0:
            alpha = float(self.lineEdit_diff_coeff.text())
        else:
            alpha = max(float(self.lineEdit_alpha_ramp_start.text()),
                        float(self.lineEdit_alpha_ramp_end_alpha.text()))

        u0 = float(self.lineEdit_boundary_left.text())
        uL = float(self.lineEdit_boundary_right.text())
        dt = float(self.lineEdit_discret_time.text())
        t = self.safe_eval(self.lineEdit_sim_time.text())
        init_conc = float(self.lineEdit_hydro_conc_init.text())
        self.textEdit_console_output.clear()
        dx_values = self.get_dx_values()  # Get the list of dx values
        max_dx = max(dx_values)
        min_dim = min(height, width)
        if min_dim / max_dx <= 3:
            QMessageBox.critical(self, "Error", "Discretization is too coarse. Consider higher discretization.")
            return  # Exit the method gracefully
        elif 3 < min_dim / max_dx < 10:
            reply = QMessageBox.warning(self, "Warning", "Unexpectedly large dx|dy detected. Consider higher "
                                                         "discretization.\nDo you want to continue?", QMessageBox.Yes |
                                        QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return  # Exit the method gracefully if the user chooses not to continue

        self.worker = Worker(width, alpha, u0, uL, dt, t, init_conc, dx_values)
        self.worker.progress.connect(self.update_console_output)
        self.worker.result.connect(self.plot_hydrogen_distribution)
        self.worker.result.connect(self.populate_tab4)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(self.handle_worker_error)
        self.worker.start()

        # 2D Stuff
        if self.checkBox_incl_animation.isChecked():
            dx = dx_values
            dim_x = float(self.lineEdit_sample_width.text())
            dim_y = float(self.lineEdit_sample_height.text())
            sim_time = t
            init_hydrogen_conc = init_conc
            # REMEMBER!! Top and Bottom is switched cuz python counts from top to bottom (and left to right)
            border_hydrogen = [float(self.lineEdit_boundary_bottom.text()), float(self.lineEdit_boundary_top.text()),
                               float(self.lineEdit_boundary_left.text()), float(self.lineEdit_boundary_right.text())]
            border_bc = [self.checkBox_neumann_bottom.isChecked(), self.checkBox_neumann_top.isChecked(),
                         self.checkBox_neumann_left.isChecked(), self.checkBox_neumann_right.isChecked()]

            if self.current_alpha_option == 0:
                diffusion_coefficient = alpha
            else:
                alpha_0 = float(self.lineEdit_alpha_ramp_start.text())
                alpha_1 = float(self.lineEdit_alpha_ramp_end_alpha.text())
                alpha_end_time = self.safe_eval(self.lineEdit_alpha_ramp_end_time.text())
                diffusion_coefficient = [alpha_0, alpha_1, alpha_end_time]

            warm_up_time = self.safe_eval(self.lineEdit_warm_up.text())
            k_value = self.safe_eval(self.lineEdit_k_value.text())

            save_every_x_s = float(self.lineEdit_save_sim_frequency.text())
            file_name = resolve_simulation_file_path(self.lineEdit_file_name.text())
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            self.worker2d = Worker2D(dx, dim_x, dim_y, sim_time, init_hydrogen_conc, border_hydrogen, border_bc,
                                     diffusion_coefficient, warm_up_time, k_value, save_every_x_s, file_name)
            self.worker2d.progress.connect(self.update_console_output)
            self.worker2d.result.connect(self.plot_hydrogen_distribution_2d)
            self.worker2d.finished.connect(self.on_worker_finished_2d)
            self.worker2d.error_occurred.connect(self.handle_worker_error)
            self.worker2d.start()

    def populate_tab4(self, result):
        x_analytical, u_analytical, _, x_numerical_list, u_numerical_list, dx_values, _, _ = result

        # Get the set of all unique x points from all numerical solutions
        all_x_points = sorted(set(x for x_numerical in x_numerical_list for x in x_numerical))

        # Set the number of rows based on the total number of unique x points
        self.tableWidget_tab4.setRowCount(len(all_x_points))

        # Set the number of columns: analytical, one for each dx, and difference
        num_columns = 2 + len(dx_values) + 1
        self.tableWidget_tab4.setColumnCount(num_columns)

        # Create a dictionary to store u_analytical values at each x point
        u_analytical_dict = {x: np.interp(x, x_analytical, u_analytical) for x in all_x_points}

        # Populate the table
        for row, x_val in enumerate(all_x_points):
            self.tableWidget_tab4.setItem(row, 0, QTableWidgetItem(f"{x_val:.8e}"))
            self.tableWidget_tab4.setItem(row, 1, QTableWidgetItem(f"{u_analytical_dict[x_val]:.8e}"))

            for col, (dx, x_numerical, u_numerical) in enumerate(zip(dx_values, x_numerical_list, u_numerical_list),
                                                                 start=2):
                if x_val in x_numerical:
                    index = np.where(x_numerical == x_val)[0][0]
                    u_num = u_numerical[index]
                    self.tableWidget_tab4.setItem(row, col, QTableWidgetItem(f"{u_num:.8e}"))
                else:
                    self.tableWidget_tab4.setItem(row, col, QTableWidgetItem(""))

            # Calculate the difference for the smallest dx
            min_dx_index = dx_values.index(min(dx_values))
            if x_val in x_numerical_list[min_dx_index]:
                index_min_dx = np.where(x_numerical_list[min_dx_index] == x_val)[0][0]
                u_num_min_dx = u_numerical_list[min_dx_index][index_min_dx]
                self.tableWidget_tab4.setItem(row, len(dx_values) + 2,
                                              QTableWidgetItem(f"{u_analytical_dict[x_val] - u_num_min_dx:.8e}"))

        # Set the column headers
        headers = ["x", "u_analytical"] + [f"u_numerical (dx={dx})" for dx in dx_values] + ["Difference (smallest dx)"]
        self.tableWidget_tab4.setHorizontalHeaderLabels(headers)

        header = self.tableWidget_tab4.horizontalHeader()
        for column in range(num_columns):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.Stretch)

    def on_stop_all_sims_button_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        if self.worker2d and self.worker2d.isRunning():
            self.worker2d.stop()
        self.pushButton_stop_all_sims.setEnabled(False)
        self.pushButton_stop_all_sims.setVisible(False)
        self.pushButton_start_all_sims.setVisible(True)
        self.pushButton_start_all_sims.setEnabled(True)

    def on_worker_finished(self):
        if not self.worker2d or not self.worker2d.isRunning():  # change the button if worker2d isn't running
            self.pushButton_stop_all_sims.setEnabled(False)
            self.pushButton_stop_all_sims.setVisible(False)
            self.pushButton_start_all_sims.setVisible(True)
            self.pushButton_start_all_sims.setEnabled(True)

    def on_worker_finished_2d(self):
        if not self.worker or not self.worker.isRunning():  # change the button if worker isn't running
            self.pushButton_stop_all_sims.setEnabled(False)
            self.pushButton_stop_all_sims.setVisible(False)
            self.pushButton_start_all_sims.setVisible(True)
            self.pushButton_start_all_sims.setEnabled(True)


    def safe_eval(self, expression):
        try:
            result = eval(expression, {"__builtins__": None}, {})
            return float(result)
        except:
            raise ValueError("Invalid input")

    def get_min_discret_space(self):
        try:
            # Get the text from the QLineEdit
            discret_space_text = self.lineEdit_discret_space.text()
            # Split the text by commas and convert each part to a float
            dx_values = [float(x.strip()) for x in discret_space_text.split(',')]
            # Find the minimum value
            dx = min(dx_values)
            # This weirdnes here with errors and dx = 1e-15 and output.clear() is cuz when people type in values
            self.textEdit_console_output.clear()
            if dx == 0:
                raise ValueError
            return dx
        except ValueError:
            self.textEdit_console_output.clear()
            print("Invalid input. Please enter valid numerical values separated by commas.")
            dx = 1e-15
            return dx

    def get_dx_values(self):
        try:
            # Get the text from the QLineEdit
            discret_space_text = self.lineEdit_discret_space.text()
            # Split the text by commas and convert each part to a float
            dx_values = [float(x.strip()) for x in discret_space_text.split(',')]
            return dx_values
        except ValueError:
            print("Invalid input. Please enter valid numerical values separated by commas.")
            return []

    def check_if_dt_is_ok(self):
        # check if the dt they put in can even be stable, warn if not
        try:
            dx = self.get_min_discret_space()  # special function to allow multipe dx!
            user_dt = float(self.lineEdit_discret_time.text())
            self.current_alpha_option = self.horizontalSlider_fixed_variable_alpha.value()

            if self.current_alpha_option == 0:
                alpha = float(self.lineEdit_diff_coeff.text())
            else:
                alpha = max(float(self.lineEdit_alpha_ramp_start.text()),
                            float(self.lineEdit_alpha_ramp_end_alpha.text()))

            dt = 0.5 * dx ** 2 / alpha  # Calculate the maximum stable time step size

            # Format dt to the same precision as user_dt
            dt_str = f"{dt:.8e}"
            formatted_dt = float(dt_str)

            if user_dt > formatted_dt:
                self.lineEdit_discret_time.setStyleSheet("background-color: red;")
                self.lineEdit_discret_time.setToolTip("WARNING: Custom dt larger than stable dt! Numerical solutions "
                                                      "highly likely to not be stable" )
            else:
                self.lineEdit_discret_time.setStyleSheet("")  # Reverts to the default style
                self.lineEdit_discret_time.setToolTip("Set custom dt or use "
                                                      "largest stable dt based on alpha and dx. (Using a custom dt "
                                                      "larger than the automatic dt will likely result in divergence "
                                                      "and an unstable numeric solution)")
        except ValueError:
            dt = 0
            self.lineEdit_discret_time.setText("Invalid input")  # Handle invalid input

    def calculate_dt_for_ui(self):
        try:
            dx = self.get_min_discret_space()  # special function to allow multipe dx!

            if self.current_alpha_option == 0:
                alpha = float(self.lineEdit_diff_coeff.text())
            else:
                alpha = max(float(self.lineEdit_alpha_ramp_start.text()),
                            float(self.lineEdit_alpha_ramp_end_alpha.text()))

            dt = 0.5 * dx ** 2 / alpha  # Calculate the maximum stable time step size
            self.lineEdit_discret_time.setText(f"{dt:.8e}")  # Set the calculated dt value in the lineEdit

        except ValueError:
            self.lineEdit_discret_time.setText("Invalid input")  # Handle invalid input

    def toggle_discret_time(self):
        if self.checkBox_discret_time.isChecked():
            self.lineEdit_discret_time.setEnabled(False)
        else:
            self.lineEdit_discret_time.setEnabled(True)

    def on_tabwidget_settings_changed(self, index):
        self.toggle_tab_visibility()

    def toggle_tab_visibility(self):

        # Hides/shows the animation tab if "include animation" is off/on
        if self.checkBox_incl_animation.isChecked():
            self.tabWidget.setTabVisible(1, True)
        else:
            self.tabWidget.setTabVisible(1, False)

        # Hides/Shows the display tabs according to which "mode tab" is currently used

        if self.tabWidget_Settings.currentIndex() == 0:
            self.tabWidget.setTabVisible(0, True)
            if self.checkBox_incl_animation.isChecked():
                self.tabWidget.setTabVisible(1, True)
            else:
                self.tabWidget.setTabVisible(1, False)
            self.tabWidget.setTabVisible(2, True)
            self.tabWidget.setTabVisible(3, True)
            self.tabWidget.setTabVisible(4, False)
            self.tabWidget.setTabVisible(5, False)
            self.tabWidget.setCurrentIndex(0)  # Show Mesh Tab

        if self.tabWidget_Settings.currentIndex() == 1:
            self.tabWidget.setTabVisible(0, False)
            self.tabWidget.setTabVisible(1, False)
            self.tabWidget.setTabVisible(2, False)
            self.tabWidget.setTabVisible(3, False)
            self.tabWidget.setTabVisible(4, True)
            self.tabWidget.setTabVisible(5, False)
            self.tabWidget.setCurrentIndex(4)  # Show Mesh Tab

    def update_drawing(self):
        try:
            width = float(self.lineEdit_sample_width.text())
            height = float(self.lineEdit_sample_height.text())
            discret_space = self.get_min_discret_space()  # special function to allow multipe dx!

            if width <= 0 or height <= 0 or discret_space <= 0:
                raise ValueError("Dimensions and discretization space must be greater than zero.")
        except ValueError:
            self.canvas.axes.clear()
            self.canvas.axes.text(0.5, 0.5, 'Invalid input', horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        # Set a threshold for the maximum number of lines to draw
        max_lines = 250

        # Calculate the number of lines based on the user input
        num_lines_x = width / discret_space
        num_lines_y = height / discret_space

        # Adjust the discretization space if the number of lines exceeds the threshold
        if num_lines_x > max_lines or num_lines_y > max_lines:
            draw_discret_space = max(width, height) / max_lines
        else:
            draw_discret_space = discret_space

        self.canvas.axes.clear()

        # Draw the rectangle
        rect = plt.Rectangle((0, 0), width, height, linewidth=1, edgecolor='black', facecolor='darkgreen')
        self.canvas.axes.add_patch(rect)

        # Draw the grid using LineCollection
        lines = []
        x = 0
        while x <= width:
            lines.append([(x, 0), (x, height)])
            x += draw_discret_space

        y = 0
        while y <= height:
            lines.append([(0, y), (width, y)])
            y += draw_discret_space

        line_collection = LineCollection(lines, colors='black', linewidths=0.5)
        self.canvas.axes.add_collection(line_collection)

        # Set the title
        title = f"Dimensions: {width}mm x {height}mm, Mesh size: {discret_space}mm"
        self.canvas.axes.set_title(title)

        # Add a warning box if the discretization space was adjusted
        if draw_discret_space != discret_space:
            warning_text = f"Display Limit: Displaying with {draw_discret_space:.2f}mm instead of {discret_space}mm"
            self.canvas.axes.text(
                1, -0.1, warning_text,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=self.canvas.axes.transAxes,
                fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        self.canvas.axes.set_xlim(0, width)
        self.canvas.axes.set_ylim(0, height)
        self.canvas.axes.set_aspect('equal', adjustable='box')  # Ensure correct aspect ratio
        self.canvas.draw()

    def plot_hydrogen_distribution(self, result):
        x_analytical, u_analytical, error_estimate, x_numerical_list, u_numerical_list, dx_values, dt_values, errors = result

        self.canvas3.axes.clear()

        # ---- Format time ----
        time_s = self.safe_eval(self.lineEdit_sim_time.text())
        if time_s >= 31557600:
            years = time_s / 31557600
            formatted_years = f"{years:.1f}"
            display_time = str(formatted_years) + " years."
        elif time_s >= 864000:
            days = time_s / 86400
            formatted_days = f"{days:.1f}"
            display_time = str(formatted_days) + " days."
        elif time_s >= 36000:
            hours = time_s / 3600
            formatted_hours = f"{hours:.1f}"
            display_time = str(formatted_hours) + " hours."
        elif time_s >= 60:
            minutes = time_s / 60
            formatted_minutes = f"{minutes:.1f}"
            display_time = str(formatted_minutes) + " min."
        elif time_s <= 1:
            ms = time_s / 1000
            formatted_ms = f"{ms:.1f}"
            display_time = str(formatted_ms) + " ms."
        else:
            formatted_s = f"{time_s:.1f}"
            display_time = str(formatted_s) + " s."

        self.canvas3.axes.plot(x_analytical, u_analytical, label='Analytical Solution')
        for x_numerical, u_numerical, dx, dt, error in zip(x_numerical_list, u_numerical_list, dx_values, dt_values,
                                                           errors):
            self.canvas3.axes.plot(x_numerical, u_numerical, 'o',
                                  label=f'Numerical Solution (dx = {dx}, dt = {dt:.2e}, MAE = {error:.2e})')
        self.canvas3.axes.set_xlabel('Position (mm)')
        self.canvas3.axes.set_ylabel('Concentration')
        self.canvas3.axes.set_title(f'Numerical: concentration distribution in the Rod at t = {display_time}\n(Analytical Error Estimate: {error_estimate:.2e})')

        # Set axis limits here. Its annoying if it scales y automatically
        high_left = float(self.lineEdit_boundary_left.text())
        high_right = float(self.lineEdit_boundary_right.text())
        high_top = float(self.lineEdit_boundary_top.text())
        high_bottom = float(self.lineEdit_boundary_bottom.text())
        high_init_conc = float(self.lineEdit_hydro_conc_init.text())
        highest_value = max(high_left, high_right, high_top, high_bottom, high_init_conc)
        self.canvas3.axes.set_ylim(0, highest_value + highest_value/100)  # Set y-axis limits here

        self.canvas3.axes.legend()
        self.canvas3.axes.grid(True)
        self.canvas3.draw()
        self.tabWidget.setCurrentIndex(2)

    def plot_hydrogen_distribution_2d(self, result):
        self.load_data()
        self.tabWidget.setCurrentIndex(1)  # activate the animation tab


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        error_message = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        error_handler.show_error_signal.emit(error_message)
