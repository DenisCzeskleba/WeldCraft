"""P3 Heat Map GUI and command-line entrypoint."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
RESOURCES_DIR = REPO_ROOT / "Resources"
if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Common.launch_ready import StartupReadySignal
from functions import (
    ConfigError,
    RESULTS_DIR,
    SimulationCancelled,
    build_initial_fields,
    compute_derived,
    create_result_view,
    ensure_config_file,
    load_settings,
    load_snapshots,
    preview_grid_info,
    render_animation,
    restore_default_config,
    run_simulation,
    update_result_view,
    validate_settings,
    write_settings,
)


APP_NAME = "WeldCraft - Heat Map"


FIELD_GROUPS = [
    (
        "Geometry",
        [
            ("le", "Left base [mm]", "float", False),
            ("ri", "Right base [mm]", "float", False),
            ("we", "Weld width [mm]", "float", False),
            ("th", "Base height [mm]", "float", False),
            ("su_h", "Support height [mm]", "float", False),
            ("su_w", "Support width [mm]", "float", False),
            ("fr_ab", "Free space above [mm]", "float", True),
            ("fr_be", "Free space below [mm]", "float", True),
            ("weld_bead_thickness", "Bead thickness [mm]", "float", False),
            ("dx", "Mesh spacing dx [mm]", "float", True),
            ("dy", "Mesh spacing dy [mm]", "float", True),
        ],
    ),
    (
        "Temperature and material",
        [
            ("t_cool", "Interpass temperature [°C]", "float", False),
            ("t_hot", "Initial weld temperature [°C]", "float", False),
            ("t_room", "Room temperature [°C]", "float", False),
            ("diff_coeff_bm", "Base-metal diffusivity [mm²/s]", "float", True),
            ("diff_coeff_wm", "Weld-metal diffusivity [mm²/s]", "float", True),
            ("diff_coeff_haz", "HAZ diffusivity [mm²/s]", "float", True),
            ("diff_coeff_air", "Air diffusivity [mm²/s]", "float", True),
            ("c", "Specific heat capacity", "float", True),
            ("rho", "Density", "float", True),
            ("conv_variable", "Cooling coefficient", "float", False),
        ],
    ),
    (
        "Weld motion",
        [
            ("weld_length", "Weld length [mm]", "float", False),
            ("weld_speed", "Weld speed [mm/min]", "float", False),
            ("weld_temp", "Heat-source temperature [°C]", "float", False),
            ("weld_spot_size", "Spot size [mm]", "float", False),
            ("time_before_weld_start", "Pre-weld delay [s]", "float", False),
        ],
    ),
    (
        "Simulation and output",
        [
            ("sim_time", "Simulation time [s]", "float", False),
            ("save_so_often_per_sec", "Save rate [snapshots/s]", "float", False),
            ("slow_down_beginning", "Extra early snapshots", "bool", False),
            ("h5_filename", "HDF5 file", "filename", False),
            ("animation_filename", "MP4 file", "filename", False),
            ("figure_filename", "Figure file", "filename", False),
            ("disable_overwrite_warning", "Disable overwrite warning", "bool", False),
        ],
    ),
    (
        "Display",
        [
            ("heatmap_style", "Color map", "choice", False),
            ("heatmap_vmin", "Minimum [°C]", "float", False),
            ("heatmap_vmax", "Maximum [°C]", "float", False),
            ("show_contours", "Show contours", "bool", False),
            ("contour_levels", "Contour levels [°C]", "list_float", False),
            ("show_monitoring_points", "Show monitoring points", "bool", False),
            ("show_mesh_lines", "Show mesh", "bool", False),
            ("monitoring_distances", "Monitoring distances [mm]", "list_float", False),
            ("monitoring_y_offset", "Monitoring Y offset [mm]", "float", True),
            ("weld_zoom_margin", "Weld zoom margin [mm]", "float", False),
            ("animation_fps", "Animation FPS", "int", True),
            ("animation_dpi", "Animation DPI", "int", True),
            ("animation_frame_stride", "Animation frame stride", "int", True),
            ("use_boundary_adjustment", "Recalculate boundary mask while welding", "bool", True),
        ],
    ),
]


class SimulationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings, output_path, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.output_path = output_path
        self.stop_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        try:
            result = run_simulation(
                self.settings,
                self.output_path,
                progress_callback=lambda fraction, message, _time: self.progress.emit(fraction, message),
                stop_event=self.stop_event,
            )
            self.completed.emit(str(result))
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class AnimationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings, loaded_data, output_path, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.loaded_data = loaded_data
        self.output_path = output_path

    def run(self):
        try:
            result = render_animation(
                self.settings,
                self.loaded_data,
                self.output_path,
                progress_callback=lambda fraction, message, _frame: self.progress.emit(fraction, message),
            )
            self.completed.emit(str(result))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.field_widgets = {}
        self.field_rows = []
        self.loading_widgets = True
        self.simulation_worker = None
        self.animation_worker = None
        self.loaded_data = None
        self.result_view = None
        self._pending_frame = None
        self._frame_update_timer = QtCore.QTimer(self)
        self._frame_update_timer.setSingleShot(True)
        self._frame_update_timer.setInterval(25)
        self._frame_update_timer.timeout.connect(self._apply_pending_frame)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QtGui.QIcon(str(REPO_ROOT / "Resources" / "Images" / "WeldCraft.ico")))
        self.resize(1500, 920)
        self._build_ui()
        self._populate_widgets()
        self.loading_widgets = False
        self._connect_widget_signals()
        self._update_preview()
        self.startup_ready_signal = StartupReadySignal(self)
        self.showMaximized()

    def _build_ui(self):
        self.setStyleSheet(
            """
            QGroupBox { font-weight: bold; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLineEdit, QComboBox { min-height: 24px; }
            QPushButton { min-height: 28px; padding: 2px 10px; }
            """
        )
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(splitter)

        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(380)
        settings_panel = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_panel)
        self.advanced_toggle = QtWidgets.QCheckBox("Show advanced settings")
        self.advanced_toggle.setToolTip("Reveal mesh, numerical, boundary, and rendering controls.")
        self.bam_logo_label = QtWidgets.QLabel()
        self.bam_logo_label.setAlignment(QtCore.Qt.AlignCenter)
        self.bam_logo_label.setMaximumHeight(60)
        bam_logo_path = REPO_ROOT / "Resources" / "Images" / "BAM Logo.png"
        if bam_logo_path.exists():
            self.bam_logo_label.setPixmap(QtGui.QPixmap(str(bam_logo_path)))
        settings_layout.addWidget(self.bam_logo_label)
        settings_layout.addWidget(self.advanced_toggle)

        for group_name, fields in FIELD_GROUPS:
            group = QtWidgets.QGroupBox(group_name)
            group_layout = QtWidgets.QVBoxLayout(group)
            group_layout.setContentsMargins(8, 6, 8, 8)
            group_layout.setSpacing(5)
            for name, label_text, kind, advanced in fields:
                label = QtWidgets.QLabel(label_text)
                label.setToolTip(self._tooltip_for(name, "label"))
                label.setWordWrap(True)
                label.setMinimumWidth(140)
                label.setMaximumWidth(220)
                widget = self._make_widget(kind)
                widget.setToolTip(self._tooltip_for(name, "input"))
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                row_layout.addWidget(label)
                row_layout.addWidget(widget, 1)
                group_layout.addWidget(row)
                self.field_widgets[name] = widget
                self.field_rows.append((row, advanced))
            settings_layout.addWidget(group)

        self.restore_button = QtWidgets.QPushButton("Restore shipped defaults")
        self.restore_button.setToolTip("Replace config.py with the shipped P3 settings.")
        settings_layout.addWidget(self.restore_button)
        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_panel)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs)
        self.setup_tab = QtWidgets.QWidget()
        setup_layout = QtWidgets.QVBoxLayout(self.setup_tab)
        self.mesh_display_banner = QtWidgets.QLabel()
        self.mesh_display_banner.setWordWrap(True)
        self.mesh_display_banner.setStyleSheet(
            "QLabel { background: #fff3cd; color: #664d03; border: 2px solid #e0a800; "
            "border-radius: 4px; padding: 8px; font-weight: 600; }"
        )
        self.mesh_display_banner.hide()
        setup_layout.addWidget(self.mesh_display_banner)
        self.preview_figure = Figure(figsize=(8, 6))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        setup_layout.addWidget(self.preview_canvas, 1)
        self.derived_label = QtWidgets.QLabel()
        self.derived_label.setWordWrap(True)
        setup_layout.addWidget(self.derived_label)
        setup_buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Start simulation")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.load_button = QtWidgets.QPushButton("Load existing HDF5")
        setup_buttons.addWidget(self.run_button)
        setup_buttons.addWidget(self.cancel_button)
        setup_buttons.addWidget(self.load_button)
        setup_layout.addLayout(setup_buttons)
        self.tabs.addTab(self.setup_tab, "Setup / Mesh")

        self.results_tab = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(self.results_tab)
        self.result_figure = Figure(figsize=(10, 7))
        self.result_canvas = FigureCanvas(self.result_figure)
        results_layout.addWidget(self.result_canvas, 1)
        results_toolbar = QtWidgets.QHBoxLayout()
        results_toolbar.addWidget(QtWidgets.QLabel("Frame:"))
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        results_toolbar.addWidget(self.frame_slider, 1)
        self.frame_label = QtWidgets.QLabel("No result loaded")
        results_toolbar.addWidget(self.frame_label)
        results_layout.addLayout(results_toolbar)
        export_buttons = QtWidgets.QHBoxLayout()
        self.export_figure_button = QtWidgets.QPushButton("Export figure")
        self.export_animation_button = QtWidgets.QPushButton("Render MP4")
        self.export_figure_button.setEnabled(False)
        self.export_animation_button.setEnabled(False)
        export_buttons.addWidget(self.export_figure_button)
        export_buttons.addWidget(self.export_animation_button)
        results_layout.addLayout(export_buttons)
        self.results_tab_index = self.tabs.addTab(self.results_tab, "Results / Animation")
        self.tabs.setTabVisible(self.results_tab_index, False)
        splitter.addWidget(right)
        splitter.addWidget(settings_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1050, 430])

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QtWidgets.QLabel("Ready")
        status_layout = QtWidgets.QHBoxLayout()
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar)
        root_layout.addLayout(status_layout)

    @staticmethod
    def _make_widget(kind):
        if kind == "bool":
            return QtWidgets.QCheckBox()
        if kind == "choice":
            widget = QtWidgets.QComboBox()
            widget.addItems(["hot", "viridis", "plasma", "jet", "inferno"])
            return widget
        return QtWidgets.QLineEdit()

    @staticmethod
    def _tooltip_for(name, target="input"):
        help_text = {
            "le": ("Width of the base metal on the left side of the weld.", "Enter a positive length in millimetres."),
            "ri": ("Width of the base metal on the right side of the weld.", "Enter a positive length in millimetres."),
            "we": ("Width of the weld region in the cross-section.", "Enter a positive length in millimetres."),
            "th": ("Height of the base metal below the support region.", "Enter a positive length in millimetres."),
            "su_h": ("Height of the raised support geometry.", "Enter a positive length in millimetres."),
            "su_w": ("Width of the raised support geometry.", "Enter a positive length in millimetres."),
            "fr_ab": ("Extra air space above the sample.", "Advanced geometry setting; use a non-negative length."),
            "fr_be": ("Extra air space below the sample.", "Advanced geometry setting; use a non-negative length."),
            "weld_bead_thickness": ("Thickness of the represented weld bead.", "Must be positive and no wider than the weld width."),
            "dx": ("Horizontal mesh spacing.", "Smaller spacing increases memory and runtime; 1 mm is recommended."),
            "dy": ("Vertical mesh spacing.", "Keep this equal to dx unless testing an anisotropic mesh."),
            "t_cool": ("Temperature of the material before welding and between passes.", "Enter a temperature in degrees Celsius."),
            "t_hot": ("Temperature assigned to the initial weld region.", "Enter a temperature in degrees Celsius."),
            "t_room": ("Temperature of the surrounding air.", "Enter a temperature in degrees Celsius."),
            "diff_coeff_bm": ("Thermal diffusivity of the base metal.", "Advanced material value in mm²/s."),
            "diff_coeff_wm": ("Thermal diffusivity of the weld metal.", "Advanced material value in mm²/s."),
            "diff_coeff_haz": ("Thermal diffusivity reserved for the HAZ material.", "Advanced material value in mm²/s."),
            "diff_coeff_air": ("Diffusivity assigned to air cells.", "Normally zero for the current boundary treatment."),
            "c": ("Specific heat capacity used by cooling.", "Advanced material value."),
            "rho": ("Density used by cooling.", "Advanced material value."),
            "conv_variable": ("Strength of the simplified cooling term.", "This is a model coefficient, not a direct surface heat-transfer coefficient."),
            "weld_length": ("Length travelled by the moving heat source.", "Enter a positive length in millimetres."),
            "weld_speed": ("Travel speed of the moving heat source.", "Enter a positive speed in millimetres per minute."),
            "weld_temp": ("Temperature assigned to the moving heat source.", "Enter a temperature in degrees Celsius."),
            "weld_spot_size": ("Length of the moving heated spot.", "Must be positive and no longer than the weld length."),
            "time_before_weld_start": ("Time simulated before the heat source starts moving.", "Enter a non-negative delay in seconds."),
            "sim_time": ("Total simulated time.", "Longer runs require more computation and storage."),
            "save_so_often_per_sec": ("Number of result snapshots saved per simulated second.", "This controls stored frames, not the internal timestep."),
            "slow_down_beginning": ("Save extra frames during the first 60 seconds.", "Useful for inspecting the moving weld, but increases file size."),
            "h5_filename": ("Name of the HDF5 snapshot file.", "Use a filename only; it is saved in P3/02_Results."),
            "animation_filename": ("Name of the exported MP4 file.", "Use a filename only; FFmpeg is required for export."),
            "figure_filename": ("Name of the exported figure file.", "Use a filename only; PNG, PDF, and SVG are supported."),
            "disable_overwrite_warning": ("Skip confirmation before replacing output files.", "Use with care; this applies to HDF5, figures, and MP4 files."),
            "heatmap_style": ("Color map used for heatmap displays.", "The same map is used for the setup preview and results."),
            "heatmap_vmin": ("Lower limit of the heatmap color scale.", "Values below this limit use the lowest color."),
            "heatmap_vmax": ("Upper limit of the heatmap color scale.", "Values above this limit use the highest color."),
            "show_contours": ("Draw temperature contour lines on the heatmap.", "Toggle contour lines on or off."),
            "contour_levels": ("Temperatures at which contour lines are drawn.", "Enter comma-separated values, for example: 200, 300, 400."),
            "show_monitoring_points": ("Mark temperature-monitoring locations on the heatmap.", "Toggle monitoring markers and labels on or off."),
            "show_mesh_lines": ("Show the computational mesh in the setup preview.", "The grid is displayed at a readable interval; dx and dy remain unchanged."),
            "monitoring_distances": ("Distances from the weld used for temperature traces.", "Enter comma-separated distances in millimetres."),
            "monitoring_y_offset": ("Vertical offset of the monitoring line.", "Advanced placement setting in millimetres."),
            "weld_zoom_margin": ("Extra space around the moving weld in the weld-area view.", "Increase this to show more surrounding material."),
            "animation_fps": ("Playback frame rate for MP4 export.", "Advanced export setting."),
            "animation_dpi": ("Resolution used for exported figures and MP4 frames.", "Higher values increase render time and file size."),
            "animation_frame_stride": ("Number of stored frames skipped between exported frames.", "A value of 1 exports every stored frame."),
            "use_boundary_adjustment": ("Recalculate the air/metal boundary while the weld moves.", "Advanced experimental solver option."),
        }
        label_help, input_help = help_text.get(name, ("P3 simulation setting.", "Enter a valid value."))
        return label_help if target == "label" else input_help

    def _populate_widgets(self):
        for name, widget in self.field_widgets.items():
            value = self.settings[name]
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(value, list):
                widget.setText(", ".join(str(item) for item in value))
            else:
                widget.setText(str(value))
        self._set_advanced_visibility(False)

    def _connect_widget_signals(self):
        self.advanced_toggle.toggled.connect(self._set_advanced_visibility)
        self.restore_button.clicked.connect(self._restore_defaults)
        self.run_button.clicked.connect(self._start_simulation)
        self.cancel_button.clicked.connect(self._cancel_simulation)
        self.load_button.clicked.connect(self._load_existing)
        self.frame_slider.valueChanged.connect(self._draw_result)
        self.export_figure_button.clicked.connect(self._export_figure)
        self.export_animation_button.clicked.connect(self._export_animation)
        for name, widget in self.field_widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(lambda _name=name: self._persist_from_widgets(_name))
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.stateChanged.connect(lambda _state, _name=name: self._persist_from_widgets(_name))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(lambda _text, _name=name: self._persist_from_widgets(_name))

    def _set_advanced_visibility(self, visible):
        for row, advanced in self.field_rows:
            row.setVisible(bool(visible or not advanced))

    def _settings_from_widgets(self):
        values = dict(self.settings)
        for name, widget in self.field_widgets.items():
            kind = next(kind for _group, fields in FIELD_GROUPS for field_name, _label, kind, _advanced in fields if field_name == name)
            if kind == "bool":
                values[name] = widget.isChecked()
            elif kind == "choice":
                values[name] = widget.currentText()
            else:
                text = widget.text().strip()
                if kind == "float":
                    values[name] = float(text)
                elif kind == "int":
                    values[name] = int(text)
                elif kind == "list_float":
                    values[name] = [float(part.strip()) for part in text.split(",") if part.strip()]
                else:
                    values[name] = text
        return validate_settings(values)

    def _persist_from_widgets(self, changed_name=""):
        if self.loading_widgets:
            return
        try:
            self.settings = write_settings(self._settings_from_widgets())
            self._clear_error(changed_name)
            self.status_label.setText(f"Saved {changed_name or 'settings'} to config.py")
            self._update_preview()
            if self.loaded_data:
                self.result_view = None
                self._render_result_frame(self.frame_slider.value())
        except Exception as exc:
            self._mark_error(changed_name)
            self.status_label.setText(f"Invalid setting: {exc}")

    def _clear_error(self, name):
        widget = self.field_widgets.get(name)
        if widget:
            widget.setStyleSheet("")

    def _mark_error(self, name):
        widget = self.field_widgets.get(name)
        if widget:
            widget.setStyleSheet("border: 1px solid #c62828;")

    def _restore_defaults(self):
        answer = QtWidgets.QMessageBox.question(
            self,
            "Restore defaults",
            "Replace the current persistent config.py with the shipped P3 defaults?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.settings = restore_default_config()
            self.loading_widgets = True
            self._populate_widgets()
            self.loading_widgets = False
            self._update_preview()
            self.result_view = None
            if self.loaded_data:
                self._render_result_frame(self.frame_slider.value())
            self.status_label.setText("Restored shipped defaults")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Restore failed", str(exc))

    def _update_preview(self):
        try:
            fields, diffusion, derived = build_initial_fields(self.settings)
            self.preview_figure.clear()
            axis = self.preview_figure.add_subplot(111)
            dx = float(self.settings["dx"])
            dy = float(self.settings["dy"])
            extent = [-(dx / 2), (fields.shape[1] - 0.5) * dx, (fields.shape[0] - 0.5) * dy, -(dy / 2)]
            image = axis.imshow(
                fields,
                cmap=self.settings["heatmap_style"],
                vmin=self.settings["heatmap_vmin"],
                vmax=self.settings["heatmap_vmax"],
                interpolation="nearest",
                aspect="equal",
                extent=extent,
            )
            grid_info = preview_grid_info(fields.shape, dx, dy)
            if self.settings["show_mesh_lines"]:
                grid_stride = grid_info["stride"]
                axis.set_xticks(np.arange(-0.5, fields.shape[1], grid_stride) * dx, minor=True)
                axis.set_yticks(np.arange(-0.5, fields.shape[0], grid_stride) * dy, minor=True)
                axis.grid(which="minor", color="#8795a1", alpha=0.55, linewidth=0.45)
                axis.tick_params(which="minor", bottom=False, left=False)
            if self.settings["show_mesh_lines"] and grid_info["limited"]:
                self.mesh_display_banner.setText(grid_info["message"])
                self.mesh_display_banner.show()
            else:
                self.mesh_display_banner.hide()
            x_values = np.arange(fields.shape[1]) * dx
            y_values = np.arange(fields.shape[0]) * dy
            axis.contour(x_values, y_values, diffusion > 0, levels=[0.5], colors="black", linewidths=1.2)
            self.preview_figure.colorbar(image, ax=axis, label="Temperature [°C]")
            axis.set_title("Mesh preview")
            axis.set_xlabel("X [mm]")
            axis.set_ylabel("Y [mm]")
            self.preview_canvas.draw_idle()
            self.derived_label.setText(
                f"Calculated stable dt: {derived['dt']:.6g} s   |   Mesh: {derived['nx']} x {derived['ny']} "
                f"({derived['nx'] * derived['ny']:,} cells)   |   Weld ends: {derived['weld_end_time']:.2f} s"
            )
        except Exception as exc:
            self.derived_label.setText(f"Preview unavailable: {exc}")

    def _result_path(self, filename):
        return RESULTS_DIR / filename

    def _confirm_overwrite(self, path):
        if not path.exists() or self.settings.get("disable_overwrite_warning", False):
            return True
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Overwrite existing output?")
        box.setText(f"The file already exists:\n{path.name}\n\nIt will be replaced.")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        result = box.exec_()
        return result == QtWidgets.QMessageBox.Yes

    def _start_simulation(self):
        try:
            self.settings = write_settings(self._settings_from_widgets())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot start simulation", str(exc))
            return
        output = self._result_path(self.settings["h5_filename"])
        if not self._confirm_overwrite(output):
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.load_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting simulation...")
        self.simulation_worker = SimulationWorker(self.settings, output, self)
        self.simulation_worker.progress.connect(self._worker_progress)
        self.simulation_worker.completed.connect(self._simulation_completed)
        self.simulation_worker.cancelled.connect(self._simulation_cancelled)
        self.simulation_worker.failed.connect(self._worker_failed)
        self.simulation_worker.finished.connect(self._simulation_thread_finished)
        self.simulation_worker.start()

    def _cancel_simulation(self):
        if self.simulation_worker:
            self.simulation_worker.request_stop()
            self.status_label.setText("Cancelling simulation...")
            self.cancel_button.setEnabled(False)

    def _worker_progress(self, fraction, message):
        self.progress_bar.setValue(min(100, max(0, int(fraction * 100))))
        self.status_label.setText(message)

    def _simulation_completed(self, path):
        self.status_label.setText(f"Simulation complete: {Path(path).name}")
        self.progress_bar.setValue(100)
        self._load_result_path(Path(path))

    def _simulation_cancelled(self):
        self.status_label.setText("Simulation cancelled")

    def _simulation_thread_finished(self):
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.simulation_worker = None

    def _worker_failed(self, message):
        self.status_label.setText("Operation failed")
        QtWidgets.QMessageBox.critical(self, "P3 Heat Map error", message)

    def _load_existing(self):
        self._load_result_path(self._result_path(self.settings["h5_filename"]))

    def _load_result_path(self, path):
        try:
            self.loaded_data = load_snapshots(path)
            self.result_view = None
            self._pending_frame = None
            self._frame_update_timer.stop()
            self.tabs.setTabVisible(self.results_tab_index, True)
            self.frame_slider.setRange(0, len(self.loaded_data["arrays"]) - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.export_figure_button.setEnabled(True)
            self.export_animation_button.setEnabled(True)
            self.tabs.setCurrentWidget(self.results_tab)
            self._render_result_frame(0)
            self.status_label.setText(f"Loaded {path.name}: {len(self.loaded_data['arrays'])} frames")
        except Exception as exc:
            self.loaded_data = None
            self.result_view = None
            self.tabs.setTabVisible(self.results_tab_index, False)
            self.status_label.setText(f"Could not load result: {exc}")
            QtWidgets.QMessageBox.warning(self, "Could not load HDF5", str(exc))

    def _draw_result(self, frame):
        if not self.loaded_data:
            return
        self._pending_frame = int(frame)
        if not self._frame_update_timer.isActive():
            self._frame_update_timer.start()

    def _apply_pending_frame(self):
        if self._pending_frame is None:
            return
        frame = self._pending_frame
        self._pending_frame = None
        self._render_result_frame(frame)

    def _render_result_frame(self, frame):
        if not self.loaded_data:
            return
        try:
            if self.result_view is None:
                self.result_view = create_result_view(self.settings, self.loaded_data, figure=self.result_figure)
            frame = update_result_view(self.result_view, int(frame))
            self.result_canvas.draw_idle()
            time_value = self.loaded_data["times"][frame]
            self.frame_label.setText(f"{int(frame) + 1}/{len(self.loaded_data['arrays'])} ({time_value:.0f} s)")
        except Exception as exc:
            self.status_label.setText(f"Could not draw result: {exc}")

    def _export_figure(self):
        if not self.loaded_data:
            return
        path = self._result_path(self.settings["figure_filename"])
        if not self._confirm_overwrite(path):
            return
        try:
            self.result_figure.savefig(path, dpi=self.settings["animation_dpi"], bbox_inches="tight")
            self.status_label.setText(f"Exported figure: {path.name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Figure export failed", str(exc))

    def _export_animation(self):
        if not self.loaded_data:
            return
        path = self._result_path(self.settings["animation_filename"])
        if not self._confirm_overwrite(path):
            return
        self.export_animation_button.setEnabled(False)
        self.status_label.setText("Starting animation render...")
        self.progress_bar.setValue(0)
        self.animation_worker = AnimationWorker(self.settings, self.loaded_data, path, self)
        self.animation_worker.progress.connect(self._worker_progress)
        self.animation_worker.completed.connect(lambda result: self.status_label.setText(f"Exported animation: {Path(result).name}"))
        self.animation_worker.failed.connect(self._worker_failed)
        self.animation_worker.finished.connect(lambda: self.export_animation_button.setEnabled(True))
        self.animation_worker.start()

    def closeEvent(self, event):
        if self.simulation_worker and self.simulation_worker.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Simulation running",
                "Cancel the running simulation and close P3 Heat Map?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.simulation_worker.request_stop()
            self.simulation_worker.wait(5000)
        if self.animation_worker and self.animation_worker.isRunning():
            self.animation_worker.wait(5000)
        event.accept()


def _show_config_failure(app, error):
    box = QtWidgets.QMessageBox()
    box.setIcon(QtWidgets.QMessageBox.Critical)
    box.setWindowTitle("P3 Heat Map configuration error")
    box.setText("config.py could not be loaded or validated.")
    box.setInformativeText(str(error))
    restore = box.addButton("Restore shipped defaults", QtWidgets.QMessageBox.AcceptRole)
    box.addButton("Exit", QtWidgets.QMessageBox.RejectRole)
    box.exec_()
    if box.clickedButton() is restore:
        try:
            restore_default_config()
            return True
        except Exception as restore_error:
            QtWidgets.QMessageBox.critical(None, "Restore failed", str(restore_error))
    return False


def launch_gui():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    try:
        ensure_config_file()
        settings = load_settings()
    except Exception as error:
        if not _show_config_failure(app, error):
            return 2
        try:
            settings = load_settings()
        except Exception as second_error:
            QtWidgets.QMessageBox.critical(None, "Configuration still invalid", str(second_error))
            return 2
    window = MainWindow(settings)
    return app.exec_()


def run_cli(render=False):
    settings = load_settings()
    output = RESULTS_DIR / settings["h5_filename"]
    print(f"Running P3 Heat Map: {output}")
    run_simulation(settings, output, progress_callback=lambda fraction, message, _time: print(f"{fraction * 100:6.2f}% {message}"))
    if render:
        data = load_snapshots(output)
        animation_path = RESULTS_DIR / settings["animation_filename"]
        render_animation(settings, data, animation_path)
        print(f"Animation written to {animation_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="WeldCraft P3 Heat Map")
    parser.add_argument("--gui", action="store_true", help="open the PyQt5 GUI")
    parser.add_argument("--cli", action="store_true", help="run the simulation directly")
    parser.add_argument("--render", action="store_true", help="render an MP4 after a CLI simulation")
    args = parser.parse_args(argv)
    if args.gui:
        return launch_gui()
    return run_cli(render=args.render)


if __name__ == "__main__":
    raise SystemExit(main())
