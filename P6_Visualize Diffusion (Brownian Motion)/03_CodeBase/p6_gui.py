"""WeldCraft P6 Brownian Motion graphical interface."""

from __future__ import annotations

import html
import sys
import threading
import shutil
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
RESOURCES_DIR = REPO_ROOT / "Resources"
if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))

from Common.launch_ready import StartupReadySignal
from p6_gui_support import (
    DIAGRAM_PRESETS_DIR,
    GUI_DEFAULTS,
    H5FrameSource,
    P6ConfigError,
    P6SimulationCancelled,
    delete_custom_diagram_preset,
    discover_diagram_presets,
    estimate_snapshot_bytes,
    find_ffmpeg,
    format_scientific_steps,
    frame_summary,
    inspect_resume_source,
    load_diagram_settings,
    load_gui_settings,
    list_custom_diagram_presets,
    parse_step_count,
    render_diagram_figure,
    render_presentation_animation,
    restore_gui_defaults,
    result_path,
    run_brownian_simulation,
    save_diagram_preset,
    settings_from_resume_metadata,
    validate_gui_settings,
    write_gui_settings,
)


APP_NAME = "WeldCraft - Brownian Motion"


FIELD_GROUPS = [
    ("Simulation", [
        ("steps", "Total simulation steps", "step"),
        ("save_every_steps", "Save every steps", "step"),
        ("h5_filename", "HDF5 filename", "text"),
        ("GUI_DISABLE_OVERWRITE_WARNING", "Disable overwrite warning", "bool"),
    ]),
    ("Random matrix", [
        ("x", "Width [pixels]", "int"),
        ("y", "Height [pixels]", "int"),
        ("max_sol_a", "Possible sites A [%]", "float"),
        ("max_sol_b", "Possible sites B [%]", "float"),
    ]),
    ("Initial hydrogen", [
        ("concentration_a", "Concentration A [%]", "float"),
        ("concentration_b", "Concentration B [%]", "float"),
        ("USE_INITIAL_CONCENTRATION_PROFILE", "Use linear profiles", "bool"),
        ("concentration_profile_a_left", "A profile left [%]", "float"),
        ("concentration_profile_a_right", "A profile right [%]", "float"),
        ("concentration_profile_b_left", "B profile left [%]", "float"),
        ("concentration_profile_b_right", "B profile right [%]", "float"),
    ]),
    ("Spot", [
        ("USE_SPOT", "Enable spot", "bool"),
        ("SPOT_DIAMETER", "Diameter [pixels]", "int"),
        ("SPOT_CENTER_X", "Center X [pixels]", "int"),
        ("SPOT_CENTER_Y", "Center Y [pixels]", "int"),
        ("max_sol_spot", "Possible sites [%]", "float"),
        ("concentration_spot", "Concentration [%]", "float"),
        ("affinity_spot", "Affinity", "float"),
        ("mobility_spot", "Mobility", "float"),
    ]),
    ("Trap layer", [
        ("USE_TRAP_LAYER", "Enable trap layer", "bool"),
        ("TRAP_LAYER_CENTER_X", "Center X [pixels]", "int"),
        ("TRAP_LAYER_WIDTH", "Width [pixels]", "int"),
        ("max_sol_trap_layer", "Possible sites [%]", "float"),
        ("concentration_trap_layer", "Concentration [%]", "float"),
        ("affinity_trap_layer", "Affinity", "float"),
        ("mobility_trap_layer", "Mobility", "float"),
    ]),
    ("Base-area characteristics", [
        ("affinity_a", "Affinity A", "float"),
        ("mobility_a", "Mobility A", "float"),
        ("affinity_b", "Affinity B", "float"),
        ("mobility_b", "Mobility B", "float"),
    ]),
    ("Source and sink", [
        ("USE_SINK_SOURCE", "Enable source/sink", "bool"),
        ("SOURCE_SIDE", "Source side", "source_side"),
        ("SINK_SOURCE_THICKNESS", "Boundary thickness", "int"),
    ]),
    ("Analysis and reproducibility", [
        ("num_subregions", "Flux-analysis regions", "int"),
        ("random_seed", "Fixed seed (blank = random)", "optional_int"),
    ]),
    ("Animation presentation", [
        ("SHOW_MAIN_SIMULATION_PANEL", "Show matrix panel", "bool"),
        ("SHOW_CONCENTRATION_PROFILE_PANEL", "Show concentration profile", "bool"),
        ("SHOW_NET_FLUX_PANEL", "Show net flux", "bool"),
        ("MAIN_RENDER_MODE", "Matrix style", "render_mode"),
        ("DOT_SIZE_AVAILABLE", "Available-site dot size", "float"),
        ("DOT_SIZE_HYDROGEN", "Hydrogen dot size", "float"),
        ("DOT_ALPHA_AVAILABLE", "Available-site opacity", "float"),
        ("DOT_ALPHA_HYDROGEN", "Hydrogen opacity", "float"),
        ("COLOR_EMPTY", "Empty/background color", "text"),
        ("COLOR_AVAILABLE_SPOT", "Available-site color", "text"),
        ("COLOR_HYDROGEN", "Hydrogen color", "text"),
        ("COLOR_CONCENTRATION_LINE", "Profile color", "text"),
        ("render_every_nth_frame", "MP4 frame stride", "int"),
        ("animation_fps", "MP4 frames per second", "int"),
        ("animation_filename", "MP4 filename", "text"),
        ("GUI_FIGURE_FILENAME", "Still-image filename", "text"),
    ]),
]


DEPENDENT_FIELDS = {
    "USE_INITIAL_CONCENTRATION_PROFILE": {
        "concentration_profile_a_left", "concentration_profile_a_right",
        "concentration_profile_b_left", "concentration_profile_b_right",
    },
    "USE_SPOT": {
        "SPOT_DIAMETER", "SPOT_CENTER_X", "SPOT_CENTER_Y", "max_sol_spot",
        "concentration_spot", "affinity_spot", "mobility_spot",
    },
    "USE_TRAP_LAYER": {
        "TRAP_LAYER_CENTER_X", "TRAP_LAYER_WIDTH", "max_sol_trap_layer",
        "concentration_trap_layer", "affinity_trap_layer", "mobility_trap_layer",
    },
    "USE_SINK_SOURCE": {"SOURCE_SIDE", "SINK_SOURCE_THICKNESS"},
}

RESUME_LOCKED_FIELDS = {
    "x", "y", "max_sol_a", "max_sol_b", "concentration_a", "concentration_b",
    "USE_INITIAL_CONCENTRATION_PROFILE", "concentration_profile_a_left",
    "concentration_profile_a_right", "concentration_profile_b_left",
    "concentration_profile_b_right", "USE_SPOT", "SPOT_DIAMETER", "SPOT_CENTER_X",
    "SPOT_CENTER_Y", "max_sol_spot", "concentration_spot", "affinity_spot",
    "mobility_spot", "USE_TRAP_LAYER", "TRAP_LAYER_CENTER_X", "TRAP_LAYER_WIDTH",
    "max_sol_trap_layer", "concentration_trap_layer", "affinity_trap_layer",
    "mobility_trap_layer", "affinity_a", "mobility_a", "affinity_b", "mobility_b",
    "USE_SINK_SOURCE", "SOURCE_SIDE", "SINK_SOURCE_THICKNESS", "num_subregions",
    "random_seed",
}


PRESENTATION_FIELDS = [
    ("RENDER_MODE", "Render mode", "diagram_render_mode", "common"),
    ("SHOW_MAIN_PANEL", "Show main panel", "bool", "common"),
    ("SHOW_CONCENTRATION_PROFILE_PANEL", "Show concentration profile", "bool", "common"),
    ("SHOW_HEATMAP_PANEL", "Show heatmap panel", "bool", "common"),
    ("SHOW_NET_FLUX_PANEL", "Show net-flux panel", "bool", "common"),
    ("SHOW_LEGEND", "Show legend", "bool", "common"),
    ("COLOR_EMPTY", "Background color", "text", "common"),
    ("COLOR_AVAILABLE_SPOT", "Available-site color", "text", "common"),
    ("COLOR_HYDROGEN", "Hydrogen color", "text", "common"),
    ("DOT_SIZE_AVAILABLE", "Available dot size", "float", "common"),
    ("DOT_SIZE_HYDROGEN", "Hydrogen dot size", "float", "common"),
    ("DOT_ALPHA_AVAILABLE", "Available opacity", "float", "common"),
    ("DOT_ALPHA_HYDROGEN", "Hydrogen opacity", "float", "common"),
    ("SHOW_SPECIAL_REGION_OUTLINES", "Special-region outlines", "bool", "common"),
    ("PROFILE_AXIS", "Profile axis", "axis", "profile"),
    ("PROFILE_X_RANGE", "Profile X range (min, max)", "optional_pair", "profile"),
    ("PROFILE_Y_RANGE", "Profile Y range (min, max)", "optional_pair", "profile"),
    ("PROFILE_BIN_SIZE", "Profile bin size", "int", "profile"),
    ("PROFILE_SMOOTHING_WINDOW", "Smoothing window", "int", "profile"),
    ("PROFILE_GAUSSIAN_SIGMA", "Gaussian sigma", "float", "profile"),
    ("SHOW_PROFILE_HALF_TRANSITION", "Show half transition", "bool", "profile"),
    ("SHOW_PROFILE_SPOT_SHADE", "Shade spot on profile", "bool", "profile"),
    ("HEATMAP_MODE", "Heatmap mode", "heatmap_mode", "heatmap"),
    ("HEATMAP_SIGMA", "Heatmap smoothing sigma", "float", "heatmap"),
    ("HEATMAP_COLORMAP", "Heatmap color map", "text", "heatmap"),
    ("HEATMAP_DEVIATION_RANGE", "Deviation range (min, max)", "float_pair", "heatmap"),
    ("HEATMAP_OCCUPANCY_RANGE", "Occupancy range (min, max)", "float_pair", "heatmap"),
    ("HEATMAP_RESPECT_AREA_BOUNDARIES", "Respect area boundaries", "bool", "heatmap"),
    ("HEATMAP_SHOW_CONTOURS", "Show contours", "bool", "heatmap"),
    ("HEATMAP_CONTOUR_LEVELS", "Contour levels (comma-separated)", "float_list", "heatmap"),
    ("HEATMAP_SHOW_COLORBAR", "Show colorbar", "bool", "heatmap"),
    ("GLYPH_BIN_SIZE", "Glyph bin size", "int", "printer"),
    ("GLYPH_MIN_RADIUS_FRACTION", "Minimum glyph radius", "float", "printer"),
    ("GLYPH_MAX_RADIUS_FRACTION", "Maximum glyph radius", "float", "printer"),
    ("GLYPH_SHOW_GRID", "Show glyph grid", "bool", "printer"),
    ("GLYPH_SHOW_EXPLANATION", "Show glyph explanation", "bool", "printer"),
    ("AREA_SUMMARY_TOTAL_DOTS", "Summary dot count", "int", "area"),
    ("AREA_SUMMARY_DENSITY_MODE", "Dot-density mode", "density_mode", "area"),
    ("AREA_SUMMARY_CONCENTRATION_MODE", "Concentration mode", "concentration_mode", "area"),
    ("AREA_SUMMARY_LINEAR_CONCENTRATION", "Linear concentration (left, right)", "float_pair", "area"),
    ("AREA_SUMMARY_CONCENTRATION_BIN_WIDTH", "Concentration bin width", "int", "area"),
    ("AREA_SUMMARY_POSITION_MODE", "Dot-position mode", "position_mode", "area"),
    ("AREA_SUMMARY_MIN_DOT_SPACING", "Minimum dot spacing", "float", "area"),
    ("AREA_SUMMARY_SHAKE_MODE", "Arrangement style", "shake_mode", "area"),
    ("AREA_SUMMARY_CLUSTER_COUNT", "Cluster count", "int", "area"),
    ("AREA_SUMMARY_CLUSTER_ATTRACTION", "Cluster attraction", "float", "area"),
    ("AREA_SUMMARY_CLUSTER_SCOPE", "Cluster scope", "cluster_scope", "area"),
    ("AREA_SUMMARY_DOT_SIZE", "Summary dot size", "float", "area"),
    ("AREA_SUMMARY_DOT_ALPHA", "Summary dot opacity", "float", "area"),
    ("AREA_SUMMARY_SHOW_AREA_LABELS", "Show area labels", "bool", "area"),
    ("AREA_SUMMARY_SHOW_EXPLANATION", "Show explanation", "bool", "area"),
    ("AREA_SUMMARY_SHOW_SOURCE_SINK_BANDS", "Show source/sink bands", "bool", "area"),
    ("AREA_SUMMARY_SHOW_HALF_DIVIDER", "Show half divider", "bool", "area"),
]


CHOICES = {
    "source_side": ["left", "right"],
    "render_mode": ["pixels", "dots"],
    "diagram_render_mode": ["pixels", "dots", "concentration_heatmap", "printer_glyphs", "area_summary_dots"],
    "axis": ["x", "y"],
    "heatmap_mode": ["deviation", "occupancy", "change_from_initial", "change_from_initial_setup"],
    "density_mode": ["available_sites", "uniform_area"],
    "concentration_mode": ["area_average", "saved_x_profile", "linear_x"],
    "position_mode": ["even_hex", "random"],
    "shake_mode": ["none", "gentle", "organic", "clustered"],
    "cluster_scope": ["per_area", "combined_a_b"],
}


class SimulationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(object, str, object, object)
    completed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings, output, parent=None):
        super().__init__(parent)
        self.settings = deepcopy(settings)
        self.output = Path(output)
        self.stop_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        try:
            result = run_brownian_simulation(
                self.settings,
                self.output,
                progress_callback=lambda fraction, message, completed, frames: self.progress.emit(
                    fraction, message, completed, frames
                ),
                stop_event=self.stop_event,
            )
            self.completed.emit(str(result))
        except P6SimulationCancelled:
            self.cancelled.emit(str(self.output))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class AnimationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(self, source, output, preset, overrides, fps, stride, ffmpeg, parent=None):
        super().__init__(parent)
        self.source = Path(source)
        self.output = Path(output)
        self.preset = preset
        self.overrides = deepcopy(overrides)
        self.fps = fps
        self.stride = stride
        self.ffmpeg = ffmpeg
        self.stop_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        try:
            result = render_presentation_animation(
                self.source, self.output, self.preset, self.overrides,
                self.fps, self.stride, self.ffmpeg,
                progress_callback=lambda fraction, message: self.progress.emit(fraction, message),
                stop_event=self.stop_event,
            )
            self.completed.emit(str(result))
        except P6SimulationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ResultRenderWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(int, str, int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(self, serial, key, source, frame_index, preset, overrides, parent=None):
        super().__init__(parent)
        self.serial = int(serial)
        self.key = str(key)
        self.source = Path(source)
        self.frame_index = int(frame_index)
        self.preset = str(preset)
        self.overrides = deepcopy(overrides)

    def run(self):
        try:
            figure = render_diagram_figure(
                self.source,
                self.frame_index,
                self.preset,
                self.overrides,
            )
            self.completed.emit(self.serial, self.key, self.frame_index, figure)
        except Exception as exc:
            self.failed.emit(self.serial, f"{type(exc).__name__}: {exc}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.widgets = {}
        self.rows = {}
        self.field_labels = {}
        self.loading_widgets = True
        self.resume_info = None
        self.frame_source = None
        self.simulation_worker = None
        self.animation_worker = None
        self.result_render_worker = None
        self.pending_result_render = None
        self.result_render_serial = 0
        self.latest_result_render_serial = 0
        self.run_progress_start_step = 0
        self.run_progress_end_step = 0
        self.run_progress_total_frames = 0
        self.current_canvas = {}
        self.presentation_widgets = {}
        self.presentation_settings = {}
        self.presentation_loading = False
        self.ffmpeg = None
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)
        self.frame_debounce = QtCore.QTimer(self)
        self.frame_debounce.setSingleShot(True)
        self.frame_debounce.setInterval(75)
        self.frame_debounce.timeout.connect(self._render_selected_result)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QtGui.QIcon(str(RESOURCES_DIR / "Images" / "WeldCraft.ico")))
        self.resize(1500, 920)
        self._build_ui()
        self._populate_widgets()
        self.loading_widgets = False
        self._connect_signals()
        self._update_dependencies()
        self._update_preview()
        self._load_presentation_controls()
        self.ffmpeg = find_ffmpeg(self._configured_ffmpeg())
        self.render_mp4_button.setEnabled(bool(self.ffmpeg))
        if not self.ffmpeg:
            self.render_mp4_button.setToolTip("FFmpeg was not found; MP4 rendering is unavailable.")
        self.startup_ready_signal = StartupReadySignal(self)
        self.showMaximized()

    def _configured_ffmpeg(self):
        try:
            import b2_Brown_Config as cfg
            return getattr(cfg, "ffmpeg_path", None)
        except Exception:
            return None

    def _build_ui(self):
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 9px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QLineEdit, QComboBox { min-height: 24px; }"
            "QPushButton { min-height: 28px; padding: 2px 10px; }"
        )
        self._build_menu()
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)
        self._build_setup_tab()
        self._build_results_tab()
        self.tabs.setTabVisible(1, False)
        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(260)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("0.0%")
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_bar)
        root.addLayout(status_row)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        self.load_result_action = file_menu.addAction("Load existing HDF5…")
        file_menu.addSeparator()
        self.resume_action = file_menu.addAction("Continue from exact checkpoint…")
        self.resume_action.setToolTip(
            "Special-purpose continuation of a new P6 file with a validated exact checkpoint."
        )
        self.clear_resume_action = file_menu.addAction("Return to new simulation")
        self.clear_resume_action.setEnabled(False)

    def _build_setup_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        banner = QtWidgets.QLabel(
            "Model: random matrix · event-driven wiggle. Advanced movement and numerical settings remain code-only."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet("background:#e8f4fd; border:1px solid #8ab6d6; padding:8px;")
        preview_layout.addWidget(banner)
        preview_style_row = QtWidgets.QHBoxLayout()
        preview_style_row.addStretch(1)
        preview_style_row.addWidget(QtWidgets.QLabel("Preview / animation matrix style:"))
        self.preview_style_combo = QtWidgets.QComboBox()
        self.preview_style_combo.addItems(CHOICES["render_mode"])
        self.preview_style_combo.setToolTip(
            "This is the same Matrix style setting shown under Animation presentation."
        )
        self.preview_style_combo.setMaximumWidth(150)
        preview_style_row.addWidget(self.preview_style_combo)
        preview_layout.addLayout(preview_style_row)
        self.preview_figure = Figure(figsize=(9, 6))
        self.preview_canvas = FigureCanvas(self.preview_figure)
        preview_layout.addWidget(self.preview_canvas, 1)
        self.derived_label = QtWidgets.QLabel()
        self.derived_label.setWordWrap(True)
        preview_layout.addWidget(self.derived_label)

        self.resume_banner = QtWidgets.QFrame()
        self.resume_banner.setStyleSheet(
            "QFrame { background:#fff3cd; border:1px solid #d6b656; border-radius:4px; padding:5px; }"
        )
        resume_layout = QtWidgets.QHBoxLayout(self.resume_banner)
        resume_layout.setContentsMargins(8, 6, 8, 6)
        self.resume_label = QtWidgets.QLabel()
        self.resume_label.setWordWrap(True)
        self.resume_label.setTextFormat(QtCore.Qt.RichText)
        resume_layout.addWidget(self.resume_label, 1)
        self.clear_resume_button = QtWidgets.QPushButton("Return to new simulation")
        self.clear_resume_button.setToolTip("Discard this continuation setup without changing the source HDF5 file.")
        self.clear_resume_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.clear_resume_button.setMaximumWidth(240)
        resume_layout.addWidget(self.clear_resume_button, 0, QtCore.Qt.AlignVCenter)
        self.resume_banner.hide()
        preview_layout.addWidget(self.resume_banner)

        run_buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Start simulation")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.load_button = QtWidgets.QPushButton("Load existing HDF5…")
        self.continue_button = QtWidgets.QPushButton("Open HDF5 to continue…")
        self.continue_button.setToolTip("Select a resumable exact checkpoint and configure a new continuation file.")
        run_buttons.addWidget(self.run_button)
        run_buttons.addWidget(self.cancel_button)
        run_buttons.addWidget(self.load_button)
        run_buttons.addWidget(self.continue_button)
        preview_layout.addLayout(run_buttons)
        splitter.addWidget(preview_panel)

        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(420)
        settings_panel = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_panel)
        logo = QtWidgets.QLabel()
        logo.setAlignment(QtCore.Qt.AlignCenter)
        logo_path = RESOURCES_DIR / "Images" / "BAM Logo.png"
        if logo_path.exists():
            logo.setPixmap(QtGui.QPixmap(str(logo_path)))
        settings_layout.addWidget(logo)
        for group_name, fields in FIELD_GROUPS:
            group = QtWidgets.QGroupBox(group_name)
            group_layout = QtWidgets.QVBoxLayout(group)
            for name, label_text, kind in fields:
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                label = QtWidgets.QLabel(label_text)
                label.setWordWrap(True)
                label.setMinimumWidth(175)
                widget = self._make_widget(kind)
                widget.setToolTip(self._tooltip(name))
                row_layout.addWidget(label)
                row_layout.addWidget(widget, 1)
                group_layout.addWidget(row)
                self.widgets[name] = widget
                self.rows[name] = row
                self.field_labels[name] = label
            settings_layout.addWidget(group)
        self.restore_button = QtWidgets.QPushButton("Restore GUI defaults")
        settings_layout.addWidget(self.restore_button)
        settings_layout.addStretch(1)
        settings_scroll.setWidget(settings_panel)
        splitter.addWidget(settings_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1050, 450])
        self.tabs.addTab(tab, "Setup")

    def _build_results_tab(self):
        tab = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(tab)
        frame_row = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_label = QtWidgets.QLabel("No result loaded")
        frame_row.addWidget(self.play_button)
        frame_row.addWidget(QtWidgets.QLabel("Frame:"))
        frame_row.addWidget(self.frame_slider, 1)
        frame_row.addWidget(self.frame_label)
        root.addLayout(frame_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)
        self.result_views = QtWidgets.QTabWidget()
        self.classic_container = QtWidgets.QWidget()
        self.classic_layout = QtWidgets.QVBoxLayout(self.classic_container)
        self.diagram_container = QtWidgets.QWidget()
        self.diagram_layout = QtWidgets.QVBoxLayout(self.diagram_container)
        self.result_views.addTab(self.classic_container, "Animation view")
        self.result_views.addTab(self.diagram_container, "Diagram view")
        splitter.addWidget(self.result_views)

        presentation_scroll = QtWidgets.QScrollArea()
        presentation_scroll.setWidgetResizable(True)
        presentation_scroll.setMinimumWidth(390)
        presentation_panel = QtWidgets.QWidget()
        self.presentation_layout = QtWidgets.QVBoxLayout(presentation_panel)
        preset_group = QtWidgets.QGroupBox("Presentation preset")
        preset_layout = QtWidgets.QVBoxLayout(preset_group)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(discover_diagram_presets())
        self.save_preset_button = QtWidgets.QPushButton("Save current settings as new preset…")
        self.manage_presets_button = QtWidgets.QPushButton("Manage saved presets…")
        self.reset_preset_button = QtWidgets.QPushButton("Reset current preset")
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.save_preset_button)
        preset_layout.addWidget(self.manage_presets_button)
        preset_layout.addWidget(self.reset_preset_button)
        self.presentation_layout.addWidget(preset_group)
        self.presentation_fields_container = QtWidgets.QWidget()
        self.presentation_fields_layout = QtWidgets.QVBoxLayout(self.presentation_fields_container)
        self.presentation_layout.addWidget(self.presentation_fields_container)
        self.presentation_layout.addStretch(1)
        presentation_scroll.setWidget(presentation_panel)
        splitter.addWidget(presentation_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1100, 400])

        export_row = QtWidgets.QHBoxLayout()
        self.export_figure_button = QtWidgets.QPushButton("Export current diagram")
        self.render_mp4_button = QtWidgets.QPushButton("Render MP4")
        self.cancel_render_button = QtWidgets.QPushButton("Cancel MP4")
        self.cancel_render_button.setEnabled(False)
        export_row.addWidget(self.export_figure_button)
        export_row.addWidget(self.render_mp4_button)
        export_row.addWidget(self.cancel_render_button)
        root.addLayout(export_row)
        self.tabs.addTab(tab, "Results")

    @staticmethod
    def _make_widget(kind):
        if kind == "bool":
            return QtWidgets.QCheckBox()
        if kind in CHOICES:
            widget = QtWidgets.QComboBox()
            widget.addItems(CHOICES[kind])
            return widget
        return QtWidgets.QLineEdit()

    @staticmethod
    def _tooltip(name):
        tips = {
            "steps": "Total event-driven simulation steps. Scientific notation such as 3e9 is accepted.",
            "save_every_steps": "Exact interval between committed HDF5 snapshots.",
            "h5_filename": "A filename only. It is always written directly to P6/02_Results.",
            "max_sol_a": "Fraction of locations in area A that can hold hydrogen, shown as percent.",
            "max_sol_b": "Fraction of locations in area B that can hold hydrogen, shown as percent.",
            "random_seed": "Leave blank for a fresh recorded seed; enter an integer for reproducibility.",
            "num_subregions": "Number of width-wise regions used to record net transport.",
            "render_every_nth_frame": "Use every nth committed HDF5 frame in exported MP4 files.",
        }
        return tips.get(name, "P6 Brownian-motion setting. Invalid values are not saved.")

    def _connect_signals(self):
        for name, widget in self.widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(lambda _name=name: self._persist_widgets(_name))
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(lambda _checked, _name=name: self._persist_widgets(_name))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(lambda _text, _name=name: self._persist_widgets(_name))
        self.restore_button.clicked.connect(self._restore_defaults)
        self.run_button.clicked.connect(self._start_simulation)
        self.cancel_button.clicked.connect(self._cancel_simulation)
        self.load_button.clicked.connect(self._choose_result)
        self.continue_button.clicked.connect(self._choose_resume)
        self.load_result_action.triggered.connect(self._choose_result)
        self.resume_action.triggered.connect(self._choose_resume)
        self.clear_resume_action.triggered.connect(self._clear_resume)
        self.clear_resume_button.clicked.connect(self._clear_resume)
        self.frame_slider.valueChanged.connect(self._frame_changed)
        self.play_button.clicked.connect(self._toggle_play)
        self.result_views.currentChanged.connect(lambda _index: self._render_selected_result())
        self.preset_combo.currentTextChanged.connect(self._preset_changed)
        self.save_preset_button.clicked.connect(self._save_current_preset)
        self.manage_presets_button.clicked.connect(self._manage_saved_presets)
        self.reset_preset_button.clicked.connect(self._reset_current_preset)
        self.export_figure_button.clicked.connect(self._export_current_figure)
        self.render_mp4_button.clicked.connect(self._start_animation_render)
        self.cancel_render_button.clicked.connect(self._cancel_animation_render)
        self.preview_style_combo.currentTextChanged.connect(self._preview_style_changed)

    def _populate_widgets(self):
        for name, widget in self.widgets.items():
            value = self.settings.get(name, "")
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.setText("" if value is None else str(value))
        self.preset_combo.setCurrentText(self.settings["GUI_DIAGRAM_PRESET"])
        self.preview_style_combo.blockSignals(True)
        self.preview_style_combo.setCurrentText(self.settings["MAIN_RENDER_MODE"])
        self.preview_style_combo.blockSignals(False)

    def _preview_style_changed(self, style):
        if self.loading_widgets:
            return
        matrix_style_widget = self.widgets["MAIN_RENDER_MODE"]
        matrix_style_widget.blockSignals(True)
        matrix_style_widget.setCurrentText(style)
        matrix_style_widget.blockSignals(False)
        self._persist_widgets("MAIN_RENDER_MODE")

    def _settings_from_widgets(self):
        values = deepcopy(self.settings)
        kind_by_name = {name: kind for _group, fields in FIELD_GROUPS for name, _label, kind in fields}
        for name, widget in self.widgets.items():
            kind = kind_by_name[name]
            if isinstance(widget, QtWidgets.QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                values[name] = widget.currentText()
            else:
                text = widget.text().strip()
                if kind in ("int", "optional_int"):
                    values[name] = None if kind == "optional_int" and not text else parse_step_count(text)
                elif kind == "float":
                    values[name] = float(text)
                else:
                    values[name] = text
        return validate_gui_settings(values)

    def _persist_widgets(self, changed_name=""):
        if self.loading_widgets:
            return
        try:
            checked = self._settings_from_widgets()
            self.settings = write_gui_settings(checked)
            self._clear_field_errors()
            self.status_label.setText(f"Saved {changed_name or 'settings'} to b2_Brown_Config.py")
            self.run_button.setEnabled(self.simulation_worker is None)
            self._update_dependencies()
            self._update_preview()
        except Exception as exc:
            widget = self.widgets.get(changed_name)
            if widget:
                widget.setStyleSheet("border:1px solid #c62828;")
            self.run_button.setEnabled(False)
            self.status_label.setText(f"Invalid setting: {exc}")

    def _clear_field_errors(self):
        for widget in self.widgets.values():
            widget.setStyleSheet("")

    def _update_dependencies(self):
        for controller, dependents in DEPENDENT_FIELDS.items():
            enabled = bool(self.settings[controller])
            for name in dependents:
                self.rows[name].setVisible(enabled)
        for name, widget in self.widgets.items():
            locked = self.resume_info is not None and name in RESUME_LOCKED_FIELDS
            widget.setEnabled(not locked)

    def _update_resume_banner(self, settings, summary=None):
        if not self.resume_info:
            return
        start_step = int(self.resume_info["step"])
        summary = summary or frame_summary(
            settings["steps"], settings["save_every_steps"], start_step
        )
        source_name = html.escape(Path(self.resume_info["path"]).name)
        output_name = html.escape(settings["h5_filename"])
        self.resume_label.setText(
            "<b>Exact continuation ready — click “Start continuation” below.</b><br>"
            f"Source: <b>{source_name}</b> at step {start_step:,} "
            f"({int(self.resume_info['frame_count']):,} committed frames) → "
            f"new file: <b>{output_name}</b> (source remains unchanged)<br>"
            f"Additional: {format_scientific_steps(settings['steps'])} steps "
            f"({int(settings['steps']):,} exact) · global range {summary['first']:,} → {summary['last']:,} · "
            f"exactly {summary['count']:,} new frames, saved every "
            f"{format_scientific_steps(settings['save_every_steps'])} steps"
        )

    def _update_preview(self):
        try:
            s = self._settings_from_widgets() if not self.loading_widgets else self.settings
        except Exception:
            return
        self.preview_figure.clear()
        axis = self.preview_figure.add_subplot(111)
        width, height = s["x"], s["y"]
        self.preview_style_combo.blockSignals(True)
        self.preview_style_combo.setCurrentText(s["MAIN_RENDER_MODE"])
        self.preview_style_combo.blockSignals(False)
        axis.add_patch(Rectangle((0, 0), width / 2, height, facecolor="#f2f5f7", edgecolor="#708090"))
        axis.add_patch(Rectangle((width / 2, 0), width / 2, height, facecolor="#dde8ef", edgecolor="#708090"))
        axis.text(
            width * 0.25,
            height * 0.045,
            "Area A",
            ha="center",
            va="top",
            weight="bold",
            fontsize=14,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
            zorder=8,
        )
        axis.text(
            width * 0.75,
            height * 0.045,
            "Area B",
            ha="center",
            va="top",
            weight="bold",
            fontsize=14,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 3},
            zorder=8,
        )
        trap_left = trap_right = None
        if s["USE_TRAP_LAYER"]:
            trap_left = s["TRAP_LAYER_CENTER_X"] - s["TRAP_LAYER_WIDTH"] / 2
            trap_right = trap_left + s["TRAP_LAYER_WIDTH"]
            axis.add_patch(Rectangle((trap_left, 0), s["TRAP_LAYER_WIDTH"], height, facecolor="#ffd580", alpha=0.65, edgecolor="#9a6700", zorder=1))
            axis.text(
                s["TRAP_LAYER_CENTER_X"], height * 0.50,
                "Trap layer",
                ha="center", va="center", fontsize=14, weight="bold", rotation=90, zorder=8,
                bbox={"facecolor": "#fff3cd", "alpha": 0.82, "edgecolor": "none", "pad": 2},
            )
        spot_radius = None
        if s["USE_SPOT"]:
            spot_radius = s["SPOT_DIAMETER"] / 2
            axis.add_patch(Circle((s["SPOT_CENTER_X"], s["SPOT_CENTER_Y"]), spot_radius, facecolor="#c6dbef", edgecolor="#2166ac", alpha=0.8, zorder=2))
            axis.annotate(
                "Spot",
                xy=(s["SPOT_CENTER_X"] + spot_radius, s["SPOT_CENTER_Y"]),
                xytext=(14, -20), textcoords="offset points", fontsize=16, weight="bold",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
                arrowprops={"arrowstyle": "-", "color": "#2166ac"}, zorder=8,
            )

        # Use deterministic random values so changing a setting changes only
        # the sites governed by that setting. Pixel mode uses a capped raster;
        # dot mode uses a fixed candidate cloud. Neither creates the real matrix.
        rng = np.random.default_rng(104729)
        if s["MAIN_RENDER_MODE"] == "pixels":
            maximum_preview_cells = 360_000
            raster_scale = min(1.0, np.sqrt(maximum_preview_cells / max(1, width * height)))
            raster_width = max(2, min(1200, int(round(width * raster_scale))))
            raster_height = max(2, min(900, int(round(height * raster_scale))))
            grid_x = (np.arange(raster_width) + 0.5) * width / raster_width
            grid_y = (np.arange(raster_height) + 0.5) * height / raster_height
            px, py = np.meshgrid(grid_x, grid_y)
            px = px.ravel()
            py = py.ravel()
        else:
            candidate_count = 20_000
            px = rng.uniform(0, width, candidate_count)
            py = rng.uniform(0, height, candidate_count)
        candidate_count = px.size
        in_a = px < width / 2
        site_percent = np.where(in_a, s["max_sol_a"], s["max_sol_b"]).astype(float)
        if s["USE_INITIAL_CONCENTRATION_PROFILE"]:
            a_fraction = np.clip(px / max(width / 2, 1), 0, 1)
            b_fraction = np.clip((px - width / 2) / max(width / 2, 1), 0, 1)
            concentration_percent = np.where(
                in_a,
                s["concentration_profile_a_left"]
                + a_fraction * (s["concentration_profile_a_right"] - s["concentration_profile_a_left"]),
                s["concentration_profile_b_left"]
                + b_fraction * (s["concentration_profile_b_right"] - s["concentration_profile_b_left"]),
            )
        else:
            concentration_percent = np.where(in_a, s["concentration_a"], s["concentration_b"]).astype(float)
        if s["USE_TRAP_LAYER"]:
            in_trap = (px >= trap_left) & (px < trap_right)
            site_percent[in_trap] = s["max_sol_trap_layer"]
            concentration_percent[in_trap] = s["concentration_trap_layer"]
        if s["USE_SPOT"]:
            in_spot = (
                (px - s["SPOT_CENTER_X"]) ** 2
                + (py - s["SPOT_CENTER_Y"]) ** 2
                <= spot_radius ** 2
            )
            site_percent[in_spot] = s["max_sol_spot"]
            concentration_percent[in_spot] = s["concentration_spot"]
        possible = rng.random(candidate_count) < site_percent / 100
        occupied = possible & (rng.random(candidate_count) < concentration_percent / 100)
        available = possible & ~occupied
        available_color = "#2166ac"
        occupied_color = "#b2182b"
        if s["MAIN_RENDER_MODE"] == "pixels":
            raster = np.zeros((candidate_count, 4), dtype=np.float32)
            raster[available] = to_rgba(available_color, 0.82)
            raster[occupied] = to_rgba(occupied_color, 0.92)
            axis.imshow(
                raster.reshape(raster_height, raster_width, 4),
                extent=(0, width, height, 0),
                interpolation="nearest",
                aspect="auto",
                zorder=4,
            )
        else:
            axis.scatter(
                px[available], py[available],
                s=4,
                color=available_color,
                alpha=0.48,
                zorder=4,
            )
            axis.scatter(
                px[occupied], py[occupied],
                s=5,
                color=occupied_color,
                alpha=0.68,
                zorder=5,
            )
        if s["USE_SINK_SOURCE"]:
            thickness = s["SINK_SOURCE_THICKNESS"]
            source_x = 0 if s["SOURCE_SIDE"] == "left" else width - thickness
            sink_x = width - thickness if s["SOURCE_SIDE"] == "left" else 0
            source_color = "#d73027"
            sink_color = "#4575b4"
            axis.add_patch(Rectangle((source_x, 0), thickness, height, facecolor=source_color, alpha=0.9, zorder=6))
            axis.add_patch(Rectangle((sink_x, 0), thickness, height, facecolor=sink_color, alpha=0.9, zorder=6))
            source_label_x = 0.035 if source_x < width / 2 else 0.965
            sink_label_x = 0.035 if sink_x < width / 2 else 0.965
            boundary_label_box = {
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "none",
                "pad": 3,
            }
            axis.text(
                source_label_x, 0.50, "Source",
                ha="center", va="center", rotation=90, color=source_color,
                fontsize=14, weight="bold", transform=axis.transAxes,
                bbox=boundary_label_box, zorder=9,
            )
            axis.text(
                sink_label_x, 0.50, "Sink",
                ha="center", va="center", rotation=90, color=sink_color,
                fontsize=14, weight="bold", transform=axis.transAxes,
                bbox=boundary_label_box, zorder=9,
            )
        legend_handles = [
            Line2D(
                [], [], linestyle="none", marker="o", markersize=12,
                markerfacecolor=available_color, markeredgecolor="none",
                label="Available Site",
            ),
            Line2D(
                [], [], linestyle="none", marker="o", markersize=12,
                markerfacecolor=occupied_color, markeredgecolor="none",
                label="H-Occupied Site",
            ),
        ]
        axis.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=2,
            frameon=True,
            framealpha=0.88,
            fontsize=14,
            handletextpad=0.7,
            columnspacing=1.8,
        )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_aspect("equal")
        axis.set_title("Deterministic setup preview (illustrative, not the generated random matrix)", fontsize=16, pad=11)
        axis.set_xlabel("X [pixels]", fontsize=14)
        axis.set_ylabel("Y [pixels]", fontsize=14)
        axis.tick_params(labelsize=13)
        self.preview_canvas.draw_idle()
        start = self.resume_info["step"] if self.resume_info else 0
        summary = frame_summary(s["steps"], s["save_every_steps"], start)
        raw_bytes = s["x"] * s["y"] * summary["count"]
        disk_free = shutil.disk_usage(result_path("")).free
        storage_warning = (
            " · WARNING: estimated output exceeds currently free result-drive space"
            if raw_bytes * 1.15 > disk_free
            else ""
        )
        self.derived_label.setText(
            f"{format_scientific_steps(s['steps'])} steps ({s['steps']:,} exact) · "
            f"save every {format_scientific_steps(s['save_every_steps'])} · "
            f"exactly {summary['count']:,} frames · global steps {summary['first']:,} → {summary['last']:,} · "
            f"approx. raw snapshots {raw_bytes / (1024 ** 3):.2f} GiB before HDF5 overhead"
            f"{storage_warning}"
        )
        self._update_resume_banner(s, summary)

    def _restore_defaults(self):
        answer = QtWidgets.QMessageBox.question(self, "Restore GUI defaults", "Restore all GUI-managed P6 settings? Hidden power-user settings will not be changed.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.settings = restore_gui_defaults(self.settings)
            self.loading_widgets = True
            self._populate_widgets()
            self.loading_widgets = False
            self._clear_resume(persist=False)
            self._update_dependencies()
            self._update_preview()
            self._load_presentation_controls()
            self.status_label.setText("Restored GUI defaults")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Restore failed", str(exc))

    def _confirm_overwrite(self, path, resume_source=None):
        path = Path(path).resolve()
        if resume_source is not None and path == Path(resume_source).resolve():
            QtWidgets.QMessageBox.warning(self, "Invalid output", "A continuation must use a different output filename from its source.")
            return False
        if not path.exists() or self.settings["GUI_DISABLE_OVERWRITE_WARNING"]:
            return True
        answer = QtWidgets.QMessageBox.warning(
            self, "Replace existing output?", f"The following file already exists and will be replaced:\n\n{path}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
        )
        return answer == QtWidgets.QMessageBox.Yes

    def _start_simulation(self):
        try:
            self.settings = write_gui_settings(self._settings_from_widgets())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot start simulation", str(exc))
            return
        output = result_path(self.settings["h5_filename"])
        source = self.resume_info["path"] if self.resume_info else None
        if not self._confirm_overwrite(output, source):
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.load_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.load_result_action.setEnabled(False)
        self.resume_action.setEnabled(False)
        self.clear_resume_action.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0.0%")
        self.run_progress_start_step = int(self.resume_info["step"]) if self.resume_info else 0
        self.run_progress_end_step = self.run_progress_start_step + int(self.settings["steps"])
        self.run_progress_total_frames = frame_summary(
            self.settings["steps"],
            self.settings["save_every_steps"],
            self.run_progress_start_step,
        )["count"]
        self.status_label.setText(
            "Starting exact continuation…" if self.resume_info else "Starting event-driven simulation…"
        )
        self.simulation_worker = SimulationWorker(self.settings, output, self)
        self.simulation_worker.progress.connect(self._simulation_progress)
        self.simulation_worker.completed.connect(self._simulation_completed)
        self.simulation_worker.cancelled.connect(self._simulation_cancelled)
        self.simulation_worker.failed.connect(self._operation_failed)
        self.simulation_worker.finished.connect(self._simulation_finished)
        self.simulation_worker.start()

    def _cancel_simulation(self):
        if self.simulation_worker:
            self.simulation_worker.request_stop()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling; rolling back to the last scheduled snapshot…")

    def _simulation_progress(self, fraction, message, completed, frames):
        if fraction is None or completed is None or frames is None:
            return
        fraction = max(0.0, min(1.0, float(fraction)))
        self.progress_bar.setValue(int(fraction * 1000))
        self.progress_bar.setFormat(f"{fraction * 100:.1f}%")
        global_step = self.run_progress_start_step + int(completed)
        activity = "Continuing" if self.resume_info else "Running"
        self.status_label.setText(
            f"{activity} event-driven simulation · global step {global_step:,} of "
            f"{self.run_progress_end_step:,} · {int(frames):,}/{self.run_progress_total_frames:,} "
            "frames safely committed"
        )

    def _simulation_completed(self, path):
        self.progress_bar.setValue(1000)
        self.progress_bar.setFormat("100.0%")
        self.status_label.setText(f"Simulation complete: {Path(path).name}")
        self._load_result(Path(path))

    def _simulation_cancelled(self, path):
        self.progress_bar.setFormat("Cancelled")
        self.status_label.setText(f"Simulation cancelled; retained exact checkpoint in {Path(path).name}")
        try:
            self._load_result(Path(path))
        except Exception:
            pass

    def _simulation_finished(self):
        self.simulation_worker = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.continue_button.setEnabled(True)
        self.load_result_action.setEnabled(True)
        self.resume_action.setEnabled(True)
        self.clear_resume_action.setEnabled(self.resume_info is not None)

    def _operation_failed(self, message):
        self.status_label.setText("Operation failed")
        QtWidgets.QMessageBox.critical(self, "P6 Brownian Motion error", message)

    def _choose_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load P6 HDF5", str(result_path("")), "HDF5 files (*.h5 *.hdf5)")
        if path:
            self._load_result(Path(path))

    def _load_result(self, path):
        try:
            self.frame_source = H5FrameSource(path)
            self.frame_slider.setRange(0, self.frame_source.frame_count - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.tabs.setTabVisible(1, True)
            self.tabs.setCurrentIndex(1)
            self._render_selected_result()
            self.status_label.setText(
                f"Loaded {self.frame_source.path.name}: {self.frame_source.frame_count} committed frames ({self.frame_source.status})"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not load HDF5", str(exc))

    def _choose_resume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Continue exact P6 checkpoint", str(result_path("")), "HDF5 files (*.h5 *.hdf5)")
        if not path:
            return
        info = inspect_resume_source(path)
        if not info["valid"]:
            QtWidgets.QMessageBox.warning(self, "File cannot be continued", info["reason"])
            return
        try:
            updated = settings_from_resume_metadata(info["metadata"], self.settings)
            updated["RESUME_FROM_H5"] = str(info["path"])
            if result_path(updated["h5_filename"]).resolve() == info["path"]:
                updated["h5_filename"] = f"continued_{info['path'].name}"
            self.settings = write_gui_settings(updated)
            self.resume_info = info
            self.loading_widgets = True
            self._populate_widgets()
            self.loading_widgets = False
            self.resume_banner.show()
            self.field_labels["steps"].setText("Additional simulation steps")
            self.run_button.setText("Start continuation")
            self.continue_button.setText("Change continuation file…")
            self.clear_resume_action.setEnabled(True)
            self._clear_field_errors()
            self.run_button.setEnabled(True)
            self._update_dependencies()
            self._update_preview()
            self.status_label.setText(
                "Continuation configured; review the yellow summary, then click Start continuation"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not prepare continuation", str(exc))

    def _clear_resume(self, _checked=False, persist=True):
        self.resume_info = None
        self.settings["RESUME_FROM_H5"] = None
        if persist:
            self.settings = write_gui_settings(self.settings)
        self.resume_label.clear()
        self.resume_banner.hide()
        self.field_labels["steps"].setText("Total simulation steps")
        self.run_button.setText("Start simulation")
        self.continue_button.setText("Open HDF5 to continue…")
        self.clear_resume_action.setEnabled(False)
        self._update_dependencies()
        self._update_preview()

    def _frame_changed(self, value):
        if self.frame_source:
            self.frame_label.setText(f"{value + 1}/{self.frame_source.frame_count} · step {int(self.frame_source.steps[value]):,}")
            self.frame_debounce.start()

    def _toggle_play(self):
        if not self.frame_source:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self.play_timer.start(max(20, int(1000 / max(1, self.settings["animation_fps"]))))
            self.play_button.setText("Pause")

    def _advance_frame(self):
        if not self.frame_source:
            return
        value = self.frame_slider.value() + 1
        if value >= self.frame_source.frame_count:
            self.play_timer.stop()
            self.play_button.setText("Play")
            value = 0
        self.frame_slider.setValue(value)

    def _classic_preset(self):
        return {
            "SHOW_MAIN_PANEL": self.settings["SHOW_MAIN_SIMULATION_PANEL"],
            "SHOW_CONCENTRATION_PROFILE_PANEL": self.settings["SHOW_CONCENTRATION_PROFILE_PANEL"],
            "SHOW_NET_FLUX_PANEL": self.settings["SHOW_NET_FLUX_PANEL"],
            "SHOW_HEATMAP_PANEL": False,
            "RENDER_MODE": self.settings["MAIN_RENDER_MODE"],
            "DOT_SIZE_AVAILABLE": self.settings["DOT_SIZE_AVAILABLE"],
            "DOT_SIZE_HYDROGEN": self.settings["DOT_SIZE_HYDROGEN"],
            "DOT_ALPHA_AVAILABLE": self.settings["DOT_ALPHA_AVAILABLE"],
            "DOT_ALPHA_HYDROGEN": self.settings["DOT_ALPHA_HYDROGEN"],
            "COLOR_EMPTY": self.settings["COLOR_EMPTY"],
            "COLOR_AVAILABLE_SPOT": self.settings["COLOR_AVAILABLE_SPOT"],
            "COLOR_HYDROGEN": self.settings["COLOR_HYDROGEN"],
            "COLOR_CONCENTRATION_LINE": self.settings["COLOR_CONCENTRATION_LINE"],
        }

    def _render_selected_result(self):
        if not self.frame_source:
            return
        index = self.frame_slider.value()
        if self.result_views.currentIndex() == 0:
            key = "classic"
            preset = "default"
            overrides = self._classic_preset()
        else:
            key = "diagram"
            preset = self.preset_combo.currentText()
            overrides = self.settings.get("GUI_DIAGRAM_OVERRIDES", {}).get(preset, {})
        self.result_render_serial += 1
        self.latest_result_render_serial = self.result_render_serial
        self.pending_result_render = (
            self.result_render_serial,
            key,
            self.frame_source.path,
            index,
            preset,
            deepcopy(overrides),
        )
        self.frame_label.setText(
            f"{index + 1}/{self.frame_source.frame_count} · step "
            f"{int(self.frame_source.steps[index]):,}"
        )
        self.status_label.setText(f"Rendering {preset} frame {index + 1}…")
        self._start_pending_result_render()

    def _start_pending_result_render(self):
        if self.result_render_worker is not None or self.pending_result_render is None:
            return
        request = self.pending_result_render
        self.pending_result_render = None
        self.result_render_worker = ResultRenderWorker(*request, parent=self)
        self.result_render_worker.completed.connect(self._result_render_completed)
        self.result_render_worker.failed.connect(self._result_render_failed)
        self.result_render_worker.finished.connect(self._result_render_finished)
        self.result_render_worker.start()

    def _result_render_completed(self, serial, key, index, figure):
        current_key = "classic" if self.result_views.currentIndex() == 0 else "diagram"
        if (
            serial == self.latest_result_render_serial
            and key == current_key
            and self.frame_source is not None
            and index == self.frame_slider.value()
        ):
            layout = self.classic_layout if key == "classic" else self.diagram_layout
            self._install_figure(key, layout, figure)
            self.status_label.setText(f"Rendered frame {index + 1}")
        else:
            figure.clear()

    def _result_render_failed(self, serial, message):
        if serial == self.latest_result_render_serial:
            self.status_label.setText(f"Could not render result: {message}")

    def _result_render_finished(self):
        worker = self.result_render_worker
        self.result_render_worker = None
        if worker is not None:
            worker.deleteLater()
        if self.pending_result_render is not None:
            QtCore.QTimer.singleShot(0, self._start_pending_result_render)

    def _install_figure(self, key, layout, figure):
        old = self.current_canvas.get(key)
        if old:
            canvas, toolbar = old
            old_figure = canvas.figure
            canvas.figure = figure
            figure.set_canvas(canvas)
            pixel_ratio = float(canvas.devicePixelRatioF())
            target_width = max(1.0, canvas.width() * pixel_ratio)
            target_height = max(1.0, canvas.height() * pixel_ratio)
            figure.set_size_inches(
                target_width / figure.dpi,
                target_height / figure.dpi,
                forward=False,
            )
            if old_figure is not figure:
                old_figure.clear()
                old_figure.set_canvas(None)
            toolbar.update()
            canvas.updateGeometry()
            canvas.draw_idle()
            return
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(canvas, 1)
        layout.addWidget(toolbar)
        self.current_canvas[key] = (canvas, toolbar)
        canvas.draw_idle()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _load_presentation_controls(self):
        preset = self.settings.get("GUI_DIAGRAM_PRESET", "default")
        if preset not in discover_diagram_presets():
            preset = "default"
        self.presentation_loading = True
        self.preset_combo.setCurrentText(preset)
        self._clear_layout(self.presentation_fields_layout)
        self.presentation_widgets = {}
        overrides = self.settings.get("GUI_DIAGRAM_OVERRIDES", {}).get(preset, {})
        self.presentation_settings = load_diagram_settings(preset, overrides)
        mode = self.presentation_settings.get("RENDER_MODE")
        groups = {name: QtWidgets.QGroupBox(name.title()) for name in ("common", "profile", "heatmap", "printer", "area")}
        layouts = {name: QtWidgets.QVBoxLayout(group) for name, group in groups.items()}
        active_groups = {"common"}
        if self.presentation_settings.get("SHOW_CONCENTRATION_PROFILE_PANEL"):
            active_groups.add("profile")
        if self.presentation_settings.get("SHOW_HEATMAP_PANEL") or mode == "concentration_heatmap":
            active_groups.add("heatmap")
        if mode == "printer_glyphs":
            active_groups.add("printer")
        if mode == "area_summary_dots":
            active_groups.add("area")
        for key, label_text, kind, group_name in PRESENTATION_FIELDS:
            if group_name not in active_groups or key not in self.presentation_settings:
                continue
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(label_text)
            label.setWordWrap(True)
            label.setMinimumWidth(165)
            widget = self._make_widget(kind)
            value = self.presentation_settings[key]
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
                widget.toggled.connect(lambda _checked, _key=key, _kind=kind: self._presentation_edited(_key, _kind))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(str(value))
                widget.currentTextChanged.connect(lambda _text, _key=key, _kind=kind: self._presentation_edited(_key, _kind))
            else:
                widget.setText(str(value))
                widget.editingFinished.connect(lambda _key=key, _kind=kind: self._presentation_edited(_key, _kind))
            row_layout.addWidget(label)
            row_layout.addWidget(widget, 1)
            layouts[group_name].addWidget(row)
            self.presentation_widgets[key] = widget
        for name in ("common", "profile", "heatmap", "printer", "area"):
            if layouts[name].count():
                self.presentation_fields_layout.addWidget(groups[name])
        self.presentation_fields_layout.addStretch(1)
        self.presentation_loading = False

    def _presentation_edited(self, key, kind):
        if self.presentation_loading:
            return
        widget = self.presentation_widgets[key]
        try:
            if isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                value = widget.currentText()
            elif kind == "int":
                value = parse_step_count(widget.text().strip())
            elif kind == "float":
                value = float(widget.text().strip())
            elif kind in ("float_pair", "optional_pair", "float_list"):
                text = widget.text().strip()
                if kind == "optional_pair" and text.lower() in ("", "none"):
                    value = None
                else:
                    parts = [float(part.strip()) for part in text.strip("()[]").split(",") if part.strip()]
                    if kind in ("float_pair", "optional_pair") and len(parts) != 2:
                        raise ValueError("Enter exactly two comma-separated values")
                    if kind == "float_list" and not parts:
                        raise ValueError("Enter at least one value")
                    value = tuple(parts)
            else:
                value = widget.text().strip()
            preset = self.preset_combo.currentText()
            overrides_by_preset = deepcopy(self.settings.get("GUI_DIAGRAM_OVERRIDES", {}))
            overrides = dict(overrides_by_preset.get(preset, {}))
            overrides[key] = value
            load_diagram_settings(preset, overrides)
            overrides_by_preset[preset] = overrides
            self.settings["GUI_DIAGRAM_OVERRIDES"] = overrides_by_preset
            self.settings = write_gui_settings(self.settings)
            widget.setStyleSheet("")
            self.status_label.setText(f"Saved {key} for {preset}")
            if key.startswith("SHOW_") or key == "RENDER_MODE":
                self._load_presentation_controls()
            self._render_selected_result()
        except Exception as exc:
            widget.setStyleSheet("border:1px solid #c62828;")
            self.status_label.setText(f"Invalid presentation setting: {exc}")

    def _preset_changed(self, preset):
        if self.presentation_loading or not preset:
            return
        self.settings["GUI_DIAGRAM_PRESET"] = preset
        try:
            self.settings = write_gui_settings(self.settings)
            self._load_presentation_controls()
            if self.frame_source:
                if self.result_views.currentIndex() != 1:
                    self.result_views.setCurrentIndex(1)
                else:
                    self._render_selected_result()
        except Exception as exc:
            self.status_label.setText(f"Could not select preset: {exc}")

    def _refresh_preset_combo(self, selected):
        self.presentation_loading = True
        try:
            self.preset_combo.clear()
            self.preset_combo.addItems(discover_diagram_presets())
            self.preset_combo.setCurrentText(selected)
        finally:
            self.presentation_loading = False

    def _save_current_preset(self):
        current = self.preset_combo.currentText()
        suggested_name = f"{current.replace('_', ' ').title()} Custom"
        display_name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Save diagram preset",
            "Name for the new preset:",
            text=suggested_name,
        )
        if not accepted:
            return
        try:
            overrides = self.settings.get("GUI_DIAGRAM_OVERRIDES", {}).get(current, {})
            resolved = load_diagram_settings(current, overrides)
            path = save_diagram_preset(display_name, resolved)
            self.settings["GUI_DIAGRAM_PRESET"] = path.stem
            self.settings = write_gui_settings(self.settings)
            self._refresh_preset_combo(path.stem)
            self._load_presentation_controls()
            if self.frame_source:
                if self.result_views.currentIndex() != 1:
                    self.result_views.setCurrentIndex(1)
                else:
                    self._render_selected_result()
            self.status_label.setText(f"Saved and selected new preset: {path.name}")
        except Exception as exc:
            self.presentation_loading = False
            QtWidgets.QMessageBox.warning(self, "Could not save preset", str(exc))

    def _manage_saved_presets(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Manage saved diagram presets")
        dialog.resize(620, 360)
        layout = QtWidgets.QVBoxLayout(dialog)
        explanation = QtWidgets.QLabel(
            "GUI-created presets are stored in:\n"
            f"{DIAGRAM_PRESETS_DIR}\n\n"
            "Only presets created with “Save current settings as new preset” are listed. "
            "Shipped and manually maintained presets are protected."
        )
        explanation.setWordWrap(True)
        explanation.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(explanation)
        preset_list = QtWidgets.QListWidget()
        preset_list.addItems(list_custom_diagram_presets())
        layout.addWidget(preset_list, 1)
        button_row = QtWidgets.QHBoxLayout()
        delete_button = QtWidgets.QPushButton("Delete selected preset")
        delete_button.setEnabled(bool(preset_list.currentItem()))
        close_button = QtWidgets.QPushButton("Close")
        button_row.addWidget(delete_button)
        button_row.addStretch(1)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        preset_list.currentItemChanged.connect(
            lambda current, _previous: delete_button.setEnabled(current is not None)
        )
        close_button.clicked.connect(dialog.accept)

        def delete_selected():
            item = preset_list.currentItem()
            if item is None:
                return
            preset = item.text()
            path = DIAGRAM_PRESETS_DIR / f"{preset}.py"
            answer = QtWidgets.QMessageBox.warning(
                dialog,
                "Permanently delete preset?",
                "Delete this GUI-created preset permanently?\n\n"
                f"{path}\n\n"
                "Its saved GUI overrides will also be removed. This cannot be undone.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
            try:
                selected = self.preset_combo.currentText()
                if selected == preset:
                    selected = "default"
                overrides = deepcopy(self.settings.get("GUI_DIAGRAM_OVERRIDES", {}))
                overrides.pop(preset, None)
                self.settings["GUI_DIAGRAM_OVERRIDES"] = overrides
                self.settings["GUI_DIAGRAM_PRESET"] = selected
                self.settings = write_gui_settings(self.settings)
                deleted_path = delete_custom_diagram_preset(preset)
                self._refresh_preset_combo(selected)
                self._load_presentation_controls()
                preset_list.takeItem(preset_list.row(item))
                if self.frame_source and self.result_views.currentIndex() == 1:
                    self._render_selected_result()
                self.status_label.setText(f"Deleted custom preset: {deleted_path.name}")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dialog, "Could not delete preset", str(exc))

        delete_button.clicked.connect(delete_selected)
        dialog.exec_()

    def _reset_current_preset(self):
        preset = self.preset_combo.currentText()
        overrides = deepcopy(self.settings.get("GUI_DIAGRAM_OVERRIDES", {}))
        overrides.pop(preset, None)
        self.settings["GUI_DIAGRAM_OVERRIDES"] = overrides
        self.settings = write_gui_settings(self.settings)
        self._load_presentation_controls()
        self._render_selected_result()
        self.status_label.setText(f"Reset {preset} to its shipped settings")

    def _export_current_figure(self):
        if not self.frame_source:
            return
        try:
            self.settings = write_gui_settings(self._settings_from_widgets())
            output = result_path(self.settings["GUI_FIGURE_FILENAME"])
            if not self._confirm_overwrite(output):
                return
            preset = self.preset_combo.currentText()
            overrides = self.settings.get("GUI_DIAGRAM_OVERRIDES", {}).get(preset, {})
            fig = render_diagram_figure(self.frame_source.path, self.frame_slider.value(), preset, overrides)
            if output.suffix.lower() == ".svg":
                with plt.rc_context({"svg.fonttype": "none"}):
                    fig.savefig(output, bbox_inches="tight")
            else:
                fig.savefig(output, dpi=300, bbox_inches="tight")
            plt.close(fig)
            self.status_label.setText(f"Exported diagram: {output.name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Figure export failed", str(exc))

    def _start_animation_render(self):
        if not self.frame_source or not self.ffmpeg:
            return
        try:
            self.settings = write_gui_settings(self._settings_from_widgets())
            output = result_path(self.settings["animation_filename"])
            if not self._confirm_overwrite(output):
                return
            preset = self.preset_combo.currentText()
            overrides = self.settings.get("GUI_DIAGRAM_OVERRIDES", {}).get(preset, {})
            self.animation_worker = AnimationWorker(
                self.frame_source.path, output, preset, overrides,
                self.settings["animation_fps"], self.settings["render_every_nth_frame"], self.ffmpeg, self,
            )
            self.animation_worker.progress.connect(self._animation_progress)
            self.animation_worker.completed.connect(self._animation_completed)
            self.animation_worker.cancelled.connect(lambda: self.status_label.setText("MP4 rendering cancelled; partial file removed"))
            self.animation_worker.failed.connect(self._operation_failed)
            self.animation_worker.finished.connect(self._animation_finished)
            self.render_mp4_button.setEnabled(False)
            self.cancel_render_button.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0.0%")
            self.status_label.setText("Starting MP4 render…")
            self.animation_worker.start()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not render MP4", str(exc))

    def _animation_progress(self, fraction, message):
        fraction = max(0.0, min(1.0, float(fraction)))
        self.progress_bar.setValue(int(fraction * 1000))
        self.progress_bar.setFormat(f"{fraction * 100:.1f}%")
        self.status_label.setText(message)

    def _animation_completed(self, path):
        self.progress_bar.setValue(1000)
        self.progress_bar.setFormat("100.0%")
        self.status_label.setText(f"Exported MP4: {Path(path).name}")

    def _animation_finished(self):
        self.animation_worker = None
        self.render_mp4_button.setEnabled(bool(self.ffmpeg))
        self.cancel_render_button.setEnabled(False)

    def _cancel_animation_render(self):
        if self.animation_worker:
            self.animation_worker.request_stop()
            self.cancel_render_button.setEnabled(False)
            self.status_label.setText("Cancelling MP4 render…")

    def closeEvent(self, event):
        if self.simulation_worker and self.simulation_worker.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "Simulation running",
                "Cancel the simulation, roll back to its last scheduled snapshot, and close?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.simulation_worker.request_stop()
            self.simulation_worker.wait(15000)
            if self.simulation_worker.isRunning():
                event.ignore()
                return
        self.pending_result_render = None
        if self.animation_worker and self.animation_worker.isRunning():
            self.animation_worker.request_stop()
        if self.result_render_worker and self.result_render_worker.isRunning():
            self.result_render_worker.wait(30000)
        if self.animation_worker and self.animation_worker.isRunning():
            self.animation_worker.wait(10000)
        self.play_timer.stop()
        event.accept()


def launch_gui():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    try:
        settings = load_gui_settings()
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "P6 configuration error", str(exc))
        return 2
    window = MainWindow(settings)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
