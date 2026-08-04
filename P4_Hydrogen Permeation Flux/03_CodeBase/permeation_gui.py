"""PyQt5 interface for P4 Hydrogen Permeation Flux."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from permeation_cases import RUNTIME_CONFIG_PATH, estimate_case_count, list_presets, load_preset, validate_settings
from permeation_diagrams import build_figure
from permeation_gui_support import (
    create_profile,
    ensure_runtime_state,
    expected_output_paths,
    export_loaded_results,
    load_result,
    normalize_gui_state,
    recover_runtime_defaults,
    rename_profile,
    result_matches_settings,
    restore_defaults,
    run_atlas_job,
    scientific_settings_hash,
    validate_profile_name,
    validate_result_name,
    write_runtime_state,
)
from permeation_model import SimulationCancelled


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "02_Results"
RESOURCES_DIR = REPO_ROOT / "Resources"
if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))

from Common.launch_ready import StartupReadySignal


APP_NAME = "WeldCraft - Hydrogen Permeation Flux"
METRIC_NAMES = ("t10", "t50", "t90", "time_lag", "peak_flux", "final_flux", "overshoot")


FIELD_GROUPS = [
    ("Reference membrane", [
        ("simulation.reference_length_mm", "Reference thickness [mm]", "float", False),
        ("simulation.reference_diffusivity_mm2_s", "Reference diffusivity [mm²/s]", "float", False),
        ("simulation.reference_concentration_mol_mm3", "Reference concentration [mol/mm³]", "optional_float", False),
        ("simulation.end_time_ref", "Simulation end [τref]", "float", False),
    ]),
    ("Ideal response sweeps", [
        ("ideal.diffusivity_ratios", "Diffusivity ratios", "list_float", False),
        ("ideal.length_ratios", "Thickness ratios", "list_float", False),
        ("ideal.solubility_ratios", "Solubility ratios", "list_float", False),
    ]),
    ("Entry condition", [
        ("surface.entry_concentration_ratios", "Entry concentration ratios", "list_float", False),
        ("surface.onset_fraction_of_ideal_t50", "Onset / ideal t50", "float", False),
        ("surface.time_constant_fraction_of_ideal_t50", "Time constant / ideal t50", "float", False),
    ]),
    ("McNabb–Foster trapping", [
        ("trapping.capacity_ratios", "Capacity ratios", "list_float", False),
        ("trapping.release_half_times_ref", "Release half-times [τref]", "list_float", False),
        ("trapping.capture_rate_refs", "Capture rates [1/τref]", "list_float", False),
        ("trapping.capture_rate_ref", "Default capture rate", "float", True),
        ("trapping.end_time_ref", "Trap simulation end [τref]", "float", True),
        ("trapping.capture_sweep_capacity_ratio", "Capture-sweep capacity", "float", True),
        ("trapping.capture_sweep_release_half_time_ref", "Capture-sweep half-time", "float", True),
        ("trapping.capacity_sweep_release_half_time_ref", "Capacity-sweep half-time", "float", True),
        ("trapping.release_sweep_capacity_ratio", "Release-sweep capacity", "float", True),
        ("trapping.map_capacity_ratios", "Map capacity ratios", "list_float", True),
        ("trapping.map_release_half_times_ref", "Map release half-times", "list_float", True),
        ("trapping.combined_cases", "Combined cases [JSON]", "json", True),
    ]),
    ("Residual prefill", [
        ("prefill.initial_fraction", "Initial fraction", "float", False),
        ("prefill.target_center_fraction", "Primary centre target", "float", False),
        ("prefill.target_center_fractions", "Centre targets", "list_float", False),
        ("prefill.maximum_age_time_ref", "Maximum ageing [τref]", "float", True),
    ]),
    ("Numerical controls", [
        ("simulation.n_nodes", "Spatial nodes (odd)", "int", True),
        ("simulation.n_output", "Stored output samples", "int", True),
        ("simulation.diffusion_safety", "Diffusion safety", "float", True),
        ("simulation.reaction_safety", "Reaction safety", "float", True),
        ("simulation.max_internal_steps", "Maximum internal steps", "int", True),
    ]),
    ("Diagram and publication", [
        ("diagram.normalization", "Flux normalization", "normalization", False),
        ("diagram.time_axis", "Time axis", "time_axis", False),
        ("diagram.comparison_window_ref", "Comparison window [τref]", "float", False),
        ("diagram.response_metric", "Response-map metric", "metric", False),
        ("diagram.dpi", "Export DPI", "int", False),
        ("diagram.figure_scale", "Figure scale", "float", True),
        ("diagram.font_scale", "Font scale", "float", True),
        ("diagram.line_width_scale", "Line-width scale", "float", True),
        ("diagram.marker_scale", "Marker scale", "float", True),
        ("diagram.grid_visible", "Show grid", "bool", True),
        ("diagram.grid_style", "Grid style", "grid_style", True),
        ("diagram.legend_mode", "Legend placement", "legend_mode", True),
        ("diagram.show_title", "Show main title", "bool", True),
        ("diagram.title_override", "Main title override", "text", True),
    ]),
]


CHOICES = {
    "normalization": [
        ("Common reference J/Jref", "common_reference"),
        ("Per curve J/Jss", "per_curve"),
        ("Physical flux", "physical"),
    ],
    "time_axis": [("Minutes", "minutes"), ("Seconds", "seconds"), ("Reference time", "reference"), ("Fourier number", "fo")],
    "metric": [(name, name) for name in METRIC_NAMES],
    "grid_style": [("Dotted", ":"), ("Dashed", "--"), ("Solid", "-"), ("Dash-dot", "-.")],
    "legend_mode": [("Original", "original"), ("Best", "best"), ("Outside", "outside"), ("Hidden", "hidden")],
}


def _get_path(mapping, dotted):
    value = mapping
    for part in dotted.split("."):
        value = value[part]
    return value


def _set_path(mapping, dotted, value):
    target = mapping
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


class RunWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, settings, preset, result_name, formats, output_directory, parent=None):
        super().__init__(parent)
        self.settings = deepcopy(settings)
        self.preset = preset
        self.result_name = result_name
        self.formats = list(formats)
        self.output_directory = Path(output_directory)
        self.cancel_flag = np.zeros(1, dtype=np.int8)

    def request_cancel(self):
        self.cancel_flag[0] = 1

    def run(self):
        try:
            outcome = run_atlas_job(
                self.settings, self.preset, self.result_name, self.formats,
                self.output_directory, self.progress.emit, self.cancel_flag,
            )
            self.completed.emit(outcome)
        except SimulationCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExportWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, results, metadata, settings, result_name, formats, output_directory, parent=None):
        super().__init__(parent)
        self.results = results
        self.metadata = deepcopy(metadata)
        self.settings = deepcopy(settings)
        self.result_name = result_name
        self.formats = list(formats)
        self.output_directory = Path(output_directory)
        self.cancel_flag = np.zeros(1, dtype=np.int8)

    def request_cancel(self):
        self.cancel_flag[0] = 1

    def run(self):
        try:
            paths = export_loaded_results(
                self.results, self.metadata, self.settings, self.result_name,
                self.formats, self.output_directory, self.progress.emit, self.cancel_flag,
            )
            self.completed.emit(paths)
        except SimulationCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings, gui_state, profiles, config_path=RUNTIME_CONFIG_PATH, results_dir=RESULTS_DIR):
        super().__init__()
        self.settings = validate_settings(settings)
        self.gui_state = normalize_gui_state(gui_state)
        self.profiles = deepcopy(profiles)
        self.config_path = Path(config_path)
        self.results_dir = Path(results_dir)
        self.field_widgets = {}
        self.field_rows = []
        self.loading_widgets = True
        self.worker = None
        self.loaded_results = None
        self.loaded_metadata = {}
        self.loaded_result_path = None
        self.preview_canvas = None
        self.preview_toolbar = None
        self.preview_figure = None
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._render_preview)

        self.setWindowTitle(APP_NAME)
        icon = RESOURCES_DIR / "Images" / "WeldCraft.ico"
        self.setWindowIcon(QtGui.QIcon(str(icon)))
        self.resize(1500, 920)
        self._build_ui()
        self._populate_all()
        self.loading_widgets = False
        self._connect_signals()
        self._restore_window_state()
        self._update_preset_summary()
        self._update_last_result_action()
        self._update_stale_banner()
        self.startup_ready_signal = StartupReadySignal(self)
        self.showMaximized()

    def _build_ui(self):
        self.setStyleSheet(
            "QGroupBox{font-weight:bold;margin-top:10px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}"
            "QLineEdit,QComboBox,QSpinBox{min-height:24px;}"
            "QPushButton{min-height:28px;padding:2px 10px;}"
        )
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(self.splitter, 1)

        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumWidth(410)
        self.settings_panel = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(self.settings_panel)

        self.bam_logo = QtWidgets.QLabel()
        self.bam_logo.setAlignment(QtCore.Qt.AlignCenter)
        self.bam_logo.setMaximumHeight(72)
        bam_path = RESOURCES_DIR / "Images" / "BAM Logo.png"
        if bam_path.exists():
            pixmap = QtGui.QPixmap(str(bam_path))
            self.bam_logo.setPixmap(pixmap.scaledToHeight(64, QtCore.Qt.SmoothTransformation))
        settings_layout.addWidget(self.bam_logo)

        profile_group = QtWidgets.QGroupBox("Working profiles")
        profile_layout = QtWidgets.QGridLayout(profile_group)
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_load = QtWidgets.QPushButton("Load")
        self.profile_save = QtWidgets.QPushButton("Save new…")
        self.profile_rename = QtWidgets.QPushButton("Rename…")
        self.profile_delete = QtWidgets.QPushButton("Delete")
        profile_layout.addWidget(self.profile_combo, 0, 0, 1, 2)
        profile_layout.addWidget(self.profile_load, 1, 0)
        profile_layout.addWidget(self.profile_save, 1, 1)
        profile_layout.addWidget(self.profile_rename, 2, 0)
        profile_layout.addWidget(self.profile_delete, 2, 1)
        settings_layout.addWidget(profile_group)

        self.advanced_toggle = QtWidgets.QCheckBox("Show advanced settings")
        settings_layout.addWidget(self.advanced_toggle)

        for group_name, fields in FIELD_GROUPS:
            group = QtWidgets.QGroupBox(group_name)
            group_layout = QtWidgets.QVBoxLayout(group)
            for path, label_text, kind, advanced in fields:
                row = QtWidgets.QWidget()
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                label = QtWidgets.QLabel(label_text)
                label.setWordWrap(True)
                label.setMinimumWidth(155)
                widget = self._make_field(kind)
                widget.setToolTip(f"Persistent P4 setting: {path}")
                row_layout.addWidget(label)
                row_layout.addWidget(widget, 1)
                group_layout.addWidget(row)
                self.field_widgets[path] = (widget, kind)
                self.field_rows.append((row, advanced))
            settings_layout.addWidget(group)

        self.restore_button = QtWidgets.QPushButton("Restore shipped defaults")
        settings_layout.addWidget(self.restore_button)
        settings_layout.addStretch(1)
        settings_scroll.setWidget(self.settings_panel)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs, 1)
        self._build_setup_tab()
        self._build_results_tab()

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.status_label = QtWidgets.QLabel("Ready")
        status = QtWidgets.QHBoxLayout()
        status.addWidget(self.status_label, 1)
        status.addWidget(self.progress_bar)
        right_layout.addLayout(status)

        self.splitter.addWidget(settings_scroll)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

    def _build_setup_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        heading = QtWidgets.QLabel("Hydrogen Permeation Flux")
        heading.setStyleSheet("font-size:20px;font-weight:600;")
        layout.addWidget(heading)
        form = QtWidgets.QFormLayout()
        self.preset_combo = QtWidgets.QComboBox()
        for name, values in list_presets().items():
            self.preset_combo.addItem(name, name)
            self.preset_combo.setItemData(self.preset_combo.count() - 1, values.get("description", ""), QtCore.Qt.ToolTipRole)
        self.result_name_edit = QtWidgets.QLineEdit()
        form.addRow("Figure preset", self.preset_combo)
        form.addRow("Result name", self.result_name_edit)
        layout.addLayout(form)
        self.preset_description = QtWidgets.QLabel()
        self.preset_description.setWordWrap(True)
        layout.addWidget(self.preset_description)
        self.case_summary = QtWidgets.QLabel()
        self.case_summary.setWordWrap(True)
        self.case_summary.setStyleSheet("background:#eef3f8;border:1px solid #9ab0c4;padding:8px;")
        layout.addWidget(self.case_summary)

        formats = QtWidgets.QGroupBox("Outputs created during Run")
        format_layout = QtWidgets.QHBoxLayout(formats)
        self.png_check = QtWidgets.QCheckBox("PNG")
        self.pdf_check = QtWidgets.QCheckBox("PDF")
        self.svg_check = QtWidgets.QCheckBox("SVG")
        format_layout.addWidget(QtWidgets.QLabel("HDF5 (always)"))
        format_layout.addStretch(1)
        format_layout.addWidget(self.png_check)
        format_layout.addWidget(self.pdf_check)
        format_layout.addWidget(self.svg_check)
        layout.addWidget(formats)

        buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.load_result_button = QtWidgets.QPushButton("Load existing HDF5…")
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.load_result_button)
        layout.addLayout(buttons)
        self.last_result_button = QtWidgets.QPushButton("Reopen last result")
        self.last_result_button.setVisible(False)
        layout.addWidget(self.last_result_button)
        layout.addWidget(QtWidgets.QLabel("Run log"))
        self.status_log = QtWidgets.QPlainTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumBlockCount(500)
        layout.addWidget(self.status_log, 1)
        self.tabs.addTab(tab, "Setup / Run")

    def _build_results_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.stale_banner = QtWidgets.QLabel()
        self.stale_banner.setWordWrap(True)
        self.stale_banner.setStyleSheet(
            "background:#fff3cd;color:#664d03;border:2px solid #e0a800;padding:8px;font-weight:600;"
        )
        self.stale_banner.hide()
        layout.addWidget(self.stale_banner)
        toolbar = QtWidgets.QHBoxLayout()
        self.result_info = QtWidgets.QLabel("No result loaded")
        self.result_info.setWordWrap(True)
        self.figure_combo = QtWidgets.QComboBox()
        self.open_directory_button = QtWidgets.QPushButton("Open result directory")
        toolbar.addWidget(self.result_info, 1)
        toolbar.addWidget(QtWidgets.QLabel("Figure"))
        toolbar.addWidget(self.figure_combo)
        toolbar.addWidget(self.open_directory_button)
        layout.addLayout(toolbar)
        self.figure_container = QtWidgets.QWidget()
        self.figure_layout = QtWidgets.QVBoxLayout(self.figure_container)
        self.figure_placeholder = QtWidgets.QLabel("Run a preset or load a P4 HDF5 result to preview its response figures.")
        self.figure_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.figure_placeholder.setWordWrap(True)
        self.figure_layout.addWidget(self.figure_placeholder, 1)
        layout.addWidget(self.figure_container, 3)
        self.metrics_table = QtWidgets.QTableWidget(0, 1 + len(METRIC_NAMES))
        self.metrics_table.setHorizontalHeaderLabels(["Case", *METRIC_NAMES])
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.metrics_table, 1)
        export = QtWidgets.QHBoxLayout()
        self.rerender_button = QtWidgets.QPushButton("Export checked formats")
        self.rerender_button.setEnabled(False)
        export.addStretch(1)
        export.addWidget(self.rerender_button)
        layout.addLayout(export)
        self.results_tab_index = self.tabs.addTab(tab, "Results / Export")

    def _make_field(self, kind):
        if kind == "bool":
            return QtWidgets.QCheckBox()
        if kind == "int":
            widget = QtWidgets.QSpinBox()
            widget.setRange(1, 2_000_000_000)
            return widget
        if kind in CHOICES:
            widget = QtWidgets.QComboBox()
            for label, value in CHOICES[kind]:
                widget.addItem(label, value)
            return widget
        return QtWidgets.QLineEdit()

    def _populate_profiles(self, selected=None):
        selected = selected or self.profile_combo.currentText()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(self.profiles, key=str.casefold))
        index = self.profile_combo.findText(selected)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)
        enabled = bool(self.profiles)
        for button in (self.profile_load, self.profile_rename, self.profile_delete):
            button.setEnabled(enabled)

    def _populate_all(self):
        self._populate_profiles()
        for path, (widget, kind) in self.field_widgets.items():
            self._set_widget_value(widget, kind, _get_path(self.settings, path))
        preset_index = self.preset_combo.findData(self.gui_state["preset"])
        self.preset_combo.setCurrentIndex(max(0, preset_index))
        self.result_name_edit.setText(self.gui_state["result_name"])
        formats = set(self.gui_state["formats"])
        self.png_check.setChecked("png" in formats)
        self.pdf_check.setChecked("pdf" in formats)
        self.svg_check.setChecked("svg" in formats)
        self.advanced_toggle.setChecked(self.gui_state["advanced_visible"])
        self._apply_advanced_visibility()

    def _set_widget_value(self, widget, kind, value):
        widget.setStyleSheet("")
        if kind == "bool":
            widget.setChecked(bool(value))
        elif kind == "int":
            widget.setValue(int(value))
        elif kind in CHOICES:
            index = widget.findData(value)
            widget.setCurrentIndex(max(0, index))
        elif kind == "list_float":
            widget.setText(", ".join(f"{float(item):g}" for item in value))
        elif kind == "json":
            widget.setText(json.dumps(value, ensure_ascii=False, separators=(", ", ": ")))
        elif kind == "optional_float":
            widget.setText("" if value is None else f"{float(value):.12g}")
        else:
            widget.setText(str(value) if kind == "text" else f"{float(value):.12g}")

    def _parse_widget(self, widget, kind):
        if kind == "bool":
            return widget.isChecked()
        if kind == "int":
            return int(widget.value())
        if kind in CHOICES:
            return widget.currentData()
        text = widget.text().strip()
        if kind == "optional_float":
            return None if not text else float(text)
        if kind == "list_float":
            values = [item.strip() for item in text.split(",") if item.strip()]
            if not values:
                raise ValueError("Enter at least one comma-separated number.")
            return [float(item) for item in values]
        if kind == "json":
            return json.loads(text)
        if kind == "text":
            return text
        return float(text)

    def _settings_from_widgets(self):
        values = deepcopy(self.settings)
        for path, (widget, kind) in self.field_widgets.items():
            _set_path(values, path, self._parse_widget(widget, kind))
        values["diagram"]["formats"] = self._checked_formats()
        return validate_settings(values)

    def _connect_signals(self):
        for path, (widget, kind) in self.field_widgets.items():
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(lambda _path=path: self._persist_field(_path))
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.valueChanged.connect(lambda _value, _path=path: self._persist_field(_path))
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(lambda _value, _path=path: self._persist_field(_path))
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.stateChanged.connect(lambda _value, _path=path: self._persist_field(_path))
        self.advanced_toggle.toggled.connect(self._advanced_changed)
        self.restore_button.clicked.connect(self._restore_defaults)
        self.profile_save.clicked.connect(self._save_profile)
        self.profile_load.clicked.connect(self._load_profile)
        self.profile_rename.clicked.connect(self._rename_profile)
        self.profile_delete.clicked.connect(self._delete_profile)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.result_name_edit.editingFinished.connect(self._state_changed)
        for checkbox in (self.png_check, self.pdf_check, self.svg_check):
            checkbox.stateChanged.connect(self._state_changed)
        self.tabs.currentChanged.connect(self._state_changed)
        self.run_button.clicked.connect(self._start_run)
        self.cancel_button.clicked.connect(self._cancel_operation)
        self.load_result_button.clicked.connect(self._choose_result)
        self.last_result_button.clicked.connect(self._reopen_last_result)
        self.figure_combo.currentIndexChanged.connect(self._figure_changed)
        self.rerender_button.clicked.connect(self._start_export)
        self.open_directory_button.clicked.connect(self._open_result_directory)

    def _persist_field(self, path):
        if self.loading_widgets:
            return
        widget = self.field_widgets[path][0]
        old_hash = scientific_settings_hash(self.settings)
        try:
            checked = self._settings_from_widgets()
            self.settings = checked
            widget.setStyleSheet("")
            self._persist_all()
        except Exception as exc:
            widget.setStyleSheet("border:2px solid #b00020;")
            self.status_label.setText(f"Setting not saved: {exc}")
            return
        if scientific_settings_hash(self.settings) != old_hash:
            self._update_stale_banner()
        if path.startswith("diagram."):
            self.preview_timer.start()
        self._update_preset_summary()
        self.status_label.setText(f"Saved {path}")

    def _state_changed(self, *_args):
        if self.loading_widgets:
            return
        try:
            self.gui_state["result_name"] = validate_result_name(self.result_name_edit.text())
            self.result_name_edit.setStyleSheet("")
        except Exception as exc:
            self.result_name_edit.setStyleSheet("border:2px solid #b00020;")
            self.status_label.setText(str(exc))
            return
        self.gui_state["preset"] = self.preset_combo.currentData()
        self.gui_state["formats"] = self._checked_formats()
        self.gui_state["selected_tab"] = self.tabs.currentIndex()
        self.settings["diagram"]["formats"] = self._checked_formats()
        self._persist_all()

    def _persist_all(self):
        self.settings, self.gui_state, self.profiles = write_runtime_state(
            self.settings, self.gui_state, self.profiles, self.config_path
        )

    def _checked_formats(self):
        return [name for name, widget in (("png", self.png_check), ("pdf", self.pdf_check), ("svg", self.svg_check)) if widget.isChecked()]

    def _advanced_changed(self, checked):
        self.gui_state["advanced_visible"] = bool(checked)
        self._apply_advanced_visibility()
        if not self.loading_widgets:
            self._persist_all()

    def _apply_advanced_visibility(self):
        visible = self.advanced_toggle.isChecked()
        for row, advanced in self.field_rows:
            row.setVisible(visible or not advanced)

    def _preset_changed(self, *_args):
        self._state_changed()
        self._update_preset_summary()

    def _update_preset_summary(self):
        try:
            name = self.preset_combo.currentData() or "overview"
            preset = load_preset(name)
            self.preset_description.setText(preset.get("description", ""))
            count = estimate_case_count(self.settings, preset["figures"])
            self.case_summary.setText(
                f"{count} numerical case(s) will be evaluated for: {', '.join(preset['figures'])}. "
                "HDF5 is always saved; checked publication formats are generated after simulation."
            )
        except Exception as exc:
            self.case_summary.setText(f"Configuration needs attention: {exc}")

    def _save_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save user profile", "Profile name")
        if not ok:
            return
        try:
            self.profiles = create_profile(name, self.settings, self.profiles)
            self._persist_all()
            self._populate_profiles(validate_profile_name(name))
            self.status_label.setText(f"Saved profile: {name.strip()}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not save profile", str(exc))

    def _load_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            return
        formats = self._checked_formats()
        self.settings = validate_settings(self.profiles[name])
        self.settings["diagram"]["formats"] = formats
        self.loading_widgets = True
        for path, (widget, kind) in self.field_widgets.items():
            self._set_widget_value(widget, kind, _get_path(self.settings, path))
        self.loading_widgets = False
        self._persist_all()
        self._update_preset_summary()
        self._update_stale_banner()
        self.preview_timer.start()
        self.status_label.setText(f"Loaded profile: {name}")

    def _rename_profile(self):
        old = self.profile_combo.currentText()
        if not old:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename user profile", "New name", text=old)
        if not ok:
            return
        try:
            self.profiles = rename_profile(old, new, self.profiles)
            self._persist_all()
            self._populate_profiles(validate_profile_name(new))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not rename profile", str(exc))

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete user profile", f"Delete '{name}' permanently?")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        del self.profiles[name]
        self._persist_all()
        self._populate_profiles()

    def _restore_defaults(self):
        answer = QtWidgets.QMessageBox.question(
            self, "Restore shipped defaults",
            "Restore all current scientific and diagram settings? User profiles will be retained.",
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.settings, self.gui_state, self.profiles = restore_defaults(
            self.gui_state, self.profiles, self.config_path
        )
        self.loading_widgets = True
        self._populate_all()
        self.loading_widgets = False
        self._update_preset_summary()
        self._update_stale_banner()
        self.preview_timer.start()

    def _confirm_collisions(self, paths):
        existing = [Path(path) for path in paths if Path(path).exists()]
        if not existing:
            return True
        names = "\n".join(f"• {path.name}" for path in existing)
        answer = QtWidgets.QMessageBox.question(
            self, "Replace existing outputs",
            f"The following completed files will be replaced only after the new operation succeeds:\n\n{names}",
        )
        return answer == QtWidgets.QMessageBox.Yes

    def _start_run(self):
        if self.worker and self.worker.isRunning():
            return
        try:
            self.settings = self._settings_from_widgets()
            name = validate_result_name(self.result_name_edit.text())
            preset_name = self.preset_combo.currentData()
            figures = load_preset(preset_name)["figures"]
            formats = self._checked_formats()
            paths = expected_output_paths(self.results_dir, name, figures, formats)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid P4 settings", str(exc))
            return
        if not self._confirm_collisions(paths):
            return
        self.gui_state.update({"preset": preset_name, "result_name": name, "formats": formats})
        self.settings["diagram"]["formats"] = formats
        self._persist_all()
        self.worker = RunWorker(self.settings, preset_name, name, formats, self.results_dir, self)
        self._connect_worker(self.worker, self._run_completed)
        self._set_busy(True)
        self.status_log.clear()
        self._append_status(f"Starting preset '{preset_name}'")
        self.worker.start()

    def _start_export(self):
        if not self.loaded_results or (self.worker and self.worker.isRunning()):
            return
        formats = self._checked_formats()
        if not formats:
            QtWidgets.QMessageBox.information(self, "No export format", "Check PNG, PDF, and/or SVG first.")
            return
        try:
            self.settings = self._settings_from_widgets()
            name = validate_result_name(self.result_name_edit.text())
            figures = list(self.loaded_metadata.get("figures") or [self.figure_combo.currentData()])
            paths = [self.results_dir / f"{name}_{figure}.{extension}" for figure in figures for extension in formats]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid export settings", str(exc))
            return
        if not self._confirm_collisions(paths):
            return
        self._persist_all()
        self.worker = ExportWorker(
            self.loaded_results, self.loaded_metadata, self.settings, name,
            formats, self.results_dir, self,
        )
        self._connect_worker(self.worker, self._export_completed)
        self._set_busy(True)
        self._append_status("Rerendering loaded result")
        self.worker.start()

    def _connect_worker(self, worker, completed_slot):
        worker.progress.connect(self._worker_progress)
        worker.completed.connect(completed_slot)
        worker.cancelled.connect(self._worker_cancelled)
        worker.failed.connect(self._worker_failed)

    def _worker_progress(self, fraction, message):
        self.progress_bar.setValue(max(0, min(1000, int(fraction * 1000))))
        self.status_label.setText(message)
        self._append_status(message)

    def _run_completed(self, outcome):
        self._set_busy(False)
        path = Path(outcome["hdf5_path"])
        self.gui_state["last_result_path"] = str(path.resolve())
        self._persist_all()
        self._update_last_result_action()
        self._load_result_path(path)
        self.tabs.setCurrentIndex(self.results_tab_index)
        self.status_label.setText(f"Completed {outcome['case_count']} cases")

    def _export_completed(self, paths):
        self._set_busy(False)
        self.status_label.setText(f"Exported {len(paths)} figure file(s)")
        self._append_status("Export complete: " + ", ".join(Path(path).name for path in paths))

    def _worker_cancelled(self, message):
        self._set_busy(False)
        self.status_label.setText("Cancelled; previous completed outputs were retained")
        self._append_status(message)

    def _worker_failed(self, message):
        self._set_busy(False)
        self.status_label.setText("P4 operation failed")
        self._append_status(message)
        QtWidgets.QMessageBox.critical(self, "P4 operation failed", message)

    def _cancel_operation(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling safely…")

    def _set_busy(self, busy):
        if busy:
            self.preview_timer.stop()
        self.run_button.setEnabled(not busy)
        self.rerender_button.setEnabled(not busy and self.loaded_results is not None)
        self.cancel_button.setEnabled(busy)
        self.load_result_button.setEnabled(not busy)
        self.figure_combo.setEnabled(not busy)
        self.open_directory_button.setEnabled(not busy)
        self.settings_panel.setEnabled(not busy)
        if not busy:
            self.progress_bar.setValue(0)

    def _append_status(self, message):
        if not self.status_log.toPlainText().endswith(str(message)):
            self.status_log.appendPlainText(str(message))

    def _choose_result(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open P4 HDF5 result", str(self.results_dir), "P4 HDF5 (*.h5 *.hdf5)"
        )
        if path:
            self._load_result_path(Path(path))

    def _reopen_last_result(self):
        path = Path(self.gui_state.get("last_result_path", ""))
        if path.is_file():
            self._load_result_path(path)
        else:
            QtWidgets.QMessageBox.warning(self, "Result unavailable", "The remembered HDF5 file no longer exists.")
            self.gui_state["last_result_path"] = ""
            self._persist_all()
            self._update_last_result_action()

    def _load_result_path(self, path):
        try:
            results, metadata = load_result(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open P4 result", str(exc))
            return
        self.loaded_results = results
        self.loaded_metadata = metadata
        self.loaded_result_path = Path(path).resolve()
        self.gui_state["last_result_path"] = str(self.loaded_result_path)
        figures = list(metadata.get("figures") or ["overview"])
        self.figure_combo.blockSignals(True)
        self.figure_combo.clear()
        for figure in figures:
            self.figure_combo.addItem(figure, figure)
        remembered = self.gui_state.get("selected_figure", "")
        index = self.figure_combo.findData(remembered)
        self.figure_combo.setCurrentIndex(max(0, index))
        self.figure_combo.blockSignals(False)
        preset = metadata.get("preset", "legacy/unknown")
        self.result_info.setText(
            f"{self.loaded_result_path.name} · preset: {preset} · {len(results)} stored case(s)"
        )
        self._populate_metrics()
        self.rerender_button.setEnabled(True)
        self._persist_all()
        self._update_last_result_action()
        self._update_stale_banner()
        self._render_preview()
        self.tabs.setCurrentIndex(self.results_tab_index)

    def _populate_metrics(self):
        self.metrics_table.setRowCount(len(self.loaded_results or {}))
        for row, (case_name, result) in enumerate((self.loaded_results or {}).items()):
            self.metrics_table.setItem(row, 0, QtWidgets.QTableWidgetItem(case_name))
            for column, name in enumerate(METRIC_NAMES, 1):
                value = result.metrics.get(name, float("nan"))
                text = "—" if not np.isfinite(value) else f"{value:.6g}"
                self.metrics_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))

    def _figure_changed(self, *_args):
        if self.loading_widgets:
            return
        self.gui_state["selected_figure"] = self.figure_combo.currentData() or ""
        self._persist_all()
        self.preview_timer.start()

    def _render_preview(self):
        if not self.loaded_results or not self.figure_combo.currentData():
            return
        try:
            figure = build_figure(
                self.loaded_results,
                self.figure_combo.currentData(),
                normalization=self.settings["diagram"]["normalization"],
                time_axis=self.settings["diagram"]["time_axis"],
                response_metric=self.settings["diagram"]["response_metric"],
                comparison_window_ref=self.settings["diagram"]["comparison_window_ref"],
                style=self.settings["diagram"],
            )
        except Exception as exc:
            self.status_label.setText(f"Preview unavailable: {exc}")
            return
        if self.preview_figure is not None:
            plt.close(self.preview_figure)
        for widget in (self.preview_toolbar, self.preview_canvas, self.figure_placeholder):
            if widget is not None:
                self.figure_layout.removeWidget(widget)
                widget.hide()
                if widget is not self.figure_placeholder:
                    widget.deleteLater()
        self.preview_figure = figure
        self.preview_canvas = FigureCanvas(figure)
        self.preview_toolbar = NavigationToolbar(self.preview_canvas, self)
        self.figure_layout.addWidget(self.preview_toolbar)
        self.figure_layout.addWidget(self.preview_canvas, 1)
        self.preview_canvas.draw_idle()

    def _update_stale_banner(self):
        if not self.loaded_results:
            self.stale_banner.hide()
            return
        try:
            match = result_matches_settings(self.loaded_metadata, self.settings)
        except Exception:
            match = False
        if match is True:
            self.stale_banner.hide()
        elif match is None:
            self.stale_banner.setText(
                "Legacy result: its scientific settings hash is unavailable. It remains viewable and exportable."
            )
            self.stale_banner.show()
        else:
            self.stale_banner.setText(
                "The loaded result was calculated with earlier scientific settings. Presentation changes apply live; Run is required to refresh numerical data."
            )
            self.stale_banner.show()

    def _update_last_result_action(self):
        raw = self.gui_state.get("last_result_path", "")
        path = Path(raw) if raw else None
        self.last_result_button.setVisible(path is not None)
        if path:
            self.last_result_button.setText(f"Reopen last result: {path.name}")

    def _open_result_directory(self):
        path = self.loaded_result_path.parent if self.loaded_result_path else self.results_dir
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _restore_window_state(self):
        geometry = self.gui_state.get("window_geometry", "")
        if geometry:
            self.restoreGeometry(QtCore.QByteArray.fromBase64(geometry.encode("ascii")))
        self.splitter.setSizes(self.gui_state.get("splitter_sizes", [430, 1100]))
        self.tabs.setCurrentIndex(min(self.tabs.count() - 1, self.gui_state.get("selected_tab", 0)))

    def _save_window_state(self):
        self.gui_state["window_geometry"] = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self.gui_state["splitter_sizes"] = self.splitter.sizes()
        self.gui_state["selected_tab"] = self.tabs.currentIndex()
        self.gui_state["advanced_visible"] = self.advanced_toggle.isChecked()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            answer = QtWidgets.QMessageBox.question(
                self, "P4 is still working", "Cancel the current operation and close after it stops safely?"
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.worker.request_cancel()
            if not self.worker.wait(10_000):
                QtWidgets.QMessageBox.information(
                    self, "Still cancelling", "P4 is still leaving the numerical loop safely. Close again after cancellation completes."
                )
                event.ignore()
                return
        self._save_window_state()
        try:
            self._persist_all()
        except Exception:
            pass
        if self.preview_figure is not None:
            plt.close(self.preview_figure)
        event.accept()


def launch_gui(config_path=RUNTIME_CONFIG_PATH, results_dir=RESULTS_DIR):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    icon = RESOURCES_DIR / "Images" / "WeldCraft.ico"
    app.setWindowIcon(QtGui.QIcon(str(icon)))
    try:
        settings, state, profiles = ensure_runtime_state(Path(config_path))
    except Exception as exc:
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "P4 configuration error", str(exc))
        restore = box.addButton("Restore shipped defaults", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Exit", QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is not restore:
            return 2
        try:
            settings, state, profiles = recover_runtime_defaults(Path(config_path))
        except Exception as recovery_error:
            QtWidgets.QMessageBox.critical(None, "P4 recovery failed", str(recovery_error))
            return 2
    window = MainWindow(settings, state, profiles, config_path, results_dir)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
