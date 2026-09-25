"""Launcher-enabled GUI for the P2 welding heat/hydrogen solver.

The GUI deliberately remains a thin client: it edits allow-listed values in
b2_Simulation_Settings.py and starts the normal solver in a child process.
"""

from __future__ import annotations

import ast
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
REPO_DIR = PROJECT_DIR.parent
RESOURCES_DIR = REPO_DIR / "Resources"
if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))

from Common.launch_ready import StartupReadySignal
from c2_GUI_Functions import (
    DEFAULT_PARAM_PATH,
    GEOMETRY_FIELDS,
    MATERIAL_FIELDS,
    PARAMETER_TOOLTIPS,
    PARAMETER_FIELDS,
    PARAM_CONFIG_PATH,
    RESULTS_DIR,
    candidate_configuration,
    configuration_hints,
    geometry_tooltip,
    load_geometry_settings,
    load_parameter_values,
    material_curves,
    recover_full_file,
    seed_missing_working_files,
    sr_preset_spec,
    write_configuration,
    write_parameter_values,
)
from c3_GUI_Results import ExportCancelled, LazyP2Result, P2ResultArtists, export_animation


JOINTS = ["butt joint", "lap joint", "iso3690"]
JOINT_LABELS = {"butt joint": "Butt joint", "lap joint": "Lap joint", "iso3690": "ISO 3690"}
CHOICES = {
    "simulation_type": JOINTS,
    "diffusion_scheme": [0, 1, 2],
    "add_bead_mode": ["regular_intervals"],
    "pipe_line_inner_hydrogen": ["variable", "constant"],
}
ILLUSTRATIONS = {
    "butt joint": PROJECT_DIR / "01_Resources" / "01_Bilder" / "P2_Butt_Joint.svg",
    "lap joint": PROJECT_DIR / "01_Resources" / "01_Bilder" / "P2_Lap_Joint.svg",
    "iso3690": PROJECT_DIR / "01_Resources" / "01_Bilder" / "P2_ISO3690.svg",
}
DEGREE_C = "\N{DEGREE SIGN}C"


class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.toggle.setStyleSheet("QToolButton { font-weight: 600; font-size: 13px; border: 0; padding: 5px; }")
        self.content = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.content)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.content.setVisible(expanded)
        self.toggle.toggled.connect(self._toggle)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def _toggle(self, expanded):
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.content.setVisible(expanded)


class TaskThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.stop_event = threading.Event()

    def run(self):
        try:
            value = self.function(self.progress.emit, self.stop_event)
        except ExportCancelled:
            self.failed.emit("Cancelled")
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.succeeded.emit(value)

    def cancel(self):
        self.stop_event.set()


class MaterialEditor(QtWidgets.QDialog):
    FIELDS = [
        ("Thermal diffusivity", "microstructure_thermal_diff", "D"),
        ("Hydrogen diffusivity", "microstructure_hydrogen_diff", "D_H"),
        ("Solubility", "microstructure_solubility", "S"),
    ]

    def __init__(self, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("P2 Expert Material Editor")
        self.resize(1120, 760)
        self.values = values
        self.materials = list(values["microstructures"])
        self.editors = {}
        self.canvases = {}
        self.statuses = {}
        self.tabs = QtWidgets.QTabWidget()
        for title, field, variable in self.FIELDS:
            page = QtWidgets.QWidget()
            split = QtWidgets.QSplitter()
            editor = QtWidgets.QPlainTextEdit(str(values[field]))
            editor.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            figure = Figure(figsize=(5, 4), constrained_layout=True)
            canvas = FigureCanvas(figure)
            status = QtWidgets.QLabel()
            status.setWordWrap(True)
            right = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right)
            right_layout.addWidget(canvas, 1)
            right_layout.addWidget(status)
            split.addWidget(editor)
            split.addWidget(right)
            split.setSizes([560, 520])
            layout = QtWidgets.QVBoxLayout(page)
            layout.addWidget(split)
            self.tabs.addTab(page, title)
            self.editors[field] = editor
            self.canvases[field] = (canvas, figure, variable)
            self.statuses[field] = status
            editor.textChanged.connect(self.schedule_refresh)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Edit the existing material/range syntax. Invalid ranges, gaps, non-finite values, and negative values cannot be saved."
        ))
        layout.addWidget(self.tabs, 1)
        layout.addWidget(buttons)
        self.timer = QtCore.QTimer(self, interval=350, singleShot=True)
        self.timer.timeout.connect(self.refresh_current)
        self.tabs.currentChanged.connect(lambda _index: self.schedule_refresh())
        self.schedule_refresh()

    def schedule_refresh(self):
        self.timer.start()

    @staticmethod
    def _draw_hydrogen_reference(axis):
        """Draw the thesis GR/SR equations directly behind the edited curves."""
        # From PhD thesis scatterband derivation and final plotting data.
        gas_constant = 8.31446261815324
        gr_color = "#666666"
        sr_color = "#984B4B"

        def arrhenius(temperature_c, prefactor, activation_energy):
            return prefactor * np.exp(
                -activation_energy / (gas_constant * (temperature_c + 273.15))
            )

        low_temperature = np.linspace(20.0, 100.0, 100)
        mid_temperature = np.linspace(100.0, 740.0, 260)
        austenite_temperature = np.linspace(740.0, 1500.0, 260)
        liquid_temperature = np.linspace(1500.0, 2000.0, 180)

        gr_segments = [
            (
                low_temperature,
                1.3119647961809571e-9 * low_temperature**2.603497088622902,
                arrhenius(low_temperature, 0.10538095190506253, 10941.32734969128),
            ),
            (
                mid_temperature,
                arrhenius(mid_temperature, 0.3358814362534934, 22869.442408034083),
                arrhenius(mid_temperature, 0.10538095190506253, 10941.32734969128),
            ),
            (
                austenite_temperature,
                arrhenius(austenite_temperature, 0.6736, 45086.0),
                arrhenius(austenite_temperature, 1.0691, 41624.0),
            ),
            (
                liquid_temperature,
                arrhenius(liquid_temperature, 0.257, 17154.4),
                arrhenius(liquid_temperature, 0.437, 17296.656),
            ),
        ]
        for index, (temperature, lower, upper) in enumerate(gr_segments):
            axis.fill_between(
                temperature, lower, upper,
                color=gr_color, alpha=0.12,
                label="General Range (GR)" if index == 0 else None,
                zorder=1,
            )
            axis.plot(temperature, lower, color=gr_color, linewidth=1.15, alpha=0.94, zorder=2)
            axis.plot(temperature, upper, color=gr_color, linewidth=1.15, alpha=0.94, zorder=2)

        guide_temperature = np.linspace(450.0, 740.0, 120)
        for prefactor, activation_energy in ((0.6736, 45086.0), (1.0691, 41624.0)):
            axis.plot(
                guide_temperature,
                arrhenius(guide_temperature, prefactor, activation_energy),
                color=gr_color, linewidth=1.0, linestyle=(0, (4.0, 3.0)), alpha=0.85, zorder=2,
            )

        sr_segments = [
            (
                low_temperature,
                gr_segments[0][1],
                9.957656e-3 * low_temperature**0.337459
                * np.exp(-9523.0 / (gas_constant * (low_temperature + 273.15))),
            ),
            (
                mid_temperature,
                gr_segments[1][1],
                arrhenius(mid_temperature, 0.085967, 11390.0),
            ),
        ]
        for index, (temperature, lower, upper) in enumerate(sr_segments):
            axis.fill_between(
                temperature, lower, upper,
                color=sr_color, alpha=0.105,
                label="Special Range (SR)" if index == 0 else None,
                zorder=2,
            )
            axis.plot(temperature, lower, color=sr_color, linewidth=1.25, alpha=0.96, zorder=3)
            axis.plot(temperature, upper, color=sr_color, linewidth=1.25, alpha=0.96, zorder=3)

        axis.plot(austenite_temperature, gr_segments[2][1], color=sr_color, linewidth=1.25, zorder=3)
        axis.plot(liquid_temperature, gr_segments[3][1], color=sr_color, linewidth=1.25, zorder=3)

    def refresh_current(self):
        _title, field, variable = self.FIELDS[self.tabs.currentIndex()]
        canvas, figure, _variable = self.canvases[field]
        figure.clear()
        axis = figure.add_subplot(111)
        temperatures = np.linspace(-20.0, 2000.0, 650)
        try:
            curves = material_curves(self.editors[field].toPlainText(), variable, self.materials, temperatures)
            if variable == "D_H":
                self._draw_hydrogen_reference(axis)
                positive = curves[curves > 0]
                lower = min(1.0e-6, float(np.min(positive)) / 2.0) if positive.size else 1.0e-6
                upper = max(3.0e-1, float(np.max(positive)) * 2.0) if positive.size else 3.0e-1
                axis.set_xlim(0.0, 2000.0)
                axis.set_ylim(max(lower, 1.0e-12), upper)
                axis.set_yscale("log")
                axis.set_ylabel(r"Diffusion coefficient $D_{\mathrm{app}}$ [$\mathrm{mm}^2/\mathrm{s}$]")
            for index, material in enumerate(self.materials):
                positive = curves[index] > 0
                if not np.any(positive):
                    continue
                label = {
                    "base_metal": "Base metal",
                    "weld_metal": "Weld metal",
                    "HAZ": "Heat-affected zone",
                }.get(material, material.replace("_", " ").title())
                axis.plot(
                    temperatures[positive], curves[index][positive],
                    label=label, linewidth=2.2, zorder=5,
                )
            axis.set_xlabel(f"Temperature [{DEGREE_C}]")
            if variable == "D":
                axis.set_ylabel(r"Thermal diffusivity $D$ [$\mathrm{mm}^2/\mathrm{s}$]")
            elif variable == "S":
                axis.set_ylabel("Solubility S")
            axis.set_axisbelow(True)
            axis.grid(True, which="major", color="#cfcfcf", linewidth=0.65, alpha=0.75)
            axis.grid(True, which="minor", color="#e6e6e6", linewidth=0.4, alpha=0.7)
            axis.legend(loc="best", fontsize=8.5)
            self.statuses[field].setText("Valid material model")
            self.statuses[field].setStyleSheet("color: #237a3b")
        except Exception as exc:
            axis.text(0.5, 0.5, str(exc), transform=axis.transAxes, ha="center", va="center", wrap=True)
            self.statuses[field].setText(str(exc))
            self.statuses[field].setStyleSheet("color: #b03030")
        canvas.draw_idle()

    def save(self):
        changes = {field: editor.toPlainText() for field, editor in self.editors.items()}
        try:
            write_parameter_values(changes)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Material model not saved", str(exc))
            return
        self.accept()


class SetupWizard(QtWidgets.QWizard):
    GOAL, JOINT, GEOMETRY, RESOLUTION, THERMAL, HYDROGEN, OUTPUTS, REVIEW = range(8)

    def __init__(self, setup, parent=None):
        super().__init__(parent)
        self.setup = setup
        self.setWindowTitle("P2 Guided Simulation Setup")
        self.resize(780, 610)
        values = setup.parameters_from_ui()
        self.goal = QtWidgets.QComboBox()
        self.goal.addItems(["Coupled heat and hydrogen", "Thermal-only calibration"])
        self.goal.setCurrentIndex(1 if values["thermal_diffusion_calibration"] else 0)
        self._add_page(self.GOAL, "Simulation goal", "Choose whether hydrogen transport is included.", self.goal)

        self.joint = QtWidgets.QComboBox()
        self.joint.addItems([JOINT_LABELS[item] for item in JOINTS])
        self.joint.setCurrentIndex(JOINTS.index(values["simulation_type"]))
        self.joint_image = QtSvg.QSvgWidget()
        self.joint_image.setMinimumHeight(300)
        joint_box = QtWidgets.QWidget()
        joint_layout = QtWidgets.QVBoxLayout(joint_box)
        joint_layout.addWidget(self.joint)
        joint_layout.addWidget(self.joint_image, 1)
        self.joint.currentIndexChanged.connect(self._update_joint_image)
        self._update_joint_image()
        self._add_page(self.JOINT, "Joint selection", "Pick the physical arrangement.", joint_box)

        self.wizard_geometry = {}
        self.geometry_stack = QtWidgets.QStackedWidget()
        current_geometry = setup.geometry_from_ui()
        for joint in JOINTS:
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            self.wizard_geometry[joint] = {}
            for name, label in GEOMETRY_FIELDS[joint].items():
                minimum = 0.0 if joint == "lap joint" and name == "we" else 0.00001
                field = QtWidgets.QDoubleSpinBox(decimals=5, minimum=minimum, maximum=1e7)
                field.setValue(current_geometry[joint][name])
                field.setSuffix(" mm")
                form.addRow(label, field)
                self.wizard_geometry[joint][name] = field
            self.geometry_stack.addWidget(page)
        self._add_page(self.GEOMETRY, "Geometry", "Set the dimensions for the selected joint.", self.geometry_stack)
        self.dx = QtWidgets.QDoubleSpinBox(decimals=4, minimum=0.0001, maximum=1000.0, value=float(values["dx"]))
        self.dx.setSuffix(" mm")
        self._add_page(self.RESOLUTION, "Resolution", "P2 currently requires square cells; dx and dy are kept equal.", self.dx)
        self.t_hot = QtWidgets.QDoubleSpinBox(decimals=2, minimum=-273.0, maximum=10000.0, value=float(values["t_hot"]))
        self.t_hot.setSuffix(" deg C")
        self.bead_time = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=1e9, value=float(values["time_for_weld_bead"]))
        self.bead_count = QtWidgets.QSpinBox(minimum=1, maximum=1000, value=int(values["no_of_weld_beads"]))
        self._update_bead_limits()
        thermal_box = QtWidgets.QWidget()
        thermal_form = QtWidgets.QFormLayout(thermal_box)
        thermal_form.addRow("Number of weld beads", self.bead_count)
        thermal_form.addRow("New bead temperature", self.t_hot)
        thermal_form.addRow("Time between beads [s]", self.bead_time)
        self._add_page(self.THERMAL, "Welding and thermal inputs", "Set the central thermal controls.", thermal_box)
        self.hydrogen = QtWidgets.QDoubleSpinBox(decimals=4, minimum=0.0, maximum=1e9, value=float(values["hydro_weld_metal"]))
        self.initial_hydrogen = QtWidgets.QDoubleSpinBox(decimals=4, minimum=0.0, maximum=1e9, value=float(values["h_cont_initial"]))
        hydrogen_box = QtWidgets.QWidget()
        hydrogen_form = QtWidgets.QFormLayout(hydrogen_box)
        hydrogen_form.addRow("New-bead hydrogen [%]", self.hydrogen)
        hydrogen_form.addRow("Initial hydrogen [%]", self.initial_hydrogen)
        self._add_page(self.HYDROGEN, "Hydrogen inputs", "Set initial and deposited-weld hydrogen.", hydrogen_box)
        self.output_h5 = QtWidgets.QLineEdit(str(values["file_name"]))
        self.output_mp4 = QtWidgets.QLineEdit(str(values["animation_name"]))
        self.animation = QtWidgets.QCheckBox("Render animation after simulation")
        self.animation.setChecked(bool(values["include_animation_after_run"]))
        output_box = QtWidgets.QWidget()
        output_form = QtWidgets.QFormLayout(output_box)
        output_form.addRow("HDF5 result", self.output_h5)
        output_form.addRow("MP4 animation", self.output_mp4)
        output_form.addRow(self.animation)
        self._add_page(self.OUTPUTS, "Outputs", "Choose result paths and optional post-run rendering.", output_box)
        self.review = QtWidgets.QLabel(wordWrap=True)
        self._add_page(self.REVIEW, "Review", "Finish applies and validates these choices in the working configuration.", self.review)
        self.currentIdChanged.connect(self._page_changed)

    def _add_page(self, page_id, title, subtitle, widget):
        page = QtWidgets.QWizardPage()
        page.setTitle(title)
        page.setSubTitle(subtitle)
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(widget)
        layout.addStretch(1)
        self.setPage(page_id, page)

    def _update_joint_image(self):
        joint = JOINTS[self.joint.currentIndex()]
        self.joint_image.load(str(ILLUSTRATIONS[joint]))
        if hasattr(self, "geometry_stack"):
            self.geometry_stack.setCurrentIndex(self.joint.currentIndex())
        if hasattr(self, "bead_count"):
            self._update_bead_limits()

    def _update_bead_limits(self):
        joint = JOINTS[self.joint.currentIndex()]
        current = self.bead_count.value()
        if joint == "butt joint":
            self.bead_count.setRange(2, 1000)
            self.bead_count.setSingleStep(2)
            if current % 2:
                current += 1
        elif joint == "lap joint":
            self.bead_count.setRange(1, 4)
            self.bead_count.setSingleStep(1)
        else:
            self.bead_count.setRange(1, 3)
            self.bead_count.setSingleStep(1)
        self.bead_count.setValue(current)

    def nextId(self):
        current = self.currentId()
        if current == self.THERMAL and self.goal.currentIndex() == 1:
            return self.OUTPUTS
        return super().nextId()

    def _page_changed(self, page_id):
        joint = JOINTS[self.joint.currentIndex()]
        if page_id == self.GEOMETRY:
            self.geometry_stack.setCurrentIndex(self.joint.currentIndex())
        elif page_id == self.RESOLUTION and joint == "lap joint":
            gap = self.wizard_geometry[joint]["we"].value()
            if self.dx.value() > gap:
                self.dx.setValue(gap)
        elif page_id == self.REVIEW:
            goal = "thermal-only calibration" if self.goal.currentIndex() == 1 else "coupled heat and hydrogen"
            self.review.setText(
                f"Goal: {goal}\nJoint: {JOINT_LABELS[joint]}\nMesh: {self.dx.value():g} mm square cells\n"
                f"Weld beads: {self.bead_count.value()}\nBead temperature: {self.t_hot.value():g} deg C\n"
                f"Time between beads: {self.bead_time.value():g} s\n"
                f"Result: {self.output_h5.text()}\nRender animation: {'yes' if self.animation.isChecked() else 'no'}"
            )

    def accept(self):
        values = {
            "thermal_diffusion_calibration": self.goal.currentIndex() == 1,
            "simulation_type": JOINTS[self.joint.currentIndex()],
            "dx": self.dx.value(), "dy": self.dx.value(),
            "no_of_weld_beads": self.bead_count.value(),
            "t_hot": self.t_hot.value(), "time_for_weld_bead": self.bead_time.value(),
            "hydro_weld_metal": self.hydrogen.value(), "h_cont_initial": self.initial_hydrogen.value(),
            "include_animation_after_run": self.animation.isChecked(),
            "file_name": self.output_h5.text().strip(), "animation_name": self.output_mp4.text().strip(),
        }
        self.setup.set_parameter_values(values)
        for joint, fields in self.wizard_geometry.items():
            for name, field in fields.items():
                self.setup.geometry_controls[joint][name].setValue(field.value())
        if self.setup.apply_configuration():
            super().accept()


class SetupPanel(QtWidgets.QWidget):
    configuration_applied = QtCore.pyqtSignal(dict)
    run_requested = QtCore.pyqtSignal()

    CATEGORY_ORDER = ["Simulation", "Geometry", "Mesh", "Weld beads", "Timing", "Thermal", "Hydrogen", "Saving and animation", "Material models", "Experimental"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        self._applying = False
        self.controls = {}
        self.geometry_controls = {}
        self.advanced_rows = []
        self.parameters = {}
        self.geometry = {}
        self._build_ui()
        self.reload()

    def _build_ui(self):
        top = QtWidgets.QHBoxLayout()
        wizard = QtWidgets.QPushButton("Guided setup...")
        wizard.clicked.connect(lambda: SetupWizard(self, self).exec_())
        self.advanced = QtWidgets.QCheckBox("Advanced settings")
        self.advanced.toggled.connect(self._show_advanced)
        apply_button = QtWidgets.QPushButton("Apply to settings file")
        apply_button.setDefault(True)
        apply_button.setToolTip("Apply immediately; valid field edits are also saved automatically.")
        apply_button.clicked.connect(lambda: self.apply_configuration(show_errors=True))
        reload_button = QtWidgets.QPushButton("Reload")
        reload_button.clicked.connect(self.reload)
        repair_button = QtWidgets.QPushButton("Repair settings...")
        repair_button.setToolTip("Restore the complete P2 settings file from the shipped default.")
        repair_button.clicked.connect(self.repair_settings)
        top.addWidget(wizard)
        top.addWidget(self.advanced)
        top.addStretch(1)
        top.addWidget(reload_button)
        top.addWidget(repair_button)
        top.addWidget(apply_button)

        self.sections = {name: CollapsibleSection(name, name in {"Simulation", "Geometry", "Mesh"}) for name in self.CATEGORY_ORDER}
        for name, (source_category, label, kind, advanced) in PARAMETER_FIELDS.items():
            category = source_category
            if source_category == "Numerics":
                category = "Experimental" if name == "guess_adaptive_stable_dt" else "Mesh"
            if source_category == "Simulation" and name in {"add_bead_mode"}:
                category = "Weld beads"
            control = self._make_control(name, kind)
            self.controls[name] = control
            label_widget = QtWidgets.QLabel(label)
            tooltip = PARAMETER_TOOLTIPS.get(name, "")
            label_widget.setToolTip(tooltip)
            control.setToolTip(tooltip)
            self.sections[category].form.addRow(label_widget, control)
            if advanced:
                self.advanced_rows.append((label_widget, control))
        self.controls["dx"].editingFinished.connect(self._sync_square_mesh)

        self.geometry_stack = QtWidgets.QStackedWidget()
        for joint in JOINTS:
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            self.geometry_controls[joint] = {}
            for name, label in GEOMETRY_FIELDS[joint].items():
                minimum = 0.0 if joint == "lap joint" and name == "we" else 0.00001
                control = QtWidgets.QDoubleSpinBox(decimals=5, minimum=minimum, maximum=1e7)
                control.setSuffix(" mm")
                control.valueChanged.connect(self.schedule_preview)
                label_widget = QtWidgets.QLabel(label)
                tooltip = geometry_tooltip(joint, name, label)
                label_widget.setToolTip(tooltip)
                control.setToolTip(tooltip)
                form.addRow(label_widget, control)
                self.geometry_controls[joint][name] = control
            self.geometry_stack.addWidget(page)
        self.sections["Geometry"].form.addRow(self.geometry_stack)

        material_bar = QtWidgets.QWidget()
        material_layout = QtWidgets.QHBoxLayout(material_bar)
        material_layout.setContentsMargins(0, 0, 0, 0)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(["Default", "Lower SR", "Upper SR"])
        preset_button = QtWidgets.QPushButton("Apply D_H preset")
        preset_button.clicked.connect(self.apply_preset)
        expert_button = QtWidgets.QPushButton("Expert Material Editor...")
        expert_button.clicked.connect(self.open_material_editor)
        material_layout.addWidget(self.preset)
        material_layout.addWidget(preset_button)
        material_layout.addWidget(expert_button)
        self.sections["Material models"].form.addRow(material_bar)

        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        for name in self.CATEGORY_ORDER:
            scroll_layout.addWidget(self.sections[name])
        scroll_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea(widgetResizable=True)
        scroll.setWidget(scroll_content)

        self.preview_figure = Figure(figsize=(6.0, 4.0), constrained_layout=True)
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.derived = QtWidgets.QLabel()
        self.derived.setWordWrap(True)
        self.derived.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #a33")
        refresh = QtWidgets.QPushButton("Refresh live geometry / mesh preview")
        refresh.clicked.connect(self.update_preview)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(self.preview_canvas, 1)
        right_layout.addWidget(self.derived)
        right_layout.addWidget(self.warning)
        right_layout.addWidget(refresh)
        run = QtWidgets.QPushButton("Run simulation")
        run.setMinimumHeight(38)
        run.clicked.connect(self.run_requested)
        right_layout.addWidget(run)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setSizes([620, 620])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        self.preview_timer = QtCore.QTimer(self, interval=300, singleShot=True)
        self.preview_timer.timeout.connect(self._auto_apply_configuration)
        self.controls["dy"].setReadOnly(True)
        self._show_advanced(False)

    def _make_control(self, name, kind):
        if kind == "bool":
            control = QtWidgets.QCheckBox()
            control.toggled.connect(self.schedule_preview)
            return control
        if kind == "choice":
            control = QtWidgets.QComboBox()
            for value in CHOICES[name]:
                control.addItem(str(value), value)
            control.currentIndexChanged.connect(self._choice_changed)
            return control
        control = QtWidgets.QLineEdit()
        control.editingFinished.connect(self.schedule_preview)
        return control

    def _choice_changed(self):
        sim = self.control_value("simulation_type") if "simulation_type" in self.controls else JOINTS[0]
        if sim in JOINTS:
            self.geometry_stack.setCurrentIndex(JOINTS.index(sim))
        thermal_only = bool(self.control_value("thermal_diffusion_calibration"))
        self.sections["Hydrogen"].setEnabled(not thermal_only)
        self.schedule_preview()

    def _show_advanced(self, visible):
        for label, control in self.advanced_rows:
            label.setVisible(visible)
            control.setVisible(visible)
        if not visible and "dx" in self.controls and "dy" in self.controls:
            self._sync_square_mesh()

    def _sync_square_mesh(self):
        self.controls["dy"].setText(self.controls["dx"].text())

    def control_value(self, name):
        control = self.controls[name]
        kind = PARAMETER_FIELDS[name][2]
        if kind == "bool":
            return control.isChecked()
        if kind == "choice":
            return control.currentData()
        text = control.text().strip()
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
        if kind == "pairs":
            value = ast.literal_eval(text)
            return value
        return text

    def parameters_from_ui(self):
        return {name: self.control_value(name) for name in PARAMETER_FIELDS}

    def geometry_from_ui(self):
        return {
            joint: {name: control.value() for name, control in controls.items()}
            for joint, controls in self.geometry_controls.items()
        }

    def set_parameter_values(self, values):
        for name, value in values.items():
            if name not in self.controls:
                continue
            control = self.controls[name]
            kind = PARAMETER_FIELDS[name][2]
            if kind == "bool":
                control.setChecked(bool(value))
            elif kind == "choice":
                index = control.findData(value)
                if index >= 0:
                    control.setCurrentIndex(index)
            else:
                control.setText(repr(value) if kind == "pairs" else str(value))
        self._choice_changed()

    def _load_or_recover(self):
        try:
            seed_missing_working_files()
            return load_parameter_values(), load_geometry_settings()
        except Exception as exc:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Critical)
            box.setWindowTitle("P2 configuration is unreadable")
            box.setText(str(exc))
            box.setInformativeText(
                "A syntax error cannot be selectively repaired. Recover the complete working settings file "
                "from the tracked shipped default?"
            )
            recover_settings = box.addButton("Recover settings file", QtWidgets.QMessageBox.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is not recover_settings:
                raise
            recover_full_file(PARAM_CONFIG_PATH, DEFAULT_PARAM_PATH)
            return load_parameter_values(), load_geometry_settings()

    def reload(self):
        self.preview_timer.stop()
        self._loading = True
        try:
            self.parameters, self.geometry = self._load_or_recover()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot load P2 configuration", str(exc))
            self._loading = False
            return
        try:
            self.set_parameter_values(self.parameters)
            for joint, controls in self.geometry_controls.items():
                for name, control in controls.items():
                    control.setValue(float(self.geometry[joint][name]))
        finally:
            self._loading = False
        self.update_preview()

    def schedule_preview(self):
        if not self._loading and not self._applying:
            self.preview_timer.start()

    def _auto_apply_configuration(self):
        self.apply_configuration(show_errors=False)

    def validated_draft(self):
        parameter_changes = self.parameters_from_ui()
        geometry_changes = self.geometry_from_ui()
        return candidate_configuration(parameter_changes, geometry_changes)

    def update_preview(self):
        self.preview_figure.clear()
        axis = self.preview_figure.add_subplot(111)
        try:
            import importlib
            import b2_Simulation_Settings
            from b3_Functions import initialize, weld_sample

            parameters = load_parameter_values()
            importlib.reload(b2_Simulation_Settings)
            sim_type = parameters["simulation_type"]
            sample = weld_sample(sim_type)
            _edges, dim_x, dim_y, le, _ri, we, th, su_h, su_w, fr_le, _fr_ri, fr_ab, _fr_be = sample
            nx, ny = int(dim_x / parameters["dx"]), int(dim_y / parameters["dy"])
            initialized = initialize(parameters["simulation_type"], nx, ny, parameters["dx"], parameters["dy"], le, we, th, su_h, su_w, fr_le, fr_ab)
            microstructure = initialized[7]
            axis.imshow(microstructure, cmap="tab20c", interpolation="nearest", origin="upper",
                        extent=[0, dim_x, dim_y, 0], vmin=0, vmax=3, aspect="equal")
            if sim_type == "butt joint":
                axis.add_patch(Rectangle(
                    (le, fr_ab), we, th, fill=False, edgecolor="#ff8c42", linewidth=2.0,
                    linestyle="--", label="planned beads",
                ))
                for bead in range(1, int(parameters["no_of_weld_beads"])):
                    y = fr_ab + th * bead / parameters["no_of_weld_beads"]
                    axis.plot([le, le + we], [y, y], color="#ff8c42", linewidth=0.45, alpha=0.75)
            else:
                limit = 4 if sim_type == "lap joint" else 3
                count = min(int(parameters["no_of_weld_beads"]), limit)
                center_x = le if sim_type == "lap joint" else fr_le + le / 2.0
                center_y = fr_ab + th if sim_type == "lap joint" else fr_ab
                for bead in range(count):
                    axis.add_patch(Ellipse(
                        (center_x, center_y - bead * parameters["bead_height"] * 0.45),
                        parameters["bead_width"], parameters["bead_height"], fill=False,
                        edgecolor="#ff8c42", linewidth=1.4, linestyle="--",
                        label="planned beads" if bead == 0 else None,
                    ))
            step_x = max(parameters["dx"], dim_x / 20.0)
            step_y = max(parameters["dy"], dim_y / 16.0)
            axis.set_xticks(np.arange(0, dim_x + step_x, step_x))
            axis.set_yticks(np.arange(0, dim_y + step_y, step_y))
            axis.grid(True, color="white", linewidth=0.45, alpha=0.55)
            axis.set_title(f"{JOINT_LABELS[parameters['simulation_type']]} -- initialized material regions")
            axis.set_xlabel("x [mm]")
            axis.set_ylabel("y [mm]")
            dt = float(parameters["dt"])
            dt_big = float(parameters["dt_big"])
            main_duration = float(parameters["total_time_to_rt"])
            rt_duration = float(parameters["time_diffusion_at_rt"])
            estimates = int(np.ceil(main_duration / dt)) + int(np.ceil(rt_duration / dt_big))
            self.derived.setText(
                f"Physical size: {dim_x:g} x {dim_y:g} mm   |   Mesh: {nx} x {ny} = {nx * ny:,} cells\n"
                f"Stable dt: {dt:.6g} s   |   RT dt: {dt_big:.6g} s   |   Estimated iterations: {estimates:,}\n"
                f"Pre-weld: {parameters['time_before_first_weld']:g} s   |   To room temperature: {main_duration:g} s   |   RT diffusion: {rt_duration:g} s"
            )
            hints = configuration_hints(parameters, load_geometry_settings())
            self.warning.setText("\n".join(f"Hint: {hint}" for hint in hints))
        except Exception as exc:
            axis.text(0.5, 0.5, "Preview unavailable", ha="center", va="center", transform=axis.transAxes)
            self.derived.setText("")
            self.warning.setText(str(exc))
        self.preview_canvas.draw()

    def apply_configuration(self, show_errors=True):
        if self._loading or self._applying:
            return False
        self._sync_square_mesh()
        try:
            ui_parameters = self.parameters_from_ui()
            ui_geometry = self.geometry_from_ui()
            parameter_changes = {
                name: value for name, value in ui_parameters.items()
                if name not in self.parameters or value != self.parameters[name]
            }
            geometry_changes = {
                joint: {
                    name: value for name, value in values.items()
                    if joint not in self.geometry or value != self.geometry[joint].get(name)
                }
                for joint, values in ui_geometry.items()
            }
            geometry_changes = {joint: values for joint, values in geometry_changes.items() if values}
            if not parameter_changes and not geometry_changes:
                self.parameters, self.geometry = candidate_configuration()
                self.update_preview()
                return True

            self._applying = True
            parameters, geometry = write_configuration(parameter_changes, geometry_changes)
        except Exception as exc:
            self.warning.setText(f"Not saved: {exc}")
            if show_errors:
                QtWidgets.QMessageBox.critical(self, "Configuration not applied", str(exc))
            return False
        finally:
            self._applying = False
        self.parameters, self.geometry = parameters, geometry
        self.configuration_applied.emit(parameters)
        self.update_preview()
        return True

    def repair_settings(self):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Repair P2 settings")
        box.setText("Replace the complete working settings file with the shipped default?")
        box.setInformativeText(
            "This discards all current settings and any manual code edits in b2_Simulation_Settings.py."
        )
        repair_button = box.addButton("Repair", QtWidgets.QMessageBox.DestructiveRole)
        cancel_button = box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(cancel_button)
        box.exec_()
        if box.clickedButton() is not repair_button:
            return
        try:
            recover_full_file(PARAM_CONFIG_PATH, DEFAULT_PARAM_PATH)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Settings not repaired", str(exc))
            return
        self.reload()

    def apply_preset(self):
        try:
            spec = sr_preset_spec(self.preset.currentText())
            write_parameter_values({"microstructure_hydrogen_diff": spec})
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Preset not applied", str(exc))
            return
        self.reload()

    def open_material_editor(self):
        try:
            values = load_parameter_values()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot open editor", str(exc))
            return
        if MaterialEditor(values, self).exec_() == QtWidgets.QDialog.Accepted:
            self.reload()


class ResultsPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.artists = None
        self.worker = None
        self.play_timer = QtCore.QTimer(self, interval=120)
        self.play_timer.timeout.connect(self._advance)
        self._build_ui()

    def _build_ui(self):
        open_button = QtWidgets.QPushButton("Open HDF5...")
        open_button.clicked.connect(self.choose_file)
        self.play = QtWidgets.QPushButton("Play")
        self.play.clicked.connect(self.toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=0, maximum=0)
        self.slider.valueChanged.connect(self.show_frame)
        self.frame_label = QtWidgets.QLabel("No result loaded")
        self.show_diffusivity = QtWidgets.QCheckBox("Show diffusivity")
        self.show_diffusivity.setChecked(True)
        self.show_diffusivity.toggled.connect(self._toggle_diffusivity)
        statistics = QtWidgets.QPushButton("Scan statistics")
        statistics.clicked.connect(self.scan_statistics)
        export_frame = QtWidgets.QPushButton("Export current frame...")
        export_frame.clicked.connect(self.export_frame)
        export_movie = QtWidgets.QPushButton("Export standard MP4...")
        export_movie.clicked.connect(self.export_movie)
        self.cancel_task = QtWidgets.QPushButton("Cancel task")
        self.cancel_task.clicked.connect(self.cancel_worker)
        self.cancel_task.setEnabled(False)
        controls = QtWidgets.QHBoxLayout()
        for widget in (open_button, self.play, self.slider, self.frame_label, self.show_diffusivity, statistics, export_frame, export_movie, self.cancel_task):
            controls.addWidget(widget, 1 if widget is self.slider else 0)
        self.figure = Figure(figsize=(12, 7), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.metadata = QtWidgets.QPlainTextEdit(readOnly=True)
        self.metadata.setMaximumWidth(330)
        self.progress = QtWidgets.QProgressBar()
        placeholders = QtWidgets.QGroupBox("Reserved analysis tools")
        placeholder_layout = QtWidgets.QHBoxLayout(placeholders)
        for text in ("ISO report", "Compare results", "Concentration-gradient animation", "Snapshot diagrams", "Vertical distributions"):
            button = QtWidgets.QPushButton(text, enabled=False)
            button.setToolTip("Reserved for a later c1-c7 integration phase; the existing script remains directly usable.")
            placeholder_layout.addWidget(button)
        split = QtWidgets.QSplitter()
        split.addWidget(self.canvas)
        split.addWidget(self.metadata)
        split.setSizes([950, 260])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(split, 1)
        layout.addWidget(self.progress)
        layout.addWidget(placeholders)

    def choose_file(self):
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open P2 result", str(RESULTS_DIR), "HDF5 results (*.h5 *.hdf5)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            self.result = LazyP2Result(path)
            self.artists = P2ResultArtists(self.result, self.figure)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot open result", str(exc))
            return False
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.result.frame_count - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        summary = self.result.settings_summary()
        metadata = "\n".join(f"{key}: {value}" for key, value in summary.items())
        if self.result.metadata:
            metadata += "\n\nMetadata\n" + "\n".join(f"{key}: {value}" for key, value in sorted(self.result.metadata.items()))
        self.metadata.setPlainText(metadata)
        self.show_frame(0)
        self._toggle_diffusivity(self.show_diffusivity.isChecked())
        return True

    def show_frame(self, position):
        if not self.artists:
            return
        frame = self.artists.update(position)
        self.frame_label.setText(f"{position + 1}/{self.result.frame_count}  t={frame['time']:.5g} s")
        self.canvas.draw_idle()

    def toggle_play(self):
        if not self.result:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play.setText("Play")
        else:
            self.play_timer.start()
            self.play.setText("Pause")

    def _advance(self):
        if not self.result:
            return
        value = self.slider.value() + 1
        if value >= self.result.frame_count:
            value = 0
        self.slider.setValue(value)

    def _toggle_diffusivity(self, visible):
        if self.artists:
            self.artists.diffusivity_axis.set_visible(visible)
            self.artists.diffusivity_colorbar.ax.set_visible(visible)
            self.canvas.draw_idle()

    def _start_worker(self, function, on_success=None):
        if self.worker and self.worker.isRunning():
            return
        self.worker = TaskThread(function, self)
        self.worker.progress.connect(lambda value, text: (self.progress.setValue(int(value * 100)), self.progress.setFormat(text)))
        self.worker.failed.connect(lambda error: QtWidgets.QMessageBox.warning(self, "Background task", error))
        self.worker.finished.connect(lambda: self.cancel_task.setEnabled(False))
        if on_success:
            self.worker.succeeded.connect(on_success)
        self.cancel_task.setEnabled(True)
        self.worker.start()

    def scan_statistics(self):
        if not self.result:
            return
        self._start_worker(
            lambda progress, stop: self.result.scan_statistics(progress, stop),
            lambda data: (self.artists.update_statistics(data), self.canvas.draw_idle()),
        )

    def export_frame(self):
        if not self.result:
            return
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(self, "Export current frame", str(self.result.path.with_suffix(".png")), "PNG image (*.png);;SVG image (*.svg)")
        if path:
            self.figure.savefig(path, dpi=180)

    def export_movie(self, suggested_path=None, stride=None):
        if not self.result:
            return
        if suggested_path:
            path = str(suggested_path)
        else:
            default = str(self.result.path.with_suffix(".mp4"))
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export P2 animation", default, "MP4 animation (*.mp4)",
            )
            if not path:
                return
        if Path(path).exists() and QtWidgets.QMessageBox.question(self, "Overwrite animation?", f"Replace {path}?") != QtWidgets.QMessageBox.Yes:
            return
        if stride is None:
            stride, accepted = QtWidgets.QInputDialog.getInt(
                self, "Frame stride", "Render every nth frame:", 1, 1, 100000,
            )
            if not accepted:
                return
        self._start_worker(
            lambda progress, stop: export_animation(self.result, path, stride=stride, progress_callback=progress, stop_event=stop),
            lambda result: QtWidgets.QMessageBox.information(self, "Animation complete", str(result)),
        )

    def cancel_worker(self):
        if self.worker:
            self.worker.cancel()


class P2MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WeldCraft P2 -- Heat and Hydrogen During Welding")
        self.resize(1380, 900)
        self.process = None
        self.tabs = QtWidgets.QTabWidget()
        self.setup = SetupPanel()
        self.results = ResultsPanel()
        self.tabs.addTab(self.setup, "Setup")
        self.tabs.addTab(self.results, "Results")
        self.setup.run_requested.connect(self.run_simulation)
        logo = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(str(RESOURCES_DIR / "Images" / "BAM Logo.png"))
        logo.setPixmap(pixmap.scaledToHeight(54, QtCore.Qt.SmoothTransformation))
        title = QtWidgets.QLabel("<h2>WeldCraft P2</h2><div>Transient heat and hydrogen diffusion during welding</div>")
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.addWidget(logo)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.output = QtWidgets.QDockWidget("Solver output", self)
        output_widget = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(output_widget)
        self.output_text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.cancel_run_button = QtWidgets.QPushButton("Cancel active simulation")
        self.cancel_run_button.clicked.connect(self.cancel_run)
        output_layout.addWidget(self.output_text, 1)
        output_layout.addWidget(self.cancel_run_button)
        self.output.setWidget(output_widget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.output)
        self.output.hide()
        self.settings = QtCore.QSettings("WeldCraft", "P2")
        self.run_cancel_requested = False
        geometry = self.settings.value("windowGeometry")
        state = self.settings.value("windowState")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)
        self.startup_ready_signal = StartupReadySignal(self)

    def run_simulation(self):
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            return
        if not self.setup.apply_configuration():
            return
        self.output.show()
        self.output_text.clear()
        self.run_cancel_requested = False
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(CODE_DIR))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process)
        self.process.finished.connect(self._run_finished)
        self.statusBar().showMessage("Simulation running...")
        self.process.start(sys.executable, [str(CODE_DIR / "b1_Main_WeldCraft.py")])

    def _read_process(self):
        if self.process:
            self.output_text.appendPlainText(bytes(self.process.readAllStandardOutput()).decode(errors="replace").rstrip())

    def _run_finished(self, code, _status):
        self.statusBar().showMessage("Simulation completed" if code == 0 else f"Simulation failed ({code})", 10000)
        if code == 0:
            try:
                values = load_parameter_values()
                path = values["file_name"]
                if self.results.load_file(path):
                    self.tabs.setCurrentWidget(self.results)
                    if values.get("include_animation_after_run"):
                        self.results.export_movie(
                            values.get("animation_name"), stride=values.get("animation_frame_stride", 1),
                        )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Result could not be opened", str(exc))
        self.process.deleteLater()
        self.process = None

    def cancel_run(self, wait=False):
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            self.run_cancel_requested = True
            self.process.terminate()
            if wait:
                if not self.process.waitForFinished(3000):
                    self.process.kill()
                    self.process.waitForFinished(2000)
            else:
                QtCore.QTimer.singleShot(3000, lambda: self.process and self.process.kill())

    def closeEvent(self, event):
        active = self.process and self.process.state() != QtCore.QProcess.NotRunning
        if active:
            answer = QtWidgets.QMessageBox.question(self, "Stop active simulation?", "Closing P2 will terminate the active solver process.")
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self.cancel_run(wait=True)
        self.results.cancel_worker()
        self.setup.preview_timer.stop()
        self.results.play_timer.stop()
        if self.results.worker and self.results.worker.isRunning():
            self.results.worker.wait(5000)
        self.settings.setValue("windowGeometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("WeldCraft P2")
    app.setWindowIcon(QtGui.QIcon(str(RESOURCES_DIR / "Images" / "WeldCraft.ico")))
    window = P2MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
