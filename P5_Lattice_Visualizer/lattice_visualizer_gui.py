"""PyQt5 floating toolbox for the WeldCraft P5 Lattice Visualizer."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from visualize_lattice import (
    Config,
    Species,
    apply_camera_preset,
    apply_visual_preset,
    dump_config,
    load_config,
    persistent_config_path,
    runtime_directory,
    ensure_config_file,
)


APP_NAME = "WeldCraft - Lattice Visualizer"
WORKSPACE_PYTHON = Path(r"F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe")
READY_FILE_ENV_VAR = "WELDCRAFT_STARTUP_READY_FILE"


class _NoWheelMixin:
    """Keep the mouse wheel for page scrolling, never value editing."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelSlider(_NoWheelMixin, QtWidgets.QSlider):
    pass


class NoWheelSpinBox(_NoWheelMixin, QtWidgets.QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheelMixin, QtWidgets.QDoubleSpinBox):
    pass


class NoWheelComboBox(_NoWheelMixin, QtWidgets.QComboBox):
    pass


BOOL_KEYS = {
    "show_axes", "enable_picking", "show_unit_cell_overlay", "draw_bravais_overlay",
    "show_overlay_legend", "save_png",
    "png_include_lattice_name", "png_avoid_overwrite", "png_transparent_background",
    "base_atom_outline", "camera_normalize_demo_atom_size",
    "camera_parallel_projection", "adaptive_resolution",
}
INT_KEYS = {
    "target_atoms", "png_scale", "sphere_theta", "axis_font_size",
    "overlay_legend_padding", "overlay_legend_font_size", "stride", "random_seed",
}
FLOAT_KEYS = {
    "r", "base_radius_scale", "overlay_alpha",
    "overlay_marker_scale", "overlay_marker_opacity", "overlay_marker_specular",
    "overlay_legend_x_offset", "sphere_specular", "sphere_ambient", "sphere_diffuse",
    "base_atom_opacity", "base_atom_outline_width", "camera_distance_scale",
    "camera_view_angle", "axis_line_width",
}
LIST_FLOAT_KEYS = {"camera_direction", "camera_view_up", "window_size"}

TOOLTIPS = {
    "target_atoms": "Approximate number of host atoms. Cell dimensions are adjusted to reach this size.",
    "r": "Physical host-atom radius in nanometres.",
    "base_radius_scale": "Displayed host-atom radius relative to the physical host radius.",
    "lattice_size_behavior": "Choose whether small examples use one cell automatically, the requested atom count is always respected, or exactly one conventional cell is shown.",
    "visual_preset": "Choose a coordinated screen, thesis, publication, or outlined appearance.",
    "stride": "Keep every nth lattice point to make a very large lattice lighter.",
    "show_unit_cell_overlay": "Show the conventional cell and available interstitial sites.",
    "enable_picking": "Allow right-click selection of hydrogen sites in the display.",
    "camera_preset": "Choose a standard camera arrangement or use the explicit camera values.",
    "save_png": "Save a PNG using the output settings when a renderer window is opened.",
    "random_seed": "Optional repeatable random seed for dopant placement. Leave blank for a fresh random layout.",
    "lattice": "Select the host crystal structure.",
    "background": "Color behind the lattice.",
    "base_color": "Color used for the host atoms.",
    "base_atom_opacity": "Transparency of the host atoms.",
    "base_atom_outline": "Add an outline around host atoms.",
    "zoom_mode": "Choose whether zoom is centered on the cursor or the camera focus.",
    "interstitial_site_view": "All shows every periodic copy, Canonical shows one representative in the conventional cell, and Picture uses the configured picture-facing copies.",
    "overlay_marker_scale": "Size of the markers for available interstitial sites.",
    "overlay_marker_opacity": "Transparency of the available interstitial-site markers.",
    "camera_direction": "Direction from which the lattice is viewed.",
    "camera_view_up": "Which direction remains upward in the display.",
    "camera_distance_scale": "Distance of the camera from the lattice.",
    "camera_view_angle": "Perspective field of view in degrees.",
    "camera_parallel_projection": "Use an orthographic view without perspective shortening.",
    "window_size": "Initial display width and height in pixels.",
    "forced_position": "Optional fixed fractional position, entered as x, y, z.",
    "site": "Restrict interstitial placement to a site family, or allow any family.",
    "base_color": "Choose the host-atom color.",
    "visual_preset": "Apply a coordinated group of colors, lighting, opacity, and outline settings.",
    "base_atom_outline_color": "Choose the color of host-atom outlines.",
    "base_atom_outline_width": "Thickness of host-atom outlines.",
    "sphere_specular": "Strength of bright reflective highlights on atom surfaces.",
    "sphere_ambient": "Amount of even fill light applied to atom surfaces.",
    "sphere_diffuse": "Strength of directional light and surface shading.",
    "overlay_color": "Choose the unit-cell line color.",
    "overlay_alpha": "Visibility of unit-cell lines.",
    "overlay_marker_specular": "Strength of reflective highlights on site markers.",
    "tetrahedral_color": "Choose the tetrahedral-site marker color.",
    "octahedral_color": "Choose the octahedral-site marker color.",
    "cubic_color": "Choose the cubic-site marker color.",
    "show_overlay_legend": "Show the legend that identifies host atoms, site families, and occupied sites.",
    "overlay_legend_loc": "Choose the corner used by the site legend.",
    "overlay_legend_text_color": "Choose the neutral text color used by the site legend.",
    "overlay_legend_font_size": "Size of legend text in pixels.",
    "overlay_legend_padding": "Space between each legend marker and its label.",
    "overlay_legend_x_offset": "Move the legend horizontally within its selected corner.",
    "show_axes": "Show the orientation triad and numbered coordinate axes.",
    "draw_bravais_overlay": "Show the characteristic internal connections for BCC and FCC cells.",
    "pick_instruction": "Instruction displayed in the renderer when hydrogen picking is enabled.",
    "camera_normalize_demo_atom_size": "Keep host atoms visually comparable when switching between one-cell lattice types.",
    "axis_location": "Choose where the numbered coordinate axes are drawn around the scene.",
    "axis_font_size": "Size of coordinate-axis titles and numbers.",
    "axis_line_width": "Thickness of coordinate axes and tick marks.",
    "anti_aliasing": "Edge-smoothing method. FXAA is a good default for translucent outlined atoms.",
    "sphere_theta": "Geometric smoothness of atom spheres. Higher values look rounder but render more slowly.",
    "adaptive_resolution": "Automatically reduce sphere detail for very large lattices to keep interaction responsive.",
    "png_path": "File name and location used for PNG output.",
    "png_include_lattice_name": "Add the lattice type to the PNG file name.",
    "png_avoid_overwrite": "Create a numbered file instead of replacing an existing PNG.",
    "png_scale": "Resolution multiplier for saved PNGs.",
    "png_transparent_background": "Save the PNG without an opaque background.",
    "basic_A_fraction": "Fraction of host sites replaced by substitutional species A.",
    "basic_A_size": "Displayed radius of species A relative to host atoms.",
    "basic_B_fraction": "Fraction of host sites replaced by substitutional species B.",
    "basic_B_size": "Displayed radius of species B relative to host atoms.",
    "basic_H_count": "Number of interstitial hydrogen atoms.",
    "basic_H_size": "Displayed hydrogen radius relative to host atoms.",
}

LABEL_OVERRIDES = {
    "lattice_size_behavior": "Lattice size behavior",
    "target_atoms": "Approximate host atoms",
    "r": "Physical host radius [nm]",
    "base_radius_scale": "Host atom size",
    "base_color": "Host atom color",
    "interstitial_site_view": "Interstitial site view",
    "overlay_marker_scale": "Site marker size",
    "overlay_marker_opacity": "Site marker visibility",
    "random_seed": "Random seed",
    "basic_A_fraction": "Concentration",
    "basic_A_size": "Relative size",
    "basic_B_fraction": "Concentration",
    "basic_B_size": "Relative size",
    "basic_H_count": "Atom count",
    "basic_H_size": "Relative size",
    "site": "Site family",
    "forced_position": "Fixed position",
    "visual_preset": "Appearance preset",
    "background": "Background color",
    "base_atom_opacity": "Host atom visibility",
    "base_atom_outline": "Show host atom outlines",
    "base_atom_outline_color": "Outline color",
    "base_atom_outline_width": "Outline thickness",
    "sphere_specular": "Surface shine",
    "sphere_ambient": "Fill lighting",
    "sphere_diffuse": "Directional shading",
    "show_axes": "Show coordinate axes",
    "enable_picking": "Enable hydrogen picking",
    "zoom_mode": "Mouse-wheel zoom target",
    "pick_instruction": "Picking instruction",
    "show_unit_cell_overlay": "Show unit-cell guides",
    "draw_bravais_overlay": "Show lattice connections",
    "overlay_color": "Unit-cell line color",
    "overlay_alpha": "Unit-cell line visibility",
    "overlay_marker_specular": "Site marker shine",
    "tetrahedral_color": "Tetrahedral site color",
    "octahedral_color": "Octahedral site color",
    "cubic_color": "Cubic site color",
    "show_overlay_legend": "Show site legend",
    "overlay_legend_loc": "Legend corner",
    "overlay_legend_text_color": "Legend text color",
    "overlay_legend_font_size": "Legend text size",
    "overlay_legend_padding": "Legend marker spacing",
    "overlay_legend_x_offset": "Legend horizontal position",
    "camera_preset": "Camera view",
    "camera_direction": "Custom view direction",
    "camera_view_up": "Custom upward direction",
    "camera_distance_scale": "Camera distance",
    "camera_normalize_demo_atom_size": "Match one-cell atom size",
    "camera_parallel_projection": "Orthographic projection",
    "camera_view_angle": "Perspective field of view",
    "axis_location": "Numbered axes position",
    "axis_font_size": "Axis text size",
    "axis_line_width": "Axis line thickness",
    "stride": "Lattice sampling step",
    "anti_aliasing": "Edge smoothing",
    "sphere_theta": "Sphere smoothness",
    "adaptive_resolution": "Simplify very large lattices",
    "save_png": "Save a PNG when opening",
    "png_path": "PNG file",
    "png_include_lattice_name": "Add lattice to file name",
    "png_avoid_overwrite": "Keep existing images",
    "png_scale": "PNG resolution multiplier",
    "png_transparent_background": "Transparent PNG background",
    "window_size": "Display dimensions",
}


def _mark_startup_ready() -> None:
    ready_path = os.environ.get(READY_FILE_ENV_VAR, "").strip()
    if ready_path:
        try:
            Path(ready_path).touch()
        except OSError:
            pass


def _nice_label(name: str) -> str:
    return LABEL_OVERRIDES.get(name, name.replace("_", " ").capitalize())


LATTICE_OPTIONS = [
    ("Simple cubic (SC)", "Simple Cubic"),
    ("Body-centred cubic (BCC)", "BCC"),
    ("Face-centred cubic (FCC)", "FCC"),
]

SITE_VIEW_OPTIONS = [
    ("All", "all"),
    ("Canonical", "canonical"),
    ("Picture", "picture"),
]

LATTICE_SIZE_OPTIONS = [
    ("Automatic for small examples", "automatic"),
    ("Use requested atom count", "requested"),
    ("Show one-cell example", "one_cell"),
]

CUSTOM_APPEARANCE_KEYS = {
    "background", "base_color", "base_atom_opacity", "base_atom_outline",
    "base_atom_outline_color", "base_atom_outline_width", "sphere_specular",
    "sphere_ambient", "sphere_diffuse", "overlay_color", "overlay_alpha",
    "overlay_marker_opacity", "overlay_marker_specular", "tetrahedral_color",
    "octahedral_color", "cubic_color", "dopant_color",
}

CUSTOM_CAMERA_KEYS = {
    "camera_direction", "camera_view_up", "camera_distance_scale",
    "camera_parallel_projection", "camera_view_angle",
}

CONFIG_COLOR_KEYS = {
    "background", "base_color", "base_atom_outline_color", "overlay_color",
    "tetrahedral_color", "octahedral_color", "cubic_color",
    "overlay_legend_text_color",
}

# These settings affect files or the next window creation, not the scene that
# is already visible. Saving them must not rebuild a million-atom display.
DEFERRED_DISPLAY_KEYS = {
    "save_png",
    "png_path",
    "png_include_lattice_name",
    "png_avoid_overwrite",
    "png_scale",
    "png_transparent_background",
    "window_size",
}


def renderer_command():
    """Return the renderer program and fixed arguments for source/frozen use."""

    if getattr(sys, "frozen", False):
        renderer = Path(sys.executable).with_name("visualize_lattice_renderer.exe")
        return str(renderer), []
    python = str(WORKSPACE_PYTHON if WORKSPACE_PYTHON.exists() else sys.executable)
    script = str(Path(__file__).resolve().with_name("visualize_lattice.py"))
    return python, [script]


def smoke_test_renderer_launch() -> int:
    """Exercise the same sibling-renderer resolution used by the toolbox."""

    program, arguments = renderer_command()
    arguments += [
        "--config",
        str(ensure_config_file()),
        "--no-show",
    ]
    try:
        completed = subprocess.run(
            [program, *arguments],
            cwd=runtime_directory(),
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 1
    return int(completed.returncode)


class Toolbox(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.loading = True
        self.cfg_path = ensure_config_file()
        self.cfg = load_config(str(self.cfg_path))
        apply_visual_preset(self.cfg)
        apply_camera_preset(self.cfg)
        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self.basic_widgets: dict[str, QtWidgets.QWidget] = {}
        self.basic_slider_widgets: dict[str, QtWidgets.QSlider] = {}
        self.advanced_tab_indices: list[int] = []
        control_handle = tempfile.NamedTemporaryFile(
            prefix="weldcraft_p5_display_", suffix=".json", delete=False
        )
        self.control_path = Path(control_handle.name)
        control_handle.write(b"{}")
        control_handle.close()
        ready_handle = tempfile.NamedTemporaryFile(
            prefix="weldcraft_p5_display_ready_", suffix=".flag", delete=False
        )
        self.renderer_ready_path = Path(ready_handle.name)
        ready_handle.close()
        self.renderer_ready_path.unlink(missing_ok=True)
        self._pending_live_keys: set[str] = set()
        self._live_update_timer = QtCore.QTimer(self)
        self._live_update_timer.setSingleShot(True)
        self._live_update_timer.timeout.connect(self._send_live_update)
        self._status_clear_timer = QtCore.QTimer(self)
        self._status_clear_timer.setSingleShot(True)
        self._status_clear_timer.timeout.connect(
            lambda: self.status_label.setText("") if hasattr(self, "status_label") else None
        )
        self.renderer = QtCore.QProcess(self)
        self.renderer.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        self._renderer_stdout = ""
        self._renderer_stderr = ""
        self._renderer_close_requested = False
        self.renderer.stateChanged.connect(self._renderer_state_changed)
        self.renderer.errorOccurred.connect(self._renderer_error)
        self.renderer.finished.connect(self._renderer_finished)
        self.renderer.readyReadStandardOutput.connect(self._capture_renderer_output)
        self.renderer.readyReadStandardError.connect(self._capture_renderer_output)
        self._renderer_ready_timer = QtCore.QTimer(self)
        self._renderer_ready_timer.setInterval(100)
        self._renderer_ready_timer.timeout.connect(self._check_renderer_ready)

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(760, 620)
        self.resize(900, 760)
        self._set_window_icon()
        self._build_ui()
        self._restore_ui_state()
        self.loading = False
        QtCore.QTimer.singleShot(0, _mark_startup_ready)

    def _set_window_icon(self):
        candidates = [
            runtime_directory() / "Resources" / "Images" / "WeldCraft.ico",
            runtime_directory() / "WeldCraft.ico",
            Path(__file__).resolve().parents[1] / "Resources" / "Images" / "WeldCraft.ico",
        ]
        for path in candidates:
            if path.exists():
                self.setWindowIcon(QtGui.QIcon(str(path)))
                break

    def _resource_image(self, name):
        candidates = []
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            candidates.append(Path(bundle_root) / "01_Resources" / "Images" / name)
        candidates.extend([
            runtime_directory() / "01_Resources" / "Images" / name,
            Path(__file__).resolve().parent / "01_Resources" / "Images" / name,
        ])
        return next((path for path in candidates if path.exists()), None)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(4, 2, 8, 4)
        header.setSpacing(0)
        logo = QtWidgets.QLabel()
        logo.setObjectName("bamLogo")
        logo_path = self._resource_image("BAM Logo.png")
        if logo_path:
            pixmap = QtGui.QPixmap(str(logo_path))
            logo.setPixmap(pixmap.scaled(124, 48, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            logo.setToolTip("Bundesanstalt für Materialforschung und -prüfung")
        header.addWidget(logo, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        header.addSpacing(18)
        heading = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Lattice Visualizer Toolbox")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("Settings are saved automatically. Changes are applied to the display as they are ready.")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading, 1)
        root.addLayout(header)

        self.advanced_toggle = QtWidgets.QCheckBox("Advanced options")
        self.advanced_toggle.setToolTip(
            "Show the additional structure, appearance, camera, guide, quality, and output controls."
        )
        self.advanced_toggle.stateChanged.connect(self._toggle_advanced)
        root.addWidget(self.advanced_toggle)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._basic_tab(), "Basic")
        self.advanced_tab_indices = [
            self.tabs.addTab(self._structure_tab(), "Structure"),
            self.tabs.addTab(self._dopants_tab(), "Dopants"),
            self.tabs.addTab(self._appearance_tab(), "Appearance"),
            self.tabs.addTab(self._guides_tab(), "Guides & Camera"),
            self.tabs.addTab(self._quality_tab(), "Quality"),
            self.tabs.addTab(self._output_tab(), "Output"),
        ]
        for index in self.advanced_tab_indices:
            self.tabs.setTabVisible(index, False)
        root.addWidget(self.tabs, 1)

        actions = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setObjectName("statusLabel")
        actions.addWidget(self.status_label, 1)
        self.restore_button = QtWidgets.QPushButton("Restore Defaults")
        self.restore_button.setToolTip("Restore every setting to the documented starting values.")
        self.restore_button.clicked.connect(self._restore_defaults)
        actions.addWidget(self.restore_button)
        self.close_display_button = QtWidgets.QPushButton("Close Display")
        self.close_display_button.setToolTip("Close the lattice display while keeping this toolbox open.")
        self.close_display_button.clicked.connect(self._close_renderer)
        actions.addWidget(self.close_display_button)
        self.render_button = QtWidgets.QPushButton("Open Display")
        self.render_button.setToolTip(
            "Open the lattice display, or apply all current settings if it is already open."
        )
        self.render_button.setDefault(True)
        self.render_button.clicked.connect(self._render)
        actions.addWidget(self.render_button)
        root.addLayout(actions)
        self._sync_renderer_controls()

        self.setCentralWidget(central)
        self.setStyleSheet(
            """
            QWidget { font-size: 10pt; }
            #titleLabel { font-size: 18pt; font-weight: 600; color: #000000; }
            #subtitleLabel { color: #687789; }
            #statusLabel { color: #4f6070; padding-left: 4px; }
            QGroupBox { font-weight: 600; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QTabWidget::pane { border: 1px solid #c6ced8; }
            QPushButton { padding: 6px 12px; }
            """
        )

    def _sync_renderer_controls(self):
        state = self.renderer.state()
        running = state == QtCore.QProcess.Running
        self.close_display_button.setEnabled(running)
        ready = running and self.renderer_ready_path.exists()
        if state == QtCore.QProcess.Starting or (running and not ready):
            self.render_button.setText("Opening Display…")
            self.render_button.setEnabled(False)
        else:
            self.render_button.setText("Update Display" if running else "Open Display")
            self.render_button.setEnabled(True)

    def _show_status(self, message, timeout=3500):
        """Show short event feedback without leaving stale instructions behind."""

        self.status_label.setText(message)
        self._status_clear_timer.stop()
        if timeout:
            self._status_clear_timer.start(int(timeout))

    def _new_form(self):
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        return form

    def _scroll_page(self, widget):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _field_label(self, key):
        label = QtWidgets.QLabel(_nice_label(key))
        label.setToolTip(self._tooltip(key))
        return label

    def _tooltip(self, key):
        return TOOLTIPS.get(
            key,
            f"Set the {_nice_label(key).lower()} used by the display.",
        )

    @staticmethod
    def _set_control_tooltip(widget, tooltip):
        widget.setToolTip(tooltip)
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.lineEdit().setToolTip(tooltip)

    def _add_bool(self, form, key, *, group=None):
        box = QtWidgets.QCheckBox()
        box.setChecked(bool(getattr(self.cfg, key)))
        box.stateChanged.connect(lambda _state, k=key: self._persist_widget_change(k))
        box.setToolTip(self._tooltip(key))
        self.widgets[key] = box
        (group or form).addRow(self._field_label(key), box)
        return box

    def _add_combo(self, form, key, values, *, group=None, current=None):
        combo = NoWheelComboBox()
        for item in values:
            if isinstance(item, tuple):
                combo.addItem(item[0], item[1])
            else:
                combo.addItem(str(item), item)
        current = getattr(self.cfg, key) if current is None else current
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _index, k=key: self._persist_widget_change(k))
        combo.setToolTip(self._tooltip(key))
        self.widgets[key] = combo
        (group or form).addRow(self._field_label(key), combo)
        return combo

    def _add_spin(self, form, key, minimum, maximum, step=1, *, value=None, basic=False):
        spin = NoWheelSpinBox()
        spin.setRange(int(minimum), int(maximum))
        spin.setSingleStep(int(step))
        spin.setGroupSeparatorShown(True)
        spin.setValue(int(getattr(self.cfg, key) if value is None else value))
        self._set_control_tooltip(spin, self._tooltip(key))
        spin.valueChanged.connect(lambda _value, k=key: self._persist_widget_change(k))
        (self.basic_widgets if basic else self.widgets)[key] = spin
        form.addRow(self._field_label(key), spin)
        return spin

    def _add_slider(self, form, key, minimum, maximum, step=0.01, *, value=None, basic=False, decimals=2):
        current = float(getattr(self.cfg, key) if value is None else value)
        scale = max(1, int(round(1.0 / step)))
        slider = NoWheelSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
        slider.setSingleStep(1)
        slider.setValue(int(round(current * scale)))
        spin = NoWheelDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(decimals)
        spin.setSingleStep(float(step))
        spin.setValue(current)
        for widget in (slider, spin):
            self._set_control_tooltip(widget, self._tooltip(key))
        slider.valueChanged.connect(lambda raw: spin.setValue(raw / scale))
        spin.valueChanged.connect(lambda number: slider.setValue(int(round(number * scale))))
        spin.valueChanged.connect(lambda _value, k=key: self._persist_widget_change(k))

        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(spin)
        (self.basic_widgets if basic else self.widgets)[key] = spin
        if basic:
            self.basic_slider_widgets[key] = slider
        form.addRow(self._field_label(key), row)
        return spin

    def _set_color_button(self, button, color):
        parsed = QtGui.QColor(str(color))
        if not parsed.isValid():
            parsed = QtGui.QColor("white")
        canonical = parsed.name(QtGui.QColor.HexRgb).upper()
        button._config_color = canonical
        button.setText(canonical)
        text_color = "#000000" if parsed.lightness() > 145 else "#FFFFFF"
        button.setStyleSheet(
            f"QPushButton {{ background: {canonical}; color: {text_color}; border: 1px solid #7d8790; }}"
        )

    def _choose_color(self, button, key):
        selected = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(button._config_color), self, f"Choose {_nice_label(key).lower()}"
        )
        if selected.isValid():
            self._set_color_button(button, selected.name())
            self._persist_widget_change(key)

    def _add_color(self, form, key):
        button = QtWidgets.QPushButton()
        button.setToolTip(self._tooltip(key))
        self._set_color_button(button, getattr(self.cfg, key))
        button.clicked.connect(lambda _checked=False, b=button, k=key: self._choose_color(b, k))
        self.widgets[key] = button
        form.addRow(self._field_label(key), button)
        return button

    def _add_vector(self, form, key, minimum=-10.0, maximum=10.0, decimals=2, integer=False):
        values = list(getattr(self.cfg, key))
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        spins = []
        for axis, value in zip(("x", "y", "z"), values):
            spin = NoWheelSpinBox() if integer else NoWheelDoubleSpinBox()
            if integer:
                spin.setRange(int(minimum), int(maximum))
            else:
                spin.setRange(minimum, maximum)
            if not integer:
                spin.setDecimals(decimals)
                spin.setSingleStep(0.05)
            spin.setValue(int(value) if integer else float(value))
            axis_label = QtWidgets.QLabel(axis.upper())
            axis_label.setToolTip(self._tooltip(key))
            row_layout.addWidget(axis_label)
            self._set_control_tooltip(spin, self._tooltip(key))
            spin.valueChanged.connect(lambda _value, k=key: self._persist_widget_change(k))
            row_layout.addWidget(spin)
            spins.append(spin)
        row._vector_spins = spins
        row.setToolTip(self._tooltip(key))
        self.widgets[key] = row
        form.addRow(self._field_label(key), row)
        return row

    def _add_dimensions(self, form, key):
        width, height = (int(value) for value in getattr(self.cfg, key))
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        spins = []
        for label_text, value in (("Width", width), ("Height", height)):
            label = QtWidgets.QLabel(label_text)
            label.setToolTip(self._tooltip(key))
            row_layout.addWidget(label)
            spin = NoWheelSpinBox()
            spin.setRange(320, 7680)
            spin.setSingleStep(10)
            spin.setValue(value)
            self._set_control_tooltip(spin, self._tooltip(key))
            spin.valueChanged.connect(lambda _value, k=key: self._persist_widget_change(k))
            row_layout.addWidget(spin)
            spins.append(spin)
        row._vector_spins = spins
        row.setToolTip(self._tooltip(key))
        self.widgets[key] = row
        form.addRow(self._field_label(key), row)
        return row

    def _add_text(self, form, key, *, group=None, placeholder=""):
        edit = QtWidgets.QLineEdit()
        value = getattr(self.cfg, key)
        if isinstance(value, dict):
            text = json.dumps(value)
        elif isinstance(value, (tuple, list)):
            text = ", ".join(str(v) for v in value)
        else:
            text = "" if value is None else str(value)
        edit.setText(text)
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(lambda k=key: self._persist_widget_change(k))
        edit.setToolTip(self._tooltip(key))
        self.widgets[key] = edit
        (group or form).addRow(self._field_label(key), edit)
        return edit

    def _section(self, title):
        group = QtWidgets.QGroupBox(title)
        group.setLayout(self._new_form())
        return group

    def _add_basic_text(self, form, key, value, *, placeholder=""):
        edit = QtWidgets.QLineEdit()
        if isinstance(value, (tuple, list)):
            text = ", ".join(str(item) for item in value)
        else:
            text = "" if value is None else str(value)
        edit.setText(text)
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(lambda _key=key: self._persist_widget_change(_key))
        edit.setToolTip(self._tooltip(key))
        self.basic_widgets[key] = edit
        form.addRow(self._field_label(key), edit)
        return edit

    def _add_basic_slider(self, form, key, minimum, maximum, step=0.01, value=None):
        """Add a slider and editable numeric value for a frequently adjusted setting."""
        return self._add_slider(
            form, key, minimum, maximum, step, value=value, basic=True
        )

    def _add_basic_combo(self, form, key, values, *, current=None):
        combo = NoWheelComboBox()
        for item in values:
            if isinstance(item, tuple):
                combo.addItem(item[0], item[1])
            else:
                combo.addItem(str(item), item)
        current = getattr(self.cfg, key) if current is None else current
        if combo.findData(current) >= 0:
            combo.setCurrentIndex(combo.findData(current))
        combo.currentIndexChanged.connect(lambda _index, _key=key: self._persist_widget_change(_key))
        combo.setToolTip(self._tooltip(key))
        self.basic_widgets[key] = combo
        form.addRow(self._field_label(key), combo)
        return combo

    def _add_basic_bool(self, form, key):
        box = QtWidgets.QCheckBox()
        box.setChecked(bool(getattr(self.cfg, key)))
        box.stateChanged.connect(lambda _state, _key=key: self._persist_widget_change(_key))
        box.setToolTip(self._tooltip(key))
        self.basic_widgets[key] = box
        form.addRow(self._field_label(key), box)
        return box

    def _add_optional_seed(self, form, *, basic=False):
        spin = NoWheelSpinBox()
        spin.setRange(-1, 2_000_000_000)
        spin.setSpecialValueText("Random")
        spin.setGroupSeparatorShown(True)
        spin.setValue(-1 if self.cfg.random_seed is None else int(self.cfg.random_seed))
        self._set_control_tooltip(spin, self._tooltip("random_seed"))
        spin.valueChanged.connect(
            lambda _value: self._persist_widget_change("random_seed")
        )
        (self.basic_widgets if basic else self.widgets)["random_seed"] = spin
        form.addRow(self._field_label("random_seed"), spin)
        return spin

    def _species_by_name(self, name):
        for species in self.cfg.dopants:
            if species.name.strip().lower() == name.lower():
                return species
        return None

    def _lattice_size_behavior(self):
        if self.cfg.demo_cell_force is True:
            return "one_cell"
        if self.cfg.demo_cell_force is False or not self.cfg.demo_cell_auto:
            return "requested"
        return "automatic"

    @staticmethod
    def _apply_lattice_size_behavior(cfg, behavior):
        if behavior == "one_cell":
            cfg.demo_cell_auto = False
            cfg.demo_cell_force = True
        elif behavior == "requested":
            cfg.demo_cell_auto = False
            cfg.demo_cell_force = False
        else:
            cfg.demo_cell_auto = True
            cfg.demo_cell_force = None

    def _basic_species_field(self, form, species_name, field_name, value):
        key = f"basic_{species_name}_{field_name}"
        if field_name == "fraction":
            return self._add_basic_slider(form, key, 0.0, 1.0, 0.01, value=value)
        if field_name in {"size", "size_scale"}:
            return self._add_basic_slider(form, key, 0.05, 2.0, 0.01, value=value)
        return self._add_basic_text(form, key, value)

    def _basic_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        mode = self._section("Mode and lattice")
        form = mode.layout()
        self._add_basic_combo(form, "lattice", LATTICE_OPTIONS)
        self._add_spin(form, "target_atoms", 1, 2_000_000_000, value=self.cfg.target_atoms, basic=True)
        self._add_basic_combo(
            form,
            "lattice_size_behavior",
            LATTICE_SIZE_OPTIONS,
            current=self._lattice_size_behavior(),
        )
        self._add_basic_slider(form, "base_radius_scale", 0.05, 2.0, 0.01)
        self._add_optional_seed(form, basic=True)
        layout.addWidget(mode)

        substitutionals = QtWidgets.QGroupBox("Substitutionals")
        substitutional_layout = QtWidgets.QHBoxLayout(substitutionals)
        species_a = self._species_by_name("A") or Species(name="A", color="red")
        species_b = self._species_by_name("B") or Species(name="B", color="black")
        for name, species in (("A", species_a), ("B", species_b)):
            species_group = self._section(f"Species {name}")
            species_form = species_group.layout()
            self._basic_species_field(species_form, name, "fraction", species.fraction)
            self._basic_species_field(species_form, name, "size", species.size_scale)
            substitutional_layout.addWidget(species_group)
        layout.addWidget(substitutionals)

        interstitial = self._section("Species H — hydrogen interstitial")
        form = interstitial.layout()
        species_h = self._species_by_name("H") or Species(name="H", color="blue", mode="interstitial")
        self._add_spin(form, "basic_H_count", 0, 100_000_000, value=species_h.count, basic=True)
        self._basic_species_field(form, "H", "size", species_h.size_scale)
        self._add_basic_combo(form, "interstitial_site_view", SITE_VIEW_OPTIONS)
        self._add_basic_slider(form, "overlay_marker_scale", 0.05, 2.0, 0.01, value=self.cfg.overlay_marker_scale)
        self._add_basic_slider(form, "overlay_marker_opacity", 0.0, 1.0, 0.01, value=self.cfg.overlay_marker_opacity)
        layout.addWidget(interstitial)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _structure_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        basic = self._section("Physical model")
        form = basic.layout()
        self._add_combo(form, "lattice", LATTICE_OPTIONS)
        self._add_spin(form, "target_atoms", 1, 2_000_000_000)
        self._add_slider(form, "r", 0.05, 0.30, 0.001, decimals=3)
        self._add_slider(form, "base_radius_scale", 0.05, 2.0, 0.01)
        self._add_color(form, "base_color")
        self._add_combo(
            form,
            "lattice_size_behavior",
            LATTICE_SIZE_OPTIONS,
            current=self._lattice_size_behavior(),
        )
        self._add_optional_seed(form)
        layout.addWidget(basic)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _dopants_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        help_label = QtWidgets.QLabel(
            "Substitutional dopants replace lattice atoms by fraction. Interstitial dopants are placed by count "
            "at legal tetrahedral, octahedral, or cubic sites."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.dopant_table = QtWidgets.QTableWidget(0, 8)
        self.dopant_table.setHorizontalHeaderLabels([
            "Species", "Color", "Placement", "Concentration", "Atom count", "Relative size", "Site family", "Fixed position"
        ])
        self.dopant_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.dopant_table.setColumnWidth(6, 145)
        self.dopant_table.setColumnWidth(7, 190)
        self.dopant_table.setToolTip(
            "Use a site family for interstitial placement. A fixed position uses fractional cell coordinates x, y, z."
        )
        self.dopant_table.itemChanged.connect(lambda _item: self._persist_widget_change("dopants"))
        layout.addWidget(self.dopant_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add Dopant")
        add.setToolTip("Add another species to the lattice model.")
        add.clicked.connect(lambda: self._add_dopant_row(Species(name="New", color="green")))
        remove = QtWidgets.QPushButton("Remove Selected")
        remove.setToolTip("Remove the currently selected species row.")
        remove.clicked.connect(self._remove_dopant_row)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._populate_dopants()
        return self._scroll_page(page)

    def _site_value_for_lattice(self, value):
        if isinstance(value, dict):
            value = value.get(self._lattice_mapping_key())
        return str(value or "any").strip().lower()

    def _position_value_for_lattice(self, value):
        if isinstance(value, dict):
            value = value.get(self._lattice_mapping_key())
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(float(part) for part in value)
        return None

    def _lattice_mapping_key(self, lattice=None):
        lattice = str(self.cfg.lattice if lattice is None else lattice).strip().lower()
        if lattice in {"simple cubic", "simple_cubic", "sc"}:
            return "SC"
        return lattice.upper()

    def _populate_dopants(self):
        self.loading = True
        self.dopant_table.setRowCount(0)
        for species in self.cfg.dopants:
            self._add_dopant_row(species)
        self.loading = False

    def _add_dopant_row(self, species):
        row = self.dopant_table.rowCount()
        self.dopant_table.insertRow(row)
        name_item = QtWidgets.QTableWidgetItem(species.name)
        name_item.setToolTip("Name shown for this species in the display and legend.")
        self.dopant_table.setItem(row, 0, name_item)

        color_button = QtWidgets.QPushButton()
        self._set_color_button(color_button, species.color)
        color_button.setToolTip("Choose this species' atom color.")
        color_button.clicked.connect(
            lambda _checked=False, button=color_button: self._choose_color(button, "dopant_color")
        )
        self.dopant_table.setCellWidget(row, 1, color_button)

        mode_combo = NoWheelComboBox()
        mode_combo.addItem("Substitutional", "substitutional")
        mode_combo.addItem("Interstitial", "interstitial")
        mode_combo.setCurrentIndex(max(0, mode_combo.findData(species.mode)))
        mode_combo.setToolTip(
            "Substitutional atoms replace host atoms; interstitial atoms occupy spaces between them."
        )
        mode_combo.currentIndexChanged.connect(lambda _index: self._persist_widget_change("dopants"))
        self.dopant_table.setCellWidget(row, 2, mode_combo)

        fraction = self._dopant_slider(
            float(species.fraction), 0.0, 1.0, 0.01, 4,
            "Fraction of host sites replaced by this species.",
        )
        self.dopant_table.setCellWidget(row, 3, fraction)

        count = NoWheelSpinBox()
        count.setRange(0, 100_000_000)
        count.setGroupSeparatorShown(True)
        count.setValue(int(species.count))
        self._set_control_tooltip(
            count, "Number of atoms placed when this species is interstitial."
        )
        count.valueChanged.connect(lambda _value: self._persist_widget_change("dopants"))
        self.dopant_table.setCellWidget(row, 4, count)

        size = self._dopant_slider(
            float(species.size_scale), 0.05, 5.0, 0.05, 2,
            "Displayed radius relative to host atoms.",
        )
        self.dopant_table.setCellWidget(row, 5, size)

        site_combo = NoWheelComboBox()
        site_values = [
            ("Any site", "any"),
            ("Tetrahedral", "tetra"),
            ("Octahedral", "octa"),
            ("Cubic", "cubic"),
        ]
        for label, raw_value in site_values:
            site_combo.addItem(label, raw_value)
        site_index = site_combo.findData(self._site_value_for_lattice(species.interstitial_site))
        site_combo.setCurrentIndex(max(0, site_index))
        site_combo._source_value = copy.deepcopy(species.interstitial_site)
        site_combo.setToolTip(self._tooltip("site"))
        site_combo.currentIndexChanged.connect(lambda _index: self._persist_widget_change("dopants"))
        self.dopant_table.setCellWidget(row, 6, site_combo)

        position = self._position_value_for_lattice(species.forced_interstitial_position)
        position_control = QtWidgets.QWidget()
        position_layout = QtWidgets.QHBoxLayout(position_control)
        position_layout.setContentsMargins(0, 0, 0, 0)
        enabled = QtWidgets.QCheckBox()
        enabled.setChecked(position is not None)
        enabled.setToolTip(
            "Use one exact fractional cell position instead of selecting a legal site automatically."
        )
        position_layout.addWidget(enabled)
        spins = []
        values = position or (0.0, 0.0, 0.0)
        for axis, value in zip(("x", "y", "z"), values):
            spin = NoWheelDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.05)
            axis_label = QtWidgets.QLabel(axis.upper())
            axis_label.setToolTip(self._tooltip("forced_position"))
            position_layout.addWidget(axis_label)
            spin.setValue(float(value))
            spin.setEnabled(enabled.isChecked())
            self._set_control_tooltip(spin, self._tooltip("forced_position"))
            spin.valueChanged.connect(lambda _value: self._persist_widget_change("dopants"))
            position_layout.addWidget(spin)
            spins.append(spin)
        enabled.toggled.connect(lambda checked, fields=spins: [field.setEnabled(checked) for field in fields])
        enabled.toggled.connect(lambda _checked: self._persist_widget_change("dopants"))
        position_control._position_enabled = enabled
        position_control._position_spins = spins
        position_control._source_value = copy.deepcopy(species.forced_interstitial_position)
        position_control.setToolTip(self._tooltip("forced_position"))
        self.dopant_table.setCellWidget(row, 7, position_control)

        def update_placement_controls():
            interstitial = mode_combo.currentData() == "interstitial"
            fraction.setEnabled(not interstitial)
            count.setEnabled(interstitial)
            site_combo.setEnabled(interstitial)
            position_control.setEnabled(interstitial)

        mode_combo.currentIndexChanged.connect(lambda _index: update_placement_controls())
        update_placement_controls()

    def _dopant_slider(self, value, minimum, maximum, step, decimals, tooltip):
        """Create a compact table slider with an editable numeric value."""

        scale = max(1, int(round(1.0 / step)))
        slider = NoWheelSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
        slider.setValue(int(round(value * scale)))
        spin = NoWheelDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(int(decimals))
        spin.setSingleStep(float(step))
        spin.setValue(float(value))
        slider.valueChanged.connect(lambda raw: spin.setValue(raw / scale))
        spin.valueChanged.connect(lambda number: slider.setValue(int(round(number * scale))))
        spin.valueChanged.connect(lambda _value: self._persist_widget_change("dopants"))
        for widget in (slider, spin):
            self._set_control_tooltip(widget, tooltip)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        container._value_spin = spin
        container.setToolTip(tooltip)
        return container

    def _remove_dopant_row(self):
        row = self.dopant_table.currentRow()
        if row >= 0:
            self.dopant_table.removeRow(row)
            self._persist_widget_change("dopants")

    def _collect_dopants(self, lattice=None):
        result = []
        lattice_key = self._lattice_mapping_key(lattice)
        for row in range(self.dopant_table.rowCount()):
            def cell(column):
                item = self.dopant_table.item(row, column)
                return item.text().strip() if item else ""

            mode_widget = self.dopant_table.cellWidget(row, 2)
            mode = mode_widget.currentData() if mode_widget else "substitutional"
            color_widget = self.dopant_table.cellWidget(row, 1)
            fraction_widget = self.dopant_table.cellWidget(row, 3)
            count_widget = self.dopant_table.cellWidget(row, 4)
            size_widget = self.dopant_table.cellWidget(row, 5)
            site_widget = self.dopant_table.cellWidget(row, 6)
            site = site_widget.currentData() if isinstance(site_widget, QtWidgets.QComboBox) else None
            source_site = getattr(site_widget, "_source_value", None)
            if isinstance(source_site, dict):
                site = copy.deepcopy(source_site)
                site[lattice_key] = site_widget.currentData()
            position_widget = self.dopant_table.cellWidget(row, 7)
            forced = None
            if position_widget._position_enabled.isChecked():
                forced = tuple(spin.value() for spin in position_widget._position_spins)
            source_position = getattr(position_widget, "_source_value", None)
            if isinstance(source_position, dict):
                forced_mapping = copy.deepcopy(source_position)
                forced_mapping[lattice_key] = forced
                forced = forced_mapping
            if mode == "substitutional":
                site = None
                forced = None
            result.append(Species(
                name=cell(0) or "Dopant",
                color=getattr(color_widget, "_config_color", "#FFFFFF"),
                mode=mode,
                fraction=float(fraction_widget._value_spin.value()),
                count=int(count_widget.value()),
                size_scale=float(size_widget._value_spin.value()),
                interstitial_site=site,
                forced_interstitial_position=forced,
            ))
        return result

    def _appearance_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        group = self._section("Rendering style")
        form = group.layout()
        self._add_combo(form, "visual_preset", [
            ("Custom", "custom"),
            ("Standard screen view", "screen"),
            ("Thesis / publication", "thesis"),
            ("Publication", "publication"),
            ("Translucent atoms with outlines", "outline"),
        ])
        self._add_color(form, "background")
        self._add_slider(form, "base_atom_opacity", 0.05, 1.0, 0.01)
        self._add_bool(form, "base_atom_outline")
        self._add_color(form, "base_atom_outline_color")
        self._add_slider(form, "base_atom_outline_width", 0.5, 8.0, 0.1, decimals=1)
        layout.addWidget(group)

        lighting = self._section("Surface lighting")
        form = lighting.layout()
        self._add_slider(form, "sphere_specular", 0.0, 1.0, 0.01)
        self._add_slider(form, "sphere_ambient", 0.0, 1.0, 0.01)
        self._add_slider(form, "sphere_diffuse", 0.0, 1.0, 0.01)
        layout.addWidget(lighting)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _guides_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        guides = self._section("Interaction and guides")
        form = guides.layout()
        self._add_bool(form, "show_axes")
        self._add_bool(form, "enable_picking")
        self._add_combo(form, "zoom_mode", [
            ("Point under the mouse", "cursor"),
            ("Center of the scene", "focal"),
        ])
        self._add_text(form, "pick_instruction")
        self._add_bool(form, "show_unit_cell_overlay")
        self._add_bool(form, "draw_bravais_overlay")
        self._add_combo(form, "interstitial_site_view", SITE_VIEW_OPTIONS)
        self._add_color(form, "overlay_color")
        self._add_slider(form, "overlay_alpha", 0.0, 1.0, 0.01)
        self._add_slider(form, "overlay_marker_scale", 0.05, 2.0, 0.01)
        self._add_slider(form, "overlay_marker_opacity", 0.0, 1.0, 0.01)
        self._add_slider(form, "overlay_marker_specular", 0.0, 1.0, 0.01)
        self._add_color(form, "tetrahedral_color")
        self._add_color(form, "octahedral_color")
        self._add_color(form, "cubic_color")
        layout.addWidget(guides)

        legend = self._section("Site legend")
        form = legend.layout()
        self._add_bool(form, "show_overlay_legend")
        self._add_combo(form, "overlay_legend_loc", [
            ("Upper right", "upper right"),
            ("Upper left", "upper left"),
            ("Lower left", "lower left"),
            ("Lower right", "lower right"),
        ])
        self._add_color(form, "overlay_legend_text_color")
        self._add_slider(form, "overlay_legend_font_size", 8, 40, 1, decimals=0)
        self._add_slider(form, "overlay_legend_padding", 0, 40, 1, decimals=0)
        self._add_slider(form, "overlay_legend_x_offset", -0.25, 0.25, 0.005, decimals=3)
        layout.addWidget(legend)

        camera = self._section("Camera and numbered axes")
        form = camera.layout()
        self._add_combo(form, "camera_preset", [
            ("Custom", "custom"),
            ("Isometric", "isometric"),
            ("Low isometric", "low_isometric"),
        ])
        self._add_vector(form, "camera_direction", -5.0, 5.0)
        self._add_vector(form, "camera_view_up", -1.0, 1.0)
        self._add_slider(form, "camera_distance_scale", 1.0, 8.0, 0.05)
        self._add_bool(form, "camera_normalize_demo_atom_size")
        self._add_bool(form, "camera_parallel_projection")
        self._add_slider(form, "camera_view_angle", 10.0, 90.0, 1.0, decimals=0)
        self._add_combo(form, "axis_location", [
            ("Outside", "outer"),
            ("Back", "back"),
            ("Front", "front"),
            ("All sides", "all"),
        ])
        self._add_slider(form, "axis_font_size", 8, 64, 1, decimals=0)
        self._add_slider(form, "axis_line_width", 0.5, 6.0, 0.05)
        layout.addWidget(camera)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _quality_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        group = self._section("Quality and large-lattice handling")
        form = group.layout()
        self._add_slider(form, "stride", 1, 100, 1, decimals=0)
        self._add_combo(form, "anti_aliasing", [
            ("Fast edge smoothing (FXAA)", "fxaa"),
            ("Multisample smoothing (MSAA)", "msaa"),
            ("High-quality smoothing (SSAA)", "ssaa"),
            ("Off", "none"),
        ])
        self._add_slider(form, "sphere_theta", 6, 64, 1, decimals=0)
        self._add_bool(form, "adaptive_resolution")
        layout.addWidget(group)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _output_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        group = self._section("PNG output")
        form = group.layout()
        self._add_bool(form, "save_png")
        self._add_text(form, "png_path")
        self._add_bool(form, "png_include_lattice_name")
        self._add_bool(form, "png_avoid_overwrite")
        self._add_slider(form, "png_scale", 1, 6, 1, decimals=0)
        self._add_bool(form, "png_transparent_background")
        self._add_dimensions(form, "window_size")
        layout.addWidget(group)
        layout.addStretch(1)
        return self._scroll_page(page)

    def _parse_value(self, key, text):
        text = text.strip()
        if key == "random_seed":
            return None if not text else int(float(text.replace(",", ".")))
        if key in INT_KEYS:
            return int(float(text.replace(",", ".")))
        if key in FLOAT_KEYS:
            return float(text.replace(",", "."))
        if key in LIST_FLOAT_KEYS:
            values = [float(part.strip()) for part in text.replace("[", "").replace("]", "").split(",") if part.strip()]
            if key == "window_size":
                return [int(value) for value in values]
            return values
        return text

    def _widget_value(self, key, widget):
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QComboBox):
            return widget.currentData()
        if isinstance(widget, QtWidgets.QPushButton) and hasattr(widget, "_config_color"):
            return widget._config_color
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            value = widget.value()
            if key == "random_seed":
                return None if value < 0 else int(value)
            return int(round(value)) if key in INT_KEYS else value
        if hasattr(widget, "_vector_spins"):
            values = [spin.value() for spin in widget._vector_spins]
            return [int(value) for value in values] if key == "window_size" else values
        return self._parse_value(key, widget.text())

    def _collect_config(self, changed_key=None):
        cfg = copy.deepcopy(self.cfg)
        for key, widget in self.widgets.items():
            if key in {"dopants", "lattice_size_behavior"}:
                continue
            setattr(cfg, key, self._widget_value(key, widget))
        if "sphere_theta" in self.widgets:
            cfg.sphere_phi = int(cfg.sphere_theta)

        if self.advanced_toggle.isChecked():
            # The table still represents the old lattice at the instant its
            # lattice combo changes. Preserve the mappings for that save, then
            # rebuild the table below using the newly selected lattice.
            if changed_key != "lattice":
                cfg.dopants = self._collect_dopants(lattice=cfg.lattice)
        else:
            basic_values = {}
            for key, widget in self.basic_widgets.items():
                if key.startswith("basic_") or key == "lattice_size_behavior":
                    continue
                basic_values[key] = self._widget_value(key, widget)
            for key, value in basic_values.items():
                setattr(cfg, key, value)

            existing = {species.name.strip().lower(): species for species in cfg.dopants}
            basic_species = {
                "A": {"fraction": "basic_A_fraction", "size_scale": "basic_A_size"},
                "B": {"fraction": "basic_B_fraction", "size_scale": "basic_B_size"},
                "H": {"count": "basic_H_count", "size_scale": "basic_H_size"},
            }
            for name, fields in basic_species.items():
                species = existing.get(name.lower())
                if species is None:
                    species = Species(
                        name=name,
                        color="blue" if name == "H" else "red" if name == "A" else "black",
                        mode="interstitial" if name == "H" else "substitutional",
                    )
                    cfg.dopants.append(species)
                for field_name, widget_key in fields.items():
                    widget = self.basic_widgets[widget_key]
                    raw = widget.value() if isinstance(
                        widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
                    ) else float(widget.text().replace(",", "."))
                    setattr(species, field_name, int(raw) if field_name == "count" else float(raw))
        behavior_widgets = self.widgets if self.advanced_toggle.isChecked() else self.basic_widgets
        behavior_widget = behavior_widgets.get("lattice_size_behavior")
        if isinstance(behavior_widget, QtWidgets.QComboBox):
            self._apply_lattice_size_behavior(cfg, behavior_widget.currentData())
        for key in CONFIG_COLOR_KEYS:
            color = QtGui.QColor(str(getattr(cfg, key)))
            if color.isValid():
                setattr(cfg, key, color.name(QtGui.QColor.HexRgb).upper())
        for species in cfg.dopants:
            color = QtGui.QColor(str(species.color))
            if color.isValid():
                species.color = color.name(QtGui.QColor.HexRgb).upper()
        return cfg

    def _persist_widget_change(self, changed_key):
        if self.loading:
            return
        try:
            if changed_key == "target_atoms":
                # An old one-cell override must not silently defeat a newly
                # entered atom count. Users can still select an override again
                # afterwards when they deliberately want a teaching view.
                for controls in (self.widgets, self.basic_widgets):
                    behavior_combo = controls.get("lattice_size_behavior")
                    if isinstance(behavior_combo, QtWidgets.QComboBox):
                        automatic_index = behavior_combo.findData("automatic")
                        behavior_combo.blockSignals(True)
                        behavior_combo.setCurrentIndex(max(0, automatic_index))
                        behavior_combo.blockSignals(False)
                self._apply_lattice_size_behavior(self.cfg, "automatic")
            self._activate_custom_mode(changed_key)
            cfg = self._collect_config(changed_key=changed_key)
            if changed_key == "visual_preset":
                apply_visual_preset(cfg)
            if changed_key == "camera_preset":
                apply_camera_preset(cfg)
            dump_config(cfg, str(self.cfg_path))
            self.cfg = cfg
            if changed_key in {"visual_preset", "camera_preset", "lattice"}:
                self.loading = True
                self._rebuild_ui(preserve_advanced=self.advanced_toggle.isChecked())
                self.loading = False
            renderer_running = self.renderer.state() == QtCore.QProcess.Running
            if renderer_running and changed_key not in DEFERRED_DISPLAY_KEYS:
                self._pending_live_keys.add(changed_key)
                self._show_status("Updating display…")
                self._live_update_timer.start(90)
            else:
                self._show_status("Settings saved.")
        except Exception as exc:
            self._show_status(f"Could not apply that setting: {exc}", timeout=8000)

    def _activate_custom_mode(self, changed_key):
        if changed_key in CUSTOM_APPEARANCE_KEYS:
            preset = self.widgets.get("visual_preset")
            if isinstance(preset, QtWidgets.QComboBox):
                index = preset.findData("custom")
                preset.blockSignals(True)
                preset.setCurrentIndex(index)
                preset.blockSignals(False)
            else:
                self.cfg.visual_preset = "custom"
        if changed_key in CUSTOM_CAMERA_KEYS:
            preset = self.widgets.get("camera_preset")
            if isinstance(preset, QtWidgets.QComboBox):
                index = preset.findData("custom")
                preset.blockSignals(True)
                preset.setCurrentIndex(index)
                preset.blockSignals(False)
            else:
                self.cfg.camera_preset = "custom"

    def _send_live_update(self):
        if not self._pending_live_keys:
            return
        changed = sorted(self._pending_live_keys)
        self._pending_live_keys.clear()
        if self.renderer.state() != QtCore.QProcess.Running:
            return
        payload = {"action": "update", "changed": changed}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(self.control_path.parent),
                prefix="p5_update_", suffix=".json", delete=False,
            ) as handle:
                json.dump(payload, handle)
                handle.flush()
                temporary = handle.name
            os.replace(temporary, str(self.control_path))
        except OSError as exc:
            if temporary:
                try:
                    Path(temporary).unlink()
                except OSError:
                    pass
            self._show_status(f"Could not update the display: {exc}", timeout=8000)

    def _toggle_advanced(self, state):
        enabled = bool(state)
        if not hasattr(self, "tabs"):
            return
        self.tabs.setTabVisible(0, not enabled)
        for index in self.advanced_tab_indices:
            self.tabs.setTabVisible(index, enabled)
        self.tabs.setCurrentIndex(self.advanced_tab_indices[0] if enabled else 0)
        if not self.loading:
            self._rebuild_ui(preserve_advanced=enabled)

    def _restore_defaults(self):
        answer = QtWidgets.QMessageBox.question(
            self,
            "Restore defaults",
            "Replace the current lattice configuration with the documented defaults?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.loading = True
            template_path = runtime_directory() / "01_Resources" / "config_default.py"
            if not template_path.exists():
                template_path = Path(__file__).resolve().parent / "01_Resources" / "config_default.py"
            default_cfg = load_config(str(template_path))
            dump_config(default_cfg, str(self.cfg_path))
            self.cfg = load_config(str(self.cfg_path))
            apply_visual_preset(self.cfg)
            apply_camera_preset(self.cfg)
            self._rebuild_ui()
            self._show_status(
                "Defaults restored; updating display…"
                if self.renderer.state() == QtCore.QProcess.Running
                else "Defaults restored."
            )
            self._pending_live_keys = {"defaults"}
            self._live_update_timer.start(0)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Restore failed", str(exc))
        finally:
            self.loading = False

    def _rebuild_ui(self, preserve_advanced=False):
        old = self.centralWidget()
        if old is not None:
            old.deleteLater()
        self.widgets.clear()
        self.basic_widgets.clear()
        self.basic_slider_widgets.clear()
        self._build_ui()
        self.advanced_toggle.blockSignals(True)
        self.advanced_toggle.setChecked(bool(preserve_advanced))
        self.advanced_toggle.blockSignals(False)
        self.tabs.setTabVisible(0, not preserve_advanced)
        for index in self.advanced_tab_indices:
            self.tabs.setTabVisible(index, bool(preserve_advanced))
        self.tabs.setCurrentIndex(self.advanced_tab_indices[0] if preserve_advanced else 0)

    def _renderer_command(self):
        return renderer_command()

    def _render(self):
        try:
            cfg = self._collect_config()
            dump_config(cfg, str(self.cfg_path))
            self.cfg = cfg
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot render", str(exc))
            return

        if self.renderer.state() == QtCore.QProcess.Running:
            self._pending_live_keys.add("manual update")
            self._send_live_update()
            self._show_status("Updating display…")
            return
        program, arguments = self._renderer_command()
        arguments += [
            "--config", str(self.cfg_path), "--force-display",
            "--control-file", str(self.control_path),
        ]
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert(READY_FILE_ENV_VAR, str(self.renderer_ready_path))
        environment.remove("PYVISTA_OFF_SCREEN")
        self.renderer.setProcessEnvironment(environment)
        self.renderer.setWorkingDirectory(str(runtime_directory()))
        self.renderer_ready_path.unlink(missing_ok=True)
        self._pending_live_keys.clear()
        self._live_update_timer.stop()
        self._renderer_stdout = ""
        self._renderer_stderr = ""
        self._renderer_close_requested = False
        self.renderer.start(program, arguments)
        self._renderer_ready_timer.start()
        self._show_status("Opening display window…", timeout=8000)
        self.render_button.setText("Opening Display…")
        self.render_button.setEnabled(False)

    def _close_renderer(self, wait=False):
        if self.renderer.state() == QtCore.QProcess.NotRunning:
            return
        self._renderer_close_requested = True
        self.renderer.terminate()
        if wait and not self.renderer.waitForFinished(3000):
            self.renderer.kill()
            self.renderer.waitForFinished(2000)

    def _renderer_state_changed(self, state):
        if state == QtCore.QProcess.Running:
            self.close_display_button.setEnabled(True)
            if self.renderer_ready_path.exists():
                self.render_button.setText("Update Display")
                self.render_button.setEnabled(True)
            else:
                self.render_button.setText("Opening Display…")
                self.render_button.setEnabled(False)
        elif state == QtCore.QProcess.NotRunning:
            self.close_display_button.setEnabled(False)
            self.render_button.setText("Open Display")
            self.render_button.setEnabled(True)

    def _capture_renderer_output(self):
        stdout = bytes(self.renderer.readAllStandardOutput()).decode("utf-8", errors="replace")
        stderr = bytes(self.renderer.readAllStandardError()).decode("utf-8", errors="replace")
        if stdout:
            self._renderer_stdout = (self._renderer_stdout + stdout)[-12000:]
        if stderr:
            self._renderer_stderr = (self._renderer_stderr + stderr)[-12000:]

    def _check_renderer_ready(self):
        if self.renderer.state() == QtCore.QProcess.NotRunning:
            self._renderer_ready_timer.stop()
            return
        if self.renderer_ready_path.exists():
            self._renderer_ready_timer.stop()
            self.render_button.setText("Update Display")
            self.render_button.setEnabled(True)
            self._show_status("Display opened.", timeout=2500)

    def _renderer_error(self, error):
        if error != QtCore.QProcess.UnknownError:
            self._show_status(f"Display error: {self.renderer.errorString()}", timeout=8000)

    def _renderer_finished(self, exit_code, exit_status):
        self._capture_renderer_output()
        self._renderer_ready_timer.stop()
        self._pending_live_keys.clear()
        self._live_update_timer.stop()
        opened = self.renderer_ready_path.exists()
        crashed = exit_status == QtCore.QProcess.CrashExit or int(exit_code) != 0
        if self.renderer.state() == QtCore.QProcess.NotRunning:
            self.render_button.setText("Open Display")
            self.render_button.setEnabled(True)
        if self._renderer_close_requested:
            self._renderer_close_requested = False
            self._show_status("Display window closed")
            return
        if opened and not crashed:
            self._show_status("Display window closed")
            return

        details = self._renderer_stderr.strip() or self._renderer_stdout.strip()
        if not details:
            details = "The display process ended before its window became ready."
        summary = (
            f"The display could not be opened (exit code {int(exit_code)}).\n\n"
            f"{details[-5000:]}"
        )
        self._show_status("Display could not be opened.", timeout=10000)
        QtWidgets.QMessageBox.critical(self, "Display failed", summary)

    def _restore_ui_state(self):
        settings = QtCore.QSettings("WeldCraft", "P5_Lattice_Visualizer")
        geometry = settings.value("toolbox_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        advanced = settings.value("advanced_options", False)
        if isinstance(advanced, str):
            advanced = advanced.lower() in {"1", "true", "yes"}
        self.advanced_toggle.blockSignals(True)
        self.advanced_toggle.setChecked(bool(advanced))
        self.advanced_toggle.blockSignals(False)
        self._toggle_advanced(bool(advanced))
        self.tabs.setCurrentIndex(int(settings.value("selected_tab", 0)) if advanced else 0)

    def closeEvent(self, event):
        settings = QtCore.QSettings("WeldCraft", "P5_Lattice_Visualizer")
        settings.setValue("toolbox_geometry", self.saveGeometry())
        settings.setValue("selected_tab", self.tabs.currentIndex())
        settings.setValue("advanced_options", self.advanced_toggle.isChecked())
        self._close_renderer(wait=True)
        try:
            self.control_path.unlink()
        except OSError:
            pass
        try:
            self.renderer_ready_path.unlink()
        except OSError:
            pass
        event.accept()


def main():
    if "--smoke-test-renderer" in sys.argv[1:]:
        return smoke_test_renderer_launch()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("WeldCraft")
    window = Toolbox()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
