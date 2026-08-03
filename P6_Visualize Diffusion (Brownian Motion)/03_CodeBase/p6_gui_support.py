"""P6 GUI configuration, execution, HDF5, and rendering services.

The graphical interface intentionally lives on top of the existing P6 model.
It updates only its managed values in ``b2_Brown_Config.py`` and executes the
simulation in a child process so Qt stays responsive while Numba is compiling
or running long kernels.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import math
import os
import pprint
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "02_Results"
CONFIG_PATH = CODE_DIR / "b2_Brown_Config.py"
SIMULATION_SCRIPT = CODE_DIR / "b1_Random_Motion.py"
DIAGRAM_PRESETS_DIR = PROJECT_DIR / "01_Resources" / "Diagram_Presets"
NEW_GUI_CHECKPOINT_SCHEMA = "brownian_exact_restart_v2"

STOP_FILE_ENV = "WELDCRAFT_P6_STOP_FILE"
GUI_RUN_ENV = "WELDCRAFT_P6_GUI_RUN"
SETTINGS_ENV = "WELDCRAFT_P6_SETTINGS_JSON"
PROGRESS_PREFIX = "P6_GUI_PROGRESS|"
CANCELLED_MARKER = "P6_GUI_CANCELLED"
DIAGRAM_RENDER_LOCK = threading.RLock()
CUSTOM_PRESET_MARKER = "Custom P6 diagram preset saved from the WeldCraft GUI."


class P6ConfigError(ValueError):
    pass


class P6SimulationCancelled(Exception):
    pass


FRACTION_FIELDS = (
    "max_sol_a",
    "max_sol_b",
    "max_sol_spot",
    "max_sol_trap_layer",
)

MANAGED_SIMPLE_FIELDS = (
    "x",
    "y",
    "steps",
    "save_every_steps",
    "concentration_a",
    "concentration_b",
    "USE_INITIAL_CONCENTRATION_PROFILE",
    "USE_SPOT",
    "SPOT_DIAMETER",
    "SPOT_CENTER_X",
    "SPOT_CENTER_Y",
    "concentration_spot",
    "USE_TRAP_LAYER",
    "TRAP_LAYER_CENTER_X",
    "TRAP_LAYER_WIDTH",
    "concentration_trap_layer",
    "USE_SINK_SOURCE",
    "SINK_SOURCE_THICKNESS",
    "SOURCE_SIDE",
    "num_subregions",
    "random_seed",
    "h5_filename",
    "SHOW_MAIN_SIMULATION_PANEL",
    "SHOW_CONCENTRATION_PROFILE_PANEL",
    "SHOW_NET_FLUX_PANEL",
    "MAIN_RENDER_MODE",
    "DOT_SIZE_AVAILABLE",
    "DOT_SIZE_HYDROGEN",
    "DOT_ALPHA_AVAILABLE",
    "DOT_ALPHA_HYDROGEN",
    "COLOR_EMPTY",
    "COLOR_AVAILABLE_SPOT",
    "COLOR_HYDROGEN",
    "COLOR_CONCENTRATION_LINE",
    "NET_FLUX_COLOR",
    "NET_FLUX_BAND_COLOR",
    "render_every_nth_frame",
    "animation_fps",
    "animation_filename",
    "GUI_DISABLE_OVERWRITE_WARNING",
    "GUI_FIGURE_FILENAME",
    "GUI_DIAGRAM_PRESET",
    "GUI_DIAGRAM_OVERRIDES",
    "RESUME_FROM_H5",
)

GUI_DEFAULTS = {
    "x": 1300,
    "y": 920,
    "steps": 3_000_000_000,
    "save_every_steps": 10_000_000,
    "max_sol_a": 10.0,
    "max_sol_b": 10.0,
    "concentration_a": 50.0,
    "concentration_b": 50.0,
    "USE_INITIAL_CONCENTRATION_PROFILE": False,
    "concentration_profile_a_left": 100.0,
    "concentration_profile_a_right": 50.0,
    "concentration_profile_b_left": 50.0,
    "concentration_profile_b_right": 0.0,
    "USE_SPOT": True,
    "SPOT_DIAMETER": 120,
    "SPOT_CENTER_X": 866,
    "SPOT_CENTER_Y": 460,
    "max_sol_spot": 100.0,
    "concentration_spot": 0.0,
    "affinity_a": 1.0,
    "mobility_a": 1.0,
    "affinity_b": 1.0,
    "mobility_b": 1.0,
    "affinity_spot": 2.0,
    "mobility_spot": 1.0,
    "USE_TRAP_LAYER": False,
    "TRAP_LAYER_CENTER_X": 325,
    "TRAP_LAYER_WIDTH": 40,
    "max_sol_trap_layer": 10.0,
    "concentration_trap_layer": 75.0,
    "affinity_trap_layer": 3.0,
    "mobility_trap_layer": 1.0,
    "USE_SINK_SOURCE": False,
    "SINK_SOURCE_THICKNESS": 10,
    "SOURCE_SIDE": "left",
    "num_subregions": 1,
    "random_seed": None,
    "h5_filename": "random_motion.h5",
    "SHOW_MAIN_SIMULATION_PANEL": True,
    "SHOW_CONCENTRATION_PROFILE_PANEL": True,
    "SHOW_NET_FLUX_PANEL": False,
    "MAIN_RENDER_MODE": "pixels",
    "DOT_SIZE_AVAILABLE": 12.0,
    "DOT_SIZE_HYDROGEN": 12.0,
    "DOT_ALPHA_AVAILABLE": 0.85,
    "DOT_ALPHA_HYDROGEN": 0.95,
    "COLOR_EMPTY": "#440154",
    "COLOR_AVAILABLE_SPOT": "#0000FF",
    "COLOR_HYDROGEN": "#FF0000",
    "COLOR_CONCENTRATION_LINE": "#0000FF",
    "NET_FLUX_COLOR": "#4A148C",
    "NET_FLUX_BAND_COLOR": "#B39DDB",
    "render_every_nth_frame": 5,
    "animation_fps": 12,
    "animation_filename": "brownian_motion_animation.mp4",
    "GUI_DISABLE_OVERWRITE_WARNING": False,
    "GUI_FIGURE_FILENAME": "brownian_diagram.png",
    "GUI_DIAGRAM_PRESET": "default",
    "GUI_DIAGRAM_OVERRIDES": {},
    "RESUME_FROM_H5": None,
}


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise P6ConfigError(f"Could not load configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise P6ConfigError(f"Could not import {path.name}: {exc}") from exc
    return module


def _percent(value) -> float:
    if isinstance(value, Fraction):
        return float(value * 100)
    return float(value) * 100.0


def load_gui_settings() -> dict:
    cfg = _load_module(CONFIG_PATH, f"p6_gui_config_{time.time_ns()}")
    values = deepcopy(GUI_DEFAULTS)
    for name in MANAGED_SIMPLE_FIELDS:
        if hasattr(cfg, name):
            values[name] = deepcopy(getattr(cfg, name))
    for name in FRACTION_FIELDS:
        values[name] = _percent(getattr(cfg, name))

    profile_a = tuple(getattr(cfg, "concentration_profile_a"))
    profile_b = tuple(getattr(cfg, "concentration_profile_b"))
    values.update(
        concentration_profile_a_left=float(profile_a[0]),
        concentration_profile_a_right=float(profile_a[1]),
        concentration_profile_b_left=float(profile_b[0]),
        concentration_profile_b_right=float(profile_b[1]),
    )
    characteristics = getattr(cfg, "AREA_CHARACTERISTICS")
    for area in ("a", "b", "spot", "trap_layer"):
        values[f"affinity_{area}"] = float(characteristics[area]["affinity"])
        values[f"mobility_{area}"] = float(characteristics[area]["mobility"])
    return validate_gui_settings(values)


def validate_filename(value, extensions, label="filename") -> str:
    text = str(value).strip()
    if (
        not text
        or Path(text).name != text
        or any(char in text for char in ("/", "\\", ":"))
        or text in (".", "..")
    ):
        raise P6ConfigError(f"{label} must be a plain filename without a folder")
    if not text.lower().endswith(tuple(ext.lower() for ext in extensions)):
        raise P6ConfigError(f"{label} must end in {' or '.join(extensions)}")
    return text


def validate_gui_settings(settings: dict) -> dict:
    s = deepcopy(settings)
    for name in ("x", "y", "SPOT_DIAMETER", "TRAP_LAYER_WIDTH", "SINK_SOURCE_THICKNESS", "num_subregions"):
        try:
            s[name] = parse_step_count(s[name])
        except (TypeError, ValueError) as exc:
            raise P6ConfigError(f"{name} must be a whole number") from exc
    if s["x"] < 10 or s["y"] < 10:
        raise P6ConfigError("Matrix width and height must each be at least 10")
    for name in ("steps", "save_every_steps"):
        s[name] = parse_step_count(s[name])
        if s[name] <= 0:
            raise P6ConfigError(f"{name} must be greater than zero")
    if s["SPOT_DIAMETER"] <= 0 or s["TRAP_LAYER_WIDTH"] <= 0:
        raise P6ConfigError("Spot diameter and trap width must be positive")
    for name in ("SPOT_CENTER_X", "SPOT_CENTER_Y", "TRAP_LAYER_CENTER_X"):
        s[name] = parse_step_count(s[name])
    if not 0 <= s["SPOT_CENTER_X"] < s["x"] or not 0 <= s["TRAP_LAYER_CENTER_X"] < s["x"]:
        raise P6ConfigError("Spot and trap X positions must be inside the matrix")
    if not 0 <= s["SPOT_CENTER_Y"] < s["y"]:
        raise P6ConfigError("Spot Y position must be inside the matrix")
    if s["SINK_SOURCE_THICKNESS"] <= 0 or s["SINK_SOURCE_THICKNESS"] * 2 >= s["x"]:
        raise P6ConfigError("Source/sink thickness must be positive and leave an interior matrix")
    if s["num_subregions"] < 1 or s["num_subregions"] > s["x"]:
        raise P6ConfigError("Flux-region count must be between 1 and the matrix width")
    if s["SOURCE_SIDE"] not in ("left", "right"):
        raise P6ConfigError("Source side must be left or right")

    percentages = list(FRACTION_FIELDS) + [
        "concentration_a", "concentration_b", "concentration_spot",
        "concentration_trap_layer", "concentration_profile_a_left",
        "concentration_profile_a_right", "concentration_profile_b_left",
        "concentration_profile_b_right",
    ]
    for name in percentages:
        s[name] = float(s[name])
        if not 0 <= s[name] <= 100:
            raise P6ConfigError(f"{name} must be between 0 and 100 percent")
    for area in ("a", "b", "spot", "trap_layer"):
        affinity = f"affinity_{area}"
        mobility = f"mobility_{area}"
        s[affinity] = float(s[affinity])
        s[mobility] = float(s[mobility])
        if s[affinity] <= 0:
            raise P6ConfigError(f"{affinity} must be greater than zero")
        if not 0 <= s[mobility] <= 1:
            raise P6ConfigError(f"{mobility} must be between 0 and 1")
    for name in ("DOT_ALPHA_AVAILABLE", "DOT_ALPHA_HYDROGEN"):
        s[name] = float(s[name])
        if not 0 <= s[name] <= 1:
            raise P6ConfigError(f"{name} must be between 0 and 1")
    for name in ("DOT_SIZE_AVAILABLE", "DOT_SIZE_HYDROGEN"):
        s[name] = float(s[name])
        if s[name] <= 0:
            raise P6ConfigError(f"{name} must be positive")
    s["render_every_nth_frame"] = parse_step_count(s["render_every_nth_frame"])
    s["animation_fps"] = parse_step_count(s["animation_fps"])
    if s["render_every_nth_frame"] < 1 or s["animation_fps"] < 1:
        raise P6ConfigError("Animation stride and FPS must be positive")
    if s["MAIN_RENDER_MODE"] not in ("pixels", "dots"):
        raise P6ConfigError("Animation render mode must be pixels or dots")
    if not any(s[name] for name in ("SHOW_MAIN_SIMULATION_PANEL", "SHOW_CONCENTRATION_PROFILE_PANEL", "SHOW_NET_FLUX_PANEL")):
        raise P6ConfigError("At least one animation panel must be enabled")
    seed = s.get("random_seed")
    if seed in (None, ""):
        s["random_seed"] = None
    else:
        s["random_seed"] = parse_step_count(seed)
        if not 0 <= s["random_seed"] < (1 << 64):
            raise P6ConfigError("Random seed must be between 0 and 2^64 - 1")
    s["h5_filename"] = validate_filename(s["h5_filename"], (".h5", ".hdf5"), "HDF5 filename")
    s["animation_filename"] = validate_filename(s["animation_filename"], (".mp4",), "MP4 filename")
    s["GUI_FIGURE_FILENAME"] = validate_filename(s["GUI_FIGURE_FILENAME"], (".png", ".pdf", ".svg"), "figure filename")
    if not isinstance(s.get("GUI_DIAGRAM_OVERRIDES", {}), dict):
        raise P6ConfigError("GUI diagram overrides must be a dictionary")
    available = discover_diagram_presets()
    if s["GUI_DIAGRAM_PRESET"] not in available:
        s["GUI_DIAGRAM_PRESET"] = "default"
    return s


def parse_step_count(value) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise P6ConfigError("Step counts must be whole numbers")
        return int(value)
    text = str(value).strip().lower().replace("_", "").replace(",", "")
    text = text.replace("×", "*").replace("^", "**").replace(" ", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\*?10\*\*([+-]?\d+)|e([+-]?\d+))?", text)
    if not match:
        raise P6ConfigError("Use a whole step count or scientific notation such as 3e9")
    try:
        coefficient = Decimal(match.group(1))
    except InvalidOperation as exc:
        raise P6ConfigError("Invalid scientific-notation value") from exc
    exponent_text = match.group(2) if match.group(2) is not None else match.group(3)
    result = coefficient * (Decimal(10) ** int(exponent_text)) if exponent_text is not None else coefficient
    if not result.is_finite() or result != result.to_integral_value():
        raise P6ConfigError("Step counts must resolve to whole numbers")
    return int(result)


def format_scientific_steps(value: int) -> str:
    value = int(value)
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10 ** exponent)
    return f"{coefficient:g} × 10^{exponent}"


def frame_summary(steps: int, save_every_steps: int, start_step: int = 0) -> dict:
    from b4_Brown_Checkpoint import build_saved_steps

    targets = build_saved_steps(int(start_step), int(steps), int(save_every_steps))
    return {
        "count": int(len(targets)),
        "first": int(targets[0]),
        "last": int(targets[-1]),
    }


def estimate_snapshot_bytes(settings: dict) -> int:
    summary = frame_summary(settings["steps"], settings["save_every_steps"])
    return int(settings["x"]) * int(settings["y"]) * summary["count"]


def _fraction_from_percent(value) -> Fraction:
    return Fraction(str(float(value))) / 100


def _config_values(settings: dict) -> dict:
    s = validate_gui_settings(settings)
    values = {name: deepcopy(s[name]) for name in MANAGED_SIMPLE_FIELDS if name in s}
    for name in FRACTION_FIELDS:
        values[name] = _fraction_from_percent(s[name])
    values["concentration_profile_a"] = (
        s["concentration_profile_a_left"], s["concentration_profile_a_right"]
    )
    values["concentration_profile_b"] = (
        s["concentration_profile_b_left"], s["concentration_profile_b_right"]
    )
    values["AREA_CHARACTERISTICS"] = {
        area: {"affinity": s[f"affinity_{area}"], "mobility": s[f"mobility_{area}"]}
        for area in ("a", "b", "spot", "trap_layer")
    }
    values["simulation_mode"] = "event_driven_wiggle"
    values["MATRIX_SOURCE"] = "random"
    return values


def _python_literal(value) -> str:
    if isinstance(value, Fraction):
        return f"Fraction({value.numerator}, {value.denominator})"
    return pprint.pformat(value, sort_dicts=False, width=100)


def write_gui_settings(settings: dict, config_path: Path = CONFIG_PATH) -> dict:
    checked = validate_gui_settings(settings)
    values = _config_values(checked)
    source = Path(config_path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise P6ConfigError(f"Cannot update invalid Python configuration: {exc}") from exc
    line_offsets = []
    offset = 0
    for line in source.splitlines(keepends=True):
        line_offsets.append(offset)
        offset += len(line)

    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node
    replacements = []
    missing = []
    for name, value in values.items():
        node = assignments.get(name)
        if node is None:
            missing.append((name, value))
            continue
        value_node = node.value
        start = line_offsets[value_node.lineno - 1] + value_node.col_offset
        end = line_offsets[value_node.end_lineno - 1] + value_node.end_col_offset
        replacements.append((start, end, _python_literal(value)))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    if missing:
        source = source.rstrip() + "\n\n# GUI-managed values\n"
        for name, value in missing:
            source += f"{name} = {_python_literal(value)}\n"
    try:
        compile(source, str(config_path), "exec")
    except SyntaxError as exc:
        raise P6ConfigError(f"Updated configuration is invalid Python: {exc}") from exc
    temporary = Path(config_path).with_suffix(".py.tmp")
    temporary.write_text(source, encoding="utf-8")
    temporary.replace(config_path)
    return checked


def restore_gui_defaults(current: dict | None = None) -> dict:
    values = load_gui_settings() if current is None else deepcopy(current)
    values.update(deepcopy(GUI_DEFAULTS))
    return write_gui_settings(values)


def discover_diagram_presets() -> list[str]:
    return sorted(
        path.stem for path in DIAGRAM_PRESETS_DIR.glob("*.py")
        if path.stem != "all_presets" and not path.stem.startswith("_")
    )


def save_diagram_preset(display_name: str, settings: dict, preset_directory=DIAGRAM_PRESETS_DIR) -> Path:
    """Atomically save resolved diagram settings as a new, never-overwritten preset."""
    name = str(display_name).strip()
    if not name:
        raise P6ConfigError("Preset name cannot be empty")
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not stem or stem == "all_presets" or stem.startswith("_"):
        raise P6ConfigError("Use a preset name containing letters or numbers")
    directory = Path(preset_directory)
    if not directory.is_dir():
        raise P6ConfigError(f"Preset directory does not exist: {directory}")
    destination = directory / f"{stem}.py"
    if destination.exists():
        raise P6ConfigError(
            f"Preset {destination.name} already exists; choose a different name"
        )

    resolved = deepcopy(settings)
    resolved["PRESET_NAME"] = name
    resolved.pop("BATCH_RENDER_ALL_PRESETS", None)
    if resolved.get("RENDER_MODE") not in (
        "pixels", "dots", "concentration_heatmap", "printer_glyphs", "area_summary_dots"
    ):
        raise P6ConfigError("Unsupported diagram render mode")
    if not any(
        bool(resolved.get(key, False))
        for key in (
            "SHOW_MAIN_PANEL",
            "SHOW_CONCENTRATION_PROFILE_PANEL",
            "SHOW_HEATMAP_PANEL",
            "SHOW_NET_FLUX_PANEL",
        )
    ):
        raise P6ConfigError("At least one diagram panel must remain enabled")

    def plain_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): plain_value(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(plain_value(item) for item in value)
        if isinstance(value, list):
            return [plain_value(item) for item in value]
        return value

    ordered_keys = ["PRESET_NAME", *sorted(key for key in resolved if key != "PRESET_NAME")]
    lines = [
        f'"""{CUSTOM_PRESET_MARKER}"""',
        "",
    ]
    for key in ordered_keys:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            continue
        lines.append(
            f"{key} = {pprint.pformat(plain_value(resolved[key]), sort_dicts=False, width=100)}"
        )
    source = "\n\n".join(lines) + "\n"
    try:
        compile(source, str(destination), "exec")
    except SyntaxError as exc:
        raise P6ConfigError(f"Generated preset is invalid: {exc}") from exc
    temporary = destination.with_suffix(".py.tmp")
    try:
        temporary.write_text(source, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def list_custom_diagram_presets(preset_directory=DIAGRAM_PRESETS_DIR) -> list[str]:
    """Return file stems for presets explicitly created by the P6 GUI."""
    directory = Path(preset_directory)
    if not directory.is_dir():
        return []
    custom = []
    for path in directory.glob("*.py"):
        if path.is_symlink():
            continue
        try:
            header = path.read_text(encoding="utf-8")[:512]
        except OSError:
            continue
        if CUSTOM_PRESET_MARKER in header:
            custom.append(path.stem)
    return sorted(custom)


def delete_custom_diagram_preset(preset_name: str, preset_directory=DIAGRAM_PRESETS_DIR) -> Path:
    """Permanently delete one GUI-created preset; shipped presets are protected."""
    stem = str(preset_name).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", stem):
        raise P6ConfigError("Invalid custom preset name")
    directory = Path(preset_directory).resolve()
    candidate = directory / f"{stem}.py"
    if candidate.is_symlink():
        raise P6ConfigError("Symbolic-link presets cannot be deleted in the GUI")
    path = candidate.resolve()
    if path.parent != directory or not path.is_file():
        raise P6ConfigError(f"Custom preset does not exist: {stem}")
    try:
        header = path.read_text(encoding="utf-8")[:512]
    except OSError as exc:
        raise P6ConfigError(f"Could not read preset {path.name}: {exc}") from exc
    if CUSTOM_PRESET_MARKER not in header:
        raise P6ConfigError("Shipped or manually maintained presets cannot be deleted in the GUI")
    path.unlink()
    return path


def result_path(filename: str) -> Path:
    return RESULTS_DIR / filename


def find_ffmpeg(configured_path=None) -> str | None:
    if configured_path:
        path = Path(str(configured_path))
        if path.is_file():
            return str(path)
    return shutil.which("ffmpeg")


def _reader_thread(stream, output_queue):
    try:
        for line in iter(stream.readline, ""):
            output_queue.put(line)
    finally:
        output_queue.put(None)


def parse_gui_progress_records(output_text):
    """Extract structured GUI records even when terminal redraw text precedes them."""
    records = []
    diagnostics = []
    cancelled = False
    for fragment in str(output_text).replace("\r", "\n").splitlines():
        text = fragment.strip()
        if not text:
            continue
        marker_index = text.find(PROGRESS_PREFIX)
        if marker_index >= 0:
            record = text[marker_index:]
            parts = record.split("|", 4)
            if len(parts) == 5:
                try:
                    records.append(
                        (float(parts[1]), parts[4], int(parts[2]), int(parts[3]))
                    )
                except ValueError:
                    diagnostics.append(record)
            else:
                diagnostics.append(record)
            text = text[:marker_index].strip()
        if CANCELLED_MARKER in text:
            cancelled = True
        if text:
            diagnostics.append(text)
    return records, diagnostics, cancelled


def run_brownian_simulation(settings, output_path, progress_callback=None, stop_event=None):
    """Run P6 through its stable script boundary while streaming GUI progress."""
    checked = validate_gui_settings(settings)
    output = Path(output_path).resolve()
    expected = result_path(checked["h5_filename"]).resolve()
    if output != expected:
        raise P6ConfigError(f"Output path must be {expected}")
    environment = os.environ.copy()
    environment[GUI_RUN_ENV] = "1"
    runtime_values = _config_values(checked)
    for name, value in list(runtime_values.items()):
        if isinstance(value, Fraction):
            runtime_values[name] = str(value)
    environment[SETTINGS_ENV] = json.dumps(runtime_values)
    descriptor, stop_path_text = tempfile.mkstemp(prefix="weldcraft_p6_stop_", suffix=".flag")
    os.close(descriptor)
    stop_path = Path(stop_path_text)
    stop_path.unlink(missing_ok=True)
    environment[STOP_FILE_ENV] = str(stop_path)
    command = [sys.executable, "-u", str(SIMULATION_SCRIPT)]
    process = subprocess.Popen(
        command,
        cwd=str(CODE_DIR),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines = queue.Queue()
    reader = threading.Thread(target=_reader_thread, args=(process.stdout, lines), daemon=True)
    reader.start()
    cancelled_requested = False
    cancelled_reported = False
    reader_done = False
    output_tail = deque(maxlen=20)
    try:
        while process.poll() is None or not reader_done:
            if stop_event is not None and stop_event.is_set() and not cancelled_requested:
                stop_path.touch()
                cancelled_requested = True
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                reader_done = True
                continue
            records, diagnostics, line_cancelled = parse_gui_progress_records(line)
            cancelled_reported = cancelled_reported or line_cancelled
            output_tail.extend(diagnostics)
            if progress_callback:
                for fraction, message, completed, frames in records:
                    progress_callback(fraction, message, completed, frames)
        return_code = process.wait()
    finally:
        stop_path.unlink(missing_ok=True)
        if process.poll() is None:
            process.terminate()
    if cancelled_requested or cancelled_reported or return_code == 2:
        raise P6SimulationCancelled(str(output))
    if return_code != 0:
        details = "\n".join(output_tail)
        suffix = f"\n\n{details}" if details else ""
        raise RuntimeError(f"P6 simulation exited with code {return_code}{suffix}")
    return output


class H5FrameSource:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as hf:
            if "snapshots" not in hf or "saved_steps" not in hf:
                raise RuntimeError("The file does not contain P6 snapshots and saved steps")
            allocated = int(hf["snapshots"].shape[0])
            recorded = int(hf.attrs.get("frames_written", allocated))
            self.frame_count = min(allocated, max(0, recorded))
            if self.frame_count < 1:
                raise RuntimeError("The file contains no committed P6 frames")
            self.shape = tuple(int(v) for v in hf["snapshots"].shape[1:])
            self.steps = hf["saved_steps"][: self.frame_count].astype(np.int64)
            self.status = str(hf.attrs.get("run_status", "legacy/unknown"))
            meta = hf.get("meta")
            raw = meta.attrs.get("brown_config_json") if meta is not None else None
            self.metadata = json.loads(raw) if raw is not None else {}

    def read_frame(self, index):
        index = min(max(int(index), 0), self.frame_count - 1)
        with h5py.File(self.path, "r") as hf:
            return hf["snapshots"][index].astype(np.int8), int(self.steps[index])


def inspect_resume_source(path) -> dict:
    source = H5FrameSource(path)
    result = {
        "path": source.path,
        "valid": False,
        "reason": "",
        "status": source.status,
        "frame_count": source.frame_count,
        "step": int(source.steps[-1]),
        "metadata": source.metadata,
    }
    try:
        with h5py.File(source.path, "r") as hf:
            if source.status not in ("complete", "cancelled"):
                raise RuntimeError(
                    f"The run status is {source.status!r}; only cleanly completed or cancelled runs can continue"
                )
            checkpoint = hf.get("checkpoint")
            if checkpoint is None:
                raise RuntimeError("No exact rolling checkpoint is present")
            if checkpoint.attrs.get("schema") != NEW_GUI_CHECKPOINT_SCHEMA:
                raise RuntimeError("This file predates the GUI rolling-checkpoint format")
            if not bool(checkpoint.attrs.get("complete", False)):
                raise RuntimeError("The checkpoint was not committed completely")
            index = int(checkpoint.attrs["snapshot_index"])
            step = int(checkpoint.attrs["step"])
            if index != source.frame_count - 1 or int(source.steps[index]) != step:
                raise RuntimeError("Checkpoint and last committed snapshot do not match")
            if str(checkpoint.attrs.get("simulation_mode", "")) != "event_driven_wiggle":
                raise RuntimeError("Only event-driven GUI runs can be continued")
            for name in ("rng_state", "ordered_hydrogen_site_ids", "event_fenwick_tree"):
                if name not in checkpoint:
                    raise RuntimeError(f"Checkpoint is missing {name}")
            for name in ("event_pending_wait_steps", "event_total_transition_weight"):
                if name not in checkpoint.attrs:
                    raise RuntimeError(f"Checkpoint is missing {name}")
            current_cfg = _load_module(CONFIG_PATH, f"p6_resume_config_{time.time_ns()}")
            for name in ("max_radius_to_jump", "base_movement_probability"):
                if name in source.metadata and source.metadata[name] != getattr(current_cfg, name):
                    raise RuntimeError(
                        f"Code-only dynamics setting {name} differs from the source file "
                        f"({getattr(current_cfg, name)!r} != {source.metadata[name]!r})"
                    )
            result["valid"] = True
            result["reason"] = "Exact GUI checkpoint verified"
    except Exception as exc:
        result["reason"] = str(exc)
    return result


def settings_from_resume_metadata(metadata: dict, current: dict) -> dict:
    values = deepcopy(current)
    direct = (
        "x", "y", "concentration_a", "concentration_b", "USE_INITIAL_CONCENTRATION_PROFILE",
        "USE_SPOT", "SPOT_DIAMETER", "SPOT_CENTER_X", "SPOT_CENTER_Y", "concentration_spot",
        "USE_TRAP_LAYER", "TRAP_LAYER_CENTER_X", "TRAP_LAYER_WIDTH", "concentration_trap_layer",
        "USE_SINK_SOURCE", "SINK_SOURCE_THICKNESS", "SOURCE_SIDE", "num_subregions",
    )
    for name in direct:
        if name in metadata:
            values[name] = metadata[name]
    for name in FRACTION_FIELDS:
        if name in metadata:
            raw = metadata[name]
            values[name] = float(Fraction(raw) * 100) if isinstance(raw, str) else float(raw) * 100
    for key, prefix in (("concentration_profile_a", "a"), ("concentration_profile_b", "b")):
        if key in metadata and len(metadata[key]) == 2:
            values[f"concentration_profile_{prefix}_left"] = metadata[key][0]
            values[f"concentration_profile_{prefix}_right"] = metadata[key][1]
    characteristics = metadata.get("AREA_CHARACTERISTICS", {})
    for area in ("a", "b", "spot", "trap_layer"):
        if area in characteristics:
            values[f"affinity_{area}"] = characteristics[area]["affinity"]
            values[f"mobility_{area}"] = characteristics[area]["mobility"]
    return validate_gui_settings(values)


def load_diagram_settings(preset_name: str, overrides=None) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        import c3_Brown_Make_Diagram as diagram
    settings = diagram.load_diagram_preset(preset_name)
    if settings.get("BATCH_RENDER_ALL_PRESETS"):
        raise P6ConfigError("The publication batch preset is not available in the GUI")
    for name, value in (overrides or {}).items():
        if name in settings:
            settings[name] = value
    if not any(
        bool(settings.get(name, False))
        for name in (
            "SHOW_MAIN_PANEL",
            "SHOW_CONCENTRATION_PROFILE_PANEL",
            "SHOW_HEATMAP_PANEL",
            "SHOW_NET_FLUX_PANEL",
        )
    ):
        raise P6ConfigError("At least one diagram panel must remain enabled")
    if settings.get("RENDER_MODE") not in (
        "pixels", "dots", "concentration_heatmap", "printer_glyphs", "area_summary_dots"
    ):
        raise P6ConfigError("Unsupported diagram render mode")
    for name in (
        "DOT_SIZE_AVAILABLE",
        "DOT_SIZE_HYDROGEN",
        "PROFILE_BIN_SIZE",
        "PROFILE_SMOOTHING_WINDOW",
        "HEATMAP_SIGMA",
        "GLYPH_BIN_SIZE",
        "AREA_SUMMARY_TOTAL_DOTS",
        "AREA_SUMMARY_CONCENTRATION_BIN_WIDTH",
        "AREA_SUMMARY_MIN_DOT_SPACING",
        "AREA_SUMMARY_CLUSTER_COUNT",
        "AREA_SUMMARY_DOT_SIZE",
    ):
        if name in settings and float(settings[name]) <= 0:
            raise P6ConfigError(f"{name} must be greater than zero")
    for name in ("DOT_ALPHA_AVAILABLE", "DOT_ALPHA_HYDROGEN", "AREA_SUMMARY_DOT_ALPHA"):
        if name in settings and not 0 <= float(settings[name]) <= 1:
            raise P6ConfigError(f"{name} must be between 0 and 1")
    for name in ("PROFILE_X_RANGE", "PROFILE_Y_RANGE", "HEATMAP_DEVIATION_RANGE", "HEATMAP_OCCUPANCY_RANGE"):
        value = settings.get(name)
        if value is not None and (len(value) != 2 or float(value[0]) >= float(value[1])):
            raise P6ConfigError(f"{name} must contain an increasing minimum and maximum")
    return settings


def render_diagram_figure(path, frame_index, preset_name, overrides=None):
    with DIAGRAM_RENDER_LOCK, contextlib.redirect_stdout(io.StringIO()):
        import c3_Brown_Make_Diagram as diagram
        preset = load_diagram_settings(preset_name, overrides)
        for name, value in preset.items():
            setattr(diagram, name, value)
        source = H5FrameSource(path)
        matrix, saved_step = source.read_frame(frame_index)
        transport = diagram.analyze_transport(source.path) if preset["SHOW_NET_FLUX_PANEL"] else None
        baseline = None
        if preset.get("HEATMAP_MODE") == "change_from_initial":
            baseline, _ = source.read_frame(int(preset.get("HEATMAP_BASELINE_SNAPSHOT_INDEX", 0)))
        shake_mode = preset.get("AREA_SUMMARY_SHAKE_MODE") if preset.get("RENDER_MODE") == "area_summary_dots" else None
        fig = diagram.create_figure(
            matrix,
            saved_step,
            int(frame_index),
            source.metadata,
            transport,
            area_summary_shake_mode=shake_mode,
            heatmap_baseline_matrix=baseline,
            managed=False,
        )
    return fig


def render_presentation_animation(path, output_path, preset_name, overrides, fps, stride, ffmpeg_path, progress_callback=None, stop_event=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    source = H5FrameSource(path)
    indices = list(range(0, source.frame_count, max(1, int(stride))))
    if indices[-1] != source.frame_count - 1:
        indices.append(source.frame_count - 1)
    output = Path(output_path)
    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    partial.unlink(missing_ok=True)
    mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)
    video_fig = None
    try:
        first = render_diagram_figure(source.path, indices[0], preset_name, overrides)
        first.canvas.draw()
        image = np.asarray(first.canvas.buffer_rgba()).copy()
        plt.close(first)
        height, width = image.shape[:2]
        video_fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        axis = video_fig.add_axes([0, 0, 1, 1])
        axis.set_axis_off()
        artist = axis.imshow(image)
        writer = FFMpegWriter(
            fps=int(fps), metadata={"artist": "WeldCraft"}, codec="libx264rgb",
            extra_args=["-crf", "0", "-preset", "slow", "-pix_fmt", "rgb24", "-movflags", "+faststart"],
        )
        with writer.saving(video_fig, str(partial), dpi=100):
            for position, index in enumerate(indices):
                if stop_event is not None and stop_event.is_set():
                    raise P6SimulationCancelled("Animation rendering cancelled")
                if position:
                    frame_fig = render_diagram_figure(source.path, index, preset_name, overrides)
                    frame_fig.canvas.draw()
                    frame_image = np.asarray(frame_fig.canvas.buffer_rgba()).copy()
                    plt.close(frame_fig)
                    if frame_image.shape != image.shape:
                        raise RuntimeError("Presentation frame dimensions changed during animation")
                    artist.set_data(frame_image)
                writer.grab_frame()
                if progress_callback:
                    progress_callback((position + 1) / len(indices), f"Rendering frame {position + 1}/{len(indices)}")
        partial.replace(output)
        return output
    finally:
        if video_fig is not None:
            plt.close(video_fig)
        if partial.exists():
            partial.unlink()
