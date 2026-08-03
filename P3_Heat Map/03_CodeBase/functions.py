"""Reusable P3 Heat Map configuration, solver, loading, and rendering code."""

from __future__ import annotations

import importlib.util
import json
import math
import pprint
import shutil
import threading
from copy import deepcopy
from pathlib import Path

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from numba import jit
from scipy.ndimage import binary_erosion


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "03_CodeBase"
RESOURCES_DIR = PROJECT_DIR / "01_Resources"
RESULTS_DIR = PROJECT_DIR / "02_Results"
CONFIG_PATH = CODE_DIR / "config.py"
DEFAULT_CONFIG_PATH = RESOURCES_DIR / "config_default.py"


class ConfigError(ValueError):
    """Raised when the user configuration is missing or invalid."""


class SimulationCancelled(Exception):
    """Raised internally when a worker requests a cooperative stop."""


def ensure_config_file() -> Path:
    """Create the persistent config from the shipped template when absent."""
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        shutil.copyfile(DEFAULT_CONFIG_PATH, CONFIG_PATH)
    return CONFIG_PATH


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"Could not load configuration file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_settings() -> dict:
    module = _load_module(DEFAULT_CONFIG_PATH, "p3_heat_map_config_default")
    return deepcopy(getattr(module, "SETTINGS"))


def validate_settings(settings: dict) -> dict:
    """Validate and normalize the complete settings mapping."""
    if not isinstance(settings, dict):
        raise ConfigError("SETTINGS must be a dictionary")

    required = set(default_settings())
    missing = sorted(required - set(settings))
    if missing:
        raise ConfigError("Missing configuration values: " + ", ".join(missing))

    numeric_positive = (
        "le", "ri", "we", "th", "su_h", "su_w", "dx", "dy", "weld_length",
        "weld_speed", "weld_spot_size", "sim_time", "save_so_often_per_sec",
        "c", "rho", "animation_fps", "animation_dpi", "animation_frame_stride",
    )
    for name in numeric_positive:
        try:
            value = float(settings[name])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be numeric") from exc
        if value <= 0:
            raise ConfigError(f"{name} must be greater than zero")

    for name in ("diff_coeff_bm", "diff_coeff_wm", "diff_coeff_haz", "diff_coeff_air"):
        try:
            if float(settings[name]) < 0:
                raise ConfigError(f"{name} cannot be negative")
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} must be numeric") from exc

    if float(settings["weld_bead_thickness"]) <= 0 or float(settings["weld_bead_thickness"]) > float(settings["we"]):
        raise ConfigError("weld_bead_thickness must be greater than zero and no wider than we")
    if float(settings["weld_spot_size"]) > float(settings["weld_length"]):
        raise ConfigError("weld_spot_size cannot exceed weld_length")
    if float(settings["heatmap_vmax"]) <= float(settings["heatmap_vmin"]):
        raise ConfigError("heatmap_vmax must be greater than heatmap_vmin")
    if not isinstance(settings["contour_levels"], list) or not settings["contour_levels"]:
        raise ConfigError("contour_levels must be a non-empty list")
    if not isinstance(settings["monitoring_distances"], list) or not settings["monitoring_distances"]:
        raise ConfigError("monitoring_distances must be a non-empty list")
    if any(float(value) < 0 for value in settings["monitoring_distances"]):
        raise ConfigError("monitoring_distances cannot contain negative values")

    for name in ("h5_filename", "animation_filename", "figure_filename"):
        value = str(settings[name]).strip()
        if not value or Path(value).name != value or any(char in value for char in ("/", "\\", ":")):
            raise ConfigError(f"{name} must be a filename only, without folders")
    if not str(settings["h5_filename"]).lower().endswith((".h5", ".hdf5")):
        raise ConfigError("h5_filename must end in .h5 or .hdf5")
    if not str(settings["animation_filename"]).lower().endswith(".mp4"):
        raise ConfigError("animation_filename must end in .mp4")
    if not str(settings["figure_filename"]).lower().endswith((".png", ".pdf", ".svg")):
        raise ConfigError("figure_filename must end in .png, .pdf, or .svg")
    return deepcopy(settings)


def load_settings() -> dict:
    ensure_config_file()
    try:
        module = _load_module(CONFIG_PATH, "p3_heat_map_user_config")
        # Add newly introduced settings to an older persistent config without
        # discarding the user's existing values. Invalid existing values still
        # fail loudly during validation below.
        merged_settings = default_settings()
        merged_settings.update(deepcopy(module.SETTINGS))
        return validate_settings(merged_settings)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Could not import {CONFIG_PATH.name}: {exc}") from exc


def get_value(name):
    """Return one current setting using the P2-style lookup API."""
    settings = load_settings()
    try:
        return settings[name]
    except KeyError as exc:
        raise ValueError(f"Parameter '{name}' not found in config") from exc


def serialize_settings(settings: dict) -> str:
    checked = validate_settings(settings)
    payload = pprint.pformat(checked, indent=4, sort_dicts=True, width=120)
    return (
        '"""Persistent user settings for P3 Heat Map. Generated by the GUI.\n'
        'Delete this file to restore the shipped defaults.\n"""\n\n'
        f"SETTINGS = {payload}\n\n"
        "def get_value(name):\n"
        "    try:\n"
        "        return SETTINGS[name]\n"
        "    except KeyError as exc:\n"
        '        raise ValueError(f"Parameter \'{name}\' not found in config") from exc\n'
    )


def write_settings(settings: dict) -> dict:
    """Validate and atomically persist settings to config.py."""
    checked = validate_settings(settings)
    temporary_path = CONFIG_PATH.with_suffix(".py.tmp")
    temporary_path.write_text(serialize_settings(checked), encoding="utf-8")
    temporary_path.replace(CONFIG_PATH)
    return checked


def restore_default_config() -> dict:
    settings = default_settings()
    return write_settings(settings)


def compute_derived(settings: dict) -> dict:
    s = validate_settings(settings)
    dx = float(s["dx"])
    dy = float(s["dy"])
    diff_max = max(float(s["diff_coeff_bm"]), float(s["diff_coeff_wm"]), float(s["diff_coeff_haz"]))
    dt = (dx * dx * dy * dy) / (2 * diff_max * (dx * dx + dy * dy))
    fr_le = float(s["le"]) - ((float(s["su_w"]) - float(s["we"])) / 2)
    fr_ri = float(s["ri"]) - ((float(s["su_w"]) - float(s["we"])) / 2)
    dim_rows = float(s["le"]) + float(s["ri"]) + float(s["we"])
    dim_columns = float(s["th"]) + (2 * float(s["su_h"])) + float(s["fr_ab"]) + float(s["fr_be"])
    last_weld_bead = float(s["le"]) + (float(s["we"]) / 2)
    weld_start = float(s["time_before_weld_start"])
    weld_end = weld_start + (float(s["weld_length"]) / float(s["weld_speed"])) * 60
    return {
        "dx2": dx * dx,
        "dy2": dy * dy,
        "dt": dt,
        "fr_le": fr_le,
        "fr_ri": fr_ri,
        "dim_rows": dim_rows,
        "dim_columns": dim_columns,
        "nx": int(dim_rows / dx),
        "ny": int(dim_columns / dy),
        "last_weld_bead": last_weld_bead,
        "weld_start_time": weld_start,
        "weld_end_time": weld_end,
    }


def preview_grid_info(array_shape, dx: float, dy: float, max_lines: int = 250) -> dict:
    """Describe the display-only grid used by the setup preview.

    The solver always uses the configured ``dx`` and ``dy``.  Large meshes can
    make a preview unreadable, so the preview may draw every nth solver cell.
    Keeping the stride in whole cells means the visible lines remain aligned
    with the actual solver mesh rather than being placed at unrelated
    positions.
    """
    ny, nx = (int(array_shape[0]), int(array_shape[1]))
    if max_lines < 1:
        raise ValueError("max_lines must be positive")

    stride = max(1, int(np.ceil(max(nx, ny) / max_lines)))
    limited = stride > 1
    display_dx = float(dx) * stride
    display_dy = float(dy) * stride
    return {
        "nx": nx,
        "ny": ny,
        "stride": stride,
        "display_dx": display_dx,
        "display_dy": display_dy,
        "limited": limited,
        "max_lines": max_lines,
        "message": (
            f"Display notice: The configured solver mesh is {float(dx):.3g} mm x "
            f"{float(dy):.3g} mm. To keep the preview readable, it shows one grid line "
            f"for every {stride} solver cells ({display_dx:.3g} mm x {display_dy:.3g} mm). "
            "This is display-only. The simulation still uses the configured mesh spacing."
        ) if limited else "",
    }


@jit(nopython=True, cache=True)
def compute_field_derivatives(field, dx2, dy2, field_dx2, field_dy2, ny, nx):
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            field_dx2[j, i] = (field[j, i + 1] - 2 * field[j, i] + field[j, i - 1]) / dx2
            field_dy2[j, i] = (field[j + 1, i] - 2 * field[j, i] + field[j - 1, i]) / dy2
    return field_dx2, field_dy2


@jit(nopython=True, cache=True)
def update_temperature(temp_map, previous, diffusion_matrix, dt, dudx2, dudy2, ny, nx):
    for j in range(ny):
        for i in range(nx):
            temp_map[j, i] = previous[j, i] + diffusion_matrix[j, i] * dt * (dudx2[j, i] + dudy2[j, i])
    return temp_map


@jit(nopython=True, cache=True)
def apply_convection(temp_map, room_temperature, convection, inner_mask, ny, nx):
    for j in range(ny):
        for i in range(nx):
            temp_map[j, i] -= (temp_map[j, i] - room_temperature) * convection * inner_mask[j, i]
    return temp_map


def apply_mask(diffusion_matrix):
    mask = diffusion_matrix > 0
    boundary = mask ^ binary_erosion(mask, structure=np.ones((3, 3)))
    boundary_indices = np.argwhere(boundary)
    inner_mask = mask & ~boundary
    inner_indices = np.argwhere(inner_mask)
    valid_neighbors = np.zeros_like(boundary_indices)
    for index, (row, col) in enumerate(boundary_indices):
        neighbours = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        candidates = [
            (r, c) for r, c in neighbours
            if 0 <= r < diffusion_matrix.shape[0]
            and 0 <= c < diffusion_matrix.shape[1]
            and mask[r, c] and not boundary[r, c]
        ]
        if not candidates:
            for radius in range(1, 4):
                candidates = [
                    (r, c)
                    for r in range(row - radius, row + radius + 1)
                    for c in range(col - radius, col + radius + 1)
                    if 0 <= r < diffusion_matrix.shape[0]
                    and 0 <= c < diffusion_matrix.shape[1]
                    and mask[r, c] and not boundary[r, c]
                ]
                if candidates:
                    break
        if candidates:
            valid_neighbors[index] = candidates[0]
        else:
            valid_neighbors[index] = (row, col)
    return inner_mask, inner_indices, boundary_indices, valid_neighbors


def build_initial_fields(settings: dict):
    s = validate_settings(settings)
    d = compute_derived(s)
    temp_map = float(s["t_cool"]) * np.ones((d["ny"], d["nx"]), dtype=np.float64)
    top_free = int(float(s["fr_ab"]) / float(s["dy"]))
    bottom_free = int(float(s["fr_be"]) / float(s["dy"]))
    left_free = int(d["fr_le"] / float(s["dx"]))
    right_free = int(d["fr_ri"] / float(s["dx"]))
    support_height = int((float(s["fr_ab"]) + float(s["su_h"])) / float(s["dy"]))
    bottom_support = int((float(s["fr_be"]) + float(s["su_h"])) / float(s["dy"]))
    room = float(s["t_room"])
    temp_map[:top_free, :] = room
    if bottom_free:
        temp_map[-bottom_free:, :] = room
    temp_map[:support_height, :left_free] = room
    temp_map[:support_height, -right_free:] = room
    temp_map[-bottom_support:, :left_free] = room
    temp_map[-bottom_support:, -right_free:] = room

    bead_left = int(d["last_weld_bead"] / float(s["dx"]))
    bead_right = int((d["last_weld_bead"] + float(s["weld_bead_thickness"])) / float(s["dx"]))
    temp_map[int(float(s["fr_ab"]) / float(s["dy"])):-int(float(s["fr_be"]) / float(s["dy"])), bead_left:bead_right] = float(s["t_cool"]) - 5

    diffusion = temp_map.copy()
    diffusion[temp_map == float(s["t_cool"])] = float(s["diff_coeff_bm"])
    diffusion[temp_map == room] = float(s["diff_coeff_air"])
    diffusion[int(float(s["fr_ab"]) / float(s["dy"])):-int(float(s["fr_be"]) / float(s["dy"])), bead_left:bead_right] = float(s["diff_coeff_wm"])
    return temp_map, diffusion, d


def _save_snapshot(handle, prefix, index, array, actual_time):
    handle.create_dataset(f"{prefix}_{index:05d}", data=array)
    handle.create_dataset(f"t_snapshot_{index:05d}", data=actual_time)


def run_simulation(settings: dict, output_path: Path | None = None, progress_callback=None, stop_event=None):
    """Run the thermal simulation and write compatible HDF5 snapshots."""
    s = validate_settings(settings)
    temp_map, diffusion, d = build_initial_fields(s)
    output = Path(output_path or RESULTS_DIR / s["h5_filename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    dx = float(s["dx"])
    dy = float(s["dy"])
    dt = d["dt"]
    ny, nx = d["ny"], d["nx"]
    inner_mask, _, boundary_indices, valid_neighbors = apply_mask(diffusion)
    first_indices = np.where(temp_map[:, 0] == float(s["t_cool"]))[0]
    last_indices = np.where(temp_map[:, -1] == float(s["t_cool"]))[0]
    boundary_first = np.column_stack((first_indices, np.zeros_like(first_indices)))
    boundary_last = np.column_stack((last_indices, (nx - 1) * np.ones_like(last_indices)))

    temp_previous = temp_map.copy()
    dudx2 = np.zeros_like(temp_map)
    dudy2 = np.zeros_like(temp_map)
    actual_time = 0.0
    save_counter = 0
    time_since_save = 0.0
    previous_end_row = -1
    total_steps = max(1, int(math.ceil(float(s["sim_time"]) / dt)))
    weld_start_col = int(d["last_weld_bead"] / dx)
    weld_end_col = int((d["last_weld_bead"] + float(s["weld_bead_thickness"])) / dx)
    mid_index = ny // 2
    base_start_row = mid_index + int((float(s["weld_length"]) / 2) / dy) - int((float(s["weld_spot_size"]) / 2) / dy)
    start_row = base_start_row
    end_row = int(start_row + float(s["weld_spot_size"]) / dy)
    initial_save_frequency = float(s["save_so_often_per_sec"])

    with h5py.File(output, "w") as handle:
        handle.attrs["p3_format_version"] = "1"
        handle.attrs["config_json"] = json.dumps(s)
        handle.attrs["convention"] = "rows: top->bottom, cols: left->right"
        _save_snapshot(handle, "temp_map", save_counter, temp_map, actual_time)
        save_counter += 1

        for step in range(total_steps):
            if stop_event is not None and stop_event.is_set():
                handle.attrs["status"] = "cancelled"
                raise SimulationCancelled

            actual_time += dt
            time_since_save += dt
            dudx2, dudy2 = compute_field_derivatives(temp_map, d["dx2"], d["dy2"], dudx2, dudy2, ny, nx)
            temp_map = update_temperature(temp_map, temp_previous, diffusion, dt, dudx2, dudy2, ny, nx)

            thickness = dx * 1e-3
            area = dx * dy * 1e-6
            volume = area * thickness
            convection = (float(s["conv_variable"]) * area * dt) / (float(s["c"]) * float(s["rho"]) * volume)
            temp_map = apply_convection(temp_map, float(s["t_room"]), convection, inner_mask, ny, nx)

            if d["weld_start_time"] <= actual_time <= d["weld_end_time"]:
                distance_moved = int((float(s["weld_speed"]) / 60) * (actual_time - d["weld_start_time"]) / dy)
                start_row = base_start_row - distance_moved
                end_row = int(start_row + float(s["weld_spot_size"]) / dy)
                clipped_start = max(0, start_row)
                clipped_end = min(ny, end_row)
                if previous_end_row != end_row and clipped_start < clipped_end:
                    temp_map[clipped_start:clipped_end, weld_start_col:weld_end_col] = float(s["weld_temp"])
                    diffusion[clipped_start:clipped_end, weld_start_col:weld_end_col] = float(s["diff_coeff_bm"])
                    if s["use_boundary_adjustment"]:
                        inner_mask, _, boundary_indices, valid_neighbors = apply_mask(diffusion)
                    previous_end_row = end_row

            if len(boundary_indices):
                temp_map[boundary_indices[:, 0], boundary_indices[:, 1]] = temp_previous[
                    valid_neighbors[:, 0], valid_neighbors[:, 1]
                ]
            if len(boundary_first):
                temp_map[boundary_first[:, 0], boundary_first[:, 1]] = temp_map[boundary_first[:, 0], boundary_first[:, 1] + 1]
            if len(boundary_last):
                temp_map[boundary_last[:, 0], boundary_last[:, 1]] = temp_map[boundary_last[:, 0], boundary_last[:, 1] - 1]

            if s["slow_down_beginning"] and actual_time <= 60:
                save_frequency = 4.0
            else:
                save_frequency = initial_save_frequency
            save_interval = 1.0 / save_frequency
            if time_since_save >= save_interval or step == total_steps - 1:
                _save_snapshot(handle, "temp_map", save_counter, temp_map, actual_time)
                save_counter += 1
                time_since_save = 0.0
            temp_previous = temp_map.copy()

            if progress_callback and (step % max(1, total_steps // 200) == 0 or step == total_steps - 1):
                progress_callback((step + 1) / total_steps, f"Simulating: {actual_time:.1f} s", actual_time)

        handle.attrs["status"] = "complete"
    return output


def load_snapshots(file_path: Path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)
    arrays, times = [], []
    with h5py.File(path, "r") as handle:
        array_keys = sorted(key for key in handle if key.startswith("temp_map_"))
        time_keys = sorted(key for key in handle if key.startswith("t_snapshot_"))
        arrays = [handle[key][:] for key in array_keys]
        times = [float(handle[key][()]) for key in time_keys]
        metadata = {key: handle.attrs[key] for key in handle.attrs}
    if not arrays or len(arrays) != len(times):
        raise ValueError("The HDF5 file does not contain matching temperature snapshots and times")
    return {"arrays": arrays, "times": np.asarray(times), "metadata": metadata}


def monitoring_positions(settings: dict, array_shape):
    s = validate_settings(settings)
    d = compute_derived(s)
    ny, nx = array_shape
    x = int((d["last_weld_bead"] + float(s["weld_bead_thickness"])) / float(s["dx"]))
    y = int((float(s["fr_ab"]) + float(s["su_h"]) + float(s["th"]) - float(s["monitoring_y_offset"])) / float(s["dy"]))
    positions = []
    for distance in s["monitoring_distances"]:
        col = int(x + float(distance) / float(s["dx"]))
        positions.append((min(max(y, 0), ny - 1), min(max(col, 0), nx - 1)))
    return positions


def moving_weld_zoom_bounds(settings: dict, actual_time: float, array_shape):
    """Return the moving weld-area crop used by the original P3 animation."""
    s = validate_settings(settings)
    d = compute_derived(s)
    dx = float(s["dx"])
    dy = float(s["dy"])
    ny, nx = array_shape
    elapsed = min(max(float(actual_time) - d["weld_start_time"], 0.0), d["weld_end_time"] - d["weld_start_time"])
    distance_moved = int((float(s["weld_speed"]) / 60.0) * elapsed / dy)
    mid_index = ny // 2
    start_row = mid_index + int((float(s["weld_length"]) / 2) / dy) - int((float(s["weld_spot_size"]) / 2) / dy) - distance_moved
    margin_x = max(1, int(float(s["weld_zoom_margin"]) / dx))
    margin_y = max(1, int(float(s["weld_zoom_margin"]) / (2 * dy)))
    r_start = start_row - margin_y
    r_end = r_start + max(2, int(2 * float(s["weld_zoom_margin"]) / dy))
    weld_left = int(d["last_weld_bead"] / dx)
    weld_right = int((d["last_weld_bead"] + float(s["weld_bead_thickness"])) / dx)
    c_start = weld_left - margin_x
    c_end = weld_right + margin_x
    r_start = min(max(r_start, 0), max(0, ny - 1))
    r_end = min(max(r_end, r_start + 1), ny)
    c_start = min(max(c_start, 0), max(0, nx - 1))
    c_end = min(max(c_end, c_start + 1), nx)
    return r_start, r_end, c_start, c_end


def create_result_view(settings: dict, loaded_data: dict, figure=None):
    """Create reusable artists for fast interactive frame browsing.

    This is deliberately separate from :func:`create_result_figure`, which is
    also used for standalone figure generation.  The interactive viewer keeps
    its axes, colorbar, traces, and labels alive and only replaces frame-sized
    data while the slider moves.
    """
    s = validate_settings(settings)
    arrays = loaded_data["arrays"]
    times = np.asarray(loaded_data["times"])
    if not arrays or len(arrays) != len(times):
        raise ValueError("The HDF5 data does not contain matching frames and times")

    field = arrays[0]
    positions = monitoring_positions(s, field.shape)
    fig = figure if figure is not None else plt.figure(figsize=(15, 8))
    fig.clear()
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 2.2, 1.5),
        height_ratios=(2.0, 1.0),
        hspace=0.35,
        wspace=0.35,
    )
    ax_zoom = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[:, 1])
    ax_trace = fig.add_subplot(grid[:, 2])
    ax_interpass = fig.add_subplot(grid[1, 0])

    dx = float(s["dx"])
    dy = float(s["dy"])
    extent = [-(dx / 2), (field.shape[1] - 0.5) * dx, (field.shape[0] - 0.5) * dy, -(dy / 2)]
    cmap = plt.get_cmap(str(s["heatmap_style"]))
    norm = mcolors.Normalize(vmin=float(s["heatmap_vmin"]), vmax=float(s["heatmap_vmax"]))
    heat_image = ax_heat.imshow(field, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal", extent=extent)
    fig.colorbar(heat_image, ax=ax_heat, label="Temperature [°C]")
    heat_title = ax_heat.set_title(f"Time: {times[0]:.0f} s")
    ax_heat.set_xlabel("X [mm]")
    ax_heat.set_ylabel("Y [mm]")

    contour_set = None
    if s["show_contours"]:
        x_values = np.arange(field.shape[1]) * dx
        y_values = np.arange(field.shape[0]) * dy
        contour_set = ax_heat.contour(x_values, y_values, field, levels=s["contour_levels"], colors="black", alpha=0.5)
        ax_heat.clabel(contour_set, inline=True, fontsize=7, fmt="%g°C")

    if s["show_monitoring_points"]:
        for index, (row, col) in enumerate(positions):
            x_coord, y_coord = col * dx, row * dy
            ax_heat.plot(x_coord, y_coord, "+", color="black", markersize=10)
            ax_heat.text(x_coord, y_coord, chr(65 + index), color="black")

    r0, r1, c0, c1 = moving_weld_zoom_bounds(s, times[0], field.shape)
    zoom_extent = [c0 * dx, c1 * dx, r1 * dy, r0 * dy]
    zoom_image = ax_zoom.imshow(
        field[r0:r1, c0:c1],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
        extent=zoom_extent,
    )
    ax_zoom.set_title("Weld Area")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    trace_lines = []
    trace_values = []
    for index, position in enumerate(positions):
        values = np.asarray([array[position[0], position[1]] for array in arrays])
        trace_values.append(values)
        distance = s["monitoring_distances"][index]
        (line,) = ax_trace.plot(times, values, label=f"{chr(65 + index)} ({distance:g} mm)")
        trace_lines.append(line)
    ax_trace.set_title("Temperature Monitoring")
    ax_trace.set_xlabel("Time [s]")
    ax_trace.set_ylabel("Temperature [°C]")
    ax_trace.grid(True)
    if trace_lines:
        ax_trace.legend()

    interpass_line = None
    interpass_dot = None
    if trace_values:
        (interpass_line,) = ax_interpass.plot(times[:1], trace_values[-1][:1])
        (interpass_dot,) = ax_interpass.plot([times[0]], [trace_values[-1][0]], "ko", markersize=5)
    ax_interpass.set_title("Point D / Last Monitoring Point")
    ax_interpass.set_xlabel("Time [s]")
    ax_interpass.set_ylabel("Temperature [°C]")
    ax_interpass.grid(True)
    if trace_values:
        ax_interpass.set_xlim(times[0], times[-1])
        value_min = float(np.min(trace_values[-1]))
        value_max = float(np.max(trace_values[-1]))
        value_margin = max(1.0, (value_max - value_min) * 0.05)
        ax_interpass.set_ylim(value_min - value_margin, value_max + value_margin)

    return {
        "settings": s,
        "figure": fig,
        "heat_image": heat_image,
        "heat_title": heat_title,
        "zoom_image": zoom_image,
        "ax_heat": ax_heat,
        "times": times,
        "arrays": arrays,
        "positions": positions,
        "dx": dx,
        "dy": dy,
        "trace_values": trace_values,
        "trace_lines": trace_lines,
        "interpass_line": interpass_line,
        "interpass_dot": interpass_dot,
        "contour_set": contour_set,
    }


def update_result_view(view: dict, frame: int):
    """Update an interactive result view without rebuilding its figure."""
    frame = min(max(int(frame), 0), len(view["arrays"]) - 1)
    field = view["arrays"][frame]
    view["heat_image"].set_data(field)
    view["heat_title"].set_text(f"Time: {view['times'][frame]:.0f} s")

    r0, r1, c0, c1 = moving_weld_zoom_bounds(view["settings"], view["times"][frame], field.shape)
    view["zoom_image"].set_data(field[r0:r1, c0:c1])
    view["zoom_image"].set_extent([c0 * view["dx"], c1 * view["dx"], r1 * view["dy"], r0 * view["dy"]])

    if view["contour_set"] is not None:
        view["contour_set"].remove()
        view["contour_set"] = None
    if view["settings"]["show_contours"]:
        x_values = np.arange(field.shape[1]) * view["dx"]
        y_values = np.arange(field.shape[0]) * view["dy"]
        contours = view["ax_heat"].contour(
            x_values,
            y_values,
            field,
            levels=view["settings"]["contour_levels"],
            colors="black",
            alpha=0.5,
        )
        view["ax_heat"].clabel(contours, inline=True, fontsize=7, fmt="%g°C")
        view["contour_set"] = contours

    if view["trace_values"]:
        values = view["trace_values"][-1]
        view["interpass_line"].set_data(view["times"][: frame + 1], values[: frame + 1])
        view["interpass_dot"].set_data([view["times"][frame]], [values[frame]])

    return frame


def create_result_figure(settings: dict, loaded_data: dict, frame: int = 0, figure=None):
    s = validate_settings(settings)
    arrays = loaded_data["arrays"]
    times = loaded_data["times"]
    frame = min(max(int(frame), 0), len(arrays) - 1)
    field = arrays[frame]
    d = compute_derived(s)
    positions = monitoring_positions(s, field.shape)
    fig = figure if figure is not None else plt.figure(figsize=(15, 8))
    fig.clear()
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 2.2, 1.5), height_ratios=(2.0, 1.0), hspace=0.35, wspace=0.35)
    ax_zoom = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[:, 1])
    ax_trace = fig.add_subplot(grid[:, 2])
    ax_interpass = fig.add_subplot(grid[1, 0])
    dx = float(s["dx"])
    dy = float(s["dy"])
    extent = [-(dx / 2), (field.shape[1] - 0.5) * dx, (field.shape[0] - 0.5) * dy, -(dy / 2)]
    cmap = plt.get_cmap(str(s["heatmap_style"]))
    norm = mcolors.Normalize(vmin=float(s["heatmap_vmin"]), vmax=float(s["heatmap_vmax"]))
    image = ax_heat.imshow(field, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal", extent=extent)
    fig.colorbar(image, ax=ax_heat, label="Temperature [°C]")
    ax_heat.set_title(f"Time: {times[frame]:.0f} s")
    ax_heat.set_xlabel("X [mm]")
    ax_heat.set_ylabel("Y [mm]")
    if s["show_contours"]:
        x_values = np.arange(field.shape[1]) * float(s["dx"])
        y_values = np.arange(field.shape[0]) * float(s["dy"])
        contours = ax_heat.contour(x_values, y_values, field, levels=s["contour_levels"], colors="black", alpha=0.5)
        ax_heat.clabel(contours, inline=True, fontsize=7, fmt="%g°C")
    if s["show_monitoring_points"]:
        for index, (row, col) in enumerate(positions):
            x_coord, y_coord = col * float(s["dx"]), row * float(s["dy"])
            ax_heat.plot(x_coord, y_coord, "+", color="black", markersize=10)
            ax_heat.text(x_coord, y_coord, chr(65 + index), color="black")

    r0, r1, c0, c1 = moving_weld_zoom_bounds(s, times[frame], field.shape)
    zoom_extent = [c0 * dx, c1 * dx, r1 * dy, r0 * dy]
    ax_zoom.imshow(
        field[r0:r1, c0:c1],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
        extent=zoom_extent,
    )
    ax_zoom.set_title("Weld Area")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    for index, position in enumerate(positions):
        values = [array[position[0], position[1]] for array in arrays]
        distance = s["monitoring_distances"][index]
        ax_trace.plot(times, values, label=f"{chr(65 + index)} ({distance:g} mm)")
    ax_trace.set_title("Temperature Monitoring")
    ax_trace.set_xlabel("Time [s]")
    ax_trace.set_ylabel("Temperature [°C]")
    ax_trace.grid(True)
    ax_trace.legend()
    if positions:
        values = [array[positions[-1][0], positions[-1][1]] for array in arrays]
        # Match the original P3 animation: only draw the trace up to the
        # selected time and place a black indicator dot at that time.
        ax_interpass.plot(times[: frame + 1], values[: frame + 1])
        ax_interpass.plot([times[frame]], [values[frame]], "ko", markersize=5)
    ax_interpass.set_title("Point D / Last Monitoring Point")
    ax_interpass.set_xlabel("Time [s]")
    ax_interpass.set_ylabel("Temperature [°C]")
    ax_interpass.grid(True)
    return fig


def render_animation(settings: dict, loaded_data: dict, output_path: Path, progress_callback=None):
    s = validate_settings(settings)
    arrays = loaded_data["arrays"][::max(1, int(s["animation_frame_stride"]))]
    times = loaded_data["times"][::max(1, int(s["animation_frame_stride"]))]
    if not arrays:
        raise ValueError("No frames are available for animation")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    norm = mcolors.Normalize(vmin=float(s["heatmap_vmin"]), vmax=float(s["heatmap_vmax"]))
    image = ax.imshow(arrays[0], cmap=plt.get_cmap(str(s["heatmap_style"])), norm=norm, interpolation="nearest", aspect="equal")
    fig.colorbar(image, ax=ax, label="Temperature [°C]")
    title = ax.set_title(f"Time: {times[0]:.0f} s")

    def update(index):
        image.set_data(arrays[index])
        title.set_text(f"Time: {times[index]:.0f} s")
        return image, title

    animation = FuncAnimation(fig, update, frames=len(arrays), repeat=False, blit=True)
    writer = FFMpegWriter(fps=int(s["animation_fps"]), metadata={"artist": "WeldCraft"})

    def callback(current, total):
        if progress_callback:
            progress_callback(current / max(total, 1), f"Rendering animation: {current}/{total}", current)

    animation.save(str(output_path), writer=writer, dpi=int(s["animation_dpi"]), progress_callback=callback)
    plt.close(fig)
    return output_path
