"""Three-grid convergence study for WeldCraft HDF5 results.

This is deliberately a standalone, power-user tool. It does not run simulations
or modify their configuration. Give it three completed HDF5 files in
coarse-to-fine order and it will:

* read grid spacing and provenance from each file's ``/meta`` group;
* warn when the runs differ in ways other than grid/output settings;
* select a small, representative set of common physical times (20 by default);
* interpolate all three runs to those exact times;
* conservatively average medium/fine cells into the same coarse control volumes
  (or optionally compare interpolated point samples);
* exclude air and interpolation stencils crossing material/air boundaries;
* calculate material-only L2 (grid RMS) and Linf differences;
* report an observed order for uniform grid-refinement ratios; and
* write a compact CSV, JSON summary, and optional figures.

Only the slices needed for the selected times are read. The entire simulations
are never loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b3_Functions import in_results


TIME_TOLERANCE_SECONDS = 1e-9
DEFAULT_SAMPLE_COUNT = 20
DEFAULT_OUTPUT_PARENT = "05_Convergence Analysis"


@dataclass(frozen=True)
class FieldDefinition:
    command_name: str
    display_name: str
    dataset_prefix: str
    unit: str


FIELD_DEFINITIONS = {
    "temperature": FieldDefinition("temperature", "Temperature", "u_snapshot_", "degC"),
    "hydrogen": FieldDefinition("hydrogen", "Hydrogen", "h_snapshot_", "%"),
}
TIME_DATASET_PREFIX = "t_snapshot_"
MATERIAL_DATASET_PREFIX = "d_snapshot_"
MATERIAL_FIELD = FieldDefinition("material", "Material", MATERIAL_DATASET_PREFIX, "")


# These may legitimately differ in a grid study without changing the physical
# problem. Everything else is compared and reported to the user.
IGNORED_METADATA_KEYS = {
    "animation_frame_stride",
    "animation_name",
    "coef_robin_x_air",
    "coef_robin_x_cu",
    "coef_robin_x_h2",
    "coef_robin_y_air",
    "coef_robin_y_cu",
    "coef_robin_y_h2",
    "debug_bead_plots",
    "dim_columns",
    "dim_rows",
    "dt",
    "dt_big",
    "dt_big_calc",
    "dx",
    "dx2",
    "dy",
    "dy2",
    "file_name",
    "inv_dx2",
    "inv_dy2",
    "safety_factor",
    "s_per_frame_just_diffusion_sparse",
    "s_per_frame_part1",
    "use_sparse_saving_in_just_diffusion",
}


@dataclass
class SimulationRun:
    path: Path
    label: str
    config: dict[str, Any]
    dx_mm: float
    dy_mm: float
    snapshot_indices: np.ndarray
    times_seconds: np.ndarray
    field_shape_y_x: tuple[int, int]
    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray
    flip_x: bool
    flip_y: bool


@dataclass(frozen=True)
class SpatialMapping:
    ix_left: np.ndarray
    ix_right: np.ndarray
    iy_lower: np.ndarray
    iy_upper: np.ndarray
    weight_x: np.ndarray
    weight_y: np.ndarray


@dataclass
class FieldResults:
    field: FieldDefinition
    times_seconds: np.ndarray
    l2_coarse_medium: np.ndarray
    l2_medium_fine: np.ndarray
    linf_coarse_medium: np.ndarray
    linf_medium_fine: np.ndarray
    rms_l2_coarse_medium: float
    rms_l2_medium_fine: float
    rms_linf_coarse_medium: float
    rms_linf_medium_fine: float
    rms_reference_fine: float
    observed_order_l2: float | None
    observed_order_linf: float | None
    compared_cell_counts: np.ndarray


class StudyMessages:
    """Collect warnings so they appear both on screen and in the JSON report."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING: {message}")


def discover_indices(h5_file: h5py.File, prefix: str) -> set[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    found: set[int] = set()
    for name in h5_file.keys():
        match = pattern.match(name)
        if match:
            found.add(int(match.group(1)))
    return found


def read_embedded_config(h5_file: h5py.File, source_path: Path) -> dict[str, Any]:
    meta = h5_file.get("/meta")
    if meta is None:
        raise RuntimeError(f"{source_path}: missing /meta group")
    raw_config = meta.attrs.get("param_config_json")
    if raw_config is None:
        raise RuntimeError(f"{source_path}: missing /meta param_config_json attribute")
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")
    config = json.loads(str(raw_config))
    if "dx" not in config or "dy" not in config:
        raise RuntimeError(f"{source_path}: embedded configuration has no dx/dy values")
    return config


def normalize_coordinates(coordinates: np.ndarray, axis_name: str, source_path: Path) -> tuple[np.ndarray, bool]:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(f"{source_path}: {axis_name} coordinates must be a one-dimensional array with >=2 values")
    differences = np.diff(coordinates)
    if np.all(differences > 0):
        return coordinates, False
    if np.all(differences < 0):
        return coordinates[::-1].copy(), True
    raise ValueError(f"{source_path}: {axis_name} coordinates are not strictly monotone")


def load_run_information(path_text: str, requested_fields: Sequence[FieldDefinition]) -> SimulationRun:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Simulation file not found: {path}")

    with h5py.File(path, "r") as h5_file:
        config = read_embedded_config(h5_file, path)
        dx_mm = float(config["dx"])
        dy_mm = float(config["dy"])
        if dx_mm <= 0 or dy_mm <= 0:
            raise ValueError(f"{path}: dx and dy must be positive")

        common_indices = discover_indices(h5_file, TIME_DATASET_PREFIX)
        common_indices &= discover_indices(h5_file, MATERIAL_DATASET_PREFIX)
        for field in requested_fields:
            common_indices &= discover_indices(h5_file, field.dataset_prefix)
        if not common_indices:
            names = ", ".join(field.dataset_prefix for field in requested_fields)
            raise RuntimeError(
                f"{path}: no snapshots common to time, material mask, and requested fields ({names})"
            )

        index_time_pairs = [
            (index, float(np.asarray(h5_file[f"{TIME_DATASET_PREFIX}{index:05d}"])))
            for index in sorted(common_indices)
        ]
        index_time_pairs.sort(key=lambda pair: pair[1])
        indices = np.asarray([pair[0] for pair in index_time_pairs], dtype=int)
        times = np.asarray([pair[1] for pair in index_time_pairs], dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{path}: snapshot times must be finite and strictly increasing")

        first_index = int(indices[0])
        sample = np.asarray(h5_file[f"{requested_fields[0].dataset_prefix}{first_index:05d}"])
        if sample.ndim != 2:
            raise ValueError(f"{path}: field snapshots must be two-dimensional")
        ny, nx = sample.shape
        for field in requested_fields[1:]:
            other_shape = h5_file[f"{field.dataset_prefix}{first_index:05d}"].shape
            if other_shape != sample.shape:
                raise ValueError(f"{path}: requested fields do not share one grid shape")

        if "/x" in h5_file and "/y" in h5_file:
            raw_x = np.asarray(h5_file["/x"])
            raw_y = np.asarray(h5_file["/y"])
        else:
            raw_x = (np.arange(nx) + 0.5) * dx_mm
            raw_y = (np.arange(ny) + 0.5) * dy_mm
        x_coordinates, flip_x = normalize_coordinates(raw_x, "x", path)
        y_coordinates, flip_y = normalize_coordinates(raw_y, "y", path)
        if x_coordinates.size != nx or y_coordinates.size != ny:
            raise ValueError(f"{path}: coordinate lengths do not match field shape {sample.shape}")

    label = f"dx={dx_mm:g}, dy={dy_mm:g} mm"
    return SimulationRun(
        path=path,
        label=label,
        config=config,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        snapshot_indices=indices,
        times_seconds=times,
        field_shape_y_x=(ny, nx),
        x_coordinates_mm=x_coordinates,
        y_coordinates_mm=y_coordinates,
        flip_x=flip_x,
        flip_y=flip_y,
    )


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in IGNORED_METADATA_KEYS}


def metadata_differences(reference: SimulationRun, candidate: SimulationRun) -> list[str]:
    reference_config = comparable_config(reference.config)
    candidate_config = comparable_config(candidate.config)
    differences: list[str] = []
    for key in sorted(set(reference_config) | set(candidate_config)):
        reference_value = reference_config.get(key, "<missing>")
        candidate_value = candidate_config.get(key, "<missing>")
        if reference_value != candidate_value:
            differences.append(f"{key}: {reference_value!r} != {candidate_value!r}")
    return differences


def validate_runs(
    runs: Sequence[SimulationRun],
    requested_fields: Sequence[FieldDefinition],
    messages: StudyMessages,
    strict_metadata: bool,
) -> float | None:
    coarse, medium, fine = runs
    if not (coarse.dx_mm > medium.dx_mm > fine.dx_mm):
        raise ValueError("Files must be supplied coarse-to-fine with strictly decreasing dx")
    if not (coarse.dy_mm > medium.dy_mm > fine.dy_mm):
        raise ValueError("Files must be supplied coarse-to-fine with strictly decreasing dy")

    for run in runs:
        ny, nx = run.field_shape_y_x
        print(
            f"  {run.path.name}: {run.label}; shape={ny}x{nx}; "
            f"snapshots={run.times_seconds.size}; time={run.times_seconds[0]:g}..{run.times_seconds[-1]:g} s"
        )

    reference_width = coarse.field_shape_y_x[1] * coarse.dx_mm
    reference_height = coarse.field_shape_y_x[0] * coarse.dy_mm
    domain_mismatches: list[str] = []
    for run in runs[1:]:
        width = run.field_shape_y_x[1] * run.dx_mm
        height = run.field_shape_y_x[0] * run.dy_mm
        if not math.isclose(width, reference_width, rel_tol=1e-6, abs_tol=1e-9):
            domain_mismatches.append(f"x extent {reference_width:g} vs {width:g} mm")
        if not math.isclose(height, reference_height, rel_tol=1e-6, abs_tol=1e-9):
            domain_mismatches.append(f"y extent {reference_height:g} vs {height:g} mm")
    if domain_mismatches:
        message = "Grid domains differ: " + "; ".join(domain_mismatches)
        if strict_metadata:
            raise ValueError(message)
        messages.warn(message)

    for run in runs[1:]:
        differences = metadata_differences(coarse, run)
        if differences:
            preview = "; ".join(differences[:8])
            if len(differences) > 8:
                preview += f"; ... and {len(differences) - 8} more"
            message = f"Physical/numerical metadata differs for {run.path.name}: {preview}"
            if strict_metadata:
                raise ValueError(message)
            messages.warn(message)

    if any(field.command_name == "hydrogen" for field in requested_fields):
        calibration_runs = [run.path.name for run in runs if run.config.get("thermal_diffusion_calibration") is True]
        if calibration_runs:
            messages.warn(
                "Hydrogen was requested, but thermal_diffusion_calibration=True in: "
                + ", ".join(calibration_runs)
            )

    ratios = (
        coarse.dx_mm / medium.dx_mm,
        medium.dx_mm / fine.dx_mm,
        coarse.dy_mm / medium.dy_mm,
        medium.dy_mm / fine.dy_mm,
    )
    if not all(math.isclose(ratio, ratios[0], rel_tol=1e-6, abs_tol=1e-9) for ratio in ratios[1:]):
        messages.warn(
            "Refinement ratios are not uniform in x/y; pairwise errors will be reported, "
            "but observed order will be left blank"
        )
        return None
    return ratios[0]


def common_time_interval(runs: Sequence[SimulationRun], messages: StudyMessages, strict_metadata: bool) -> tuple[float, float]:
    common_start = max(run.times_seconds[0] for run in runs)
    common_end = min(run.times_seconds[-1] for run in runs)
    if common_end <= common_start:
        raise ValueError("The three simulations have no overlapping time interval")

    starts = [run.times_seconds[0] for run in runs]
    ends = [run.times_seconds[-1] for run in runs]
    if max(starts) - min(starts) > TIME_TOLERANCE_SECONDS or max(ends) - min(ends) > TIME_TOLERANCE_SECONDS:
        message = (
            "Simulation time coverage differs; analysis is restricted to the common interval "
            f"{common_start:g}..{common_end:g} s"
        )
        if strict_metadata:
            raise ValueError(message)
        messages.warn(message)
    return common_start, common_end


def phase_anchor_times(config: dict[str, Any], common_start: float, common_end: float) -> list[float]:
    anchors = [common_start, common_end]
    for key in ("total_time_to_first_weld", "total_time_to_cooling", "total_time_to_rt"):
        value = config.get(key)
        if isinstance(value, (int, float)) and common_start <= float(value) <= common_end:
            anchors.append(float(value))
    return sorted(set(anchors))


def select_representative_times(
    reference_run: SimulationRun,
    common_start: float,
    common_end: float,
    requested_count: int,
) -> np.ndarray:
    available = reference_run.times_seconds
    available = available[(available >= common_start) & (available <= common_end)]
    if available.size == 0:
        raise ValueError("No coarse-grid snapshots exist inside the common time interval")
    anchors = phase_anchor_times(reference_run.config, common_start, common_end)
    if len(anchors) >= requested_count:
        selected_indices = np.linspace(0, len(anchors) - 1, requested_count).round().astype(int)
        return np.asarray([anchors[index] for index in selected_indices], dtype=float)

    selected = list(anchors)
    target_count = min(requested_count, available.size + len(anchors))
    # Work in snapshot-rank space rather than physical-time space. This respects
    # intentional dense saving around welding while phase anchors ensure that
    # cooling and room-temperature diffusion are not lost.
    selected_ranks = [float(np.searchsorted(available, value, side="left")) for value in selected]
    candidate_ranks = np.arange(available.size, dtype=float)
    while len(selected) < target_count:
        distances = np.min(np.abs(candidate_ranks[:, None] - np.asarray(selected_ranks)[None, :]), axis=1)
        best_rank = int(np.argmax(distances))
        value = float(available[best_rank])
        if any(math.isclose(value, old, rel_tol=0.0, abs_tol=TIME_TOLERANCE_SECONDS) for old in selected):
            distances[best_rank] = -1.0
            best_rank = int(np.argmax(distances))
            value = float(available[best_rank])
        selected.append(value)
        selected_ranks.append(float(best_rank))
    return np.asarray(sorted(selected), dtype=float)


def read_raw_slice(h5_file: h5py.File, run: SimulationRun, field: FieldDefinition, position: int) -> np.ndarray:
    index = int(run.snapshot_indices[position])
    values = np.asarray(h5_file[f"{field.dataset_prefix}{index:05d}"], dtype=float)
    if run.flip_y:
        values = values[::-1, :]
    if run.flip_x:
        values = values[:, ::-1]
    return values


def read_field_at_time(
    h5_file: h5py.File,
    run: SimulationRun,
    field: FieldDefinition,
    target_time_seconds: float,
) -> np.ndarray:
    times = run.times_seconds
    if target_time_seconds < times[0] - TIME_TOLERANCE_SECONDS or target_time_seconds > times[-1] + TIME_TOLERANCE_SECONDS:
        raise ValueError(f"{run.path}: target time {target_time_seconds:g} s is outside available data")

    right = int(np.searchsorted(times, target_time_seconds, side="left"))
    if right < times.size and math.isclose(times[right], target_time_seconds, rel_tol=0.0, abs_tol=TIME_TOLERANCE_SECONDS):
        return read_raw_slice(h5_file, run, field, right)
    if right == 0:
        return read_raw_slice(h5_file, run, field, 0)
    if right == times.size:
        return read_raw_slice(h5_file, run, field, times.size - 1)

    left = right - 1
    left_time = times[left]
    right_time = times[right]
    weight = (target_time_seconds - left_time) / (right_time - left_time)
    left_values = read_raw_slice(h5_file, run, field, left)
    right_values = read_raw_slice(h5_file, run, field, right)
    return (1.0 - weight) * left_values + weight * right_values


def build_spatial_mapping(fine: SimulationRun, coarse: SimulationRun) -> SpatialMapping:
    fine_x = fine.x_coordinates_mm
    fine_y = fine.y_coordinates_mm
    coarse_x = coarse.x_coordinates_mm
    coarse_y = coarse.y_coordinates_mm

    ix_left_1d = np.clip(np.searchsorted(fine_x, coarse_x, side="right") - 1, 0, fine_x.size - 2)
    iy_lower_1d = np.clip(np.searchsorted(fine_y, coarse_y, side="right") - 1, 0, fine_y.size - 2)
    ix_right_1d = ix_left_1d + 1
    iy_upper_1d = iy_lower_1d + 1

    x1 = fine_x[ix_left_1d]
    x2 = fine_x[ix_right_1d]
    y1 = fine_y[iy_lower_1d]
    y2 = fine_y[iy_upper_1d]
    weight_x_1d = np.divide(coarse_x - x1, x2 - x1, out=np.zeros_like(coarse_x), where=x2 != x1)
    weight_y_1d = np.divide(coarse_y - y1, y2 - y1, out=np.zeros_like(coarse_y), where=y2 != y1)

    ix_left, iy_lower = np.meshgrid(ix_left_1d, iy_lower_1d)
    ix_right, iy_upper = np.meshgrid(ix_right_1d, iy_upper_1d)
    weight_x, weight_y = np.meshgrid(weight_x_1d, weight_y_1d)
    return SpatialMapping(ix_left, ix_right, iy_lower, iy_upper, weight_x, weight_y)


def map_fine_to_coarse(fine_values: np.ndarray, mapping: SpatialMapping) -> np.ndarray:
    q11 = fine_values[mapping.iy_lower, mapping.ix_left]
    q21 = fine_values[mapping.iy_lower, mapping.ix_right]
    q12 = fine_values[mapping.iy_upper, mapping.ix_left]
    q22 = fine_values[mapping.iy_upper, mapping.ix_right]
    return (
        (1.0 - mapping.weight_y) * ((1.0 - mapping.weight_x) * q11 + mapping.weight_x * q21)
        + mapping.weight_y * ((1.0 - mapping.weight_x) * q12 + mapping.weight_x * q22)
    )


def integer_grid_ratio(fine: SimulationRun, coarse: SimulationRun) -> tuple[int, int]:
    fine_ny, fine_nx = fine.field_shape_y_x
    coarse_ny, coarse_nx = coarse.field_shape_y_x
    if fine_nx % coarse_nx or fine_ny % coarse_ny:
        raise ValueError(
            f"Cell-average restriction requires nested integer grid shapes, got "
            f"{fine.field_shape_y_x} and {coarse.field_shape_y_x}"
        )
    ratio_x = fine_nx // coarse_nx
    ratio_y = fine_ny // coarse_ny
    if ratio_x < 1 or ratio_y < 1:
        raise ValueError("Cell-average restriction requires the source grid to be at least as fine")
    return ratio_y, ratio_x


def restrict_cell_averages(
    fine_values: np.ndarray,
    fine: SimulationRun,
    coarse: SimulationRun,
) -> np.ndarray:
    """Area-average nested fine cells into each coarse control volume."""
    ratio_y, ratio_x = integer_grid_ratio(fine, coarse)
    coarse_ny, coarse_nx = coarse.field_shape_y_x
    expected_shape = (coarse_ny * ratio_y, coarse_nx * ratio_x)
    if fine_values.shape != expected_shape:
        raise ValueError(f"Expected fine field shape {expected_shape}, got {fine_values.shape}")
    return fine_values.reshape(coarse_ny, ratio_y, coarse_nx, ratio_x).mean(axis=(1, 3))


def restriction_blocks_are_material(
    material_values: np.ndarray,
    fine: SimulationRun,
    coarse: SimulationRun,
) -> np.ndarray:
    """Require every fine cell in a coarse control volume to be material."""
    ratio_y, ratio_x = integer_grid_ratio(fine, coarse)
    coarse_ny, coarse_nx = coarse.field_shape_y_x
    blocks = material_values.reshape(coarse_ny, ratio_y, coarse_nx, ratio_x)
    return np.all(blocks > 0.0, axis=(1, 3))


def interpolation_stencil_is_material(material_values: np.ndarray, mapping: SpatialMapping) -> np.ndarray:
    """Return where every fine-grid donor used by bilinear interpolation is material."""
    return (
        (material_values[mapping.iy_lower, mapping.ix_left] > 0.0)
        & (material_values[mapping.iy_lower, mapping.ix_right] > 0.0)
        & (material_values[mapping.iy_upper, mapping.ix_left] > 0.0)
        & (material_values[mapping.iy_upper, mapping.ix_right] > 0.0)
    )


def error_metrics(reference: np.ndarray, comparison: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    if not np.any(valid):
        raise ValueError("No common material cells remain for comparison")
    difference = reference[valid] - comparison[valid]
    return float(np.sqrt(np.mean(difference * difference))), float(np.max(np.abs(difference)))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def observed_order(error_coarse_medium: float, error_medium_fine: float, refinement_ratio: float | None) -> float | None:
    if refinement_ratio is None or error_coarse_medium <= 0 or error_medium_fine <= 0:
        return None
    return float(math.log(error_coarse_medium / error_medium_fine) / math.log(refinement_ratio))


def analyze_field(
    field: FieldDefinition,
    runs: Sequence[SimulationRun],
    h5_files: Sequence[h5py.File],
    selected_times: np.ndarray,
    refinement_ratio: float | None,
    spatial_method: str,
) -> FieldResults:
    coarse, medium, fine = runs
    mapping_coarse_medium = None
    mapping_coarse_fine = None
    if spatial_method == "point-interpolate":
        mapping_coarse_medium = build_spatial_mapping(medium, coarse)
        mapping_coarse_fine = build_spatial_mapping(fine, coarse)

    l2_coarse_medium: list[float] = []
    l2_medium_fine: list[float] = []
    linf_coarse_medium: list[float] = []
    linf_medium_fine: list[float] = []
    reference_fine_rms: list[float] = []
    compared_cell_counts: list[int] = []

    for target_time in selected_times:
        coarse_values = read_field_at_time(h5_files[0], coarse, field, float(target_time))
        medium_values = read_field_at_time(h5_files[1], medium, field, float(target_time))
        fine_values = read_field_at_time(h5_files[2], fine, field, float(target_time))
        coarse_material = read_field_at_time(h5_files[0], coarse, MATERIAL_FIELD, float(target_time))
        medium_material = read_field_at_time(h5_files[1], medium, MATERIAL_FIELD, float(target_time))
        fine_material = read_field_at_time(h5_files[2], fine, MATERIAL_FIELD, float(target_time))

        if spatial_method == "cell-average":
            medium_on_coarse = restrict_cell_averages(medium_values, medium, coarse)
            fine_on_coarse = restrict_cell_averages(fine_values, fine, coarse)
            valid = (
                (coarse_material > 0.0)
                & restriction_blocks_are_material(medium_material, medium, coarse)
                & restriction_blocks_are_material(fine_material, fine, coarse)
            )
        elif spatial_method == "point-interpolate":
            assert mapping_coarse_medium is not None and mapping_coarse_fine is not None
            medium_on_coarse = map_fine_to_coarse(medium_values, mapping_coarse_medium)
            fine_on_coarse = map_fine_to_coarse(fine_values, mapping_coarse_fine)
            valid = (
                (coarse_material > 0.0)
                & interpolation_stencil_is_material(medium_material, mapping_coarse_medium)
                & interpolation_stencil_is_material(fine_material, mapping_coarse_fine)
            )
        else:
            raise ValueError(f"Unknown spatial comparison method: {spatial_method}")
        l2_cm, linf_cm = error_metrics(coarse_values, medium_on_coarse, valid)
        l2_mf, linf_mf = error_metrics(medium_on_coarse, fine_on_coarse, valid)
        l2_coarse_medium.append(l2_cm)
        l2_medium_fine.append(l2_mf)
        linf_coarse_medium.append(linf_cm)
        linf_medium_fine.append(linf_mf)
        reference_fine_rms.append(float(np.sqrt(np.mean(fine_on_coarse[valid] ** 2))))
        compared_cell_counts.append(int(np.count_nonzero(valid)))

    l2_cm_array = np.asarray(l2_coarse_medium)
    l2_mf_array = np.asarray(l2_medium_fine)
    linf_cm_array = np.asarray(linf_coarse_medium)
    linf_mf_array = np.asarray(linf_medium_fine)
    rms_l2_cm = rms(l2_cm_array)
    rms_l2_mf = rms(l2_mf_array)
    rms_linf_cm = rms(linf_cm_array)
    rms_linf_mf = rms(linf_mf_array)
    rms_reference_fine = rms(np.asarray(reference_fine_rms))
    return FieldResults(
        field=field,
        times_seconds=selected_times,
        l2_coarse_medium=l2_cm_array,
        l2_medium_fine=l2_mf_array,
        linf_coarse_medium=linf_cm_array,
        linf_medium_fine=linf_mf_array,
        rms_l2_coarse_medium=rms_l2_cm,
        rms_l2_medium_fine=rms_l2_mf,
        rms_linf_coarse_medium=rms_linf_cm,
        rms_linf_medium_fine=rms_linf_mf,
        rms_reference_fine=rms_reference_fine,
        observed_order_l2=observed_order(rms_l2_cm, rms_l2_mf, refinement_ratio),
        observed_order_linf=observed_order(rms_linf_cm, rms_linf_mf, refinement_ratio),
        compared_cell_counts=np.asarray(compared_cell_counts, dtype=int),
    )


def optional_number(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.8g}"


def write_csv(output_path: Path, results: Sequence[FieldResults], runs: Sequence[SimulationRun]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "field",
                "metric",
                "sample",
                "time_seconds",
                f"{runs[0].label} vs {runs[1].label}",
                f"{runs[1].label} vs {runs[2].label}",
                "summary_value",
            ]
        )
        for field_result in results:
            for metric_name, first_values, second_values in (
                ("L2", field_result.l2_coarse_medium, field_result.l2_medium_fine),
                ("Linf", field_result.linf_coarse_medium, field_result.linf_medium_fine),
            ):
                for sample_number, (time_value, first_value, second_value) in enumerate(
                    zip(field_result.times_seconds, first_values, second_values), start=1
                ):
                    writer.writerow(
                        [
                            field_result.field.display_name,
                            metric_name,
                            sample_number,
                            f"{time_value:.9g}",
                            f"{first_value:.9g}",
                            f"{second_value:.9g}",
                            "",
                        ]
                    )
            summary_rows = (
                ("RMS_L2", field_result.rms_l2_coarse_medium, field_result.rms_l2_medium_fine, None),
                ("RMS_Linf", field_result.rms_linf_coarse_medium, field_result.rms_linf_medium_fine, None),
                ("ObservedOrder_L2", None, None, field_result.observed_order_l2),
                ("ObservedOrder_Linf", None, None, field_result.observed_order_linf),
            )
            for metric_name, first_value, second_value, summary_value in summary_rows:
                writer.writerow(
                    [
                        field_result.field.display_name,
                        metric_name,
                        "",
                        "",
                        optional_number(first_value),
                        optional_number(second_value),
                        optional_number(summary_value),
                    ]
                )


def result_as_dict(result: FieldResults) -> dict[str, Any]:
    reduction_l2 = None
    if result.rms_l2_coarse_medium > 0:
        reduction_l2 = 100.0 * (1.0 - result.rms_l2_medium_fine / result.rms_l2_coarse_medium)
    relative_l2_coarse_medium = None
    relative_l2_medium_fine = None
    if result.rms_reference_fine > 0:
        relative_l2_coarse_medium = 100.0 * result.rms_l2_coarse_medium / result.rms_reference_fine
        relative_l2_medium_fine = 100.0 * result.rms_l2_medium_fine / result.rms_reference_fine
    return {
        "unit": result.field.unit,
        "rms_l2_coarse_medium": result.rms_l2_coarse_medium,
        "rms_l2_medium_fine": result.rms_l2_medium_fine,
        "rms_linf_coarse_medium": result.rms_linf_coarse_medium,
        "rms_linf_medium_fine": result.rms_linf_medium_fine,
        "rms_reference_fine": result.rms_reference_fine,
        "relative_rms_l2_coarse_medium_percent": relative_l2_coarse_medium,
        "relative_rms_l2_medium_fine_percent": relative_l2_medium_fine,
        "observed_order_l2": result.observed_order_l2,
        "observed_order_linf": result.observed_order_linf,
        "l2_error_reduction_percent": reduction_l2,
        "compared_cell_count_min": int(np.min(result.compared_cell_counts)),
        "compared_cell_count_max": int(np.max(result.compared_cell_counts)),
    }


def write_json_report(
    output_path: Path,
    runs: Sequence[SimulationRun],
    selected_times: np.ndarray,
    refinement_ratio: float | None,
    results: Sequence[FieldResults],
    messages: StudyMessages,
    spatial_method: str,
    selection_method: str,
) -> None:
    report = {
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "inputs": [
            {
                "path": str(run.path),
                "label": run.label,
                "dx_mm": run.dx_mm,
                "dy_mm": run.dy_mm,
                "shape_y_x": list(run.field_shape_y_x),
                "snapshot_count": int(run.times_seconds.size),
                "time_start_seconds": float(run.times_seconds[0]),
                "time_end_seconds": float(run.times_seconds[-1]),
                "model_version": run.config.get("model_version"),
                "simulation_type": run.config.get("simulation_type"),
                "dt_seconds": run.config.get("dt"),
                "dt_big_seconds": run.config.get("dt_big"),
                "safety_factor": run.config.get("safety_factor"),
            }
            for run in runs
        ],
        "refinement_ratio": refinement_ratio,
        "selected_times_seconds": selected_times.tolist(),
        "selection_method": selection_method,
        "spatial_method": spatial_method,
        "spatial_comparison": (
            "medium and fine cells area-averaged into identical coarse control volumes; "
            "only volumes made entirely of material in all three runs"
            if spatial_method == "cell-average"
            else "medium and fine bilinearly interpolated to identical coarse-grid cell centres; "
            "only centres whose interpolation stencils are material in all three runs"
        ),
        "aggregation": "material-only grid RMS, then unweighted RMS across representative common times",
        "error_interpretation": (
            "Pairwise values are absolute adjacent-grid discrepancies in the field unit, not errors against "
            "an exact solution. Relative values divide that discrepancy by the restricted fine-grid RMS."
        ),
        "warnings": messages.warnings,
        "results": {result.field.display_name: result_as_dict(result) for result in results},
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_plots(output_directory: Path, results: Sequence[FieldResults], runs: Sequence[SimulationRun]) -> None:
    for field_result in results:
        time_scale = 3600.0 if field_result.times_seconds[-1] >= 7200 else 1.0
        time_unit = "h" if time_scale == 3600.0 else "s"
        plot_times = field_result.times_seconds / time_scale
        for metric_name, first_values, second_values in (
            ("L2", field_result.l2_coarse_medium, field_result.l2_medium_fine),
            ("Linf", field_result.linf_coarse_medium, field_result.linf_medium_fine),
        ):
            figure, axis = plt.subplots()
            axis.plot(plot_times, first_values, "o-", label=f"{runs[0].label} vs {runs[1].label}")
            axis.plot(plot_times, second_values, "o-", label=f"{runs[1].label} vs {runs[2].label}")
            axis.set_xlabel(f"Time [{time_unit}]")
            axis.set_ylabel(f"{metric_name} difference [{field_result.field.unit}]")
            axis.set_title(f"{field_result.field.display_name} grid comparison ({metric_name})")
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.savefig(
                output_directory / f"errors_{metric_name}_{field_result.field.command_name}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(figure)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare three WeldCraft HDF5 simulations in coarse-to-fine order.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("coarse_h5", help="coarsest-grid HDF5 file")
    parser.add_argument("medium_h5", help="middle-grid HDF5 file")
    parser.add_argument("fine_h5", help="finest-grid HDF5 file")
    parser.add_argument(
        "--snapshots",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="number of representative common times to compare",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=sorted(FIELD_DEFINITIONS),
        default=["temperature", "hydrogen"],
        help="fields to compare",
    )
    parser.add_argument(
        "--output-dir",
        help="output directory; default is a timestamped folder in 02_Results/05_Convergence Analysis",
    )
    parser.add_argument("--time-start", type=float, help="first physical time to include [s]")
    parser.add_argument("--time-end", type=float, help="last physical time to include [s]")
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        help="explicit physical comparison time(s); permits a single-time study",
    )
    parser.add_argument(
        "--exact-saved-times",
        action="store_true",
        help="require every explicit --times value to be a saved timestamp in every input file",
    )
    parser.add_argument(
        "--spatial-method",
        choices=("cell-average", "point-interpolate"),
        default="cell-average",
        help="compare equal coarse control volumes or interpolated values at coarse cell centres",
    )
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="fail instead of warning when non-grid metadata or time coverage differs",
    )
    parser.add_argument("--no-plots", action="store_true", help="write CSV/JSON only")
    parser.add_argument("--overwrite", action="store_true", help="replace report files in an existing output directory")
    return parser


def resolve_output_directory(output_text: str | None, overwrite: bool = False) -> Path:
    if output_text:
        output_directory = Path(output_text).expanduser().resolve()
        if output_directory.exists():
            if not output_directory.is_dir():
                raise ValueError(f"Output path is not a directory: {output_directory}")
            expected_outputs = ("comparison_summary.csv", "convergence_summary.json")
            collisions = [name for name in expected_outputs if (output_directory / name).exists()]
            if collisions and not overwrite:
                raise FileExistsError(
                    f"Output directory already contains convergence results: {', '.join(collisions)}"
                )
            return output_directory
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = in_results(DEFAULT_OUTPUT_PARENT, timestamp)
    output_directory.mkdir(parents=True, exist_ok=False)
    return output_directory


def run_study(arguments: argparse.Namespace) -> Path:
    if arguments.times is None and arguments.snapshots < 3:
        raise ValueError("--snapshots must be at least 3")
    requested_fields = [FIELD_DEFINITIONS[name] for name in arguments.fields]
    messages = StudyMessages()

    print("Reading embedded run metadata:")
    runs = [
        load_run_information(arguments.coarse_h5, requested_fields),
        load_run_information(arguments.medium_h5, requested_fields),
        load_run_information(arguments.fine_h5, requested_fields),
    ]
    refinement_ratio = validate_runs(runs, requested_fields, messages, arguments.strict_metadata)
    common_start, common_end = common_time_interval(runs, messages, arguments.strict_metadata)
    if arguments.time_start is not None:
        common_start = max(common_start, arguments.time_start)
    if arguments.time_end is not None:
        common_end = min(common_end, arguments.time_end)
    if common_end <= common_start:
        raise ValueError("Requested comparison window does not overlap the common simulation interval")
    if arguments.times is not None:
        selected_times = np.asarray(sorted(set(arguments.times)), dtype=float)
        if selected_times.size == 0 or not np.isfinite(selected_times).all():
            raise ValueError("--times must contain at least one finite value")
        if selected_times[0] < common_start or selected_times[-1] > common_end:
            raise ValueError(f"Explicit comparison times must lie inside {common_start:g}..{common_end:g} s")
        if arguments.exact_saved_times:
            for run in runs:
                for value in selected_times:
                    if not np.any(np.isclose(run.times_seconds, value, rtol=0.0, atol=TIME_TOLERANCE_SECONDS)):
                        raise ValueError(f"{run.path.name}: {value:g} s is not an exact saved timestamp")
        selection_method = "explicit user-supplied physical times"
    else:
        if arguments.exact_saved_times:
            raise ValueError("--exact-saved-times requires --times")
        selected_times = select_representative_times(runs[0], common_start, common_end, arguments.snapshots)
        selection_method = "phase anchors plus approximately uniform spacing over coarse-run saved-snapshot rank"
    print(f"Comparing {selected_times.size} representative times in {common_start:g}..{common_end:g} s")

    with ExitStack() as stack:
        h5_files = [stack.enter_context(h5py.File(run.path, "r")) for run in runs]
        results = [
            analyze_field(field, runs, h5_files, selected_times, refinement_ratio, arguments.spatial_method)
            for field in requested_fields
        ]

    output_directory = resolve_output_directory(arguments.output_dir, arguments.overwrite)
    write_csv(output_directory / "comparison_summary.csv", results, runs)
    write_json_report(
        output_directory / "convergence_summary.json",
        runs,
        selected_times,
        refinement_ratio,
        results,
        messages,
        arguments.spatial_method,
        selection_method,
    )
    if not arguments.no_plots:
        write_plots(output_directory, results, runs)

    print("\nSummary:")
    for result in results:
        p_l2 = "n/a" if result.observed_order_l2 is None else f"{result.observed_order_l2:.3f}"
        p_linf = "n/a" if result.observed_order_linf is None else f"{result.observed_order_linf:.3f}"
        relative_mf = 100.0 * result.rms_l2_medium_fine / result.rms_reference_fine
        print(
            f"  {result.field.display_name}: p(L2)={p_l2}, p(Linf)={p_linf}; "
            f"RMS L2 {result.rms_l2_coarse_medium:.6g} -> {result.rms_l2_medium_fine:.6g}; "
            f"medium-fine relative L2={relative_mf:.2f}%"
        )
    print(f"Results written to: {output_directory}")
    return output_directory


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        run_study(arguments)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
