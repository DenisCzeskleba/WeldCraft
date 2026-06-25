"""
Weld & Hydrogen Diffusion — Grid Convergence / Consistency Checker
=================================================================

Purpose
-------
This script compares **three** simulation runs of a 2D weld + hydrogen diffusion
model that were executed with different spatial resolutions (e.g., dx = 1.0,
0.5, 0.25). The goal is to provide a simple, engineer-friendly way to:

1) Load snapshot data (temperature and hydrogen concentration) from HDF5 files.
2) Align snapshots across runs in time (either by nearest-neighbor or by optional
   linear interpolation in time on the finer run).
3) Map the finer solutions onto the coarser grid in space (bilinear interpolation),
   so fields are comparable on the same grid.
4) Compute **global error metrics** (L2 / Linf) over time between adjacent pairs
   of runs (coarse–medium, medium–fine).
5) Extract and compare **time series at a few engineer-relevant probe points**
   (e.g., weld centerline, HAZ, fusion boundary), again for adjacent pairs.
6) Report a concise CSV summary you can quickly inspect or plot elsewhere.

Design Philosophy
-----------------
- **Simplicity is king.** Descriptive variable names, explicit steps, and
  minimal abstractions. The code favors readability over cleverness.
- **Zero magic.** Functions do exactly one obvious thing. Most intermediate
  arrays are named verbosely.
- **Stable defaults.** By default the script keeps the CFL idea in mind but
  does not enforce it; it simply compares the outputs you already saved.

Assumptions About Your HDF5 Files
---------------------------------
- Snapshots are saved as **individual datasets** with numeric suffixes, e.g.:
    - 'u_snapshot_00042'   (temperature)
    - 'h_snapshot_00042'   (hydrogen concentration)
    - 't_snapshot_00042'   (scalar, simulation time for this snapshot in seconds)
    - 'd_snapshot_00042'   (optional field: diffusion coefficient map)
- The file **also** contains spatial coordinate arrays:
    - '/x' → shape (nx,)
    - '/y' → shape (ny,)
  If these are not present in your files, please add them. It makes life easier
  and avoids guessing grid spacing.

Time Alignment Options
----------------------
- **Nearest snapshot matching (default):** Good enough for engineering work.
  The script pairs each coarse snapshot time with the closest fine snapshot time.
- **Linear-in-time interpolation (optional):** If you want to be a bit more
  "mathy", set USE_TEMPORAL_INTERPOLATION=True and the script will interpolate
  the finer run in time onto the coarse snapshot times before doing spatial
  mapping.

Outputs
-------
- Prints observed (coarse–medium vs. medium–fine) error levels and a rough
  observed order via RMS-over-time.
- Writes a CSV 'comparison_summary.csv' containing per-snapshot errors for
  both fields (T and H) and differences at probe points over time.

How To Use
----------
1) Set the three HDF5 paths and their dx values in the CONFIG block.
2) Adjust PROBE_POINTS if needed.
3) Run:  python weld_diffusion_convergence_eval.py

"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional
import re
import math
import warnings
import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from b4_functions import in_results


@dataclass
class SimulationRunDescription:
    file_path: str
    label_for_plots_and_csv: str
    spatial_step_dx_mm: float
    spatial_step_dy_mm: float | None = None  # if None, assume same as dx


# ============================= CONFIGURATION =============================

# Automatically detect the directory where this script lives
DEFAULT_BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# All inputs and outputs for a comparison go into this subfolder next to your base dir.
# We'll create a timestamped subfolder inside it for each run, e.g. "250908_0806".
ANALYSIS_PARENT_FOLDER_NAME: str = r"05_Convergence Analysis"  # folder will be created if missing

# Output CSV filename (will live inside timestamped subfolder)
OUTPUT_CSV_FILENAME = "comparison_summary.csv"

# Subfolder for figures inside timestamped analysis folder
OUTPUT_FIGURES_SUBFOLDER_NAME = "figures"

# Save figures toggle
SAVE_FIGURES: bool = True

# Order: coarse → medium → fine
SIMULATION_RUNS_LIST: List[SimulationRunDescription] = [
    # Put ONLY filenames here; the script will resolve them inside the active analysis folder.
    # This keeps everything "in one spot" and avoids absolute paths.
    SimulationRunDescription(file_path="XXX.h5", label_for_plots_and_csv="dx=1.00", spatial_step_dx_mm=1.00),
    SimulationRunDescription(file_path="XXX.h5", label_for_plots_and_csv="dx=0.50", spatial_step_dx_mm=0.50),
    SimulationRunDescription(file_path="XXX.h5", label_for_plots_and_csv="dx=0.25", spatial_step_dx_mm=0.25),
]

# Dataset name prefixes exactly as used by your saver
DATASET_PREFIX_TEMPERATURE: str = "u_snapshot_"      # temperature field snapshots
DATASET_PREFIX_HYDROGEN: str    = "h_snapshot_"      # hydrogen concentration snapshots
DATASET_PREFIX_TIME: str        = "t_snapshot_"      # scalar time stamps
DATASET_PREFIX_DIFFUSIVITY: str = "d_snapshot_"      # optional, diffusion coefficient map

# Coordinate dataset names (1D arrays)
DATASET_X_COORDINATES: str = "/x"
DATASET_Y_COORDINATES: str = "/y"

# Engineer-style probe points (physical coordinates). Change as needed.
# Example: centerline, fusion line, HAZ, quarter-thickness, etc.
PROBE_POINTS: List[Tuple[str, Tuple[float, float]]] = [
    ("Centerline", (70, 20.0)),
    ("FusionLine", (12.5, 6.0)),
    ("HAZ_5mm",    (20.5, 6.0)),
    ("Mid_Bead",   (17.5, 5.0)),
]

# Choose which fields to compare (label, dataset_prefix)
FIELDS_TO_COMPARE: List[Tuple[str, str]] = [
    ("Temperature", DATASET_PREFIX_TEMPERATURE),
    ("Hydrogen",    DATASET_PREFIX_HYDROGEN),
]

# Time alignment behavior
USE_TEMPORAL_INTERPOLATION: bool = True   # False → nearest snapshot; True → linear in time on the finer run
MAX_TIME_MISMATCH_SECONDS: float = 1e2     # only used for sanity checks when nearest-matching (set large to ignore)
FLOAT_TIME_TOLERANCE: float = 1e-12  # small epsilon for time comparisons
# ============================= DATA CONTAINERS =============================

@dataclass
class SnapshotSeries:
    """Container for a time series of 2D fields and associated metadata.

    Attributes
    ----------
    times_seconds : (nt,) float array of snapshot times (seconds)
    field_stack   : (nt, ny, nx) array for the primary field (Temperature or Hydrogen)
    x_coordinates : (nx,) 1D array of x positions (mm), strictly increasing internally
    y_coordinates : (ny,) 1D array of y positions (mm), strictly increasing internally
    optional_diffusivity_stack : (nt, ny, nx) array or None (loaded but not used in metrics)
    """
    times_seconds: np.ndarray
    field_stack: np.ndarray
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    optional_diffusivity_stack: Optional[np.ndarray]

# ============================= PATH HELPERS =============================

def resolve_active_analysis_folder() -> str:
    """Create (if needed) the parent folder and a timestamped analysis subfolder.

    Example output path:
        <base>/Convergence Analysis/250908_0806
    """
    parent_folder_abspath = str(in_results(ANALYSIS_PARENT_FOLDER_NAME))
    os.makedirs(parent_folder_abspath, exist_ok=True)

    now = datetime.now()
    folder_name = now.strftime("%y%m%d_%H%M")  # YYMMDD_HHMM
    active_folder_abspath = os.path.join(parent_folder_abspath, folder_name)
    os.makedirs(active_folder_abspath, exist_ok=True)
    return active_folder_abspath

# ============================= LOADING HELPERS =============================

def discover_snapshot_indices(h5_handle: h5py.File, prefix: str) -> List[int]:
    """Return a sorted list of integer indices for datasets like 'prefix_00042'."""
    regex_pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    discovered_indices: List[int] = []
    for dataset_name in h5_handle.keys():
        match = regex_pattern.match(dataset_name)
        if match:
            discovered_indices.append(int(match.group(1)))
    discovered_indices.sort()
    return discovered_indices


# def _ensure_increasing_coordinates_and_align_stack(
#     coords: np.ndarray,
#     stack_time_y_x: np.ndarray,
#     axis: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """If coords are strictly decreasing, reverse both coords and corresponding data axis.
#
#     Parameters
#     ----------
#     coords : array of shape (n,)
#     stack_time_y_x : array of shape (nt, ny, nx)
#     axis : 1 for y, 2 for x (matching stack dimensions)
#     """
#     if coords.ndim != 1:
#         raise ValueError("Coordinate arrays must be 1D.")
#     if coords.size < 2:
#         return coords, stack_time_y_x
#
#     is_increasing = np.all(np.diff(coords) > 0)
#     is_decreasing = np.all(np.diff(coords) < 0)
#     if is_decreasing:
#         coords = coords[::-1].copy()
#         stack_time_y_x = np.flip(stack_time_y_x, axis=axis)
#     elif not is_increasing:
#         raise ValueError("Coordinate array must be strictly monotone (increasing or decreasing).")
#     return coords, stack_time_y_x


def read_coordinate_arrays_or_construct(
    h5_handle: h5py.File,
    run_desc: SimulationRunDescription,
    sample_field_slice_y_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read `/x` and `/y` if present; otherwise, construct from dx/dy in mm (cell centers).

    Returns `(x_coords, y_coords)`, both strictly increasing.
    """
    ny, nx = sample_field_slice_y_x.shape

    if DATASET_X_COORDINATES in h5_handle and DATASET_Y_COORDINATES in h5_handle:
        x_coords = np.asarray(h5_handle[DATASET_X_COORDINATES])
        y_coords = np.asarray(h5_handle[DATASET_Y_COORDINATES])
    else:
        dx_mm = run_desc.spatial_step_dx_mm
        dy_mm = run_desc.spatial_step_dy_mm if run_desc.spatial_step_dy_mm is not None else dx_mm
        x_coords = (np.arange(nx) + 0.5) * float(dx_mm)
        y_coords = (np.arange(ny) + 0.5) * float(dy_mm)

    return x_coords.astype(float), y_coords.astype(float)


def load_snapshot_series_for_field(
    simulation_run: SimulationRunDescription,
    dataset_prefix_main_field: str,
    analysis_parent_abspath: str,
) -> SnapshotSeries:
    """Load one field (Temperature or Hydrogen) as a time series from a run.

    This function assembles a (time, y, x) stack by scanning datasets with the
    required prefixes and sorting by their numeric suffixes. It also loads and
    returns the matching time stamps and coordinate arrays. Coordinates are
    normalized to be strictly increasing internally; data stacks are flipped
    accordingly if needed so spatial mapping is consistent.
    """
    # Resolve file path relative to the **parent** analysis folder (inputs live there)
    absolute_file_path = simulation_run.file_path
    if not os.path.isabs(absolute_file_path):
        absolute_file_path = os.path.join(analysis_parent_abspath, absolute_file_path)

    with h5py.File(absolute_file_path, "r") as h5f:
        # Discover indices where both the main field and time exist
        indices_main = set(discover_snapshot_indices(h5f, dataset_prefix_main_field))
        indices_time = set(discover_snapshot_indices(h5f, DATASET_PREFIX_TIME))
        common_indices_sorted = sorted(indices_main.intersection(indices_time))
        if len(common_indices_sorted) == 0:
            raise RuntimeError(
                f"No common snapshots (field '{dataset_prefix_main_field}' and time) found in {simulation_run.file_path}"
            )

        # Read one field slice to get shape (ny, nx) for coord handling
        first_idx = common_indices_sorted[0]
        sample_field = np.asarray(h5f[f"{dataset_prefix_main_field}{first_idx:05d}"])
        ny, nx = sample_field.shape

        # Read /x and /y, or build fallback from dx/dy (mm)
        x_coords, y_coords = read_coordinate_arrays_or_construct(h5f, simulation_run, sample_field)

        # Allocate stacks
        nt = len(common_indices_sorted)
        times_seconds = np.empty(nt, dtype=float)
        field_stack = np.empty((nt, ny, nx), dtype=sample_field.dtype)

        # Optional diffusivity stack (loaded but not used in metrics)
        has_any_diffusivity = any(
            f"{DATASET_PREFIX_DIFFUSIVITY}{idx:05d}" in h5f for idx in common_indices_sorted
        )
        optional_diffusivity_stack = None
        if has_any_diffusivity:
            optional_diffusivity_stack = np.empty((nt, ny, nx), dtype=sample_field.dtype)

        # Load series
        for k, snapshot_index in enumerate(common_indices_sorted):
            time_dataset_name = f"{DATASET_PREFIX_TIME}{snapshot_index:05d}"
            field_dataset_name = f"{dataset_prefix_main_field}{snapshot_index:05d}"
            times_seconds[k] = float(np.asarray(h5f[time_dataset_name]))
            field_stack[k, :, :] = np.asarray(h5f[field_dataset_name])
            if optional_diffusivity_stack is not None:
                diffusivity_dataset_name = f"{DATASET_PREFIX_DIFFUSIVITY}{snapshot_index:05d}"
                if diffusivity_dataset_name in h5f:
                    optional_diffusivity_stack[k, :, :] = np.asarray(h5f[diffusivity_dataset_name])
                else:
                    # Carry forward last known map or set NaN on the very first
                    if k > 0:
                        optional_diffusivity_stack[k, :, :] = optional_diffusivity_stack[k - 1, :, :]
                    else:
                        optional_diffusivity_stack[k, :, :] = np.nan

    # --- Normalize coordinate direction to increasing, and apply same flips to both stacks ---
    def _ensure_increasing_and_get_flip(coords: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (coords_increasing, flipped?) where flipped indicates original was decreasing."""
        if coords.ndim != 1:
            raise ValueError("Coordinate arrays must be 1D.")
        if coords.size < 2:
            return coords, False
        diffs = np.diff(coords)
        if np.all(diffs > 0):
            return coords, False
        if np.all(diffs < 0):
            return coords[::-1].copy(), True
        raise ValueError("Coordinate array must be strictly monotone (increasing or decreasing).")

    # Decide flips once from the raw coords we just read/built
    x_coords, flip_x = _ensure_increasing_and_get_flip(x_coords)
    y_coords, flip_y = _ensure_increasing_and_get_flip(y_coords)

    # Apply flips to the main field stack (t, y, x)
    if flip_x:
        field_stack = np.flip(field_stack, axis=2)
    if flip_y:
        field_stack = np.flip(field_stack, axis=1)

    # Apply the exact same flips to the optional diffusivity stack
    if optional_diffusivity_stack is not None:
        if flip_x:
            optional_diffusivity_stack = np.flip(optional_diffusivity_stack, axis=2)
        if flip_y:
            optional_diffusivity_stack = np.flip(optional_diffusivity_stack, axis=1)

    # Optional: warn if NaN/Inf present (you said your sims should avoid this)
    if not np.isfinite(field_stack).all():
        n_bad = np.size(field_stack) - int(np.isfinite(field_stack).sum())
        warnings.warn(f"Field stack contains {n_bad} non‑finite entries; results may be affected.")

    return SnapshotSeries(
        times_seconds=times_seconds,
        field_stack=field_stack,
        x_coordinates=x_coords,
        y_coordinates=y_coords,
        optional_diffusivity_stack=optional_diffusivity_stack,
    )

# ============================= INTERPOLATION HELPERS =============================

def find_nearest_time_index(target_time_seconds: float, available_times_seconds: np.ndarray) -> int:
    absolute_differences = np.abs(available_times_seconds - target_time_seconds)
    return int(np.argmin(absolute_differences))


def linear_temporal_interpolation(
    coarse_target_time_seconds: float,
    fine_times_seconds: np.ndarray,
    fine_field_stack_time_y_x: np.ndarray,
) -> np.ndarray:
    """Return (ny, nx) slice interpolated in time from fine run onto target time.

    Out‑of‑range targets clamp to nearest endpoint. Exact hits return that slice.
    """
    if coarse_target_time_seconds <= fine_times_seconds[0] + FLOAT_TIME_TOLERANCE:
        return fine_field_stack_time_y_x[0]
    if coarse_target_time_seconds >= fine_times_seconds[-1] - FLOAT_TIME_TOLERANCE:
        return fine_field_stack_time_y_x[-1]

    right_index = int(np.searchsorted(fine_times_seconds, coarse_target_time_seconds, side="right"))
    left_index = right_index - 1
    left_time = fine_times_seconds[left_index]
    right_time = fine_times_seconds[right_index]

    if abs(right_time - left_time) < FLOAT_TIME_TOLERANCE:
        return fine_field_stack_time_y_x[left_index]

    interpolation_weight = (coarse_target_time_seconds - left_time) / (right_time - left_time)
    return (
        (1.0 - interpolation_weight) * fine_field_stack_time_y_x[left_index]
        + interpolation_weight * fine_field_stack_time_y_x[right_index]
    )

# ---------- Vectorized bilinear mapping fine→coarse ----------

@dataclass
class SpatialMappingCache:
    """Precomputed mapping indices and weights for vectorized bilinear interpolation.

    Given fine (x_f, y_f) and coarse (x_c, y_c), we cache left/right indices and
    interpolation weights for both axes so each time slice can be mapped quickly.
    """
    ix_left_2d: np.ndarray  # shape (ny_c, nx_c)
    ix_right_2d: np.ndarray # shape (ny_c, nx_c)
    iy_bot_2d: np.ndarray   # shape (ny_c, nx_c)
    iy_top_2d: np.ndarray   # shape (ny_c, nx_c)
    tx_2d: np.ndarray       # shape (ny_c, nx_c)
    ty_2d: np.ndarray       # shape (ny_c, nx_c)


def build_spatial_mapping_cache(
    fine_x_coords: np.ndarray,
    fine_y_coords: np.ndarray,
    coarse_x_coords: np.ndarray,
    coarse_y_coords: np.ndarray,
) -> SpatialMappingCache:
    """Precompute 2D index/weight arrays for bilinear interpolation from fine→coarse.

    Coordinates must be strictly increasing; inputs are assumed normalized.
    """
    nx_f = fine_x_coords.size
    ny_f = fine_y_coords.size

    # For every coarse x, find the bracketing fine x indices
    ix_left = np.searchsorted(fine_x_coords, coarse_x_coords, side="right") - 1
    ix_left = np.clip(ix_left, 0, nx_f - 2)
    ix_right = ix_left + 1

    # For every coarse y, find the bracketing fine y indices
    iy_bot = np.searchsorted(fine_y_coords, coarse_y_coords, side="right") - 1
    iy_bot = np.clip(iy_bot, 0, ny_f - 2)
    iy_top = iy_bot + 1

    # Compute 1D weights for x and y, then broadcast to 2D grids
    x1 = fine_x_coords[ix_left]
    x2 = fine_x_coords[ix_right]
    with np.errstate(divide='ignore', invalid='ignore'):
        tx = np.where(x2 == x1, 0.0, (coarse_x_coords - x1) / (x2 - x1))
    y1 = fine_y_coords[iy_bot]
    y2 = fine_y_coords[iy_top]
    with np.errstate(divide='ignore', invalid='ignore'):
        ty = np.where(y2 == y1, 0.0, (coarse_y_coords - y1) / (y2 - y1))

    # Mesh into 2D arrays for vectorized gather
    ix_left_2d, iy_bot_2d = np.meshgrid(ix_left, iy_bot)
    ix_right_2d, iy_top_2d = ix_left_2d + 1, iy_bot_2d + 1
    tx_2d, ty_2d = np.meshgrid(tx, ty)

    return SpatialMappingCache(
        ix_left_2d=ix_left_2d.astype(int),
        ix_right_2d=ix_right_2d.astype(int),
        iy_bot_2d=iy_bot_2d.astype(int),
        iy_top_2d=iy_top_2d.astype(int),
        tx_2d=tx_2d.astype(float),
        ty_2d=ty_2d.astype(float),
    )


def map_fine_to_coarse_grid_single_slice_vectorized(
    fine_field_slice_y_x: np.ndarray,
    cache: SpatialMappingCache,
) -> np.ndarray:
    """Vectorized bilinear interpolation of one slice from fine grid to coarse grid.

    Returns (ny_c, nx_c) array.
    """
    q11 = fine_field_slice_y_x[cache.iy_bot_2d,  cache.ix_left_2d]
    q21 = fine_field_slice_y_x[cache.iy_bot_2d,  cache.ix_right_2d]
    q12 = fine_field_slice_y_x[cache.iy_top_2d,  cache.ix_left_2d]
    q22 = fine_field_slice_y_x[cache.iy_top_2d,  cache.ix_right_2d]

    one_minus_tx = (1.0 - cache.tx_2d)
    one_minus_ty = (1.0 - cache.ty_2d)

    # (1 - ty) * ((1 - tx) * q11 + tx * q21) + ty * ((1 - tx) * q12 + tx * q22)
    return (
        one_minus_ty * (one_minus_tx * q11 + cache.tx_2d * q21)
        + cache.ty_2d * (one_minus_tx * q12 + cache.tx_2d * q22)
    )

# ============================= ERROR METRICS =============================

def l2_rms_error_over_grid(coarse_values_y_x: np.ndarray, mapped_fine_values_y_x: np.ndarray) -> float:
    difference_y_x = coarse_values_y_x - mapped_fine_values_y_x
    return float(np.sqrt(np.mean(difference_y_x ** 2)))


def linf_max_error_over_grid(coarse_values_y_x: np.ndarray, mapped_fine_values_y_x: np.ndarray) -> float:
    return float(np.max(np.abs(coarse_values_y_x - mapped_fine_values_y_x)))

# ============================= PROBES =============================

def bilinear_spatial_interpolation_single_point(
    field_slice_y_x: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    physical_x_position: float,
    physical_y_position: float,
) -> float:
    """Return field value at (x,y) by bilinear interpolation on a uniform, monotone grid.

    Coordinates are assumed strictly increasing due to normalization.
    """
    # Clamp inside domain for safety
    clamped_x = float(np.clip(physical_x_position, x_coordinates[0], x_coordinates[-1]))
    clamped_y = float(np.clip(physical_y_position, y_coordinates[0], y_coordinates[-1]))

    ix_left = int(np.searchsorted(x_coordinates, clamped_x, side="right") - 1)
    iy_bot  = int(np.searchsorted(y_coordinates, clamped_y, side="right") - 1)

    ix_left = int(np.clip(ix_left, 0, len(x_coordinates) - 2))
    iy_bot  = int(np.clip(iy_bot,  0, len(y_coordinates) - 2))

    x1, x2 = x_coordinates[ix_left], x_coordinates[ix_left + 1]
    y1, y2 = y_coordinates[iy_bot],  y_coordinates[iy_bot + 1]

    q11 = field_slice_y_x[iy_bot,     ix_left]
    q21 = field_slice_y_x[iy_bot,     ix_left + 1]
    q12 = field_slice_y_x[iy_bot + 1, ix_left]
    q22 = field_slice_y_x[iy_bot + 1, ix_left + 1]

    tx = 0.0 if x2 == x1 else (clamped_x - x1) / (x2 - x1)
    ty = 0.0 if y2 == y1 else (clamped_y - y1) / (y2 - y1)

    return (1 - ty) * ((1 - tx) * q11 + tx * q21) + ty * ((1 - tx) * q12 + tx * q22)


def extract_probe_time_series(
    snapshot_series: SnapshotSeries,
    probe_points_name_and_xy_list: Sequence[Tuple[str, Tuple[float, float]]],
) -> Dict[str, np.ndarray]:
    """Return dict: probe_name → series(t) by bilinear sampling at each snapshot time."""
    out_dict: Dict[str, np.ndarray] = {}
    for probe_name, (probe_x, probe_y) in probe_points_name_and_xy_list:
        time_series_values = np.empty_like(snapshot_series.times_seconds, dtype=float)
        for time_index in range(len(snapshot_series.times_seconds)):
            time_series_values[time_index] = bilinear_spatial_interpolation_single_point(
                field_slice_y_x=snapshot_series.field_stack[time_index],
                x_coordinates=snapshot_series.x_coordinates,
                y_coordinates=snapshot_series.y_coordinates,
                physical_x_position=probe_x,
                physical_y_position=probe_y,
            )
        out_dict[probe_name] = time_series_values
    return out_dict


def get_field_slice_at_time(
    snapshot_series: SnapshotSeries,
    target_time_seconds: float,
    use_temporal_interpolation: bool,
) -> np.ndarray:
    """Return a (ny, nx) slice from snapshot_series at the requested time."""
    if use_temporal_interpolation:
        return linear_temporal_interpolation(
            coarse_target_time_seconds=target_time_seconds,
            fine_times_seconds=snapshot_series.times_seconds,
            fine_field_stack_time_y_x=snapshot_series.field_stack,
        )
    nearest_index = find_nearest_time_index(target_time_seconds, snapshot_series.times_seconds)
    return snapshot_series.field_stack[nearest_index]


def extract_probe_time_series_aligned(
    snapshot_series: SnapshotSeries,
    target_times_seconds: np.ndarray,
    probe_x: float,
    probe_y: float,
    use_temporal_interpolation: bool,
) -> np.ndarray:
    """Sample one run at (probe_x,probe_y) for a list of target times."""
    values = np.empty_like(target_times_seconds, dtype=float)
    for k, tval in enumerate(target_times_seconds):
        slice_at_t = get_field_slice_at_time(snapshot_series, tval, use_temporal_interpolation)
        values[k] = bilinear_spatial_interpolation_single_point(
            field_slice_y_x=slice_at_t,
            x_coordinates=snapshot_series.x_coordinates,
            y_coordinates=snapshot_series.y_coordinates,
            physical_x_position=probe_x,
            physical_y_position=probe_y,
        )
    return values

# ============================= COMPARISON CORE =============================

def compare_two_runs_over_common_times(
    coarse_run: SnapshotSeries,
    fine_run: SnapshotSeries,
    field_display_name: str,
    use_temporal_interpolation: bool,
    mapping_cache: Optional[SpatialMappingCache] = None,
) -> Tuple[List[float], List[float], np.ndarray]:
    """Compute L2 and L∞ errors for all coarse snapshot times, mapping fine→coarse.

    Returns `(list_l2_errors, list_linf_errors, coarse_times_seconds)`
    """
    if mapping_cache is None:
        mapping_cache = build_spatial_mapping_cache(
            fine_x_coords=fine_run.x_coordinates,
            fine_y_coords=fine_run.y_coordinates,
            coarse_x_coords=coarse_run.x_coordinates,
            coarse_y_coords=coarse_run.y_coordinates,
        )

    list_l2_errors: List[float] = []
    list_linf_errors: List[float] = []

    for coarse_time_index, coarse_time_value in enumerate(coarse_run.times_seconds):
        if use_temporal_interpolation:
            fine_slice_at_coarse_time = linear_temporal_interpolation(
                coarse_target_time_seconds=coarse_time_value,
                fine_times_seconds=fine_run.times_seconds,
                fine_field_stack_time_y_x=fine_run.field_stack,
            )
        else:
            nearest_index_in_fine = find_nearest_time_index(coarse_time_value, fine_run.times_seconds)
            time_mismatch = abs(fine_run.times_seconds[nearest_index_in_fine] - coarse_time_value)
            if time_mismatch > MAX_TIME_MISMATCH_SECONDS:
                warnings.warn(
                    f"[{field_display_name}] Time mismatch {time_mismatch:.3f}s exceeds "
                    f"MAX_TIME_MISMATCH_SECONDS={MAX_TIME_MISMATCH_SECONDS:.3f}s at coarse t={coarse_time_value:.3f}s."
                )
            fine_slice_at_coarse_time = fine_run.field_stack[nearest_index_in_fine]

        mapped_fine_on_coarse = map_fine_to_coarse_grid_single_slice_vectorized(
            fine_field_slice_y_x=fine_slice_at_coarse_time,
            cache=mapping_cache,
        )

        coarse_slice = coarse_run.field_stack[coarse_time_index]
        l2_error_value = l2_rms_error_over_grid(coarse_slice, mapped_fine_on_coarse)
        linf_error_value = linf_max_error_over_grid(coarse_slice, mapped_fine_on_coarse)

        list_l2_errors.append(l2_error_value)
        list_linf_errors.append(linf_error_value)

    return list_l2_errors, list_linf_errors, coarse_run.times_seconds


def observed_order_from_two_error_series(
    coarse_vs_medium_errors: Sequence[float],
    medium_vs_fine_errors: Sequence[float],
) -> float:
    """Estimate observed order p using RMS‑over‑time of error series (robust to noise)."""
    rms_coarse_medium = float(np.sqrt(np.mean(np.asarray(coarse_vs_medium_errors) ** 2)))
    rms_medium_fine  = float(np.sqrt(np.mean(np.asarray(medium_vs_fine_errors) ** 2)))
    rms_coarse_medium = max(rms_coarse_medium, 1e-30)
    rms_medium_fine  = max(rms_medium_fine,  1e-30)
    return float((math.log(rms_coarse_medium) - math.log(rms_medium_fine)) / math.log(2.0))

# ============================= MAIN =============================

def main() -> None:
    # Resolve output locations
    active_analysis_folder_abspath = resolve_active_analysis_folder()
    print(f"Active analysis folder: {active_analysis_folder_abspath}")

    analysis_parent_abspath = os.path.join(DEFAULT_BASE_DIRECTORY, ANALYSIS_PARENT_FOLDER_NAME)
    output_csv_abspath = os.path.join(active_analysis_folder_abspath, OUTPUT_CSV_FILENAME)
    output_figures_directory_abspath = os.path.join(active_analysis_folder_abspath, OUTPUT_FIGURES_SUBFOLDER_NAME)

    # Load the three runs for each field separately (Temperature and Hydrogen)
    loaded_runs_per_field: Dict[str, List[SnapshotSeries]] = {}
    for field_display_name, dataset_prefix in FIELDS_TO_COMPARE:
        field_runs_list: List[SnapshotSeries] = []
        for simulation_run in SIMULATION_RUNS_LIST:
            snapshot_series_for_run = load_snapshot_series_for_field(
                simulation_run=simulation_run,
                dataset_prefix_main_field=dataset_prefix,
                analysis_parent_abspath=analysis_parent_abspath,
            )
            field_runs_list.append(snapshot_series_for_run)
        loaded_runs_per_field[field_display_name] = field_runs_list

    # Prepare CSV lines
    csv_lines: List[str] = []
    header_columns = [
        "field",
        "metric",
        "time_seconds",
        f"{SIMULATION_RUNS_LIST[0].label_for_plots_and_csv} vs {SIMULATION_RUNS_LIST[1].label_for_plots_and_csv}",
        f"{SIMULATION_RUNS_LIST[1].label_for_plots_and_csv} vs {SIMULATION_RUNS_LIST[2].label_for_plots_and_csv}",
    ]
    csv_lines.append(",".join(header_columns))

    # For each field, compute global errors over time and observed orders
    for field_display_name, _dataset_prefix in FIELDS_TO_COMPARE:
        coarse_series, medium_series, fine_series = loaded_runs_per_field[field_display_name]

        # Precompute spatial mapping caches to speed up slice mapping
        cache_cm = build_spatial_mapping_cache(
            fine_x_coords=medium_series.x_coordinates,
            fine_y_coords=medium_series.y_coordinates,
            coarse_x_coords=coarse_series.x_coordinates,
            coarse_y_coords=coarse_series.y_coordinates,
        )
        cache_mf = build_spatial_mapping_cache(
            fine_x_coords=fine_series.x_coordinates,
            fine_y_coords=fine_series.y_coordinates,
            coarse_x_coords=medium_series.x_coordinates,
            coarse_y_coords=medium_series.y_coordinates,
        )

        l2_coarse_medium, linf_coarse_medium, times_cm = compare_two_runs_over_common_times(
            coarse_run=coarse_series,
            fine_run=medium_series,
            field_display_name=field_display_name,
            use_temporal_interpolation=USE_TEMPORAL_INTERPOLATION,
            mapping_cache=cache_cm,
        )
        l2_medium_fine, linf_medium_fine, times_mf = compare_two_runs_over_common_times(
            coarse_run=medium_series,
            fine_run=fine_series,
            field_display_name=field_display_name,
            use_temporal_interpolation=USE_TEMPORAL_INTERPOLATION,
            mapping_cache=cache_mf,
        )

        # Write per‑time errors to CSV (times_cm and times_mf match respective coarse times)
        for time_value, e1, e2 in zip(times_cm, l2_coarse_medium, l2_medium_fine[: len(times_cm)]):
            csv_lines.append(",".join([
                field_display_name, "L2", f"{time_value:.6f}", f"{e1:.6e}", f"{e2:.6e}"
            ]))
        for time_value, e1, e2 in zip(times_cm, linf_coarse_medium, linf_medium_fine[: len(times_cm)]):
            csv_lines.append(",".join([
                field_display_name, "Linf", f"{time_value:.6f}", f"{e1:.6e}", f"{e2:.6e}"
            ]))

        # Observed orders (RMS over time) — include in CSV as summary rows
        observed_order_l2 = observed_order_from_two_error_series(l2_coarse_medium, l2_medium_fine)
        observed_order_linf = observed_order_from_two_error_series(linf_coarse_medium, linf_medium_fine)
        print(f"{field_display_name}: observed order (RMS over time)  L2≈{observed_order_l2:.2f}, Linf≈{observed_order_linf:.2f}")

        csv_lines.append(",".join([
            field_display_name, "ObservedOrder_L2", "", f"{observed_order_l2:.6f}", f"{observed_order_l2:.6f}"
        ]))
        csv_lines.append(",".join([
            field_display_name, "ObservedOrder_Linf", "", f"{observed_order_linf:.6f}", f"{observed_order_linf:.6f}"
        ]))

        # Probe comparisons at engineer‑relevant spots
        probe_time_series_coarse = extract_probe_time_series(coarse_series, PROBE_POINTS)
        probe_time_series_medium = extract_probe_time_series(medium_series, PROBE_POINTS)
        probe_time_series_fine = extract_probe_time_series(fine_series, PROBE_POINTS)

        # Dump probe absolute differences (coarse–medium at coarse times, medium–fine at medium times)
        for probe_name, (probe_x, probe_y) in PROBE_POINTS:
            # coarse vs medium at coarse times
            coarse_at_coarse = extract_probe_time_series_aligned(
                snapshot_series=coarse_series,
                target_times_seconds=coarse_series.times_seconds,
                probe_x=probe_x,
                probe_y=probe_y,
                use_temporal_interpolation=False,  # exact sampling on its own times
            )
            medium_at_coarse = extract_probe_time_series_aligned(
                snapshot_series=medium_series,
                target_times_seconds=coarse_series.times_seconds,
                probe_x=probe_x,
                probe_y=probe_y,
                use_temporal_interpolation=USE_TEMPORAL_INTERPOLATION,
            )
            for time_value, dval_abs in zip(coarse_series.times_seconds, np.abs(coarse_at_coarse - medium_at_coarse)):
                csv_lines.append(",".join([
                    field_display_name, f"probe:{probe_name}", f"{time_value:.6f}", f"{dval_abs:.6e}", ""
                ]))

            # medium vs fine at medium times
            medium_at_medium = extract_probe_time_series_aligned(
                snapshot_series=medium_series,
                target_times_seconds=medium_series.times_seconds,
                probe_x=probe_x,
                probe_y=probe_y,
                use_temporal_interpolation=False,
            )
            fine_at_medium = extract_probe_time_series_aligned(
                snapshot_series=fine_series,
                target_times_seconds=medium_series.times_seconds,
                probe_x=probe_x,
                probe_y=probe_y,
                use_temporal_interpolation=USE_TEMPORAL_INTERPOLATION,
            )
            for time_value, dval_abs in zip(medium_series.times_seconds, np.abs(medium_at_medium - fine_at_medium)):
                csv_lines.append(",".join([
                    field_display_name, f"probe:{probe_name}", f"{time_value:.6f}", "", f"{dval_abs:.6e}"
                ]))

        # ============================= PLOTTING =============================
        if SAVE_FIGURES:
            os.makedirs(output_figures_directory_abspath, exist_ok=True)

            # 1) Global error plots over time
            plt.figure()
            plt.plot(times_cm, l2_coarse_medium, label=f"L2 {SIMULATION_RUNS_LIST[0].label_for_plots_and_csv} vs {SIMULATION_RUNS_LIST[1].label_for_plots_and_csv}")
            plt.plot(times_mf, l2_medium_fine, label=f"L2 {SIMULATION_RUNS_LIST[1].label_for_plots_and_csv} vs {SIMULATION_RUNS_LIST[2].label_for_plots_and_csv}")
            plt.xlabel("Time [s]")
            plt.ylabel("L2 error (RMS over grid)")
            plt.title(f"Global L2 error over time — {field_display_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig_path_l2 = os.path.join(output_figures_directory_abspath, f"errors_over_time_L2_{field_display_name}.png")
            plt.savefig(fig_path_l2, dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(times_cm, linf_coarse_medium, label=f"Linf {SIMULATION_RUNS_LIST[0].label_for_plots_and_csv} vs {SIMULATION_RUNS_LIST[1].label_for_plots_and_csv}")
            plt.plot(times_mf, linf_medium_fine, label=f"Linf {SIMULATION_RUNS_LIST[1].label_for_plots_and_csv} vs {SIMULATION_RUNS_LIST[2].label_for_plots_and_csv}")
            plt.xlabel("Time [s]")
            plt.ylabel("Linf error (max over grid)")
            plt.title(f"Global Linf error over time — {field_display_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            fig_path_linf = os.path.join(output_figures_directory_abspath, f"errors_over_time_Linf_{field_display_name}.png")
            plt.savefig(fig_path_linf, dpi=150, bbox_inches="tight")
            plt.close()

            # 2) Probe time series overlays: coarse vs medium vs fine
            for probe_name, (_px, _py) in PROBE_POINTS:
                plt.figure()
                plt.plot(coarse_series.times_seconds, probe_time_series_coarse[probe_name], label=SIMULATION_RUNS_LIST[0].label_for_plots_and_csv)
                plt.plot(medium_series.times_seconds, probe_time_series_medium[probe_name], label=SIMULATION_RUNS_LIST[1].label_for_plots_and_csv)
                plt.plot(fine_series.times_seconds,   probe_time_series_fine[probe_name],   label=SIMULATION_RUNS_LIST[2].label_for_plots_and_csv)
                plt.xlabel("Time [s]")
                plt.ylabel(field_display_name)
                plt.title(f"Probe '{probe_name}' — {field_display_name}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                fig_probe_path = os.path.join(output_figures_directory_abspath, f"probe_{probe_name}_{field_display_name}.png")
                plt.savefig(fig_probe_path, dpi=150, bbox_inches="tight")
                plt.close()

    # Save summary CSV at the very end
    with open(output_csv_abspath, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(csv_lines))
    print(f"Wrote summary to {output_csv_abspath}")


if __name__ == "__main__":
    main()
