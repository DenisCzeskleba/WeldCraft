"""
Create a still Brownian-motion diagram from one saved HDF5 snapshot.

Diagram options live in this file on purpose. Simulation settings are read from
the HDF5 metadata when present, so old config edits do not change old diagrams.
"""

from pathlib import Path
import contextlib
import importlib.util
import io
import json

import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Rectangle, Wedge

with contextlib.redirect_stdout(io.StringIO()):
    from b3_Brown_Functions import (
        create_spot_mask,
        create_trap_layer_mask,
        in_results,
        load_brown_config_json,
        read_saved_steps,
        resources_dir,
        results_dir,
    )
    from c2_Brown_Get_Speed import analyze_transport


# ---------------------- Input Snapshot ---------------------- #
# Used by ordinary presets. The special "all_presets" mode instead always uses
# 02_Results/Examples/published_examples_source.h5 for reproducible public examples.
INPUT_H5_FILENAME = ("O2 V2 narrower version but short time wise.h5")
SNAPSHOT_INDEX = -1  # HDF5 saved-frame index to plot; -1 means the last saved frame, 0 means first saved frame.


# ---------------------- Output ---------------------- #
SHOW_PLOT = True
SAVE_PNG = True
SAVE_PDF = False
SAVE_SVG = False
# False keeps labels as editable text in Inkscape; True converts them to paths.
SVG_TEXT_AS_PATHS = False
OUTPUT_FOLDER = ""  # Relative to 02_Results; leave empty to save directly in 02_Results.
OUTPUT_BASENAME = "brownian_diagram"
SAVE_DPI = 300


# ---------------------- Diagram Profile ---------------------- #
# Available diagram presets:
#   "default"                               - Detailed pixel-by-pixel simulation matrix.
#   "all_presets"                           - Render every preset from Examples/published_examples_source.h5 into Examples.
#   "two_regions_w_solubility"              - Pixel matrix emphasizing both regions and solubilities.
#   "simple_1_region_source_sink"           - Pixel matrix configured for a single-region source/sink view.
#   "simple_1_region_source_with_heatmap"   - Purple source/sink matrix paired with the difference heatmap.
#   "simple_concentration_profile"          - Saved concentration profile without matrix or flux panels.
#   "concentration_profile_with_heatmap"    - Clean concentration profile paired with the difference heatmap.
#   "depletion_heatmap"                     - Smoothed local enrichment/depletion heatmap.
#   "printer_friendly"                      - Spatially binned, larger occupancy glyphs for print.
#   "area_summary"                          - Stylized non-overlapping dots using measured area averages.
#   "chapter_2_3_brown_overview"            - Stylized dots following a transient saved x-profile.
#   "area_summary_transient"                - Stylized non-overlapping dots using measured area averages, with transient x-profile.
# When adding a preset, also add it to BATCH_PRESET_ORDER in the "all_presets"
# preset file. Treat existing entries as append-only to keep public numbering stable.
DIAGRAM_PRESET = "simple_1_region_source_with_heatmap"  # File stem in 01_Resources/Diagram_Presets.

REQUIRED_DIAGRAM_PRESET_KEYS = [
    "PRESET_NAME",
    "RENDER_MODE",
    "FIGURE_SIZE",
    "MATCH_SIDE_PANEL_HEIGHT_TO_MAIN",
    "TITLE",
    "X_LABEL",
    "Y_LABEL",
    "COLOR_EMPTY",
    "COLOR_AVAILABLE_SPOT",
    "COLOR_HYDROGEN",
    "COLOR_CONCENTRATION_LINE",
    "DOT_SIZE_AVAILABLE",
    "DOT_SIZE_HYDROGEN",
    "DOT_ALPHA_AVAILABLE",
    "DOT_ALPHA_HYDROGEN",
    "SHOW_MAIN_PANEL",
    "SHOW_CONCENTRATION_PROFILE_PANEL",
    "SHOW_NET_FLUX_PANEL",
    "PROFILE_AXIS",
    "PROFILE_X_RANGE",
    "PROFILE_Y_RANGE",
    "PROFILE_BIN_SIZE",
    "PROFILE_SMOOTHING_WINDOW",
    "PROFILE_GAUSSIAN_SIGMA",
    "SHOW_PROFILE_HALF_TRANSITION",
    "PROFILE_HALF_TRANSITION_COLOR",
    "PROFILE_AREA_1_LABEL",
    "PROFILE_AREA_2_LABEL",
    "SHOW_PROFILE_SPOT_SHADE",
    "PROFILE_SPOT_SHADE_COLOR",
    "PROFILE_SPOT_SHADE_ALPHA",
    "PROFILE_SPOT_SHADE_LABEL",
    "PROFILE_SPOT_SHADE_LABEL_COLOR",
    "SHOW_REGION_ANNOTATIONS",
    "SHOW_LEFT_RIGHT_ANNOTATIONS",
    "SHOW_LEFT_RIGHT_WITHOUT_SINK_SOURCE_ANNOTATIONS",
    "SHOW_SOURCE_SINK_ANNOTATIONS",
    "SHOW_SPOT_ANNOTATION",
    "SHOW_TRAP_LAYER_ANNOTATION",
    "ANNOTATION_FONT_SIZE",
    "ANNOTATION_COLOR",
    "CUSTOM_RECT_REGIONS",
]

OPTIONAL_DIAGRAM_PRESET_DEFAULTS = {
    "SHOW_LEGEND": True,
    "SHOW_HEATMAP_PANEL": False,
    # Per-preset panel typography. None keeps Matplotlib's current default.
    "PANEL_TITLE_FONT_SIZE": None,
    "AXIS_LABEL_FONT_SIZE": None,
    "TICK_LABEL_FONT_SIZE": None,
    "SHOW_CONCENTRATION_PROFILE_TITLE": True,
    "SHOW_SITE_STATE_LEGEND": False,
    "SITE_STATE_AVAILABLE_LABEL": "Available Lattice Sites",
    "SITE_STATE_OCCUPIED_LABEL": "Occupied Sites",
    "SITE_STATE_UNAVAILABLE_LABEL": "Unavailable Locations",
    "SITE_STATE_LEGEND_LOCATION": "upper right",
    "SITE_STATE_LEGEND_ANCHOR": (0.98, 0.98),
    "SITE_STATE_LEGEND_FONT_SIZE": 18,
    "SITE_STATE_LEGEND_MARKER_AREA": 128,
    # Smoothed concentration-map settings.
    "HEATMAP_MODE": "deviation",  # Options: "deviation", "occupancy"
    "HEATMAP_SIGMA": 18.0,
    "HEATMAP_COLORMAP": "RdBu_r",
    "HEATMAP_DEVIATION_LIMIT": 20.0,
    "HEATMAP_OCCUPANCY_RANGE": (0.0, 100.0),
    "HEATMAP_REFERENCE_MODE": "regional_bulk",  # Options: "regional_bulk", "global_bulk"
    "HEATMAP_RESPECT_AREA_BOUNDARIES": True,
    "HEATMAP_SEPARATE_BASE_AREAS": False,
    "HEATMAP_SHOW_CONTOURS": True,
    "HEATMAP_CONTOUR_LEVELS": (-15, -10, -5, 5, 10, 15),
    "HEATMAP_CONTOUR_COLOR": "#303030",
    "HEATMAP_CONTOUR_ALPHA": 0.35,
    "HEATMAP_SHOW_COLORBAR": True,
    # Printer-friendly, spatially binned occupancy glyphs.
    "GLYPH_BIN_SIZE": 32,
    "GLYPH_MIN_RADIUS_FRACTION": 0.38,
    "GLYPH_MAX_RADIUS_FRACTION": 0.90,
    "GLYPH_CAPACITY_GAMMA": 0.30,
    "GLYPH_EDGE_COLOR": "#202020",
    "GLYPH_EDGE_WIDTH": 0.45,
    "GLYPH_BACKGROUND_COLOR": "#F5F5F5",
    "GLYPH_SHOW_GRID": True,
    "GLYPH_GRID_COLOR": "#D8D8D8",
    "GLYPH_SHOW_EXPLANATION": True,
    "GLYPH_SHOW_SPECIAL_REGION_VALUES": True,
    # Matrix-shaped area-average reconstruction with synthetic dot positions.
    "AREA_SUMMARY_TOTAL_DOTS": 5000,
    "AREA_SUMMARY_DENSITY_MODE": "available_sites",  # Options: "available_sites", "uniform_area"
    "AREA_SUMMARY_MIN_DOTS_PER_AREA": 30,
    "AREA_SUMMARY_RANDOM_SEED": 104729,
    "AREA_SUMMARY_CONCENTRATION_MODE": "area_average",  # Options: "area_average", "saved_x_profile", "linear_x"
    "AREA_SUMMARY_LINEAR_CONCENTRATION": (100, 0),
    "AREA_SUMMARY_X_PROFILE_SIGMA": 20.0,
    "AREA_SUMMARY_CONCENTRATION_BIN_WIDTH": 40,
    "AREA_SUMMARY_POSITION_MODE": "even_hex",  # Options: "even_hex", "random"
    "AREA_SUMMARY_MIN_DOT_SPACING": 8.0,
    "AREA_SUMMARY_SHAKE_OUTPUT_MODES": None,
    "AREA_SUMMARY_SHAKE_MODE": "none",  # Options: "none", "gentle", "organic", "clustered"
    "AREA_SUMMARY_SHAKE_STRENGTH": None,
    "AREA_SUMMARY_SHAKE_PASSES": None,
    "AREA_SUMMARY_SHAKE_ATTEMPTS": 8,
    "AREA_SUMMARY_CLUSTER_COUNT": 8,
    "AREA_SUMMARY_CLUSTER_ATTRACTION": 0.70,
    "AREA_SUMMARY_CLUSTER_SCOPE": "per_area",  # Options: "per_area", "combined_a_b"
    "AREA_SUMMARY_POSITION_JITTER": 0.42,
    "AREA_SUMMARY_POSITION_INSET": 5,
    "AREA_SUMMARY_DOT_SIZE": 18,
    "AREA_SUMMARY_DOT_ALPHA": 0.96,
    "AREA_SUMMARY_DOT_EDGE_COLOR": "#FFFFFF",
    "AREA_SUMMARY_DOT_EDGE_WIDTH": 0.25,
    "AREA_SUMMARY_BACKGROUND_COLOR": "#FAFAFA",
    "AREA_SUMMARY_SHOW_AREA_LABELS": True,
    "AREA_SUMMARY_SHOW_EXPLANATION": True,
    "AREA_SUMMARY_SHOW_DOT_LEGEND": False,
    "AREA_SUMMARY_VACANT_LABEL": "Available Lattice Sites",
    "AREA_SUMMARY_OCCUPIED_LABEL": "Occupied Sites",
    "AREA_SUMMARY_DOT_LEGEND_LOCATION": "upper right",
    "AREA_SUMMARY_DOT_LEGEND_ANCHOR": (0.98, 0.98),
    "AREA_SUMMARY_DOT_LEGEND_FONT_SIZE": 12,
    "AREA_SUMMARY_DOT_LEGEND_SIZE_SCALE": 1.0,
    "AREA_SUMMARY_SHOW_SOURCE_SINK_BANDS": True,
    "AREA_SUMMARY_SHOW_SOURCE_SINK_LABELS": False,
    "AREA_SUMMARY_SOURCE_SINK_BAND_WIDTH_SCALE": 1.0,
    "AREA_SUMMARY_SOURCE_SINK_EDGE_COLOR": "#303030",
    "AREA_SUMMARY_SOURCE_SINK_EDGE_WIDTH": 0.8,
    "AREA_SUMMARY_SHOW_HALF_DIVIDER": True,
    "AREA_SUMMARY_HALF_DIVIDER_COLOR": "#A0A0A0",
    # Shared visual aids.
    "SHOW_SPECIAL_REGION_OUTLINES": False,
    "SPECIAL_REGION_OUTLINE_COLOR": "#202020",
    "SPECIAL_REGION_OUTLINE_WIDTH": 1.2,
    "SPECIAL_REGION_OUTLINE_STYLE": "--",
    "SHOW_SPOT_REGION_FILL": False,
    "SPOT_REGION_FILL_COLOR": "#D9D9D9",
    "SPOT_REGION_FILL_ALPHA": 1.0,
    "SPOT_REGION_OUTLINE_COLOR": "#202020",
    "SPOT_REGION_OUTLINE_WIDTH": 1.2,
    "SPOT_REGION_OUTLINE_STYLE": "--",
    "ANNOTATION_STROKE_COLOR": "#000000",
    "ANNOTATION_STROKE_WIDTH": 2.5,
    "ANNOTATION_BOX_ENABLED": False,
    "ANNOTATION_BOX_FACE_COLOR": "#FFFFFF",
    "ANNOTATION_BOX_EDGE_COLOR": "#707070",
    "ANNOTATION_BOX_ALPHA": 0.88,
    "SHOW_FRAME_SUPTITLE": True,
    "NET_FLUX_COLOR": "#4A148C",
    "NET_FLUX_BAND_COLOR": "#B39DDB",
}


def diagram_presets_dir():
    return resources_dir() / "Diagram_Presets"


def load_diagram_preset_module(preset_file_stem):
    preset_path = diagram_presets_dir() / f"{preset_file_stem}.py"
    if not preset_path.exists():
        available = sorted(path.stem for path in diagram_presets_dir().glob("*.py"))
        raise FileNotFoundError(
            f"Diagram preset not found: {preset_path}. Available presets: {', '.join(available)}"
        )

    spec = importlib.util.spec_from_file_location(f"brown_diagram_preset_{preset_file_stem}", preset_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load diagram preset: {preset_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, preset_path


def load_diagram_preset(preset_file_stem):
    module, preset_path = load_diagram_preset_module(preset_file_stem)
    if getattr(module, "BATCH_RENDER_ALL_PRESETS", False):
        required_batch_keys = (
            "PRESET_NAME",
            "BATCH_INPUT_H5_FILENAME",
            "BATCH_SNAPSHOT_INDEX",
            "BATCH_OUTPUT_FOLDER",
            "BATCH_SAVE_DPI",
            "BATCH_MANIFEST_FILENAME",
            "BATCH_PRESET_ORDER",
        )
        missing = [key for key in required_batch_keys if not hasattr(module, key)]
        if missing:
            raise RuntimeError(
                f"Batch diagram preset {preset_path} is missing required settings: "
                f"{', '.join(missing)}"
            )
        return {
            "PRESET_NAME": module.PRESET_NAME,
            "BATCH_RENDER_ALL_PRESETS": True,
            "BATCH_INPUT_H5_FILENAME": module.BATCH_INPUT_H5_FILENAME,
            "BATCH_SNAPSHOT_INDEX": module.BATCH_SNAPSHOT_INDEX,
            "BATCH_OUTPUT_FOLDER": module.BATCH_OUTPUT_FOLDER,
            "BATCH_SAVE_DPI": module.BATCH_SAVE_DPI,
            "BATCH_MANIFEST_FILENAME": module.BATCH_MANIFEST_FILENAME,
            "BATCH_PRESET_ORDER": tuple(module.BATCH_PRESET_ORDER),
        }

    missing = [key for key in REQUIRED_DIAGRAM_PRESET_KEYS if not hasattr(module, key)]
    if missing:
        raise RuntimeError(f"Diagram preset {preset_path} is missing required settings: {', '.join(missing)}")

    preset = dict(OPTIONAL_DIAGRAM_PRESET_DEFAULTS)
    preset.update({key: getattr(module, key) for key in REQUIRED_DIAGRAM_PRESET_KEYS})
    for key in OPTIONAL_DIAGRAM_PRESET_DEFAULTS:
        if hasattr(module, key):
            preset[key] = getattr(module, key)
    return preset


def apply_diagram_preset(preset_file_stem):
    preset = load_diagram_preset(preset_file_stem)
    globals()["BATCH_RENDER_ALL_PRESETS"] = bool(
        preset.get("BATCH_RENDER_ALL_PRESETS", False)
    )
    for key, value in preset.items():
        globals()[key] = value


apply_diagram_preset(DIAGRAM_PRESET)


def resolve_h5_path():
    path = Path(INPUT_H5_FILENAME)
    if path.is_absolute():
        return path
    return in_results(INPUT_H5_FILENAME)


def resolve_output_dir():
    output_dir = results_dir() / OUTPUT_FOLDER if OUTPUT_FOLDER else results_dir()
    if not output_dir.exists():
        raise FileNotFoundError(f"Expected output directory does not exist: {output_dir}")
    return output_dir


def normalize_frame_index(frame_index, frame_count):
    normalized_index = frame_index
    if frame_index < 0:
        normalized_index = frame_count + frame_index

    if normalized_index < 0 or normalized_index >= frame_count:
        raise IndexError(
            f"SNAPSHOT_INDEX {frame_index} is outside the available HDF5 frame range. "
            f"Use -{frame_count}..-1 or 0..{frame_count - 1}."
        )

    return normalized_index


def load_snapshot_and_context(h5_path, requested_frame_index):
    metadata = load_brown_config_json(h5_path, required=True)

    with h5py.File(h5_path, "r") as hf:
        if "snapshots" not in hf:
            raise RuntimeError(f"No 'snapshots' dataset found in {h5_path}")

        snapshots = hf["snapshots"]
        saved_steps = read_saved_steps(hf)
        frame_count = snapshots.shape[0]
        frame_index = normalize_frame_index(requested_frame_index, frame_count)
        matrix = snapshots[frame_index]
        saved_step = int(saved_steps[frame_index])

    transport_analysis = (
        analyze_transport(h5_path)
        if SHOW_NET_FLUX_PANEL
        else None
    )

    print(f"Loaded: {h5_path}")
    print(f"Available saved frames: {frame_count} (valid indices: -{frame_count}..-1 or 0..{frame_count - 1})")
    print(f"Saved steps: {int(saved_steps[0])} -> {int(saved_steps[-1])}")
    print(f"Matrix shape: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
    print("Metadata: found")
    print(f"Plotting saved-frame index {frame_index}, simulation step {saved_step}")

    return matrix, saved_step, frame_index, metadata, transport_analysis


def concentration_percent(matrix, mask):
    region = matrix[mask]
    available_or_full = region > 0
    denominator = int(np.sum(available_or_full))
    if denominator == 0:
        return None
    numerator = int(np.sum(region == 2))
    return 100 * numerator / denominator


def apply_l_fraction_ticks(axis, x_length=None, y_length=None):
    fractions = [0, 0.25, 0.5, 0.75, 1.0]

    if x_length is not None:
        x_ticks = [fraction * (x_length - 1) for fraction in fractions]
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(["0", "L/4", "L/2", "3L/4", "L"])

    if y_length is not None:
        y_ticks = [fraction * (y_length - 1) for fraction in fractions]
        axis.set_yticks(y_ticks)
        axis.set_yticklabels(["0", "H/4", "H/2", "3H/4", "H"])


def dimension_label_with_pixels(label, pixel_count):
    return f"{label} ({int(pixel_count)} px)"


def clamp_range(range_value, upper_bound, label):
    if range_value is None:
        return 0, upper_bound

    start, end = int(range_value[0]), int(range_value[1])
    start = max(0, min(upper_bound, start))
    end = max(0, min(upper_bound, end))
    if end <= start:
        raise ValueError(f"Invalid {label} range: {range_value}")
    return start, end


def smooth_profile(profile):
    if len(profile) == 0:
        return profile

    window = max(1, int(PROFILE_SMOOTHING_WINDOW))
    if window > 1 and len(profile) >= window:
        padded = np.pad(profile, (window, window), mode="edge")
        smoothed = np.convolve(padded, np.ones(window) / window, mode="same")[window:-window]
    else:
        smoothed = profile

    sigma = float(PROFILE_GAUSSIAN_SIGMA)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(smoothed, sigma=sigma, mode="reflect")

    return smoothed


def bin_profile(coordinates, total_spots, filled_spots):
    bin_size = max(1, int(PROFILE_BIN_SIZE))
    if bin_size == 1:
        return coordinates, total_spots, filled_spots

    binned_coordinates = []
    binned_total = []
    binned_filled = []

    for start in range(0, len(coordinates), bin_size):
        end = min(start + bin_size, len(coordinates))
        binned_coordinates.append(float(np.mean(coordinates[start:end])))
        binned_total.append(int(np.sum(total_spots[start:end])))
        binned_filled.append(int(np.sum(filled_spots[start:end])))

    return (
        np.asarray(binned_coordinates, dtype=float),
        np.asarray(binned_total, dtype=float),
        np.asarray(binned_filled, dtype=float),
    )


def compute_profile(matrix, metadata):
    rows, cols = matrix.shape
    x_start, x_end = clamp_range(PROFILE_X_RANGE, cols, "PROFILE_X_RANGE")
    y_start, y_end = clamp_range(PROFILE_Y_RANGE, rows, "PROFILE_Y_RANGE")
    cropped = matrix[y_start:y_end, x_start:x_end]
    included = np.ones(matrix.shape, dtype=bool)
    spot_mask = get_spot_mask(matrix.shape, metadata)
    if spot_mask is not None:
        included &= ~spot_mask
    cropped_included = included[y_start:y_end, x_start:x_end]

    if PROFILE_AXIS == "x":
        total_spots = np.sum((cropped > 0) & cropped_included, axis=0)
        filled_spots = np.sum((cropped == 2) & cropped_included, axis=0)
        coordinates = np.arange(x_start, x_end)
        axis_label = "Length"
    elif PROFILE_AXIS == "y":
        total_spots = np.sum((cropped > 0) & cropped_included, axis=1)
        filled_spots = np.sum((cropped == 2) & cropped_included, axis=1)
        coordinates = np.arange(y_start, y_end)
        axis_label = "Height"
    else:
        raise ValueError("PROFILE_AXIS must be 'x' or 'y'")

    coordinates, total_spots, filled_spots = bin_profile(coordinates, total_spots, filled_spots)
    profile = np.zeros_like(filled_spots, dtype=float)
    mask = total_spots > 0
    profile[mask] = filled_spots[mask] / total_spots[mask]

    return coordinates, smooth_profile(profile), axis_label


def rectangle_mask(shape, x_start, x_end, y_start=None, y_end=None):
    rows, cols = shape
    y_start = 0 if y_start is None else y_start
    y_end = rows if y_end is None else y_end
    x_start = max(0, min(cols, int(x_start)))
    x_end = max(0, min(cols, int(x_end)))
    y_start = max(0, min(rows, int(y_start)))
    y_end = max(0, min(rows, int(y_end)))

    mask = np.zeros(shape, dtype=bool)
    if x_end > x_start and y_end > y_start:
        mask[y_start:y_end, x_start:x_end] = True
    return mask


def metadata_bool(metadata, key):
    return bool(metadata[key])


def metadata_int(metadata, key):
    return int(metadata[key])


def metadata_str(metadata, key):
    return str(metadata[key])


def value_to_percent(value):
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return 100 * float(numerator) / float(denominator)
    return 100 * float(value)


def get_spot_settings(metadata):
    use_spot = metadata_bool(metadata, "USE_SPOT")
    if not use_spot:
        return None
    return {
        "center_x": metadata_int(metadata, "SPOT_CENTER_X"),
        "center_y": metadata_int(metadata, "SPOT_CENTER_Y"),
        "diameter": metadata_int(metadata, "SPOT_DIAMETER"),
    }


def get_spot_mask(matrix_shape, metadata):
    spot_settings = get_spot_settings(metadata)
    if spot_settings is None:
        return None
    return create_spot_mask(
        matrix_shape,
        center_x=spot_settings["center_x"],
        center_y=spot_settings["center_y"],
        diameter=spot_settings["diameter"],
    )


def get_trap_layer_settings(matrix_shape, metadata):
    if not metadata_bool(metadata, "USE_TRAP_LAYER"):
        return None
    _, cols = matrix_shape
    return {
        "center_x": int(metadata.get("TRAP_LAYER_CENTER_X", cols // 2)),
        "width": metadata_int(metadata, "TRAP_LAYER_WIDTH"),
        "max_solubility": value_to_percent(
            metadata.get("max_sol_trap_layer", 1)
        ),
    }


def get_trap_layer_mask(matrix_shape, metadata):
    trap_settings = get_trap_layer_settings(matrix_shape, metadata)
    if trap_settings is None:
        return None
    return create_trap_layer_mask(
        matrix_shape,
        width=trap_settings["width"],
        center_x=trap_settings["center_x"],
    )


def build_visualization_area_masks(matrix_shape, metadata):
    """Return non-overlapping A/B/special-area masks using simulation precedence."""
    rows, cols = matrix_shape
    mid_x = cols // 2
    area_a = rectangle_mask(matrix_shape, 0, mid_x)
    area_b = rectangle_mask(matrix_shape, mid_x, cols)
    areas = {"Area A": area_a, "Area B": area_b}

    trap_mask = get_trap_layer_mask(matrix_shape, metadata)
    if trap_mask is not None:
        areas["Area A"] &= ~trap_mask
        areas["Area B"] &= ~trap_mask
        areas["Trap layer"] = trap_mask.copy()

    spot_mask = get_spot_mask(matrix_shape, metadata)
    if spot_mask is not None:
        for mask in areas.values():
            mask &= ~spot_mask
        areas["Spot"] = spot_mask.copy()

    return {name: mask for name, mask in areas.items() if np.any(mask)}


def get_special_region_mask(matrix_shape, metadata):
    special_mask = np.zeros(matrix_shape, dtype=bool)
    spot_mask = get_spot_mask(matrix_shape, metadata)
    trap_mask = get_trap_layer_mask(matrix_shape, metadata)
    if trap_mask is not None:
        special_mask |= trap_mask
    if spot_mask is not None:
        special_mask |= spot_mask
    return special_mask


def draw_special_region_backgrounds(axis, metadata):
    """Draw optional region fills below dots and other simulation data."""
    if not SHOW_SPOT_REGION_FILL:
        return

    spot_settings = get_spot_settings(metadata)
    if spot_settings is not None:
        axis.add_patch(Circle(
            (spot_settings["center_x"], spot_settings["center_y"]),
            spot_settings["diameter"] / 2,
            facecolor=SPOT_REGION_FILL_COLOR,
            edgecolor="none",
            alpha=SPOT_REGION_FILL_ALPHA,
            zorder=0.5,
        ))


def draw_special_region_outlines(axis, matrix_shape, metadata):
    if not SHOW_SPECIAL_REGION_OUTLINES:
        return

    spot_settings = get_spot_settings(metadata)
    if spot_settings is not None:
        axis.add_patch(Circle(
            (spot_settings["center_x"], spot_settings["center_y"]),
            spot_settings["diameter"] / 2,
            fill=False,
            edgecolor=SPOT_REGION_OUTLINE_COLOR,
            linewidth=SPOT_REGION_OUTLINE_WIDTH,
            linestyle=SPOT_REGION_OUTLINE_STYLE,
            zorder=8,
        ))

    trap_settings = get_trap_layer_settings(matrix_shape, metadata)
    if trap_settings is not None:
        rows, _ = matrix_shape
        left = trap_settings["center_x"] - trap_settings["width"] / 2
        axis.add_patch(Rectangle(
            (left, -0.5),
            trap_settings["width"],
            rows,
            fill=False,
            edgecolor=SPECIAL_REGION_OUTLINE_COLOR,
            linewidth=SPECIAL_REGION_OUTLINE_WIDTH,
            linestyle=SPECIAL_REGION_OUTLINE_STYLE,
            zorder=8,
        ))


def compute_smoothed_occupancy_field(matrix, metadata):
    """Smooth H/site counts, optionally without bleeding across material boundaries."""
    from scipy.ndimage import gaussian_filter

    sigma = max(0.0, float(HEATMAP_SIGMA))
    hydrogen = (matrix == 2).astype(np.float32)
    active = (matrix > 0).astype(np.float32)
    occupancy = np.full(matrix.shape, np.nan, dtype=np.float32)

    if HEATMAP_RESPECT_AREA_BOUNDARIES:
        areas = build_visualization_area_masks(matrix.shape, metadata)
        if HEATMAP_SEPARATE_BASE_AREAS:
            area_masks = list(areas.values())
        else:
            bulk_mask = np.zeros(matrix.shape, dtype=bool)
            area_masks = []
            for name, area_mask in areas.items():
                if name in {"Area A", "Area B"}:
                    bulk_mask |= area_mask
                else:
                    area_masks.append(area_mask)
            if np.any(bulk_mask):
                area_masks.insert(0, bulk_mask)
    else:
        area_masks = [np.ones(matrix.shape, dtype=bool)]

    for area_mask in area_masks:
        masked_hydrogen = hydrogen * area_mask
        masked_active = active * area_mask
        if sigma > 0:
            local_hydrogen = gaussian_filter(masked_hydrogen, sigma=sigma, mode="reflect")
            local_active = gaussian_filter(masked_active, sigma=sigma, mode="reflect")
        else:
            local_hydrogen = masked_hydrogen
            local_active = masked_active

        valid = area_mask & (local_active > 1e-7)
        occupancy[valid] = local_hydrogen[valid] / local_active[valid]

    return occupancy


def compute_bulk_reference_field(matrix, metadata):
    """Create a per-pixel bulk reference without letting spot/trap values define it."""
    rows, cols = matrix.shape
    mid_x = cols // 2
    special_mask = get_special_region_mask(matrix.shape, metadata)
    active = matrix > 0
    hydrogen = matrix == 2
    bulk_mask = active & ~special_mask

    global_count = int(np.sum(bulk_mask))
    global_reference = (
        float(np.sum(hydrogen & bulk_mask)) / global_count
        if global_count
        else float(np.sum(hydrogen)) / max(1, int(np.sum(active)))
    )
    reference = np.full(matrix.shape, global_reference, dtype=np.float32)

    if HEATMAP_REFERENCE_MODE == "global_bulk":
        return reference, {"Bulk": global_reference}
    if HEATMAP_REFERENCE_MODE != "regional_bulk":
        raise ValueError("HEATMAP_REFERENCE_MODE must be 'regional_bulk' or 'global_bulk'")

    references = {}
    for name, x_start, x_end in [("Area A", 0, mid_x), ("Area B", mid_x, cols)]:
        region_mask = bulk_mask[:, x_start:x_end]
        denominator = int(np.sum(region_mask))
        value = (
            float(np.sum(hydrogen[:, x_start:x_end] & region_mask)) / denominator
            if denominator
            else global_reference
        )
        reference[:, x_start:x_end] = value
        references[name] = value

    return reference, references


def draw_concentration_heatmap(axis, matrix, metadata):
    occupancy = compute_smoothed_occupancy_field(matrix, metadata)
    reference, references = compute_bulk_reference_field(matrix, metadata)

    if HEATMAP_MODE == "deviation":
        displayed = (occupancy - reference) * 100
        limit = max(0.1, float(HEATMAP_DEVIATION_LIMIT))
        vmin, vmax = -limit, limit
        colorbar_label = "Local occupancy minus bulk reference (percentage points)"
    elif HEATMAP_MODE == "occupancy":
        displayed = occupancy * 100
        vmin, vmax = map(float, HEATMAP_OCCUPANCY_RANGE)
        colorbar_label = "Smoothed local H occupancy (%)"
    else:
        raise ValueError("HEATMAP_MODE must be 'deviation' or 'occupancy'")

    colormap = plt.get_cmap(HEATMAP_COLORMAP).copy()
    colormap.set_bad(COLOR_EMPTY)
    image = axis.imshow(
        np.ma.masked_invalid(displayed),
        origin="lower",
        interpolation="bilinear",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )

    if HEATMAP_SHOW_CONTOURS:
        levels = [
            float(level)
            for level in HEATMAP_CONTOUR_LEVELS
            if vmin < float(level) < vmax
        ]
        if levels:
            axis.contour(
                displayed,
                levels=levels,
                colors=HEATMAP_CONTOUR_COLOR,
                alpha=HEATMAP_CONTOUR_ALPHA,
                linewidths=0.55,
                origin="lower",
            )

    if HEATMAP_SHOW_COLORBAR:
        colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
        colorbar.set_label(colorbar_label)

    reference_text = ", ".join(
        f"{name}: {value * 100:.1f}%"
        for name, value in references.items()
    )
    axis.text(
        0.01,
        0.02,
        f"Reference occupancy — {reference_text}",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#202020",
        bbox={"facecolor": "#FFFFFF", "edgecolor": "#707070", "alpha": 0.88, "pad": 3},
        zorder=9,
    )


def draw_printer_glyphs(axis, matrix, metadata):
    """Draw non-overlapping binned circles: sector=occupancy, size=site capacity."""
    rows, cols = matrix.shape
    bin_size = max(2, int(GLYPH_BIN_SIZE))
    radius_floor = float(np.clip(GLYPH_MIN_RADIUS_FRACTION, 0.0, 1.0))
    radius_ceiling = float(np.clip(GLYPH_MAX_RADIUS_FRACTION, radius_floor, 1.0))
    capacity_gamma = max(0.01, float(GLYPH_CAPACITY_GAMMA))
    axis.set_facecolor(GLYPH_BACKGROUND_COLOR)

    # Distribute remainder cells over all bins instead of creating tiny edge bins.
    x_bin_count = max(1, int(round(cols / bin_size)))
    y_bin_count = max(1, int(round(rows / bin_size)))
    x_edges = np.rint(np.linspace(0, cols, x_bin_count + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, rows, y_bin_count + 1)).astype(int)

    for y_start, y_end in zip(y_edges[:-1], y_edges[1:]):
        for x_start, x_end in zip(x_edges[:-1], x_edges[1:]):
            block = matrix[y_start:y_end, x_start:x_end]
            active_count = int(np.sum(block > 0))
            if active_count == 0:
                continue

            area = block.size
            capacity = active_count / area
            occupancy = float(np.sum(block == 2)) / active_count
            compressed_capacity = capacity ** capacity_gamma
            radius_scale = radius_floor + (
                radius_ceiling - radius_floor
            ) * compressed_capacity
            radius = 0.5 * min(x_end - x_start, y_end - y_start) * radius_scale
            center = ((x_start + x_end - 1) / 2, (y_start + y_end - 1) / 2)

            axis.add_patch(Circle(
                center,
                radius,
                facecolor=COLOR_AVAILABLE_SPOT,
                edgecolor="none",
                alpha=DOT_ALPHA_AVAILABLE,
                zorder=2,
            ))
            if occupancy > 0:
                axis.add_patch(Wedge(
                    center,
                    radius,
                    90,
                    90 + 360 * occupancy,
                    facecolor=COLOR_HYDROGEN,
                    edgecolor="none",
                    alpha=DOT_ALPHA_HYDROGEN,
                    zorder=3,
                ))
            axis.add_patch(Circle(
                center,
                radius,
                fill=False,
                edgecolor=GLYPH_EDGE_COLOR,
                linewidth=GLYPH_EDGE_WIDTH,
                zorder=4,
            ))

    if GLYPH_SHOW_GRID:
        for x_position in x_edges:
            axis.axvline(x_position - 0.5, color=GLYPH_GRID_COLOR, linewidth=0.25, zorder=1)
        for y_position in y_edges:
            axis.axhline(y_position - 0.5, color=GLYPH_GRID_COLOR, linewidth=0.25, zorder=1)

    if GLYPH_SHOW_EXPLANATION:
        explanation_lines = [
            (
                f"Each circle ≈ one {bin_size}×{bin_size}-cell bin   •   "
                "red sector = H occupancy"
            ),
            "circle size = relative available-site density",
        ]
        if GLYPH_SHOW_SPECIAL_REGION_VALUES:
            region_values = []
            spot_mask = get_spot_mask(matrix.shape, metadata)
            if spot_mask is not None:
                spot_concentration = concentration_percent(matrix, spot_mask)
                if spot_concentration is not None:
                    region_values.append(
                        f"dashed spot = {spot_concentration:.1f}% H"
                    )

            trap_mask = get_trap_layer_mask(matrix.shape, metadata)
            if trap_mask is not None:
                if spot_mask is not None:
                    trap_mask = trap_mask & ~spot_mask
                trap_concentration = concentration_percent(matrix, trap_mask)
                if trap_concentration is not None:
                    region_values.append(
                        f"dashed trap layer = {trap_concentration:.1f}% H"
                    )
            if region_values:
                explanation_lines[-1] += "   •   " + "   •   ".join(region_values)

        axis.text(
            0.01,
            0.02,
            "\n".join(explanation_lines),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#202020",
            bbox={"facecolor": "#FFFFFF", "edgecolor": "#707070", "alpha": 0.92, "pad": 3},
            zorder=9,
        )


def get_source_sink_ranges(matrix_shape, metadata):
    """Return source and sink x ranges, or None when the boundary mode is off."""
    if not metadata_bool(metadata, "USE_SINK_SOURCE"):
        return None

    _, cols = matrix_shape
    thickness = int(np.clip(
        metadata_int(metadata, "SINK_SOURCE_THICKNESS"),
        0,
        cols,
    ))
    width_scale = float(AREA_SUMMARY_SOURCE_SINK_BAND_WIDTH_SCALE)
    if not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError(
            "AREA_SUMMARY_SOURCE_SINK_BAND_WIDTH_SCALE must be a positive number"
        )
    thickness = int(np.clip(round(thickness * width_scale), 0, cols // 2))
    if thickness == 0:
        return None

    left_range = (0, thickness)
    right_range = (cols - thickness, cols)
    if metadata_str(metadata, "SOURCE_SIDE") == "left":
        return {"Source": left_range, "Sink": right_range}
    return {"Source": right_range, "Sink": left_range}


def build_stylized_area_masks(matrix_shape, metadata):
    """Return display masks, excluding solid source/sink bands when requested."""
    masks = build_visualization_area_masks(matrix_shape, metadata)
    source_sink_ranges = (
        get_source_sink_ranges(matrix_shape, metadata)
        if AREA_SUMMARY_SHOW_SOURCE_SINK_BANDS
        else None
    )
    if source_sink_ranges is None:
        return masks, None

    boundary_mask = np.zeros(matrix_shape, dtype=bool)
    for x_start, x_end in source_sink_ranges.values():
        boundary_mask |= rectangle_mask(matrix_shape, x_start, x_end)
    for mask in masks.values():
        mask &= ~boundary_mask

    return {
        name: mask
        for name, mask in masks.items()
        if np.any(mask)
    }, source_sink_ranges


def allocate_stylized_dot_counts(weights):
    """Allocate a fixed visual dot budget while retaining small visible areas."""
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights > 0
    allocation = np.zeros(len(weights), dtype=np.int64)
    if not np.any(positive):
        return allocation

    minimum = max(0, int(AREA_SUMMARY_MIN_DOTS_PER_AREA))
    total = max(
        int(AREA_SUMMARY_TOTAL_DOTS),
        minimum * int(np.sum(positive)),
    )
    allocation[positive] = minimum
    remaining = total - int(np.sum(allocation))
    if remaining <= 0:
        return allocation

    normalized = weights[positive] / np.sum(weights[positive])
    raw_extra = normalized * remaining
    extra = np.floor(raw_extra).astype(np.int64)
    allocation[positive] += extra
    unassigned = remaining - int(np.sum(extra))
    if unassigned:
        positive_indices = np.flatnonzero(positive)
        order = np.argsort(-(raw_extra - extra))
        allocation[positive_indices[order[:unassigned]]] += 1
    return allocation


def generate_hex_candidates(mask, spacing, phase_x, phase_y):
    """Generate one staggered lattice and retain points whose centres lie in mask."""
    row_indices = np.flatnonzero(np.any(mask, axis=1))
    column_indices = np.flatnonzero(np.any(mask, axis=0))
    if not len(row_indices) or not len(column_indices):
        return np.empty(0), np.empty(0)

    y_min, y_max = row_indices[[0, -1]]
    x_min, x_max = column_indices[[0, -1]]
    vertical_spacing = spacing * np.sqrt(3) / 2
    global_y_origin = phase_y * vertical_spacing
    first_lattice_row = int(np.floor(
        (y_min - global_y_origin) / vertical_spacing
    )) - 1
    last_lattice_row = int(np.ceil(
        (y_max - global_y_origin) / vertical_spacing
    )) + 1
    lattice_rows = np.arange(first_lattice_row, last_lattice_row + 1)
    y_values = global_y_origin + lattice_rows * vertical_spacing

    x_parts = []
    y_parts = []
    for lattice_row, y_position in zip(lattice_rows, y_values):
        stagger = (lattice_row % 2) * spacing / 2
        global_x_origin = phase_x * spacing + stagger
        first_lattice_column = int(np.floor(
            (x_min - global_x_origin) / spacing
        )) - 1
        last_lattice_column = int(np.ceil(
            (x_max - global_x_origin) / spacing
        )) + 1
        lattice_columns = np.arange(
            first_lattice_column,
            last_lattice_column + 1,
        )
        x_values = global_x_origin + lattice_columns * spacing
        rounded_x = np.rint(x_values).astype(int)
        rounded_y = np.full(len(x_values), int(round(y_position)), dtype=int)
        in_bounds = (
            (rounded_x >= 0)
            & (rounded_x < mask.shape[1])
            & (rounded_y >= 0)
            & (rounded_y < mask.shape[0])
        )
        if not np.any(in_bounds):
            continue
        rounded_x = rounded_x[in_bounds]
        rounded_y = rounded_y[in_bounds]
        inside = mask[rounded_y, rounded_x]
        if np.any(inside):
            x_parts.append(x_values[in_bounds][inside])
            y_parts.append(np.full(np.sum(inside), y_position))

    if not x_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(x_parts), np.concatenate(y_parts)


def generate_even_hex_positions(
    mask,
    requested_count,
    rng,
    exclusion_positions=None,
    lattice_phase=None,
):
    """Fit an even staggered lattice, capping only when print spacing requires it."""
    requested_count = max(0, int(requested_count))
    if requested_count == 0 or not np.any(mask):
        return np.empty(0), np.empty(0)

    minimum_spacing = max(0.5, float(AREA_SUMMARY_MIN_DOT_SPACING))
    exclusion_tree = None
    if exclusion_positions is not None:
        exclusion_positions = np.asarray(exclusion_positions, dtype=np.float64)
        if len(exclusion_positions):
            from scipy.spatial import cKDTree
            exclusion_tree = cKDTree(exclusion_positions)
    mask_area = int(np.sum(mask))
    ideal_spacing = np.sqrt(
        2 * mask_area / (np.sqrt(3) * requested_count)
    )
    if lattice_phase is None:
        phase_x, phase_y = rng.random(2)
    else:
        lattice_phase = np.asarray(lattice_phase, dtype=np.float64)
        if lattice_phase.shape != (2,):
            raise ValueError("lattice_phase must contain exactly two values")
        phase_x, phase_y = lattice_phase

    def candidates(spacing):
        candidate_x, candidate_y = generate_hex_candidates(
            mask,
            spacing,
            phase_x,
            phase_y,
        )
        if exclusion_tree is not None and len(candidate_x):
            candidate_positions = np.column_stack([candidate_x, candidate_y])
            nearest_distance, _ = exclusion_tree.query(candidate_positions, k=1)
            keep = nearest_distance + 1e-9 >= minimum_spacing
            candidate_x = candidate_x[keep]
            candidate_y = candidate_y[keep]
        return candidate_x, candidate_y

    minimum_candidates = candidates(minimum_spacing)
    minimum_count = len(minimum_candidates[0])
    if minimum_count <= requested_count:
        # This is the area's non-overlapping visual capacity at print size.
        return minimum_candidates

    low_spacing = max(minimum_spacing, ideal_spacing)
    low_candidates = candidates(low_spacing)
    if len(low_candidates[0]) < requested_count:
        high_spacing = low_spacing
        low_spacing = minimum_spacing
        best_candidates = minimum_candidates
    else:
        best_candidates = low_candidates
        high_spacing = low_spacing * 1.25
        high_candidates = candidates(high_spacing)
        while len(high_candidates[0]) >= requested_count:
            low_spacing = high_spacing
            best_candidates = high_candidates
            high_spacing *= 1.25
            high_candidates = candidates(high_spacing)

    # Find the sparsest lattice that still supplies the requested count.
    for _ in range(18):
        middle_spacing = (low_spacing + high_spacing) / 2
        middle_candidates = candidates(middle_spacing)
        if len(middle_candidates[0]) >= requested_count:
            low_spacing = middle_spacing
            best_candidates = middle_candidates
        else:
            high_spacing = middle_spacing

    candidate_count = len(best_candidates[0])
    if candidate_count == requested_count:
        return best_candidates

    # Remove the small excess evenly across the row-major lattice.
    selected = np.floor(
        (np.arange(requested_count) + 0.5)
        * candidate_count
        / requested_count
    ).astype(int)
    return best_candidates[0][selected], best_candidates[1][selected]


def shake_profile_settings(shake_mode=None):
    profiles = {
        "none": (0.0, 0, 0.0),
        "gentle": (1.4, 1, 0.0),
        "organic": (2.8, 2, 0.0),
        "clustered": (4.2, 4, float(AREA_SUMMARY_CLUSTER_ATTRACTION)),
    }
    mode = str(AREA_SUMMARY_SHAKE_MODE if shake_mode is None else shake_mode)
    if mode not in profiles:
        raise ValueError(
            "AREA_SUMMARY_SHAKE_MODE must be 'none', 'gentle', "
            "'organic', or 'clustered'"
        )

    strength, passes, attraction = profiles[mode]
    if AREA_SUMMARY_SHAKE_STRENGTH is not None:
        strength = max(0.0, float(AREA_SUMMARY_SHAKE_STRENGTH))
    if AREA_SUMMARY_SHAKE_PASSES is not None:
        passes = max(0, int(AREA_SUMMARY_SHAKE_PASSES))
    return mode, strength, passes, attraction


def create_mask_cluster_centers(mask, rng):
    """Choose deterministic attraction centres across one combined geometry mask."""
    candidate_pixels = np.flatnonzero(mask)
    if len(candidate_pixels) == 0:
        return None

    cluster_count = int(np.clip(
        AREA_SUMMARY_CLUSTER_COUNT,
        1,
        len(candidate_pixels),
    ))
    selected_pixels = rng.choice(
        candidate_pixels,
        size=cluster_count,
        replace=False,
    )
    center_y, center_x = np.divmod(selected_pixels, mask.shape[1])
    return np.column_stack([
        center_x.astype(np.float64),
        center_y.astype(np.float64),
    ])


def shake_even_positions(
    x_coordinates,
    y_coordinates,
    mask,
    rng,
    shake_mode=None,
    cluster_centers=None,
    fixed_positions=None,
):
    """Disturb an even layout while preserving area ownership and hard spacing."""
    mode, strength, pass_count, attraction = shake_profile_settings(shake_mode)
    if mode == "none" or strength == 0 or pass_count == 0:
        return x_coordinates, y_coordinates

    movable_positions = np.column_stack([
        np.asarray(x_coordinates, dtype=np.float64),
        np.asarray(y_coordinates, dtype=np.float64),
    ])
    point_count = len(movable_positions)
    if point_count == 0:
        return movable_positions[:, 0], movable_positions[:, 1]

    if fixed_positions is None:
        fixed_positions = np.empty((0, 2), dtype=np.float64)
    else:
        fixed_positions = np.asarray(fixed_positions, dtype=np.float64)
        if fixed_positions.ndim != 2 or fixed_positions.shape[1] != 2:
            raise ValueError("fixed_positions must have shape (count, 2)")
    positions = np.vstack([movable_positions, fixed_positions])

    minimum_spacing = max(0.5, float(AREA_SUMMARY_MIN_DOT_SPACING))
    minimum_distance_squared = minimum_spacing ** 2
    cell_size = minimum_spacing
    attempt_count = max(1, int(AREA_SUMMARY_SHAKE_ATTEMPTS))

    def cell_for(position):
        return (
            int(np.floor(position[0] / cell_size)),
            int(np.floor(position[1] / cell_size)),
        )

    spatial_buckets = {}
    for point_index, position in enumerate(positions):
        spatial_buckets.setdefault(cell_for(position), set()).add(point_index)

    if attraction <= 0:
        cluster_centers = None
    elif cluster_centers is None:
        cluster_count = int(np.clip(
            AREA_SUMMARY_CLUSTER_COUNT,
            1,
            point_count,
        ))
        center_indices = rng.choice(
            point_count,
            size=cluster_count,
            replace=False,
        )
        cluster_centers = movable_positions[center_indices].copy()
    else:
        cluster_centers = np.asarray(cluster_centers, dtype=np.float64)
        if cluster_centers.ndim != 2 or cluster_centers.shape[1] != 2:
            raise ValueError("cluster_centers must have shape (count, 2)")

    for _ in range(pass_count):
        for point_index in rng.permutation(point_count):
            current = positions[point_index]
            current_cell = cell_for(current)

            for _ in range(attempt_count):
                angle = rng.uniform(0, 2 * np.pi)
                radius = strength * np.sqrt(rng.random())
                displacement = radius * np.array([
                    np.cos(angle),
                    np.sin(angle),
                ])

                if cluster_centers is not None:
                    differences = cluster_centers - current
                    nearest_center = differences[
                        np.argmin(np.sum(differences ** 2, axis=1))
                    ]
                    distance = np.linalg.norm(nearest_center)
                    if distance > 1e-12:
                        displacement = (
                            (1 - attraction) * displacement
                            + attraction
                            * strength
                            * rng.random()
                            * nearest_center
                            / distance
                        )

                candidate = current + displacement
                candidate_x = int(round(candidate[0]))
                candidate_y = int(round(candidate[1]))
                if (
                    candidate_x < 0
                    or candidate_x >= mask.shape[1]
                    or candidate_y < 0
                    or candidate_y >= mask.shape[0]
                    or not mask[candidate_y, candidate_x]
                ):
                    continue

                candidate_cell = cell_for(candidate)
                is_clear = True
                for x_offset in (-1, 0, 1):
                    for y_offset in (-1, 0, 1):
                        nearby = spatial_buckets.get((
                            candidate_cell[0] + x_offset,
                            candidate_cell[1] + y_offset,
                        ))
                        if not nearby:
                            continue
                        for other_index in nearby:
                            if other_index == point_index:
                                continue
                            difference = positions[other_index] - candidate
                            if np.dot(difference, difference) < minimum_distance_squared:
                                is_clear = False
                                break
                        if not is_clear:
                            break
                    if not is_clear:
                        break
                if not is_clear:
                    continue

                if candidate_cell != current_cell:
                    spatial_buckets[current_cell].remove(point_index)
                    if not spatial_buckets[current_cell]:
                        del spatial_buckets[current_cell]
                    spatial_buckets.setdefault(candidate_cell, set()).add(
                        point_index
                    )
                positions[point_index] = candidate
                break

    return positions[:point_count, 0], positions[:point_count, 1]


def draw_source_sink_bands(axis, matrix_shape, source_sink_ranges):
    if source_sink_ranges is None:
        return

    rows, _ = matrix_shape
    colors = {
        "Source": COLOR_HYDROGEN,
        "Sink": COLOR_AVAILABLE_SPOT,
    }
    for name, (x_start, x_end) in source_sink_ranges.items():
        axis.add_patch(Rectangle(
            (x_start - 0.5, -0.5),
            x_end - x_start,
            rows,
            facecolor=colors[name],
            edgecolor=AREA_SUMMARY_SOURCE_SINK_EDGE_COLOR,
            linewidth=AREA_SUMMARY_SOURCE_SINK_EDGE_WIDTH,
            zorder=2,
        ))
        if AREA_SUMMARY_SHOW_SOURCE_SINK_LABELS:
            axis.text(
                (x_start + x_end - 1) / 2,
                rows / 2,
                name,
                ha="center",
                va="center",
                rotation=90,
                fontsize=8,
                color="#FFFFFF",
                zorder=8,
            )


def stylized_area_label_position(name, mask, matrix_shape, metadata):
    rows, _ = matrix_shape
    columns = np.flatnonzero(np.any(mask, axis=0))
    center_x = float(np.mean(columns))
    if name in {"Area A", "Area B"}:
        return center_x, rows * 0.90
    if name == "Trap layer":
        return center_x, rows * 0.13
    if name == "Spot":
        spot = get_spot_settings(metadata)
        if spot is not None:
            return (
                spot["center_x"],
                spot["center_y"] + spot["diameter"] * 0.72,
            )
    return center_x, rows * 0.50


def compute_area_summary_base_x_profile(matrix, concentration_masks):
    """Return a smoothed x-profile for owned A/B sites without special-area bleed."""
    from scipy.ndimage import gaussian_filter1d

    base_mask = np.zeros(matrix.shape, dtype=bool)
    for name in ("Area A", "Area B"):
        if name in concentration_masks:
            base_mask |= concentration_masks[name]

    total_sites = np.sum((matrix > 0) & base_mask, axis=0).astype(np.float64)
    hydrogen_sites = np.sum((matrix == 2) & base_mask, axis=0).astype(np.float64)
    sigma = max(0.0, float(AREA_SUMMARY_X_PROFILE_SIGMA))
    if sigma > 0:
        total_sites = gaussian_filter1d(total_sites, sigma=sigma, mode="nearest")
        hydrogen_sites = gaussian_filter1d(
            hydrogen_sites,
            sigma=sigma,
            mode="nearest",
        )

    profile = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = total_sites > 1e-12
    profile[valid] = hydrogen_sites[valid] / total_sites[valid]
    if not np.any(valid):
        return np.zeros(matrix.shape[1], dtype=np.float64)

    # Fill columns hidden by a full-height special area from the nearest valid
    # bulk values, so the illustrative A/B gradient remains continuous.
    coordinates = np.arange(matrix.shape[1], dtype=np.float64)
    profile[~valid] = np.interp(
        coordinates[~valid],
        coordinates[valid],
        profile[valid],
    )
    return np.clip(profile, 0.0, 1.0)


def validate_area_summary_linear_concentration():
    values = AREA_SUMMARY_LINEAR_CONCENTRATION
    if not isinstance(values, (tuple, list, np.ndarray)) or len(values) != 2:
        raise ValueError(
            "AREA_SUMMARY_LINEAR_CONCENTRATION must contain "
            "(left_percent, right_percent)"
        )
    left_percent, right_percent = map(float, values)
    if (
        not np.isfinite(left_percent)
        or not np.isfinite(right_percent)
        or left_percent < 0
        or left_percent > 100
        or right_percent < 0
        or right_percent > 100
    ):
        raise ValueError(
            "AREA_SUMMARY_LINEAR_CONCENTRATION percentages must be between 0 and 100"
        )
    return left_percent / 100.0, right_percent / 100.0


def select_occupied_dots_by_x_bins(x_coordinates, probabilities, rng):
    """Assign an exact rounded red count per vertical slice, then mix it randomly."""
    x_coordinates = np.asarray(x_coordinates, dtype=np.float64)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        0.0,
        1.0,
    )
    if x_coordinates.shape != probabilities.shape:
        raise ValueError("x_coordinates and probabilities must have the same shape")

    bin_width = max(1.0, float(AREA_SUMMARY_CONCENTRATION_BIN_WIDTH))
    bin_ids = np.floor(x_coordinates / bin_width).astype(np.int64)
    selected = np.zeros(len(probabilities), dtype=bool)
    rounding_carry = 0.0

    for bin_id in np.unique(bin_ids):
        candidates = np.flatnonzero(bin_ids == bin_id)
        expected_count = float(np.sum(probabilities[candidates])) + rounding_carry
        occupied_count = int(np.floor(expected_count + 0.5))
        occupied_count = int(np.clip(occupied_count, 0, len(candidates)))
        rounding_carry = expected_count - occupied_count
        if occupied_count:
            occupied_indices = rng.choice(
                candidates,
                size=occupied_count,
                replace=False,
            )
            selected[occupied_indices] = True

    return np.flatnonzero(selected)


def draw_stylized_area_label(
    axis,
    name,
    position_mask,
    concentration_mask,
    matrix,
    metadata,
    concentration_override=None,
):
    concentration = (
        concentration_percent(matrix, concentration_mask)
        if concentration_override is None
        else float(concentration_override)
    )
    if concentration is None:
        text = f"{name}: n/a"
    else:
        text = f"{name}: {concentration:.1f}% H"
    position = stylized_area_label_position(
        name,
        position_mask,
        matrix.shape,
        metadata,
    )
    label = axis.text(
        position[0],
        position[1],
        text,
        ha="center",
        va="center",
        fontsize=ANNOTATION_FONT_SIZE,
        color=ANNOTATION_COLOR,
        bbox={
            "facecolor": ANNOTATION_BOX_FACE_COLOR,
            "edgecolor": ANNOTATION_BOX_EDGE_COLOR,
            "alpha": ANNOTATION_BOX_ALPHA,
            "pad": 2.5,
        },
        zorder=9,
    )
    label.set_path_effects([
        path_effects.withStroke(
            linewidth=ANNOTATION_STROKE_WIDTH,
            foreground=ANNOTATION_STROKE_COLOR,
        )
    ])


def prepare_combined_base_cluster_positions(
    masks,
    area_names,
    dot_counts,
    combined_position_mask,
    cluster_centers,
    base_seed,
    base_lattice_phase,
):
    """Shake A+B in one union-mask pass, then classify their final positions."""
    initial_parts = []
    initial_positions = np.empty((0, 2), dtype=np.float64)
    for area_index, (name, dot_count) in enumerate(zip(area_names, dot_counts)):
        if name not in {"Area A", "Area B"} or dot_count <= 0:
            continue

        position_mask = combined_position_mask & masks[name]
        position_rng = np.random.default_rng(np.random.SeedSequence([
            base_seed,
            area_index,
            0,
        ]))
        x_coordinates, y_coordinates = generate_even_hex_positions(
            position_mask,
            dot_count,
            position_rng,
            exclusion_positions=initial_positions,
            lattice_phase=base_lattice_phase,
        )
        area_positions = np.column_stack([x_coordinates, y_coordinates])
        if len(area_positions):
            initial_parts.append(area_positions)
            initial_positions = np.vstack([
                initial_positions,
                area_positions,
            ])

    if not initial_parts:
        return {}

    combined_positions = np.vstack(initial_parts)
    combined_rng = np.random.default_rng(np.random.SeedSequence([
        base_seed,
        536_870_911,
    ]))
    combined_x, combined_y = shake_even_positions(
        combined_positions[:, 0],
        combined_positions[:, 1],
        combined_position_mask,
        combined_rng,
        shake_mode="clustered",
        cluster_centers=cluster_centers,
    )

    midpoint = combined_position_mask.shape[1] / 2
    area_a = combined_x < midpoint
    return {
        "Area A": (combined_x[area_a], combined_y[area_a]),
        "Area B": (combined_x[~area_a], combined_y[~area_a]),
    }


def draw_area_summary_dots(axis, matrix, metadata, shake_mode=None):
    """Reconstruct area averages as a clean randomized dot field in matrix space."""
    from scipy.ndimage import binary_erosion

    rows, cols = matrix.shape
    axis.set_facecolor(AREA_SUMMARY_BACKGROUND_COLOR)
    masks, source_sink_ranges = build_stylized_area_masks(matrix.shape, metadata)
    concentration_masks = build_visualization_area_masks(matrix.shape, metadata)
    area_names = [
        name
        for name in ["Area A", "Area B", "Spot", "Trap layer"]
        if name in masks
    ]
    concentration_mode = str(AREA_SUMMARY_CONCENTRATION_MODE)
    if concentration_mode not in {"area_average", "saved_x_profile", "linear_x"}:
        raise ValueError(
            "AREA_SUMMARY_CONCENTRATION_MODE must be 'area_average', "
            "'saved_x_profile', or 'linear_x'"
        )
    base_x_probabilities = None
    if concentration_mode == "saved_x_profile":
        base_x_probabilities = compute_area_summary_base_x_profile(
            matrix,
            concentration_masks,
        )
    elif concentration_mode == "linear_x":
        left_probability, right_probability = (
            validate_area_summary_linear_concentration()
        )
        base_x_probabilities = np.linspace(
            left_probability,
            right_probability,
            cols,
            dtype=np.float64,
        )

    if AREA_SUMMARY_DENSITY_MODE == "available_sites":
        weights = [
            int(np.sum((matrix > 0) & masks[name]))
            for name in area_names
        ]
    elif AREA_SUMMARY_DENSITY_MODE == "uniform_area":
        weights = [int(np.sum(masks[name])) for name in area_names]
    else:
        raise ValueError(
            "AREA_SUMMARY_DENSITY_MODE must be 'available_sites' or 'uniform_area'"
        )
    dot_counts = allocate_stylized_dot_counts(weights)

    draw_source_sink_bands(axis, matrix.shape, source_sink_ranges)

    base_seed = int(AREA_SUMMARY_RANDOM_SEED)
    base_lattice_rng = np.random.default_rng(np.random.SeedSequence([
        base_seed,
        1_073_741_823,
    ]))
    base_lattice_phase = base_lattice_rng.random(2)
    jitter = float(np.clip(AREA_SUMMARY_POSITION_JITTER, 0.0, 0.49))
    resolved_shake_mode, _, _, _ = shake_profile_settings(shake_mode)
    cluster_scope = str(AREA_SUMMARY_CLUSTER_SCOPE)
    if cluster_scope not in {"per_area", "combined_a_b"}:
        raise ValueError(
            "AREA_SUMMARY_CLUSTER_SCOPE must be 'per_area' or 'combined_a_b'"
        )

    position_inset = max(0, int(AREA_SUMMARY_POSITION_INSET))
    combined_base_mask = np.zeros(matrix.shape, dtype=bool)
    for base_area_name in ("Area A", "Area B"):
        if base_area_name in masks:
            combined_base_mask |= masks[base_area_name]
    combined_base_position_mask = combined_base_mask
    if position_inset:
        inset_combined_mask = binary_erosion(
            combined_base_mask,
            iterations=position_inset,
            border_value=0,
        )
        if np.any(inset_combined_mask):
            combined_base_position_mask = inset_combined_mask

    shared_base_cluster_centers = None
    if resolved_shake_mode == "clustered" and cluster_scope == "combined_a_b":
        cluster_rng = np.random.default_rng(np.random.SeedSequence([
            base_seed,
            2_147_483_647,
        ]))
        shared_base_cluster_centers = create_mask_cluster_centers(
            combined_base_position_mask,
            cluster_rng,
        )

    combined_base_positions = {}
    if (
        resolved_shake_mode == "clustered"
        and cluster_scope == "combined_a_b"
        and AREA_SUMMARY_POSITION_MODE == "even_hex"
    ):
        combined_base_positions = prepare_combined_base_cluster_positions(
            masks,
            area_names,
            dot_counts,
            combined_base_position_mask,
            shared_base_cluster_centers,
            base_seed,
            base_lattice_phase,
        )

    placed_positions = np.empty((0, 2), dtype=np.float64)
    for area_index, (name, dot_count) in enumerate(zip(area_names, dot_counts)):
        if dot_count <= 0:
            continue

        mask = masks[name]
        if name in {"Area A", "Area B"}:
            position_mask = combined_base_position_mask & mask
        elif position_inset:
            inset_mask = binary_erosion(
                mask,
                iterations=position_inset,
                border_value=0,
            )
            position_mask = inset_mask if np.any(inset_mask) else mask
        else:
            position_mask = mask
        candidate_pixels = np.flatnonzero(position_mask)
        position_rng = np.random.default_rng(np.random.SeedSequence([
            base_seed,
            area_index,
            0,
        ]))
        color_rng = np.random.default_rng(np.random.SeedSequence([
            base_seed,
            area_index,
            1,
        ]))
        if name in combined_base_positions:
            x_coordinates, y_coordinates = combined_base_positions[name]
            dot_count = len(x_coordinates)
        elif AREA_SUMMARY_POSITION_MODE == "even_hex":
            x_coordinates, y_coordinates = generate_even_hex_positions(
                position_mask,
                dot_count,
                position_rng,
                exclusion_positions=placed_positions,
                lattice_phase=(
                    base_lattice_phase
                    if name in {"Area A", "Area B"}
                    else None
                ),
            )
            x_coordinates, y_coordinates = shake_even_positions(
                x_coordinates,
                y_coordinates,
                position_mask,
                position_rng,
                shake_mode=resolved_shake_mode,
                cluster_centers=(
                    shared_base_cluster_centers
                    if name in {"Area A", "Area B"}
                    else None
                ),
                fixed_positions=placed_positions,
            )
            dot_count = len(x_coordinates)
        elif AREA_SUMMARY_POSITION_MODE == "random":
            selected_pixels = position_rng.choice(
                candidate_pixels,
                size=int(dot_count),
                replace=dot_count > len(candidate_pixels),
            )
            y_coordinates, x_coordinates = np.divmod(selected_pixels, cols)
            x_coordinates = x_coordinates + position_rng.uniform(
                -jitter,
                jitter,
                size=dot_count,
            )
            y_coordinates = y_coordinates + position_rng.uniform(
                -jitter,
                jitter,
                size=dot_count,
            )
        else:
            raise ValueError(
                "AREA_SUMMARY_POSITION_MODE must be 'even_hex' or 'random'"
            )

        if dot_count == 0:
            continue

        placed_positions = np.vstack([
            placed_positions,
            np.column_stack([x_coordinates, y_coordinates]),
        ])
        concentration_mask = concentration_masks[name]
        colors = np.full(dot_count, COLOR_AVAILABLE_SPOT, dtype=object)
        displayed_concentration = None
        if name in {"Area A", "Area B"} and base_x_probabilities is not None:
            dot_probabilities = np.interp(
                x_coordinates,
                np.arange(cols, dtype=np.float64),
                base_x_probabilities,
            )
            occupied_indices = select_occupied_dots_by_x_bins(
                x_coordinates,
                dot_probabilities,
                color_rng,
            )
            displayed_concentration = 100.0 * len(occupied_indices) / dot_count
        else:
            concentration = concentration_percent(matrix, concentration_mask)
            occupied_dot_count = (
                int(np.clip(
                    np.floor(concentration / 100 * dot_count + 0.5),
                    0,
                    dot_count,
                ))
                if concentration is not None
                else 0
            )
            occupied_indices = (
                color_rng.choice(
                    dot_count,
                    size=occupied_dot_count,
                    replace=False,
                )
                if occupied_dot_count
                else np.empty(0, dtype=int)
            )
        if len(occupied_indices):
            colors[occupied_indices] = COLOR_HYDROGEN
        axis.scatter(
            x_coordinates,
            y_coordinates,
            s=AREA_SUMMARY_DOT_SIZE,
            c=colors,
            alpha=AREA_SUMMARY_DOT_ALPHA,
            marker="o",
            edgecolors=AREA_SUMMARY_DOT_EDGE_COLOR,
            linewidths=AREA_SUMMARY_DOT_EDGE_WIDTH,
            zorder=3,
        )

        if AREA_SUMMARY_SHOW_AREA_LABELS:
            draw_stylized_area_label(
                axis,
                name,
                mask,
                concentration_mask,
                matrix,
                metadata,
                concentration_override=displayed_concentration,
            )

    if AREA_SUMMARY_SHOW_HALF_DIVIDER:
        axis.axvline(
            cols / 2 - 0.5,
            color=AREA_SUMMARY_HALF_DIVIDER_COLOR,
            linewidth=0.7,
            alpha=0.65,
            zorder=4,
        )

    if SHOW_LEGEND and AREA_SUMMARY_SHOW_DOT_LEGEND:
        legend_handles = [
            axis.scatter(
                [],
                [],
                s=(
                    AREA_SUMMARY_DOT_SIZE
                    * AREA_SUMMARY_DOT_LEGEND_SIZE_SCALE
                ),
                c=COLOR_AVAILABLE_SPOT,
                alpha=AREA_SUMMARY_DOT_ALPHA,
                marker="o",
                edgecolors=AREA_SUMMARY_DOT_EDGE_COLOR,
                linewidths=AREA_SUMMARY_DOT_EDGE_WIDTH,
                label=AREA_SUMMARY_VACANT_LABEL,
            ),
            axis.scatter(
                [],
                [],
                s=(
                    AREA_SUMMARY_DOT_SIZE
                    * AREA_SUMMARY_DOT_LEGEND_SIZE_SCALE
                ),
                c=COLOR_HYDROGEN,
                alpha=AREA_SUMMARY_DOT_ALPHA,
                marker="o",
                edgecolors=AREA_SUMMARY_DOT_EDGE_COLOR,
                linewidths=AREA_SUMMARY_DOT_EDGE_WIDTH,
                label=AREA_SUMMARY_OCCUPIED_LABEL,
            ),
        ]
        legend = axis.legend(
            handles=legend_handles,
            loc=AREA_SUMMARY_DOT_LEGEND_LOCATION,
            bbox_to_anchor=AREA_SUMMARY_DOT_LEGEND_ANCHOR,
            fontsize=AREA_SUMMARY_DOT_LEGEND_FONT_SIZE,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#707070",
            framealpha=0.90,
        )
        legend.set_zorder(10)

    if AREA_SUMMARY_SHOW_EXPLANATION:
        density_text = (
            "dot density = measured available-site density"
            if AREA_SUMMARY_DENSITY_MODE == "available_sites"
            else "dot density = uniform display density"
        )
        concentration_text = {
            "area_average": "red fraction = measured area-average H occupancy",
            "saved_x_profile": "red probability = measured saved x-profile",
            "linear_x": "red probability = configured linear x-profile",
        }[concentration_mode]
        axis.text(
            0.01,
            0.02,
            (
                "Synthetic positions for clear printing   •   "
                f"{concentration_text}   •   "
                f"{density_text}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#202020",
            bbox={
                "facecolor": "#FFFFFF",
                "edgecolor": "#707070",
                "alpha": 0.92,
                "pad": 3,
            },
            zorder=9,
        )


def max_solubility_for_x(metadata, x_position, matrix_width):
    key = "max_sol_a" if x_position < matrix_width / 2 else "max_sol_b"
    return value_to_percent(metadata[key])


def build_annotation_regions(matrix_shape, metadata):
    rows, cols = matrix_shape
    mid_x = cols // 2
    regions = []
    use_sink_source = metadata_bool(metadata, "USE_SINK_SOURCE")
    sink_source_thickness = metadata_int(metadata, "SINK_SOURCE_THICKNESS")
    source_side = metadata_str(metadata, "SOURCE_SIDE")
    bulk_left_x = sink_source_thickness if use_sink_source else 0
    bulk_right_x = cols - sink_source_thickness if use_sink_source else cols

    if SHOW_LEFT_RIGHT_ANNOTATIONS:
        regions.extend([
            {
                "name": "Average Regional Concentration",
                "mask": rectangle_mask(matrix_shape, bulk_left_x, mid_x),
                "xy": (cols * 0.25, rows * 0.9),
                "max_solubility": max_solubility_for_x(metadata, cols * 0.25, cols),
            },
            {
                "name": "Average Regional Concentration",
                "mask": rectangle_mask(matrix_shape, mid_x, bulk_right_x),
                "xy": (cols * 0.75, rows * 0.9),
                "max_solubility": max_solubility_for_x(metadata, cols * 0.75, cols),
            },
        ])

    if use_sink_source and sink_source_thickness > 0:
        if SHOW_LEFT_RIGHT_WITHOUT_SINK_SOURCE_ANNOTATIONS:
            regions.extend([
                {
                    "name": "Average Regional Concentration",
                    "mask": rectangle_mask(matrix_shape, sink_source_thickness, mid_x),
                    "xy": (cols * 0.25, rows * 0.78),
                    "max_solubility": max_solubility_for_x(metadata, cols * 0.25, cols),
                },
                {
                    "name": "Average Regional Concentration",
                    "mask": rectangle_mask(matrix_shape, mid_x, cols - sink_source_thickness),
                    "xy": (cols * 0.75, rows * 0.78),
                    "max_solubility": max_solubility_for_x(metadata, cols * 0.75, cols),
                },
            ])

        if SHOW_SOURCE_SINK_ANNOTATIONS:
            if source_side == "left":
                source_x = (0, sink_source_thickness)
                sink_x = (cols - sink_source_thickness, cols)
            else:
                source_x = (cols - sink_source_thickness, cols)
                sink_x = (0, sink_source_thickness)

            regions.extend([
                {
                    "name": "Average Regional Concentration",
                    "mask": rectangle_mask(matrix_shape, source_x[0], source_x[1]),
                    "xy": (sum(source_x) / 2, rows * 0.5),
                },
                {
                    "name": "Average Regional Concentration",
                    "mask": rectangle_mask(matrix_shape, sink_x[0], sink_x[1]),
                    "xy": (sum(sink_x) / 2, rows * 0.65),
                },
            ])
    for region in CUSTOM_RECT_REGIONS:
        regions.append({
            "name": region["name"],
            "mask": rectangle_mask(
                matrix_shape,
                region["x_start"],
                region["x_end"],
                region.get("y_start"),
                region.get("y_end"),
            ),
            "xy": (
                (region["x_start"] + region["x_end"]) / 2,
                (region.get("y_start", 0) + region.get("y_end", rows)) / 2,
            ),
        })

    spot_settings = get_spot_settings(metadata)
    spot_mask = get_spot_mask(matrix_shape, metadata)
    trap_settings = get_trap_layer_settings(matrix_shape, metadata)
    trap_mask = get_trap_layer_mask(matrix_shape, metadata)

    excluded_mask = np.zeros(matrix_shape, dtype=bool)
    if trap_mask is not None:
        excluded_mask |= trap_mask
    if spot_mask is not None:
        excluded_mask |= spot_mask
    if np.any(excluded_mask):
        for region in regions:
            region["mask"] &= ~excluded_mask

    if trap_mask is not None and spot_mask is not None:
        trap_mask = trap_mask & ~spot_mask

    if (
        SHOW_TRAP_LAYER_ANNOTATION
        and trap_settings is not None
        and np.any(trap_mask)
    ):
        trap_columns = np.flatnonzero(np.any(trap_mask, axis=0))
        regions.append({
            "name": "Trap Layer Concentration",
            "mask": trap_mask,
            "xy": (float(np.mean(trap_columns)), rows * 0.15),
            "max_solubility": trap_settings["max_solubility"],
        })

    if SHOW_SPOT_ANNOTATION and spot_settings is not None:
        regions.append({
            "name": "Spot Concentration",
            "mask": spot_mask,
            "xy": (
                spot_settings["center_x"],
                spot_settings["center_y"] + spot_settings["diameter"] * 0.9,
            ),
        })

    return regions


def draw_site_state_legend(axis):
    legend_handles = []
    for color, label in (
        (COLOR_EMPTY, SITE_STATE_UNAVAILABLE_LABEL),
        (COLOR_AVAILABLE_SPOT, SITE_STATE_AVAILABLE_LABEL),
        (COLOR_HYDROGEN, SITE_STATE_OCCUPIED_LABEL),
    ):
        legend_handles.append(
            axis.scatter(
                [],
                [],
                s=SITE_STATE_LEGEND_MARKER_AREA,
                c=color,
                marker="s",
                edgecolors="#303030",
                linewidths=0.5,
                label=label,
            )
        )
    legend = axis.legend(
        handles=legend_handles,
        loc=SITE_STATE_LEGEND_LOCATION,
        bbox_to_anchor=SITE_STATE_LEGEND_ANCHOR,
        fontsize=SITE_STATE_LEGEND_FONT_SIZE,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#707070",
        framealpha=0.90,
    )
    legend.set_zorder(10)


def draw_main_panel(axis, matrix, saved_step, metadata, area_summary_shake_mode=None):
    rows, cols = matrix.shape

    draw_special_region_backgrounds(axis, metadata)

    if RENDER_MODE == "pixels":
        cmap = ListedColormap([COLOR_EMPTY, COLOR_AVAILABLE_SPOT, COLOR_HYDROGEN])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        axis.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", origin="lower")
    elif RENDER_MODE == "dots":
        axis.set_facecolor(COLOR_EMPTY)
        available_y, available_x = np.where(matrix == 1)
        hydrogen_y, hydrogen_x = np.where(matrix == 2)
        axis.scatter(
            available_x,
            available_y,
            s=DOT_SIZE_AVAILABLE,
            c=COLOR_AVAILABLE_SPOT,
            alpha=DOT_ALPHA_AVAILABLE,
            marker="o",
            edgecolors="none",
        )
        axis.scatter(
            hydrogen_x,
            hydrogen_y,
            s=DOT_SIZE_HYDROGEN,
            c=COLOR_HYDROGEN,
            alpha=DOT_ALPHA_HYDROGEN,
            marker="o",
            edgecolors="none",
        )
    elif RENDER_MODE == "concentration_heatmap":
        draw_concentration_heatmap(axis, matrix, metadata)
    elif RENDER_MODE == "printer_glyphs":
        draw_printer_glyphs(axis, matrix, metadata)
    elif RENDER_MODE == "area_summary_dots":
        draw_area_summary_dots(
            axis,
            matrix,
            metadata,
            shake_mode=area_summary_shake_mode,
        )
    else:
        raise ValueError(
            "RENDER_MODE must be 'pixels', 'dots', 'concentration_heatmap', "
            "'printer_glyphs', or 'area_summary_dots'"
        )

    draw_special_region_outlines(axis, matrix.shape, metadata)
    axis.set_xlim(-0.5, cols - 0.5)
    axis.set_ylim(-0.5, rows - 0.5)
    axis.set_aspect("equal")
    title = TITLE
    if title and RENDER_MODE == "area_summary_dots" and area_summary_shake_mode is not None:
        title = f"{title} - {area_summary_shake_mode.title()} layout"
    if RENDER_MODE == "area_summary_dots":
        concentration_title = {
            "area_average": None,
            "saved_x_profile": "Saved x-profile",
            "linear_x": "Linear x-profile",
        }.get(str(AREA_SUMMARY_CONCENTRATION_MODE))
        if title and concentration_title is not None:
            title = f"{title} - {concentration_title}"
    if title:
        axis.set_title(f"{title} (Step: {saved_step})")
    axis.set_xlabel(dimension_label_with_pixels(X_LABEL, cols))
    axis.set_ylabel(dimension_label_with_pixels(Y_LABEL, rows))
    apply_l_fraction_ticks(axis, x_length=cols, y_length=rows)

    if SHOW_REGION_ANNOTATIONS:
        for region in build_annotation_regions(matrix.shape, metadata):
            concentration = concentration_percent(matrix, region["mask"])
            if concentration is None:
                text = f"{region['name']}: n/a"
            else:
                text = f"{region['name']}: {concentration:.1f}%"
            if "max_solubility" in region:
                text = f"{text}\nMax. Solubility: {region['max_solubility']:.0f}%"
            label = axis.text(
                region["xy"][0],
                region["xy"][1],
                text,
                color=ANNOTATION_COLOR,
                fontsize=ANNOTATION_FONT_SIZE,
                ha="center",
                va="center",
                bbox=(
                    {
                        "facecolor": ANNOTATION_BOX_FACE_COLOR,
                        "edgecolor": ANNOTATION_BOX_EDGE_COLOR,
                        "alpha": ANNOTATION_BOX_ALPHA,
                        "pad": 2.5,
                    }
                    if ANNOTATION_BOX_ENABLED
                    else None
                ),
            )
            label.set_path_effects([
                path_effects.withStroke(
                    linewidth=ANNOTATION_STROKE_WIDTH,
                    foreground=ANNOTATION_STROKE_COLOR,
                )
            ])

    if SHOW_LEGEND and SHOW_SITE_STATE_LEGEND:
        draw_site_state_legend(axis)


def draw_concentration_profile(axis, matrix, metadata):
    coordinates, profile, axis_label = compute_profile(matrix, metadata)
    rows, cols = matrix.shape
    axis.plot(
        coordinates,
        profile * 100,
        color=COLOR_CONCENTRATION_LINE,
        clip_on=False,
        zorder=3,
    )
    if SHOW_CONCENTRATION_PROFILE_TITLE:
        axis.set_title("Concentration Profile")
    profile_pixel_count = cols if PROFILE_AXIS == "x" else rows
    axis.set_xlabel(dimension_label_with_pixels(axis_label, profile_pixel_count))
    axis.set_ylabel("Concentration (%)")
    axis.set_xlim(coordinates[0], coordinates[-1])
    axis.set_ylim(0, 100)
    axis.set_xticks(np.linspace(coordinates[0], coordinates[-1], 5))
    if PROFILE_AXIS == "x":
        axis.set_xticklabels(["0", "L/4", "L/2", "3L/4", "L"])
    else:
        axis.set_xticklabels(["0", "H/4", "H/2", "3H/4", "H"])

    if PROFILE_AXIS == "x" and SHOW_PROFILE_HALF_TRANSITION:
        axis.axvline(cols / 2, color=PROFILE_HALF_TRANSITION_COLOR, linestyle="--", linewidth=1)
        axis.text(
            cols / 4,
            94,
            PROFILE_AREA_1_LABEL,
            ha="center",
            va="top",
            color=PROFILE_HALF_TRANSITION_COLOR,
        )
        axis.text(
            3 * cols / 4,
            94,
            PROFILE_AREA_2_LABEL,
            ha="center",
            va="top",
            color=PROFILE_HALF_TRANSITION_COLOR,
        )

    spot_settings = get_spot_settings(metadata)
    if PROFILE_AXIS == "x" and SHOW_PROFILE_SPOT_SHADE and spot_settings is not None:
        radius = spot_settings["diameter"] // 2
        shade_start = spot_settings["center_x"] - radius
        shade_end = spot_settings["center_x"] + radius
        axis.axvspan(
            shade_start,
            shade_end,
            color=PROFILE_SPOT_SHADE_COLOR,
            alpha=PROFILE_SPOT_SHADE_ALPHA,
            linewidth=0,
        )
        axis.text(
            spot_settings["center_x"],
            50,
            PROFILE_SPOT_SHADE_LABEL,
            ha="center",
            va="center",
            rotation=90,
            color=PROFILE_SPOT_SHADE_LABEL_COLOR,
        )
    elif PROFILE_AXIS == "y" and SHOW_PROFILE_SPOT_SHADE and spot_settings is not None:
        radius = spot_settings["diameter"] // 2
        shade_start = spot_settings["center_y"] - radius
        shade_end = spot_settings["center_y"] + radius
        axis.axvspan(
            shade_start,
            shade_end,
            color=PROFILE_SPOT_SHADE_COLOR,
            alpha=PROFILE_SPOT_SHADE_ALPHA,
            linewidth=0,
        )
        axis.text(
            spot_settings["center_y"],
            50,
            PROFILE_SPOT_SHADE_LABEL,
            ha="center",
            va="center",
            rotation=90,
            color=PROFILE_SPOT_SHADE_LABEL_COLOR,
        )


def draw_heatmap_panel(axis, matrix, metadata):
    rows, cols = matrix.shape
    draw_concentration_heatmap(axis, matrix, metadata)
    draw_special_region_outlines(axis, matrix.shape, metadata)
    axis.set_xlim(-0.5, cols - 0.5)
    axis.set_ylim(-0.5, rows - 0.5)
    axis.set_aspect("equal")
    axis.set_xlabel(dimension_label_with_pixels(X_LABEL, cols))
    axis.set_ylabel(dimension_label_with_pixels(Y_LABEL, rows))
    apply_l_fraction_ticks(axis, x_length=cols, y_length=rows)


def draw_net_flux(axis, transport_analysis, saved_step):
    if not transport_analysis:
        axis.text(
            0.5,
            0.5,
            "No net-flux data found",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return

    time_values = np.asarray(transport_analysis["time"], dtype=float)
    net_flux = np.asarray(transport_analysis["net_flux"], dtype=float)
    flux_low = np.asarray(transport_analysis["flux_low"], dtype=float)
    flux_high = np.asarray(transport_analysis["flux_high"], dtype=float)

    # A file sparsified to its first and last snapshots contains one measured
    # interval-average flux. Draw that value across its interval instead of as
    # an invisible one-point line at the final saved step.
    finite_flux_indices = np.flatnonzero(np.isfinite(net_flux))
    if len(time_values) == 2 and len(finite_flux_indices) == 1:
        flux_index = int(finite_flux_indices[0])
        interval_flux = net_flux[flux_index]
        interval_low = flux_low[flux_index]
        interval_high = flux_high[flux_index]
        time_values = np.asarray([time_values[0], time_values[-1]], dtype=float)
        net_flux = np.full(2, interval_flux, dtype=float)
        flux_low = np.full(2, interval_low, dtype=float)
        flux_high = np.full(2, interval_high, dtype=float)

    axis.fill_between(
        time_values,
        flux_low,
        flux_high,
        color=NET_FLUX_BAND_COLOR,
        alpha=0.35,
        label="10th–90th spatial percentile",
    )
    axis.plot(
        time_values,
        net_flux,
        color=NET_FLUX_COLOR,
        linewidth=1.7,
        label="Net flux",
        clip_on=False,
        zorder=3,
    )
    axis.axhline(0, color="#505050", linewidth=0.8)
    axis.axvline(saved_step, color="#000000", linestyle="--", linewidth=1)
    axis.set_title("Net Diffusive Flux")
    axis.set_xlabel("Step")
    axis.set_ylabel("H particles / step\n(+x is positive)")
    if SHOW_LEGEND:
        axis.legend(fontsize=8)


def match_side_panel_heights_to_main(fig, axes_by_panel):
    if not MATCH_SIDE_PANEL_HEIGHT_TO_MAIN or "main" not in axes_by_panel:
        return

    # Equal-aspect image axes shrink after layout; align side plots to that final visual height.
    fig.canvas.draw()
    main_position = axes_by_panel["main"].get_position()
    for panel_name, axis in axes_by_panel.items():
        if panel_name == "main":
            continue
        position = axis.get_position()
        axis.set_position([position.x0, main_position.y0, position.width, main_position.height])


def apply_panel_typography(axis):
    if PANEL_TITLE_FONT_SIZE is not None:
        axis.title.set_fontsize(PANEL_TITLE_FONT_SIZE)
    if AXIS_LABEL_FONT_SIZE is not None:
        axis.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
        axis.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    if TICK_LABEL_FONT_SIZE is not None:
        axis.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)


def create_figure(
    matrix,
    saved_step,
    frame_index,
    metadata,
    transport_analysis,
    area_summary_shake_mode=None,
):
    panels = []
    if SHOW_MAIN_PANEL:
        panels.append(("main", 5))
    if SHOW_HEATMAP_PANEL:
        panels.append(("heatmap", 5))
    if SHOW_CONCENTRATION_PROFILE_PANEL:
        panels.append(("profile", 2))
    if SHOW_NET_FLUX_PANEL:
        panels.append(("flux", 3))

    if not panels:
        raise ValueError("At least one diagram panel must be enabled.")

    fig, axes_array = plt.subplots(
        1,
        len(panels),
        figsize=FIGURE_SIZE,
        gridspec_kw={"width_ratios": [panel[1] for panel in panels], "wspace": 0.35},
    )
    axes_array = np.atleast_1d(axes_array)
    axes_by_panel = {}

    for (panel_name, _), axis in zip(panels, axes_array):
        axes_by_panel[panel_name] = axis
        if panel_name == "main":
            draw_main_panel(
                axis,
                matrix,
                saved_step,
                metadata,
                area_summary_shake_mode=area_summary_shake_mode,
            )
        elif panel_name == "profile":
            draw_concentration_profile(axis, matrix, metadata)
        elif panel_name == "heatmap":
            draw_heatmap_panel(axis, matrix, metadata)
        elif panel_name == "flux":
            draw_net_flux(axis, transport_analysis, saved_step)

    for axis in fig.axes:
        apply_panel_typography(axis)

    if SHOW_FRAME_SUPTITLE:
        fig.suptitle(f"Saved Frame {frame_index}", fontsize=14)
        fig.subplots_adjust(top=0.88)
    else:
        fig.subplots_adjust(top=0.94)
    match_side_panel_heights_to_main(fig, axes_by_panel)
    return fig


def save_outputs(fig, output_dir, frame_index, saved_step, output_variant=None):
    variant_suffix = f"_{output_variant}" if output_variant else ""
    basename = (
        f"{OUTPUT_BASENAME}{variant_suffix}_frame_{frame_index}_step_{saved_step}"
    )
    if SAVE_PNG:
        png_path = output_dir / f"{basename}.png"
        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved PNG: {png_path}")
    if SAVE_PDF:
        pdf_path = output_dir / f"{basename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved PDF: {pdf_path}")
    if SAVE_SVG:
        svg_path = output_dir / f"{basename}.svg"
        svg_fonttype = "path" if SVG_TEXT_AS_PATHS else "none"
        with plt.rc_context({"svg.fonttype": svg_fonttype}):
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Saved SVG: {svg_path}")


def requested_area_summary_outputs():
    """Return ``(shake_mode, filename_suffix)`` variants for this invocation."""
    if RENDER_MODE != "area_summary_dots":
        return [(None, None)]

    configured_modes = AREA_SUMMARY_SHAKE_OUTPUT_MODES
    if configured_modes is None:
        mode, _, _, _ = shake_profile_settings(AREA_SUMMARY_SHAKE_MODE)
        return [(mode, None)]

    if isinstance(configured_modes, str):
        configured_modes = (configured_modes,)
    else:
        try:
            configured_modes = tuple(configured_modes)
        except TypeError as exc:
            raise ValueError(
                "AREA_SUMMARY_SHAKE_OUTPUT_MODES must be None, a mode string, "
                "or a sequence of mode strings"
            ) from exc
    if not configured_modes:
        raise ValueError("AREA_SUMMARY_SHAKE_OUTPUT_MODES cannot be empty")

    outputs = []
    seen_modes = set()
    for configured_mode in configured_modes:
        mode, _, _, _ = shake_profile_settings(configured_mode)
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        outputs.append((mode, mode))
    return outputs


def discover_batch_diagram_presets():
    """Return ordinary presets in documented order, then newly added presets."""
    discovered = {
        path.stem
        for path in diagram_presets_dir().glob("*.py")
        if path.stem != "all_presets" and not path.stem.startswith("_")
    }
    ordered = [
        preset_name
        for preset_name in BATCH_PRESET_ORDER
        if preset_name in discovered
    ]
    ordered.extend(sorted(discovered.difference(ordered)))
    return ordered


def read_batch_manifest(manifest_path):
    """Return files managed by a previous successful all-presets run."""
    if not manifest_path.exists():
        return set()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read batch manifest: {manifest_path}") from exc

    generated_files = manifest.get("generated_files")
    if not isinstance(generated_files, list):
        raise RuntimeError(
            f"Batch manifest has no valid generated_files list: {manifest_path}"
        )

    managed_names = set()
    for name in generated_files:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.lower().endswith(".png")
        ):
            raise RuntimeError(
                f"Unsafe generated filename in batch manifest: {name!r}"
            )
        managed_names.add(name)
    return managed_names


def publish_batch_manifest(
    manifest_path,
    *,
    output_dir,
    previous_generated_files,
    saved_paths,
    saved_step,
):
    """Atomically publish the new manifest, then remove obsolete managed PNGs."""
    current_names = [path.name for path in saved_paths]
    manifest = {
        "generated_files": current_names,
        "simulation_step": int(saved_step),
        "source_h5": BATCH_INPUT_H5_FILENAME,
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    obsolete_names = sorted(previous_generated_files.difference(current_names))
    for obsolete_name in obsolete_names:
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.is_file():
            obsolete_path.unlink()
            print(f"Deleted obsolete published example: {obsolete_path}")

    temporary_manifest.replace(manifest_path)
    print(f"Updated published-example manifest: {manifest_path}")


def render_all_diagram_presets():
    """Render Examples/published_examples_source.h5 through every ordinary preset."""
    input_path = Path(BATCH_INPUT_H5_FILENAME)
    h5_path = input_path if input_path.is_absolute() else in_results(
        BATCH_INPUT_H5_FILENAME
    )
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    output_dir = results_dir() / BATCH_OUTPUT_FOLDER
    output_dir.mkdir(parents=False, exist_ok=True)
    manifest_name = str(BATCH_MANIFEST_FILENAME)
    if Path(manifest_name).name != manifest_name:
        raise ValueError("BATCH_MANIFEST_FILENAME must be a plain filename")
    manifest_path = diagram_presets_dir() / manifest_name
    previous_generated_files = read_batch_manifest(manifest_path)

    preset_names = discover_batch_diagram_presets()
    if not preset_names:
        raise RuntimeError("No ordinary diagram presets were found.")

    # Validate every preset before creating output and load transport only when
    # at least one of the discovered presets needs it.
    preset_settings = {
        preset_name: load_diagram_preset(preset_name)
        for preset_name in preset_names
    }
    globals()["SHOW_NET_FLUX_PANEL"] = any(
        settings["SHOW_NET_FLUX_PANEL"]
        for settings in preset_settings.values()
    )
    matrix, saved_step, frame_index, metadata, transport_analysis = (
        load_snapshot_and_context(h5_path, BATCH_SNAPSHOT_INDEX)
    )

    saved_paths = []
    for preset_number, preset_name in enumerate(preset_names, start=1):
        apply_diagram_preset(preset_name)
        outputs = requested_area_summary_outputs()
        multiple_variants = len(outputs) > 1

        for variant_index, (shake_mode, output_variant) in enumerate(outputs):
            number_label = (
                str(preset_number)
                if variant_index == 0
                else f"{preset_number}.{variant_index}"
            )
            variant_suffix = (
                f"_{output_variant}"
                if multiple_variants and output_variant
                else ""
            )
            output_path = output_dir / (
                f"{number_label}_{preset_name}{variant_suffix}.png"
            )
            style_description = (
                f" ({output_variant})" if output_variant else ""
            )
            print(
                f"Rendering {number_label}: {preset_name}{style_description} "
                f"at frame {frame_index}, step {saved_step}"
            )

            fig = create_figure(
                matrix,
                saved_step,
                frame_index,
                metadata,
                transport_analysis,
                area_summary_shake_mode=shake_mode,
            )
            fig.savefig(
                output_path,
                dpi=BATCH_SAVE_DPI,
                bbox_inches="tight",
            )
            plt.close(fig)
            saved_paths.append(output_path)
            print(f"Saved PNG: {output_path}")

    publish_batch_manifest(
        manifest_path,
        output_dir=output_dir,
        previous_generated_files=previous_generated_files,
        saved_paths=saved_paths,
        saved_step=saved_step,
    )
    print(
        f"Rendered {len(saved_paths)} diagram files from "
        f"{len(preset_names)} presets into {output_dir}"
    )
    return saved_paths


def main():
    if BATCH_RENDER_ALL_PRESETS:
        render_all_diagram_presets()
        return

    h5_path = resolve_h5_path()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    matrix, saved_step, frame_index, metadata, transport_analysis = load_snapshot_and_context(
        h5_path,
        SNAPSHOT_INDEX,
    )
    figures = []
    output_dir = resolve_output_dir() if SAVE_PNG or SAVE_PDF or SAVE_SVG else None
    for shake_mode, output_variant in requested_area_summary_outputs():
        if shake_mode is not None:
            print(f"Rendering area-summary shake style: {shake_mode}")
        fig = create_figure(
            matrix,
            saved_step,
            frame_index,
            metadata,
            transport_analysis,
            area_summary_shake_mode=shake_mode,
        )
        figures.append(fig)

        if output_dir is not None:
            save_outputs(
                fig,
                output_dir,
                frame_index,
                saved_step,
                output_variant=output_variant,
            )

    if SHOW_PLOT:
        plt.show()
    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
