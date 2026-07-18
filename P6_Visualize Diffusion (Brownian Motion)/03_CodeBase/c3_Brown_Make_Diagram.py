"""
Create a still Brownian-motion diagram from one saved HDF5 snapshot.

Diagram options live in this file on purpose. Simulation settings are read from
the HDF5 metadata when present, so old config edits do not change old diagrams.
"""

from pathlib import Path
import contextlib
import io

import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

with contextlib.redirect_stdout(io.StringIO()):
    from b3_Brown_Functions import (
        in_results,
        load_brown_config_json,
        read_saved_steps,
        results_dir,
    )


# ---------------------- Input Snapshot ---------------------- #
INPUT_H5_FILENAME = "random_motion.h5"  # Set to a sparse H5 name such as "random_motion_sparse.h5" when needed.
SNAPSHOT_INDEX = -1  # HDF5 saved-frame index to plot; -1 means the last saved frame.


# ---------------------- Output ---------------------- #
SHOW_PLOT = True
SAVE_PNG = False
SAVE_PDF = False
OUTPUT_FOLDER = ""  # Relative to 02_Results; leave empty to save directly in 02_Results.
OUTPUT_BASENAME = "brownian_diagram"
SAVE_DPI = 300


# ---------------------- Main Diagram ---------------------- #
RENDER_MODE = "pixels"  # Options: "pixels", "dots"
FIGURE_SIZE = (12, 6)
MATCH_SIDE_PANEL_HEIGHT_TO_MAIN = True
TITLE = "Diffusion as a Result of Random Motion"
X_LABEL = "Width"
Y_LABEL = "Height"

COLOR_EMPTY = "#440154"  # "#440154" 
COLOR_AVAILABLE_SPOT = "#0000FF"
COLOR_HYDROGEN = "#FF0000"
COLOR_CONCENTRATION_LINE = "#0000FF"
DIFFUSION_SPEED_COLORS = [
    "#FF0000",
    "#FFA500",
    "#008000",
    "#800080",
    "#A52A2A",
    "#00FFFF",
    "#000000",
]

DOT_SIZE_AVAILABLE = 8
DOT_SIZE_HYDROGEN = 8
DOT_ALPHA_AVAILABLE = 0.8
DOT_ALPHA_HYDROGEN = 0.95


# ---------------------- Panels ---------------------- #
SHOW_MAIN_PANEL = True
SHOW_CONCENTRATION_PROFILE_PANEL = True
SHOW_DIFFUSION_SPEED_PANEL = False
PROFILE_AXIS = "x"  # Options: "x" for column-wise profile, "y" for row-wise profile.
PROFILE_X_RANGE = None  # Optional tuple: (x_start, x_end). Use None for full width.
PROFILE_Y_RANGE = None  # Optional tuple: (y_start, y_end). Use None for full height.
PROFILE_BIN_SIZE = 3  # Average every N rows/columns before plotting; 1 keeps the original resolution.
PROFILE_SMOOTHING_WINDOW = 5  # Moving-average window applied after binning; 1 disables this smoothing.
PROFILE_GAUSSIAN_SIGMA = 1.5  # Gaussian smoothing applied after binning; 0 disables this smoothing.
SHOW_PROFILE_HALF_TRANSITION = True
PROFILE_HALF_TRANSITION_COLOR = "#808080"
PROFILE_AREA_1_LABEL = "Area 1"
PROFILE_AREA_2_LABEL = "Area 2"
SHOW_PROFILE_SPOT_SHADE = True
PROFILE_SPOT_SHADE_COLOR = "#B0B0B0"
PROFILE_SPOT_SHADE_ALPHA = 0.25
PROFILE_SPOT_SHADE_LABEL = "Spot Region"
PROFILE_SPOT_SHADE_LABEL_COLOR = "#606060"


# ---------------------- Annotations ---------------------- #
SHOW_REGION_ANNOTATIONS = True
SHOW_LEFT_RIGHT_ANNOTATIONS = True
SHOW_LEFT_RIGHT_WITHOUT_SINK_SOURCE_ANNOTATIONS = False
SHOW_SOURCE_SINK_ANNOTATIONS = False
SHOW_SPOT_ANNOTATION = True
ANNOTATION_FONT_SIZE = 11
ANNOTATION_COLOR = "#FFFFFF"

# Optional custom rectangular annotations in matrix coordinates.
CUSTOM_RECT_REGIONS = [
    # {"name": "Custom Region", "x_start": 0, "x_end": 100, "y_start": 0, "y_end": 100},
]


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

        diffusion_data = {}
        for key in sorted(hf.keys()):
            if not key.startswith("region_"):
                continue
            if "mean_disp" not in hf[key]:
                continue
            if "time" in hf[key]:
                time_values = hf[key]["time"][:]
            else:
                time_values = saved_steps
            diffusion_data[key] = {
                "time": np.asarray(time_values),
                "mean_disp": hf[key]["mean_disp"][:],
            }

    print(f"Loaded: {h5_path}")
    print(f"Available saved frames: {frame_count} (valid indices: -{frame_count}..-1 or 0..{frame_count - 1})")
    print(f"Saved steps: {int(saved_steps[0])} -> {int(saved_steps[-1])}")
    print(f"Matrix shape: {matrix.shape[0]} rows x {matrix.shape[1]} columns")
    print("Metadata: found")
    print(f"Plotting saved-frame index {frame_index}, simulation step {saved_step}")

    return matrix, saved_step, frame_index, metadata, diffusion_data


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


def compute_profile(matrix):
    rows, cols = matrix.shape
    x_start, x_end = clamp_range(PROFILE_X_RANGE, cols, "PROFILE_X_RANGE")
    y_start, y_end = clamp_range(PROFILE_Y_RANGE, rows, "PROFILE_Y_RANGE")
    cropped = matrix[y_start:y_end, x_start:x_end]

    if PROFILE_AXIS == "x":
        total_spots = np.sum(cropped > 0, axis=0)
        filled_spots = np.sum(cropped == 2, axis=0)
        coordinates = np.arange(x_start, x_end)
        axis_label = "Width"
    elif PROFILE_AXIS == "y":
        total_spots = np.sum(cropped > 0, axis=1)
        filled_spots = np.sum(cropped == 2, axis=1)
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


def circle_mask(shape, center_x, center_y, diameter):
    rows, cols = shape
    yy, xx = np.ogrid[:rows, :cols]
    radius = diameter / 2
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 < radius ** 2


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


def max_solubility_for_x(metadata, x_position, matrix_width):
    key = "max_sol_a" if x_position < matrix_width / 2 else "max_sol_b"
    return value_to_percent(metadata[key])


def build_annotation_regions(matrix_shape, metadata):
    rows, cols = matrix_shape
    mid_x = cols // 2
    regions = []

    if SHOW_LEFT_RIGHT_ANNOTATIONS:
        regions.extend([
            {
                "name": "Average Regional Concentration",
                "mask": rectangle_mask(matrix_shape, 0, mid_x),
                "xy": (cols * 0.25, rows * 0.9),
                "max_solubility": max_solubility_for_x(metadata, cols * 0.25, cols),
            },
            {
                "name": "Average Regional Concentration",
                "mask": rectangle_mask(matrix_shape, mid_x, cols),
                "xy": (cols * 0.75, rows * 0.9),
                "max_solubility": max_solubility_for_x(metadata, cols * 0.75, cols),
            },
        ])

    use_sink_source = metadata_bool(metadata, "USE_SINK_SOURCE")
    sink_source_thickness = metadata_int(metadata, "SINK_SOURCE_THICKNESS")
    source_side = metadata_str(metadata, "SOURCE_SIDE")
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
    spot_settings = get_spot_settings(metadata)
    if SHOW_SPOT_ANNOTATION and spot_settings is not None:
        regions.append({
            "name": "Spot Concentration",
            "mask": circle_mask(
                matrix_shape,
                spot_settings["center_x"],
                spot_settings["center_y"],
                spot_settings["diameter"],
            ),
            "xy": (
                spot_settings["center_x"],
                spot_settings["center_y"] + spot_settings["diameter"] * 0.9,
            ),
        })
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

    return regions


def draw_main_panel(axis, matrix, saved_step, metadata):
    rows, cols = matrix.shape

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
    else:
        raise ValueError("RENDER_MODE must be 'pixels' or 'dots'")

    axis.set_xlim(-0.5, cols - 0.5)
    axis.set_ylim(-0.5, rows - 0.5)
    axis.set_aspect("equal")
    axis.set_title(f"{TITLE} (Step: {saved_step})")
    axis.set_xlabel(X_LABEL)
    axis.set_ylabel(Y_LABEL)
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
            )
            label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="#000000")])


def draw_concentration_profile(axis, matrix, metadata):
    coordinates, profile, axis_label = compute_profile(matrix)
    rows, cols = matrix.shape
    axis.plot(coordinates, profile * 100, color=COLOR_CONCENTRATION_LINE)
    axis.set_title("Concentration Profile")
    axis.set_xlabel(axis_label)
    axis.set_ylabel("Concentration (%)")
    axis.set_xlim(coordinates[0], coordinates[-1])
    axis.set_ylim(0, 100)
    axis.set_xticks(np.linspace(coordinates[0], coordinates[-1], 5))
    axis.set_xticklabels(["0", "L/4", "L/2", "3L/4", "L"])

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
        radius = spot_settings["diameter"] / 2
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
        radius = spot_settings["diameter"] / 2
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


def draw_diffusion_speed(axis, diffusion_data, saved_step):
    if not diffusion_data:
        axis.text(0.5, 0.5, "No diffusion speed data found", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    max_speed = 0.0
    for index, (region_name, values) in enumerate(diffusion_data.items()):
        color = DIFFUSION_SPEED_COLORS[index % len(DIFFUSION_SPEED_COLORS)]
        time_values = values["time"]
        mean_disp = values["mean_disp"]
        axis.plot(time_values, mean_disp, label=region_name, color=color)
        if len(mean_disp):
            max_speed = max(max_speed, float(np.nanmax(mean_disp)))

    axis.axvline(saved_step, color="#000000", linestyle="--", linewidth=1)
    axis.set_title("Diffusion Speed")
    axis.set_xlabel("Step")
    axis.set_ylabel("Mean Displacement")
    axis.set_ylim(0, max(max_speed * 1.1, 1))
    axis.legend()


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


def create_figure(matrix, saved_step, frame_index, metadata, diffusion_data):
    panels = []
    if SHOW_MAIN_PANEL:
        panels.append(("main", 5))
    if SHOW_CONCENTRATION_PROFILE_PANEL:
        panels.append(("profile", 2))
    if SHOW_DIFFUSION_SPEED_PANEL:
        panels.append(("speed", 2))

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
            draw_main_panel(axis, matrix, saved_step, metadata)
        elif panel_name == "profile":
            draw_concentration_profile(axis, matrix, metadata)
        elif panel_name == "speed":
            draw_diffusion_speed(axis, diffusion_data, saved_step)

    fig.suptitle(f"Saved Frame {frame_index}", fontsize=14)
    fig.subplots_adjust(top=0.88)
    match_side_panel_heights_to_main(fig, axes_by_panel)
    return fig


def save_outputs(fig, output_dir, frame_index, saved_step):
    basename = f"{OUTPUT_BASENAME}_frame_{frame_index}_step_{saved_step}"
    if SAVE_PNG:
        png_path = output_dir / f"{basename}.png"
        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved PNG: {png_path}")
    if SAVE_PDF:
        pdf_path = output_dir / f"{basename}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved PDF: {pdf_path}")


def main():
    h5_path = resolve_h5_path()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    matrix, saved_step, frame_index, metadata, diffusion_data = load_snapshot_and_context(h5_path, SNAPSHOT_INDEX)
    fig = create_figure(matrix, saved_step, frame_index, metadata, diffusion_data)

    if SAVE_PNG or SAVE_PDF:
        save_outputs(fig, resolve_output_dir(), frame_index, saved_step)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
