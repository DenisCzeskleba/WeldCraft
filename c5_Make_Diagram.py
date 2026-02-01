"""
Make Diagram (consolidated / animation-consistent).

- Loads snapshots exactly like c1_Make_Animation.py (u_snapshot_*, h_snapshot_*, t_snapshot_*)
- Uses mm scaling via: extent = [0, nx*dx, ny*dy, 0]
- Uses mm-based annotations (coords_mm) with one-time preprocessing to indices
- Can auto-use weld phase times from HDF5 attrs (RT, etc.) so no manual guessing

Output: saves PNGs (one per selected time) into the current working directory.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import io
import contextlib

# Suppress config prints during import
with contextlib.redirect_stdout(io.StringIO()):
    from b4_functions import get_value


# -------------------------------------------------
# USER OPTIONS
# -------------------------------------------------

# If True, pick target times from HDF5 attrs:
#   total_time_to_first_weld, total_time_to_cooling, total_time_to_rt, total_max_time
USE_PHASE_TIMES = True

# If USE_PHASE_TIMES is False (or attrs missing), use this manual list (seconds)
manual_target_times_seconds = [970, 1430, 1700, 2950]

# Tick grid style in mm (keep consistent across scripts)
MAJOR_MM = 10
MINOR_MM = 2

# Save DPI for paper-ready images
SAVE_DPI = 300

# --- Discrete (stepped) color options ---
TEMP_STEPPED_COLORS = True
TEMP_STEP_EDGES_C = list(range(25, 851, 25))   # 0,50,100,...,850 (edit to taste)

H_STEPPED_COLORS = True
H_STEP_EDGES_PCT = list(range(0, 41, 2))    # 0,10,20,...,100 (edit to taste)

# --- Fake zoom (crop in x) ---
CROP_X_ENABLE = True
CROP_LEFT_MM = 30.0   # "hide" first 20 mm
CROP_RIGHT_MM = 30.0   # "hide" last 30 mm

# Diffusion annotation mode:
# "fixed_points" = use annotation_points['diffusion'] (mm coords)
# "max_circle"   = draw a white circle around the max H location and label it
DIFFUSION_ANNOTATION_MODE = "max_circle"  # or "max_circle"
MAX_TEMP_SEARCH_Y_MAX_MM = 25.0   # set None to disable

MAX_CIRCLE_RADIUS_MM = 3.0   # radius in mm (tweak to taste)
MAX_CIRCLE_LINEWIDTH = 2.0
MAX_LABEL_OFFSET_MM = (-6.0, 8.0)  # (dx, dy) in mm for text label position

# Optional: draw a white circle around the maximum temperature location for selected output images (0-based indices)
show_max_temp_circle_for = {
    0,  # e.g. enable for first output image only
    1,  # and maybe for the 2nd image
    # 2,  # and maybe for the third image
}

MAX_TEMP_CIRCLE_RADIUS_MM = 3.0
MAX_TEMP_CIRCLE_LINEWIDTH = 2.0
MAX_TEMP_LABEL_OFFSET_MM = (-6.0, 8.0)  # (dx, dy) in mm

# Annotation points (in mm)
annotation_points = {
    "heat": [
        # example values – adjust to your specimen geometry
        # {"coords_mm": (10.0, 6.5), "text_offset": (15, -20)},
        # {"coords_mm": (130.0, 6.5), "text_offset": (-50, -20)},
    ],
    "diffusion": [
        {"coords_mm": (22.5, 6.5), "text_offset": (25, -12)},
        {"coords_mm": (22.5, 5.0), "text_offset": (-120, -30)},
        {"coords_mm": (22.5, 9.8), "text_offset": (25, 3)},
    ],
}

marker_style = dict(marker="o", s=36, facecolors="white", edgecolors="white")


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def compute_x_crop_indices(nx, dx, crop_left_mm, crop_right_mm):
    x0 = int(round(crop_left_mm / dx))
    x1 = nx - int(round(crop_right_mm / dx))
    x0 = max(0, min(nx, x0))
    x1 = max(0, min(nx, x1))
    if x1 <= x0:
        raise ValueError("Invalid crop: right crop overlaps left crop. Reduce CROP_LEFT_MM / CROP_RIGHT_MM.")
    return x0, x1


def preprocess_annotation_points(points_mm_list, dx, dy, array_shape, x_offset_mm=0.0):
    """Return list of dicts with coords_mm (shifted for view) and coords_idx (clipped)."""
    ny, nx = array_shape
    processed = []
    for p in points_mm_list:
        x_mm_raw, y_mm = p["coords_mm"]

        # shift x so the cropped view starts at 0
        x_mm = x_mm_raw - x_offset_mm

        # If point lies outside the current view, skip it (prevents axis auto-expanding)
        x_max_mm = (nx - 1) * dx
        y_max_mm = (ny - 1) * dy
        if (x_mm < 0) or (x_mm > x_max_mm) or (y_mm < 0) or (y_mm > y_max_mm):
            continue

        x_idx = int(round(x_mm / dx))
        y_idx = int(round(y_mm / dy))

        x_idx = max(0, min(nx - 1, x_idx))
        y_idx = max(0, min(ny - 1, y_idx))

        processed.append(
            {
                "coords_mm": (x_mm, y_mm),   # shifted coord used for plotting
                "coords_idx": (x_idx, y_idx),
                "text_offset": p.get("text_offset", (0, 0)),
            }
        )
    return processed


def create_annotations(ax, points, loaded_array, label):
    """
    points: list of dicts with coords_mm, coords_idx, text_offset
    label: "Temp" or "HD"
    """
    annotations = []
    for p in points:
        (x_mm, y_mm) = p["coords_mm"]
        (x_idx, y_idx) = p["coords_idx"]
        (x_off, y_off) = p["text_offset"]

        ax.scatter(x_mm, y_mm, **marker_style)

        val = loaded_array[y_idx, x_idx]
        if label == "Temp":
            text_val = f"{val:.0f}°C"
        else:
            text_val = f"{val:.1f}%"

        ann = ax.annotate(
            text_val,
            xy=(x_mm, y_mm),
            xycoords="data",
            xytext=(x_off, y_off),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color="white"),
            fontsize=12,
            color="white",
        )
        annotations.append(ann)
    return annotations


def load_snapshots_like_animation(file_name):
    """Load u/h/t snapshots in sorted-key order (exactly like c1_Make_Animation.py)."""
    loaded_u_arrays = []
    loaded_h_arrays = []
    loaded_t_values = []
    phase_times = {}

    with h5py.File(file_name, "r") as hf:
        # Phase times (if present)
        phase_times = {
            "total_time_to_first_weld": hf.attrs.get("total_time_to_first_weld", None),
            "total_time_to_cooling": hf.attrs.get("total_time_to_cooling", None),
            "total_time_to_rt": hf.attrs.get("total_time_to_rt", None),
            "total_max_time": hf.attrs.get("total_max_time", None),
        }

        for key in sorted(hf.keys()):
            if key.startswith("u_snapshot_"):
                loaded_u_arrays.append(hf[key][:])
            elif key.startswith("h_snapshot_"):
                loaded_h_arrays.append(hf[key][:])
            elif key.startswith("t_snapshot_"):
                loaded_t_values.append(hf[key][()])

    if not loaded_u_arrays or not loaded_h_arrays or not loaded_t_values:
        raise ValueError("No u/h/t snapshots found. Expected u_snapshot_*, h_snapshot_*, t_snapshot_* datasets.")

    if not (len(loaded_u_arrays) == len(loaded_h_arrays) == len(loaded_t_values)):
        raise ValueError("Snapshot lists have different lengths. Check file integrity (u/h/t mismatch).")

    return loaded_u_arrays, loaded_h_arrays, np.array(loaded_t_values, dtype=float), phase_times


def nearest_frame_index(times_s, target_time_s):
    """Return index of nearest time stamp."""
    return int(np.argmin(np.abs(times_s - target_time_s)))


# -------------------------------------------------
# Main
# -------------------------------------------------

file_name = r'02_Results\00_diffusion_array.h5'
dx, dy = get_value("dx"), get_value("dy")  # Change if you ran another simulation inbetween

loaded_u_arrays, loaded_h_arrays, times_s, phase_times = load_snapshots_like_animation(file_name)

# mm scaling: EXACTLY the trick you requested (consistent with c1_Make_Animation.py)
ny, nx = loaded_u_arrays[0].shape

x_crop0 = 0
x_crop1 = nx
x_offset_mm = 0.0

if CROP_X_ENABLE:
    x_crop0, x_crop1 = compute_x_crop_indices(nx, dx, CROP_LEFT_MM, CROP_RIGHT_MM)
    x_offset_mm = CROP_LEFT_MM  # used to shift annotation x-coordinates

nx_view = x_crop1 - x_crop0
extent = [0, nx_view * dx, ny * dy, 0]  # x starts at 0 again

# Decide which target times to render (seconds)
if USE_PHASE_TIMES:
    candidate_times = [
        phase_times.get("total_time_to_first_weld", None),
        phase_times.get("total_time_to_cooling", None),
        phase_times.get("total_time_to_rt", None),
        phase_times.get("total_max_time", None),
    ]
    candidate_times[0] = 3852  # manual override for the first entry
    candidate_times[1] = 7212  # manual override for the 2nd entry
    # candidate_times[2] = 966  # manual override for the 3rd entry
    candidate_times[3] = 101253  # manual override for the 4th entry

    # A new weld bead will be added at the following times (s): 5, 485, 965, 1445, 1925, 2405, 2885, 3365, 3845, 4325, 4805, 5285, 5765, 6245, 6725, 7205
    # A new weld bead will be added at the following times (s): 5, 665, 1325, 1985, 2645, 3305, 3965, 4625, 5285, 5945

    target_times_seconds = [t for t in candidate_times if t is not None]
    # Fallback if attrs are missing
    if not target_times_seconds:
        target_times_seconds = manual_target_times_seconds
else:
    target_times_seconds = manual_target_times_seconds

# Preprocess annotation points once (mm → indices)
view_shape = (ny, nx_view)
heat_points = preprocess_annotation_points(annotation_points["heat"], dx, dy, view_shape, x_offset_mm=x_offset_mm)
diff_points = preprocess_annotation_points(annotation_points["diffusion"], dx, dy, view_shape, x_offset_mm=x_offset_mm)

# Render one diagram per selected time
for i, t_target in enumerate(target_times_seconds, start=1):
    fi = nearest_frame_index(times_s, t_target)

    u_full = loaded_u_arrays[fi]
    h_full = loaded_h_arrays[fi]

    u = u_full[:, x_crop0:x_crop1]
    h = h_full[:, x_crop0:x_crop1]

    t_real = float(times_s[fi])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=120)

    # Temperature plot
    cmap1 = plt.get_cmap("hot").copy()
    cmap1.set_under("0.85")

    if TEMP_STEPPED_COLORS:
        bounds_T = np.array(TEMP_STEP_EDGES_C, dtype=float)
        norm1 = mcolors.BoundaryNorm(bounds_T, ncolors=cmap1.N, clip=False)
    else:
        bounds_T = None
        norm1 = mcolors.Normalize(vmin=24, vmax=800)

    im1 = ax1.imshow(u, cmap=cmap1, norm=norm1, extent=extent, origin="upper", aspect="equal")
    ax1.set_title(f"Temperature (t ≈ {t_real:.1f} s)")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")

    if TEMP_STEPPED_COLORS:
        cbar1 = fig.colorbar(
            im1, ax=ax1, fraction=0.046, pad=0.04, label="$T$ [°C]",
            ticks=bounds_T, boundaries=bounds_T
        )
        # Optional: make the labels nicer (no decimals)
        cbar1.set_ticklabels([f"{int(b)}" for b in bounds_T])
    else:
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="$T$ [°C]")

    # Hydrogen plot
    cmap2 = plt.get_cmap("viridis").copy()
    cmap2.set_under("0.85")

    if H_STEPPED_COLORS:
        bounds_H = np.array(H_STEP_EDGES_PCT, dtype=float)
        norm2 = mcolors.BoundaryNorm(bounds_H, ncolors=cmap2.N, clip=False)
    else:
        bounds_H = None
        norm2 = mcolors.Normalize(vmin=-0.01, vmax=100)

    im2 = ax2.imshow(h, cmap=cmap2, norm=norm2, extent=extent, origin="upper", aspect="equal")

    # Optional: override the hydrogen title per output image index (0-based)
    # If an index is not in here, the default title with time is used.
    hydrogen_title_override = {
        0: "Welding Process (Middle)",
        1: "Welding Process (End)",
        2: "Reached RT",
        3: "24h after Welding",
    }

    # Default title includes time, but you can override it per figure index
    if (i - 1) in hydrogen_title_override:
        ax2.set_title(r"Hydrogen Concentration Field" + "\n" + hydrogen_title_override[i - 1])
    else:
        ax2.set_title(r"Hydrogen Concentration Field" + "\n" + f"(t ≈ {t_real:.1f} s)")

    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")

    # Colorbar (with clean bin-edge ticks if stepped)
    if H_STEPPED_COLORS:
        cbar2 = fig.colorbar(
            im2, ax=ax2, fraction=0.046, pad=0.04, label="Hydrogen Concentration [%]",
            ticks=bounds_H, boundaries=bounds_H
        )
        # Optional: nicer tick labels (no decimals)
        cbar2.set_ticklabels([f"{int(b)}" for b in bounds_H])
    else:
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Hydrogen Concentration [%]")

    # mm tick styling (same idea as your other scripts)
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MultipleLocator(MAJOR_MM))
        ax.xaxis.set_minor_locator(MultipleLocator(MINOR_MM))
        ax.yaxis.set_major_locator(MultipleLocator(MAJOR_MM))
        ax.yaxis.set_minor_locator(MultipleLocator(MINOR_MM))
        ax.tick_params(axis="both", which="major", length=6, width=1.2)
        ax.tick_params(axis="both", which="minor", length=3, width=0.8)

    # Annotations (mm coords)
    create_annotations(ax1, heat_points, u, "Temp")

    # Optional: highlight maximum temperature with a circle (per output image index)
    out_idx = i - 1  # convert enumerate(start=1) to 0-based output index

    if out_idx in show_max_temp_circle_for:

        # --- build validity mask for searching max temperature ---
        # 1) Exclude "background/outside": use your temperature under-threshold (or use u > 0)
        valid = (u >= 24)  # or: valid = (u > 0)

        # 2) Optional limiter: only search in y <= MAX_TEMP_SEARCH_Y_MAX_MM (in mm)
        if MAX_TEMP_SEARCH_Y_MAX_MM is not None:
            y_max_idx = int(np.floor(MAX_TEMP_SEARCH_Y_MAX_MM / dy))
            y_max_idx = max(0, min(u.shape[0] - 1, y_max_idx))
            valid[y_max_idx + 1:, :] = False  # exclude rows with y > max

        # (Optional) x-limits too, if you add them later:
        # if MAX_TEMP_SEARCH_X_MIN_MM is not None:
        #     x_min_idx = int(np.ceil(MAX_TEMP_SEARCH_X_MIN_MM / dx))
        #     x_min_idx = max(0, min(u.shape[1], x_min_idx))
        #     valid[:, :x_min_idx] = False
        # if MAX_TEMP_SEARCH_X_MAX_MM is not None:
        #     x_max_idx = int(np.floor(MAX_TEMP_SEARCH_X_MAX_MM / dx))
        #     x_max_idx = max(0, min(u.shape[1] - 1, x_max_idx))
        #     valid[:, x_max_idx + 1:] = False

        if np.any(valid):
            u_valid = np.where(valid, u, -np.inf)
            y_idx, x_idx = np.unravel_index(np.argmax(u_valid), u.shape)
            max_temp = u[y_idx, x_idx]

            # idx -> mm
            x_mm = x_idx * dx
            y_mm = y_idx * dy

            circle = plt.Circle(
                (x_mm, y_mm),
                radius=MAX_TEMP_CIRCLE_RADIUS_MM,
                color="white",
                fill=False,
                lw=MAX_TEMP_CIRCLE_LINEWIDTH,
                zorder=10
            )
            ax1.add_patch(circle)

            ax1.annotate(
                f"{max_temp:.0f}°C",
                xy=(x_mm, y_mm),
                xycoords="data",
                xytext=(x_mm + MAX_TEMP_LABEL_OFFSET_MM[0], y_mm + MAX_TEMP_LABEL_OFFSET_MM[1]),
                textcoords="data",
                fontsize=12,
                color="white",
                zorder=11
            )

    # Diffusion annotations (choose mode)
    if DIFFUSION_ANNOTATION_MODE == "fixed_points":
        create_annotations(ax2, diff_points, h, "HD")

    elif DIFFUSION_ANNOTATION_MODE == "max_circle":
        # Find max location within physical domain (your convention: h >= 0 is inside part)
        valid = (h >= 0)
        if np.any(valid):
            # Use masked array so max ignores invalid region
            h_valid = np.where(valid, h, -np.inf)
            y_idx, x_idx = np.unravel_index(np.argmax(h_valid), h.shape)
            max_val = h[y_idx, x_idx]

            # Convert idx -> mm (pixel corner convention consistent with extent)
            x_mm = x_idx * dx
            y_mm = y_idx * dy

            # Convert radius mm -> radius in data units (mm)
            circle = plt.Circle(
                (x_mm, y_mm),
                radius=MAX_CIRCLE_RADIUS_MM,
                color="white",
                fill=False,
                lw=MAX_CIRCLE_LINEWIDTH,
                zorder=10
            )
            ax2.add_patch(circle)

            # Label near the circle (in mm coords)
            ax2.annotate(
                f"{max_val:.1f}%",
                xy=(x_mm, y_mm),
                xycoords="data",
                xytext=(x_mm + MAX_LABEL_OFFSET_MM[0], y_mm + MAX_LABEL_OFFSET_MM[1]),
                textcoords="data",
                fontsize=12,
                color="white",
                zorder=11
            )
    else:
        raise ValueError(f"Unknown DIFFUSION_ANNOTATION_MODE: {DIFFUSION_ANNOTATION_MODE}")

    fig.tight_layout()

    out_name = rf"02_Results\04_Diagrams\diagram_{i:02d}_t{t_real:.0f}s.png"
    fig.savefig(out_name, dpi=SAVE_DPI)
    plt.close(fig)

print("Done. Rendered diagrams for times (s):", [float(t) for t in target_times_seconds])