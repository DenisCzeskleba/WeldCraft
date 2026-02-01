"""
Adapted from make_animation on 17.10.2025 to change annotations and add discription.
"""

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import numpy as np
import math
import io
import contextlib

# Suppress config prints during import, so no double prints. They’ll still appear when scripts run directly.
with contextlib.redirect_stdout(io.StringIO()):
    from b4_functions import get_value

# -------------------------------------------------
# USER OPTION: Track individual beads?
# -------------------------------------------------
track_individual_beads = True  # Recomended: False (set to False to only track weld metal as a whole)


def draw_static_outline(ax, data, dx, dy, color='red', linewidth=1.0, alpha=0.9, zorder=10):
    """
    Draw a static outline around regions where the input data == 0.
    Works even if the data never crosses 0 smoothly.
    - ax: Matplotlib axis to draw on
    - data: 2D numpy array (e.g. first snapshot)
    - dx, dy: physical spacing in mm
    """
    ny, nx = data.shape
    x = np.linspace(0, nx * dx, nx)
    y = np.linspace(0, ny * dy, ny)

    # Boolean mask of the "part" (where value == 0)
    mask = (data == 0)

    # Detect edges of that mask
    edges = np.zeros_like(mask, dtype=bool)
    edges[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edges[:, :-1] |= mask[:, :-1] != mask[:, 1:]

    # Convert boolean edges to float (for contour)
    edges = edges.astype(float)

    # Draw outline
    ax.contour(
        x, y, edges,
        levels=[0.5],
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder
    )


# ---------------------------- 1. Load Data from HDF5 File ----------------------------
file_name = get_value("file_name")  # 'diffusion_array.h5'
animation_file_name = get_value("animation_name")
loaded_u_arrays = []
loaded_h_arrays = []
loaded_t_values = []
dx, dy = get_value("dx"), get_value("dy")

with h5py.File(file_name, 'r') as hf:
    for key in sorted(hf.keys()):
        if key.startswith('u_snapshot_'):
            loaded_u_arrays.append(hf[key][:])
        elif key.startswith('h_snapshot_'):
            loaded_h_arrays.append(hf[key][:])
        elif key.startswith('t_snapshot_'):
            t_value = hf[key][()]
            loaded_t_values.append(t_value)
    # --- Load key weld phase times ---
    total_time_to_first_weld = hf.attrs.get('total_time_to_first_weld', None)
    total_time_to_cooling = hf.attrs.get('total_time_to_cooling', None)
    total_time_to_rt = hf.attrs.get('total_time_to_rt', None)
    total_max_time = hf.attrs.get('total_max_time', None)

ny, nx = loaded_u_arrays[0].shape
extent = [0, nx*dx, ny*dy, 0]  # x_min, x_max, y_max, y_min (origin="upper")
base_mask = loaded_h_arrays[0] >= 0

# -------------------------------------------------
# STEP 0: Automatic bead detection and edge data
# -------------------------------------------------
bead_masks = []
bead_start_frames = []
bead_edge_data = []  # list of (x_coords, y_coords, edges_float)

if track_individual_beads:
    previous_mask = base_mask.copy()

    for frame_idx, h_arr in enumerate(loaded_h_arrays):
        current_mask = (h_arr >= 0)
        new_area = current_mask & (~previous_mask)

        if np.any(new_area):
            bead_masks.append(new_area.copy())
            bead_start_frames.append(frame_idx)
            print(f"[DEBUG] Detected bead {len(bead_masks)} at frame {frame_idx}")
            previous_mask |= new_area

            if len(bead_masks) > 3:
                print("WARNING: More than 3 beads detected in mask evolution!")

    if bead_masks:
        # Precompute x/y grids for contours
        x_coords = np.linspace(0, nx * dx, nx)
        y_coords = np.linspace(0, ny * dy, ny)

        # Step 1 — build all raw edge masks
        for bmask in bead_masks:
            b = bmask
            edges_bool = np.zeros_like(b, dtype=bool)

            edges_bool[:-1, :] |= (b[:-1, :] & ~b[1:, :])  # bottom
            edges_bool[1:, :] |= (b[1:, :] & ~b[:-1, :])  # top
            edges_bool[:, :-1] |= (b[:, :-1] & ~b[:, 1:])  # right
            edges_bool[:, 1:] |= (b[:, 1:] & ~b[:, :-1])  # left

            edges = edges_bool.astype(float)
            bead_edge_data.append((x_coords, y_coords, edges))

        # Step 2 — clean overlapping edges
        cleaned_edges = []
        occupied = np.zeros_like(bead_edge_data[0][2], dtype=bool)

        for (x, y, edges) in bead_edge_data:
            e = edges > 0.5

            # Remove if directly adjacent (4-neighbor) to any already used edge pixel
            adj = (
                    np.pad(occupied[:, 1:], ((0, 0), (0, 1)), constant_values=False) |  # left neighbor
                    np.pad(occupied[:, :-1], ((0, 0), (1, 0)), constant_values=False) |  # right neighbor
                    np.pad(occupied[1:, :], ((0, 1), (0, 0)), constant_values=False) |  # top neighbor
                    np.pad(occupied[:-1, :], ((1, 0), (0, 0)), constant_values=False)  # bottom neighbor
            )

            e[adj] = False

            # Accept remaining pixels normally
            occupied |= e

            cleaned_edges.append((x, y, e.astype(float)))

        bead_edge_data = cleaned_edges

else:
    bead_masks = []
    bead_start_frames = []
    bead_edge_data = []

# ---------------------------- 2. Set Up the Figure and Axes ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=96)

# Create a 2x2 grid, bottom one spans both columns
fig = plt.figure(figsize=(16, 9), dpi=96)
gs = GridSpec(2, 2, height_ratios=[1.0, 0.8], figure=fig)
ax1 = fig.add_subplot(gs[0, 0])  # Temperature heatmap
ax2 = fig.add_subplot(gs[0, 1])  # Hydrogen heatmap
ax3 = fig.add_subplot(gs[1, :])  # New bottom plot (spans both columns)

# First subplot: Diffusion Heatmap
norm1 = mcolors.Normalize(vmin=0, vmax=800)
cmap1 = plt.get_cmap('hot').copy()
cmap1.set_under('0.85')
im1 = ax1.imshow(loaded_u_arrays[0], cmap=cmap1, norm=norm1,
                 extent=extent, origin="upper", aspect="equal")
ax1.set_title('Temperature')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('y [mm]')
cbar_ax1 = fig.add_axes([0.03, 0.15, 0.03, 0.7])
cbar_ax1.set_xlabel('$T$ [°C]', labelpad=20)
fig.colorbar(im1, cax=cbar_ax1)

# Second subplot: Diffusion Hydrogen
norm2 = mcolors.Normalize(vmin=-0.01, vmax=100)
cmap2 = plt.get_cmap('viridis').copy()
cmap2.set_under('0.85')
im2 = ax2.imshow(loaded_h_arrays[0], cmap=cmap2, norm=norm2,
                 extent=extent, origin="upper", aspect="equal")

# --- Draw static outline around regions where h == 0 ---
# draw_static_outline(ax2, loaded_h_arrays[0], dx, dy, color='darkgrey', linewidth=1.0, alpha=0.8)

diff_coeff_h = get_value("diff_coeff_h")
exponent = int(math.floor(math.log10(diff_coeff_h)))
mantissa = diff_coeff_h / 10**exponent
ax2.set_title(r"Hydrogen Diffusion" + "\n" + fr"$D_{{H}} = {mantissa:.2f} \times 10^{{{exponent}}}  [\,\mathrm{{mm}}^2/\mathrm{{s}}$]")
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('y [mm]')
cbar_ax2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax2.set_xlabel('Hydrogen Concentration [%]', labelpad=20)
fig.colorbar(im2, cax=cbar_ax2)

# ---------------------------- 2b. Max Temperature vs Time ----------------------------
# Compute maximum temperature from each frame
max_temps = [np.max(arr) for arr in loaded_u_arrays]

# Or use all times available:
last_idx = len(loaded_t_values)

# Compute average hydrogen concentration in WM bead (once, not animated)
avg_h_conc = []

for h_array in loaded_h_arrays:
    # 1. Sum all hydrogen values > 0 in the whole domain
    total_hydrogen = np.nansum(h_array[h_array > 0])

    # 2. figure out pixel amount in bead area
    current_mask = h_array >= 0
    bead_pixel_count = np.sum(current_mask & ~base_mask)

    # 3. Divide by the bead pixel count to normalize
    avg_conc = total_hydrogen / bead_pixel_count if bead_pixel_count > 0 else 0.0

    avg_h_conc.append(avg_conc)

# Base metal average (mask taken from initial state loaded_h_arrays[0])
base_pixel_count = np.sum(base_mask)
base_avg_h_conc = []
for h_array in loaded_h_arrays:
    if base_pixel_count > 0:
        base_avg_h_conc.append(np.nansum(h_array[base_mask]) / base_pixel_count)
    else:
        base_avg_h_conc.append(0.0)
base_avg_h_conc = np.array(base_avg_h_conc)

# Per-bead averages (dynamic: 0–3 beads depending on detection)
bead_avg_h_concs = []  # list of np.arrays, one per bead
if bead_masks:
    num_frames = len(loaded_h_arrays)
    for bmask in bead_masks:
        bead_pixels = np.sum(bmask)
        if bead_pixels == 0:
            # still keep length consistent
            bead_avg_h_concs.append(np.zeros(num_frames))
            continue

        series = []
        for h_array in loaded_h_arrays:
            series.append(np.nansum(h_array[bmask]) / bead_pixels)
        bead_avg_h_concs.append(np.array(series))

ax3.plot(np.array(loaded_t_values)[:last_idx],
         np.array(max_temps)[:last_idx],
         color='darkred', linewidth=2)
# Create a twin y-axis for hydrogen concentration
ax3b = ax3.twinx()
ax3b.plot(np.array(loaded_t_values),
          np.array(avg_h_conc),
          color='steelblue', linewidth=2, label='Avg H₂ Concentration')
ax3b.set_ylabel('H₂ Concentration [%]', color='black')
ax3b.tick_params(axis='y', labelcolor='black')

# Colors for bead averages (supports up to 3 beads)
bead_colors = ['tab:orange', 'tab:green', 'tab:purple']

for i, series in enumerate(bead_avg_h_concs):
    color = bead_colors[i % len(bead_colors)]
    ax3b.plot(
        np.array(loaded_t_values),
        series,
        linestyle='--',
        linewidth=1.5,
        color=color
        # label intentionally omitted here; legend is handled via dummy lines below
    )

# Base metal average (initial base_mask region)
base_color = 'black'
ax3b.plot(
    np.array(loaded_t_values),
    base_avg_h_conc,
    linestyle='-.',
    linewidth=1.5,
    color=base_color
    # label intentionally omitted here; legend is handled via dummy lines below
)

ax3.set_title('Maximum Temperature over Time')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Max Temperature [°C]')

ax3.set_xlim(0, max(loaded_t_values))
ax3.set_ylim(0, max(max_temps) * 1.01)
# --- NEW: y-limit must cover all hydrogen curves (WM, beads, base) ---
all_h_series = [np.array(avg_h_conc), base_avg_h_conc] + bead_avg_h_concs
max_h = max(np.max(series) for series in all_h_series) if all_h_series else max(avg_h_conc)
ax3b.set_ylim(0, max_h * 1.05)

# --- Add minor ticks every 2 seconds ---
ax3.xaxis.set_minor_locator(MultipleLocator(2))

# Optional: make minor ticks shorter or thinner for readability
ax3.tick_params(axis='x', which='minor', length=4, width=0.8)

# Grid style
ax3.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)

# ---------------------------- Add shaded weld phases ----------------------------
phase_start = 0
phases = []

if total_time_to_first_weld is not None:
    phases.append((phase_start, total_time_to_first_weld, 'Pre-Weld', 'lightgrey'))
    phase_start = total_time_to_first_weld

if total_time_to_cooling is not None:
    phases.append((phase_start, total_time_to_cooling, 'Welding', 'orange'))
    phase_start = total_time_to_cooling

if total_time_to_rt is not None:
    phases.append((phase_start, total_time_to_rt, 'Ice Water Drop', 'lightblue'))
    phase_start = total_time_to_rt

# Final phase (diffusion at RT)
max_time = max(loaded_t_values)
phases.append((phase_start, max_time, 'Ice Water', 'grey'))

# Plot shaded regions and centered labels
y_top = ax3.get_ylim()[1]
y_label = y_top * 0.95  # 5% down from top

for start, end, label, color in phases:
    ax3.axvspan(start, end, color=color, alpha=0.15)
    mid_x = (start + end) / 2
    duration = end - start

    # Choose time unit and format nicely
    if duration < 60:
        duration_str = f"({duration:.0f} s)"
    elif duration < 3600:
        duration_str = f"({duration/60:.1f} min)"
    else:
        duration_str = f"({duration/3600:.1f} h)"

    ax3.text(
        mid_x, y_label, f"{label}\n{duration_str}",
        ha='center', va='top', color='black',
        fontsize=11, weight='bold', alpha=0.8,
        linespacing=1.1
    )

ax3.axvspan(0, total_time_to_first_weld, color='lightgrey', alpha=0.2, label='Pre-weld')
ax3.axvspan(total_time_to_first_weld, total_time_to_cooling, color='orange', alpha=0.1, label='Welding')
ax3.axvspan(total_time_to_cooling, total_time_to_rt, color='lightblue', alpha=0.1, label='Cooling')

# Legend showing the main quantities + per-bead and base averages
lines1 = ax3.plot([], [], color='darkred', linewidth=2, label='Max. Temperature [°C]')
lines2 = ax3b.plot([], [], color='steelblue', linewidth=2, label='H₂ Concentration (total WM) [%]')

legend_handles = [lines1[0], lines2[0]]

# Dummy lines for bead averages (so legend matches the new curves)
bead_legend_colors = ['tab:orange', 'tab:green', 'tab:purple']
for i, _ in enumerate(bead_avg_h_concs):
    line, = ax3b.plot([], [], linestyle='--', linewidth=1.5,
                      color=bead_legend_colors[i % len(bead_legend_colors)],
                      label=f'H₂ Concentration (Bead {i+1}) [%]')
    legend_handles.append(line)

# Dummy line for base metal average
base_legend_line, = ax3b.plot([], [], linestyle='-.', linewidth=1.5,
                              color='black', label='H₂ Concentration (Base) [%]')
legend_handles.append(base_legend_line)

ax3.legend(handles=legend_handles, loc='center right', fontsize=10, frameon=True)

# --- General adjustments ---
fig.subplots_adjust(right=0.85, hspace=0.35)
real_time_text = ax2.text(-0.25, 1.2, f'Time: {0} s', transform=ax2.transAxes, color='black', fontsize=20)
# Axis scaling: show major ticks every 5 mm and minor ticks every 1 mm

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8)

# Add average HD indicator
avg_text = ax2.text(
    0.15, 0.15, "",
    transform=ax2.transAxes,
    color='black',
    fontsize=12,
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
)

# Define marker style with corrected 's', 'facecolors', and 'edgecolors'
marker_style = dict(marker='o', s=36, facecolors='white', edgecolors='white')  # s=36 for markersize=6

# Define annotation points and their settings
# # old for butt weld
space_above_in_pixels = 10

annotation_points = {
    'heat': [
        {'coords_mm': (27.0, 10.0), 'text_offset': (15, -20)},
        {'coords_mm': (17.5, 4.0), 'text_offset': (-50, -20)}
    ],
    'diffusion': [
        {'coords_mm': (17.5, 5.5), 'text_offset': (25, -12)},
        {'coords_mm': (17.5, 4.0), 'text_offset': (-120, -30)},
        {'coords_mm': (17.5, 10.0), 'text_offset': (25, 3)}
    ]
}


def preprocess_annotation_points(points_mm_list, dx, dy, array_shape):
    """Return a list with coords_mm (float mm) and coords_idx (int indices, clipped)."""
    ny_local, nx_local = array_shape
    processed = []
    for p in points_mm_list:
        x_mm, y_mm = p['coords_mm']
        # indices for sampling
        x_idx = int(round(x_mm / dx))
        y_idx = int(round(y_mm / dy))
        # clip to valid range to avoid IndexError
        x_idx = max(0, min(nx_local - 1, x_idx))
        y_idx = max(0, min(ny_local - 1, y_idx))
        processed.append({
            'coords_mm': (x_mm, y_mm),
            'coords_idx': (x_idx, y_idx),
            'text_offset': p.get('text_offset', (0, 0)),
        })
    return processed


# Function to create annotations
def create_annotations(ax, points, loaded_array, label):
    """
    points: list of dicts with:
      - coords_mm: (x_mm, y_mm)
      - coords_idx: (x_idx, y_idx)
      - text_offset: (dx_pts, dy_pts)
    """

    annotations = []
    for p in points:
        (x_mm, y_mm) = p['coords_mm']
        (x_idx, y_idx) = p['coords_idx']
        (x_off, y_off) = p['text_offset']

        # dot
        pt = ax.scatter(x_mm, y_mm, **marker_style)

        # label text
        text_val = (f'{loaded_array[y_idx, x_idx]:.0f}°C'
                    if label == 'Temp'
                    else f'{loaded_array[y_idx, x_idx]:.1f}%')

        ann = ax.annotate(
            text_val,
            xy=(x_mm, y_mm), xycoords='data',
            xytext=(x_off, y_off), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.3",
                            color='white'),
            fontsize=12, color='white'
        )

        annotations.append((pt, ann))

    return annotations



# -------------------------------------------------
# STEP 2: Bead outlines + bead annotations (setup)
# -------------------------------------------------
# Globals to be used in update()
bead_contours = []
bead_scatters = []
bead_annots = []
bead_centers_idx = []  # list of (x_idx, y_idx) for each bead


# Function to update the plot for each frame
def update(frame, loaded_u_arrays, loaded_h_arrays, heat_annotations, diff_annotations):
    loaded_u_array = loaded_u_arrays[frame]
    loaded_h_array = loaded_h_arrays[frame]
    real_time = loaded_t_values[frame]

    # time label
    if real_time > 18000:
        real_time_text.set_text(f'Time: {int(real_time / 3600)} h')
    elif real_time > 900:
        real_time_text.set_text(f'Time: {int(real_time / 60)} min')
    else:
        real_time_text.set_text(f'Time: {int(real_time)} s')

    # Update annotation texts with indices (temperature)
    for (pt, ann), p in zip(heat_annotations, heat_points):
        x_idx, y_idx = p['coords_idx']
        ann.set_text(f'{loaded_u_array[y_idx, x_idx]:.0f}°C')

    # Update annotation texts with indices (original diffusion points)
    for (pt, ann), p in zip(diff_annotations, diff_points):
        if not ann.get_visible():
            continue  # skip hidden ones (top two when bead tracking is on)
        x_idx, y_idx = p['coords_idx']
        ann.set_text(f'{loaded_h_array[y_idx, x_idx]:.1f}%')

    # 1. Sum all hydrogen values > 0 in the whole domain
    total_hydrogen = np.nansum(loaded_h_array[loaded_h_array > 0])

    # 2. Determine pixel count in the weld bead
    current_mask = loaded_h_array >= 0
    bead_pixel_count = np.sum(current_mask & ~base_mask)

    # 3. Normalize
    avg_conc = total_hydrogen / bead_pixel_count if bead_pixel_count > 0 else 0.0

    avg_text.set_text(f"Avg H₂ concentration in WM: {avg_conc:.1f}%")

    # --- STEP 1 & 2: per-bead average + visuals (if enabled) ---
    if track_individual_beads and bead_masks:
        # show/hide bead visuals depending on time
        for i in range(len(bead_masks)):
            visible_now = (frame >= bead_start_frames[i])
            if i < len(bead_edge_overlays):
                bead_edge_overlays[i].set_visible(visible_now)
            if i < len(bead_scatters):
                bead_scatters[i].set_visible(visible_now)
            if i < len(bead_annots):
                bead_annots[i].set_visible(visible_now)

            # Use value at centroid point instead of bead average
            if visible_now and i < len(bead_annots) and i < len(bead_centers_idx):
                cx_idx, cy_idx = bead_centers_idx[i]
                local_val = loaded_h_array[cy_idx, cx_idx]
                bead_annots[i].set_text(f'{local_val:.1f}%')

    # Update images
    im1.set_data(loaded_u_array)
    im2.set_data(loaded_h_array)

    # Debug, comment out if you like the annotation setup and what not.
    # if frame == 100:
    #     plt.show()

    return [im1, im2, real_time_text] + heat_annotations + diff_annotations + bead_annots


# Precompute idx from mm once
heat_points = preprocess_annotation_points(annotation_points['heat'], dx, dy, loaded_u_arrays[0].shape)
diff_points = preprocess_annotation_points(annotation_points['diffusion'], dx, dy, loaded_h_arrays[0].shape)

# Create annotations using first frame
heat_annotations = create_annotations(ax1, heat_points, loaded_u_arrays[0], 'Temp')
diff_annotations = create_annotations(ax2, diff_points, loaded_h_arrays[0], 'HD')

# -------------------------------------------------
# STEP 2 (continued): Create bead outlines & bead annotations,
# and hide top two original diffusion annotations if needed
# -------------------------------------------------
if track_individual_beads and bead_masks:
    # 1) Hide two static hydrogen annotations, keep the bottom-most one
    y_vals = [p['coords_mm'][1] for p in diff_points]  # y in mm
    bottom_idx = int(np.argmax(y_vals))  # largest y = bottom-most

    hide_indices = [i for i in range(len(diff_points)) if i != bottom_idx]

    for idx in hide_indices:
        pt, ann = diff_annotations[idx]
        pt.set_visible(False)
        ann.set_visible(False)

    # 2) Create pixel-level edge overlays (initially invisible)
    bead_edge_overlays = []

    for (x_coords, y_coords, edges) in bead_edge_data:
        # edges is already a float array with 1 where edges exist and 0 elsewhere
        overlay = np.zeros((*edges.shape, 4))  # RGBA image

        # Color only the edge pixels
        overlay[edges > 0.5] = [0, 0, 0, 0.8]  # black, alpha=0.8

        img = ax2.imshow(
            overlay,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            zorder=8
        )
        img.set_visible(False)
        bead_edge_overlays.append(img)

    # 3) Create bead annotations (centroid-based), initially invisible
    offsets = [(-60, -30), (30, -30), (0, -50)]  # rough offsets; can be tuned later

    for i, bmask in enumerate(bead_masks):
        ys, xs = np.where(bmask)
        if xs.size == 0:
            continue

        cx_idx = int(round(xs.mean()))
        cy_idx = int(round(ys.mean()))
        bead_centers_idx.append((cx_idx, cy_idx))

        cx_mm = cx_idx * dx
        cy_mm = cy_idx * dy
        x_off, y_off = offsets[i] if i < len(offsets) else (20, -20)

        # Scatter marker (white dot), hidden initially
        pt = ax2.scatter(cx_mm, cy_mm, **marker_style)
        pt.set_visible(False)

        # Initial text uses first frame's value, then updated in update()
        text_val = f'{loaded_h_arrays[0][cy_idx, cx_idx]:.1f}%'
        ann = ax2.annotate(
            text_val,
            xy=(cx_mm, cy_mm), xycoords='data',
            xytext=(x_off, y_off), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='white'),
            fontsize=12, color='white', zorder=12
        )
        ann.set_visible(False)

        bead_scatters.append(pt)
        bead_annots.append(ann)

# ---------------------------- 3. Set Up the Video Writer ----------------------------
# Define the video writer with desired settings
writer = FFMpegWriter(fps=30, metadata=dict(artist='Your Name'), bitrate=1800)

# ---------------------------- 4. Create and Save the Animation with Progress Bar ----------------------------
print("Converting to .mp4 now. This may take some time. Please wait...")

with writer.saving(fig, animation_file_name, dpi=96):
    for frame in tqdm(range(len(loaded_u_arrays)), desc="Rendering Animation Frames"):
        # Update the plot with the current frame
        update(frame, loaded_u_arrays, loaded_h_arrays, heat_annotations, diff_annotations)

        # Grab the current frame and write it to the video
        writer.grab_frame()

print("Animation saved successfully!")

# ---------------------------- 5. Clean Up ----------------------------
plt.close(fig)

# # Render a specific frame (e.g., frame 10)
# frame_to_render = 10
#
# # Create annotations for the specific frame
# heat_annotations = create_annotations(ax1, annotation_points['heat'], loaded_u_arrays[frame_to_render], 'Temp')
# diff_annotations = create_annotations(ax2, annotation_points['diffusion'], loaded_h_arrays[frame_to_render], 'HD')
#
# # Update the plot with the chosen frame
# update(frame_to_render, loaded_u_arrays, loaded_h_arrays, heat_annotations, diff_annotations)
#
# # Save the frame as an image
# plt.savefig(r"02_Results\frame_10.png", dpi=96)
# plt.close(fig)
