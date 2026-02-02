import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import math
import io
import contextlib
import numpy as np

# Suppress config prints during import, so no double prints. They’ll still appear when scripts run directly.
with contextlib.redirect_stdout(io.StringIO()):
    from b4_functions import in_results, load_param_config_json

# ---------------------------- 1. Load Data from HDF5 File ----------------------------
file_name = str(in_results("00_diffusion_array.h5"))  # <-- set the file you want
param_cfg = load_param_config_json(file_name)
animation_file_name = param_cfg["animation_name"]
loaded_u_arrays = []
loaded_h_arrays = []
loaded_t_values = []
dx, dy = param_cfg["dx"], param_cfg["dy"]
frame_stride = param_cfg["animation_frame_stride"]   # only render every n-th frame

with h5py.File(file_name, 'r') as hf:
    for key in sorted(hf.keys()):
        if key.startswith('u_snapshot_'):
            loaded_u_arrays.append(hf[key][:])
        elif key.startswith('h_snapshot_'):
            loaded_h_arrays.append(hf[key][:])
        elif key.startswith('t_snapshot_'):
            t_value = hf[key][()]
            loaded_t_values.append(t_value)

if len(loaded_t_values) >= 2:
    N = min(20, len(loaded_t_values) - 1)
    dt_sim = float(np.mean(np.diff(loaded_t_values[:N + 1])))
else:
    dt_sim = float("nan")

fps = 30
dt_frame = dt_sim * frame_stride
speedup = dt_frame * fps   # sim-seconds per video-second
print(f"Video Render: dt_sim={dt_sim:.3g}s | stride={frame_stride} → dt_frame={dt_frame:.3g}s | speed≈{speedup:.1f}×")

ny, nx = loaded_u_arrays[0].shape
extent = [0, nx*dx, ny*dy, 0]  # x_min, x_max, y_max, y_min (origin="upper")

# ---------------------------- 2. Set Up the Figure and Axes ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=96)

# First subplot: Diffusion Heatmap
norm1 = mcolors.Normalize(vmin=24, vmax=400)
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
norm2 = mcolors.Normalize(vmin=-0.1, vmax=100)
cmap2 = plt.get_cmap('viridis').copy()
cmap2.set_under('0.85')
im2 = ax2.imshow(loaded_h_arrays[0], cmap=cmap2, norm=norm2,
                 extent=extent, origin="upper", aspect="equal")
diff_coeff_h = param_cfg["diff_coeff_h"]
exponent = int(math.floor(math.log10(diff_coeff_h)))
mantissa = diff_coeff_h / 10**exponent
# ax2.set_title(r"Hydrogen Diffusion" + "\n" + fr"$D_{{H}} = {mantissa:.2f} \times 10^{{{exponent}}}  [\,\mathrm{{mm}}^2/\mathrm{{s}}$]")
ax2.set_title(r"Hydrogen Diffusion")
ax2.set_xlabel('x [mm]')
ax2.set_ylabel('y [mm]')
cbar_ax2 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax2.set_xlabel('Hydrogen Concentration [%]', labelpad=20)
fig.colorbar(im2, cax=cbar_ax2)

# General adjustments
fig.subplots_adjust(right=0.85)
real_time_text = ax2.text(-0.25, 1.5, f'Time: {0} s', transform=ax2.transAxes, color='black', fontsize=20)
# Axis scaling: show major ticks every 5 mm and minor ticks every 1 mm
for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8)


# Define marker style with corrected 's', 'facecolors', and 'edgecolors'
marker_style = dict(marker='o', s=36, facecolors='white', edgecolors='white')  # s=36 for markersize=6

# Define annotation points and their settings
# # old for butt weld
space_above_in_pixels = 10

annotation_points = {
    'heat': [
        {'coords_mm': (10.0, 6.5), 'text_offset': (15, -20)},
        {'coords_mm': (130.0, 6.5), 'text_offset': (-65, -20)}

    ],
    'diffusion': [
        {'coords_mm': (70, 6.5), 'text_offset': (25, -12)},
        {'coords_mm': (70, 15.0), 'text_offset': (-60, -30)},
        {'coords_mm': (70, 25), 'text_offset': (25, 3)}
    ]
}


def preprocess_annotation_points(points_mm_list, dx, dy, array_shape):
    """Return a list with coords_mm (float mm) and coords_idx (int indices, clipped)."""
    ny, nx = array_shape
    processed = []
    for p in points_mm_list:
        x_mm, y_mm = p['coords_mm']
        # indices for sampling
        x_idx = int(round(x_mm / dx))
        y_idx = int(round(y_mm / dy))
        # clip to valid range to avoid IndexError
        x_idx = max(0, min(nx - 1, x_idx))
        y_idx = max(0, min(ny - 1, y_idx))
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

        ax.scatter(x_mm, y_mm, **marker_style)

        text_val = (f'{loaded_array[y_idx, x_idx]:.0f}°C'
                    if label == 'Temp'
                    else f'{loaded_array[y_idx, x_idx]:.1f}%')

        ann = ax.annotate(
            text_val,
            xy=(x_mm, y_mm), xycoords='data',
            xytext=(x_off, y_off), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='white'),
            fontsize=12, color='white'
        )
        annotations.append(ann)
    return annotations


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

    # Update annotation texts with indices
    for ann, p in zip(heat_annotations, heat_points):
        x_idx, y_idx = p['coords_idx']
        ann.set_text(f'{loaded_u_array[y_idx, x_idx]:.0f}°C')

    for ann, p in zip(diff_annotations, diff_points):
        x_idx, y_idx = p['coords_idx']
        ann.set_text(f'{loaded_h_array[y_idx, x_idx]:.1f}%')

    # Update images
    im1.set_data(loaded_u_array)
    im2.set_data(loaded_h_array)

    # Debug, comment out if you like the annotation setup and what not.
    # if frame == 100:
    #     plt.show()

    return [im1, im2, real_time_text] + heat_annotations + diff_annotations


# Precompute idx from mm once
heat_points = preprocess_annotation_points(annotation_points['heat'], dx, dy, loaded_u_arrays[0].shape)
diff_points = preprocess_annotation_points(annotation_points['diffusion'], dx, dy, loaded_h_arrays[0].shape)

# Create annotations using first frame
heat_annotations = create_annotations(ax1, heat_points, loaded_u_arrays[0], 'Temp')
diff_annotations = create_annotations(ax2, diff_points, loaded_h_arrays[0], 'HD')


# ---------------------------- 3. Set Up the Video Writer ----------------------------
# Define the video writer with desired settings
writer = FFMpegWriter(fps=fps, metadata=dict(artist='Your Name'), bitrate=1800)

# ---------------------------- 4. Create and Save the Animation with Progress Bar ----------------------------
print("Converting to .mp4 now. This may take some time. Please wait...")

with writer.saving(fig, animation_file_name, dpi=96):
    for frame in tqdm(range(0, len(loaded_u_arrays), frame_stride), desc="Rendering Animation Frames"):
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
# plt.savefig(str(in_results("frame_10.png", mkdir=True)), dpi=96)
# plt.close(fig)
