import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import numpy as np
from b4_functions import in_results

# ---------------------------- 0. Configuration ----------------------------
# Animation Mode
animation_mode = "hydrogen comparison"  # Only "hydrogen comparison" is handled in this script

# HDF5 File Name
file_name = str(in_results("00_diffusion_array.h5"))

# Output Video File Name
output_file = str(in_results("result_visualize_gradient_animation.mp4", mkdir=True))

# Frames Per Second for the Output Video
fps = 60

# Bitrate for the Output Video
bitrate = 1800

# Relative Threshold for Detecting Significant dt Changes (e.g., 10%)
dt_threshold = 0.10

# ---------------------------- 1. Load Data from HDF5 File ----------------------------
loaded_h_arrays = []  # Hydrogen Diffusion Coefficients
loaded_d_arrays = []  # Heat Diffusion Coefficients
loaded_t_values = []  # Time Snapshots

with h5py.File(file_name, 'r') as hf:
    # Ensure keys are sorted to maintain chronological order
    snapshot_keys = [
        k for k in hf.keys()
        if any(k.startswith(prefix) for prefix in ('h_snapshot_', 'd_snapshot_', 't_snapshot_'))
    ]
    sorted_keys = sorted(snapshot_keys, key=lambda x: int(x.split('_')[-1]))

    for key in sorted_keys:
        if key.startswith('h_snapshot_'):
            loaded_h_arrays.append(hf[key][:])
        elif key.startswith('d_snapshot_'):
            loaded_d_arrays.append(hf[key][:])
        elif key.startswith('t_snapshot_'):
            t_value = hf[key][()]
            loaded_t_values.append(t_value)

# ---------------------------- 2. Verify Data Consistency ----------------------------
if not (len(loaded_h_arrays) == len(loaded_d_arrays) == len(loaded_t_values)):
    raise ValueError("Mismatch in the number of h, d, and t snapshots.")

if not loaded_h_arrays:
    raise ValueError("No data found in the HDF5 file.")

# ---------------------------- 3. Precompute Normalization Parameters ----------------------------
print("Precomputing normalized changes and determining color scaling parameters...")

changes = []
dt_seconds = []  # Time differences between frames in seconds

prev_dt = None
previous_change = None

for frame in range(len(loaded_h_arrays)):
    if frame == 0:
        # No previous frame to compare; set change to zero
        change = np.zeros_like(loaded_h_arrays[0])
        dt = 1  # Default to 1 second to prevent division by zero
    elif frame == 1:
        # For frame 1, compute the normalized change as usual without checking dt change
        dt = loaded_t_values[frame] - loaded_t_values[frame - 1]
        if dt <= 0:
            print(f"Warning: Non-positive dt ({dt}) at frame {frame}. Setting dt to 1 second.")
            dt = 1  # Prevent division by zero or negative dt
        change = (loaded_h_arrays[frame] - loaded_h_arrays[frame - 1]) / dt
        previous_change = change.copy()  # Initialize previous_change
    else:
        # For frame >=2, check for significant dt change
        dt = loaded_t_values[frame] - loaded_t_values[frame - 1]
        if dt <= 0:
            print(f"Warning: Non-positive dt ({dt}) at frame {frame}. Setting dt to 1 second.")
            dt = 1  # Prevent division by zero or negative dt

        if prev_dt is not None:
            relative_change = abs(dt - prev_dt) / prev_dt
            if relative_change > dt_threshold:
                print(f"Significant dt change detected at frame {frame}: dt={dt}, prev_dt={prev_dt}")
                if previous_change is not None:
                    # Reuse the previous change to prevent flickering
                    change = previous_change.copy()
                else:
                    # If previous_change is None (shouldn't happen), compute as usual
                    change = (loaded_h_arrays[frame] - loaded_h_arrays[frame - 1]) / dt
            else:
                # Compute the normalized change
                change = (loaded_h_arrays[frame] - loaded_h_arrays[frame - 1]) / dt
                previous_change = change.copy()
        else:
            # If prev_dt is None (only possible for frame=1), compute as usual
            change = (loaded_h_arrays[frame] - loaded_h_arrays[frame - 1]) / dt
            previous_change = change.copy()

    changes.append(change)
    dt_seconds.append(dt)
    prev_dt = dt

# Stack changes to compute global min and max excluding top 20% extreme changes
changes_stack = np.stack(changes)
flattened_changes = np.abs(changes_stack.flatten())

# Determine the 80th percentile of the absolute changes
percentile_80 = np.percentile(flattened_changes, 80)

# Set abs_max to the 80th percentile value for symmetric normalization
abs_max = percentile_80
norm_diff = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)

print(f"Normalization based on the 80th percentile: ±{abs_max:.2f}")

# ---------------------------- 4. Set Up the Figure and Axes ----------------------------
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=96)

# Precompute min and max for d_snapshot_ (Heat Diffusion Coefficients) to set base image normalization
d_min = min(arr.min() for arr in loaded_d_arrays)
d_max = max(arr.max() for arr in loaded_d_arrays)

# Initialize the base image using d_snapshot_ (Heat Diffusion Coefficients)
im_base = ax.imshow(loaded_d_arrays[0], cmap='gray', norm=mcolors.Normalize(vmin=d_min, vmax=d_max))
ax.set_title('Hydrogen Diffusion Change Overlay')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Initialize the change image with semi-transparency
cmap_diff = plt.get_cmap('coolwarm').copy()
# Removed set_under and set_over to allow natural color mapping for out-of-bounds values
im_diff = ax.imshow(changes[0], cmap=cmap_diff, norm=norm_diff, alpha=0.5)

# Colorbar for change image only
cbar_diff = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_diff.set_xlabel('Change in HD per second', labelpad=20)
fig.colorbar(im_diff, cax=cbar_diff)

# General adjustments
fig.subplots_adjust(right=0.85)

# Initialize real_time_text and position it higher with a bounding box for better visibility
real_time_text = ax.text(0.02, 1.05, f'Real Time: {0} s', transform=ax.transAxes, color='black', fontsize=12,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


# ---------------------------- 5. Function to Update the Plot ----------------------------
def update(frame, loaded_d_arrays, changes, loaded_t_values):
    if frame >= len(loaded_d_arrays):
        print(f"Frame {frame} is out of range. Skipping.")
        return [im_base, im_diff, real_time_text]

    # Update the base image
    im_base.set_data(loaded_d_arrays[frame])

    # Update the change image
    im_diff.set_data(changes[frame])

    # Update the real_time_text
    real_time = loaded_t_values[frame]
    if real_time > 18000:
        real_time_formatted = f'{int(real_time / 3600)} h'
    elif real_time > 900:
        real_time_formatted = f'{int(real_time / 60)} min'
    else:
        real_time_formatted = f'{int(real_time)} s'
    real_time_text.set_text(f'Real Time: {real_time_formatted}')

    return [im_base, im_diff, real_time_text]


# ---------------------------- 6. Set Up the Video Writer ----------------------------
writer = FFMpegWriter(fps=fps, metadata=dict(artist='Your Name'), bitrate=bitrate)

# ---------------------------- 7. Create and Save the Animation with Progress Bar ----------------------------
print(f"Converting to {output_file} now. This may take some time. Please wait...")

with writer.saving(fig, output_file, dpi=96):
    for frame in tqdm(range(len(loaded_h_arrays)), desc="Saving Animation Frames"):
        # Update the plot with the current frame
        update(frame, loaded_d_arrays, changes, loaded_t_values)

        # Grab the current frame and write it to the video
        writer.grab_frame()

print(f"Animation saved successfully as {output_file}!")

# ---------------------------- 8. Clean Up ----------------------------
plt.close(fig)
