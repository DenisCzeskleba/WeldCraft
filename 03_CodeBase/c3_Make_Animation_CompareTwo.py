import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import numpy as np
import time

# Load two .h5 files for comparison
file1_name = r"02_Results\03_Batch-Executions\XXX.h5"
file2_name = r"02_Results\03_Batch-Executions\XXX.h5"

# Optional end time for limiting the data used
end_time = 0  # Set to 0 for using all data, or a specific time limit (e.g., 60000 seconds)


# ------------------ 1. Load data arrays and time values --------------------------------
def load_diffusion_arrays(file_name, end_time):
    loaded_u_arrays = []
    loaded_h_arrays = []
    loaded_t_values = []
    with h5py.File(file_name, 'r') as hf:
        for key in hf.keys():
            if key.startswith('u_snapshot_'):
                loaded_u_arrays.append(hf[key][:])  # Temperature array
            elif key.startswith('h_snapshot_'):
                loaded_h_arrays.append(hf[key][:])  # Hydrogen array
            elif key.startswith('t_snapshot_'):
                t_value = hf[key][()]
                if end_time == 0 or t_value <= end_time:
                    loaded_t_values.append(t_value)

    # Trim arrays based on end_time if needed
    if end_time > 0:
        max_index = len(loaded_t_values)
        loaded_u_arrays = loaded_u_arrays[:max_index]
        loaded_h_arrays = loaded_h_arrays[:max_index]

    return loaded_u_arrays, loaded_h_arrays, loaded_t_values


loaded_u_arrays_1, loaded_h_arrays_1, loaded_t_values_1 = load_diffusion_arrays(file1_name, end_time)
loaded_u_arrays_2, loaded_h_arrays_2, loaded_t_values_2 = load_diffusion_arrays(file2_name, end_time)

# ------------------ 2. Create the figure and subplots ---------------------------------
fig = plt.figure(figsize=(16, 9), dpi=96)

# A GridSpec layout: 2 rows, 6 columns (2 columns per animation, 3 columns per diagram)
gs = GridSpec(
    2, 6,
    height_ratios=[2, 1],
    width_ratios=[1, 1, 1, 1, 1, 1],
    hspace=0.2, wspace=0.1
)

# Temperature map in the middle (top row, columns 2-3)
ax_top = fig.add_subplot(gs[0, 2:4])

# Hydrogen maps on left (columns 0-1) and right (columns 4-5)
ax_bottom_left  = fig.add_subplot(gs[0, 0:2])
ax_bottom_right = fig.add_subplot(gs[0, 4:6])

# Max hydrogen vs time plot below (second row, columns 0-6)
ax_hydrogen_max = fig.add_subplot(gs[1, 0:6])

# ------------------ 3. Set up colormaps and imshow objects -----------------------------
norm_heat      = mcolors.Normalize(vmin=24,   vmax=400)
norm_hydrogen = mcolors.Normalize(vmin=-0.1, vmax=100)

cmap_heat = plt.get_cmap('hot')
cmap_heat.set_under('0.85')

cmap_hydrogen = plt.get_cmap('viridis')
cmap_hydrogen.set_under('0.85')

im_top          = ax_top.imshow(loaded_u_arrays_1[0],  cmap=cmap_heat,     norm=norm_heat)
im_bottom_left  = ax_bottom_left.imshow(loaded_h_arrays_1[0], cmap=cmap_hydrogen, norm=norm_hydrogen)
im_bottom_right = ax_bottom_right.imshow(loaded_h_arrays_2[0], cmap=cmap_hydrogen, norm=norm_hydrogen)

# Titles and axis labels
ax_top.         set_title('Heat map')
ax_bottom_left. set_title(r'Hydrogen Diffusion ($D_{min}$)')
ax_bottom_right.set_title(r'Hydrogen Diffusion ($D_{max}$)')
ax_bottom_left.set_xlabel('');  ax_bottom_left.set_ylabel('weld height [0.5mm]')
ax_bottom_right.set_xlabel('')
ax_top.set_xlabel('weld width [0.5mm]')

# Hide Y ticks on the middle (top) and right axes for clarity
ax_top.set_yticklabels([])
ax_bottom_right.set_yticklabels([])

# ------------------ 4. Add colorbars ---------------------------------------------------
cbar_ax_heat      = fig.add_axes([0.05, 0.35, 0.02, 0.5])
cbar_ax_hydrogen  = fig.add_axes([0.92, 0.35, 0.02, 0.5])

cbar_top = fig.colorbar(im_top, cax=cbar_ax_heat)
cbar_top.set_label('$T$ [C°]', labelpad=20, loc='top')

cbar_bottom_right = fig.colorbar(im_bottom_right, cax=cbar_ax_hydrogen)
cbar_bottom_right.set_label('Hydrogen Concentration %', labelpad=20, loc='top')

# ------------------ 5. “Heat” Annotations ----------------------------------------------
marker_style = dict(marker='o', markersize=6, markerfacecolor='white', markeredgecolor='white')

annotation_points = {
    # We keep only 'heat' – remove 'diffusion1' and 'diffusion2'
    'heat': [{'coords': (20, 13), 'text_offset': (0, 20)}],
}


def create_annotations(ax, points, loaded_array, label, frame=0):
    annotations = []
    for point in points:
        x, y = point['coords']
        x_offset, y_offset = point['text_offset']
        arr = loaded_array[frame]
        # Plot the marker
        ax.scatter(x, y, s=marker_style['markersize'],
                   c=marker_style['markerfacecolor'],
                   edgecolors=marker_style['markeredgecolor'])
        # Create the annotation
        annotation = ax.annotate(
            f'{label}: {arr[y, x]:.0f}',
            xy=(x, y),
            xycoords='data',
            xytext=(x + x_offset, y + y_offset),
            textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='white'),
            fontsize=12,
            color='white'
        )
        annotations.append(annotation)
    return annotations

heat_annotations = create_annotations(ax_top, annotation_points['heat'], loaded_u_arrays_1, 'Temp')


# ------------------ 6. Calculate hydrogen stats for circles/plots ----------------------
def calculate_hydrogen_stats(loaded_h_arrays):
    total_hydrogen = []
    max_hydrogen   = []
    max_locations  = []
    for h_array in loaded_h_arrays:
        total_h = np.sum(h_array[h_array >= 0])
        max_h   = np.max(h_array[h_array >= 0])
        max_loc = np.unravel_index(np.argmax(h_array, axis=None), h_array.shape)
        total_hydrogen.append(total_h)
        max_hydrogen.append(max_h)
        max_locations.append(max_loc)
    return total_hydrogen, max_hydrogen, max_locations


total_hydrogen_1, max_hydrogen_1, max_locations_1 = calculate_hydrogen_stats(loaded_h_arrays_1)
total_hydrogen_2, max_hydrogen_2, max_locations_2 = calculate_hydrogen_stats(loaded_h_arrays_2)

# ------------------ 7. Plot max hydrogen concentration over time -----------------------
ax_hydrogen_max.plot(loaded_t_values_1, max_hydrogen_1, label=r'No Hydrogen on Inside Wall', color='red')
ax_hydrogen_max.plot(loaded_t_values_2, max_hydrogen_2, label=r"Diffusion limited on Inside Wall (Sieverts law)", color='black')
ax_hydrogen_max.set_title('Maximum Hydrogen Concentration')
ax_hydrogen_max.set_xlabel('Time [s]')
ax_hydrogen_max.set_ylabel('Concentration [%]')
ax_hydrogen_max.legend()

ax_hydrogen_max.set_xlim([0, 2950])
ax_hydrogen_max.set_ylim([0, 500])
ax_hydrogen_max.grid(True)

# Add a shaded area to highlight a region in time
highlight_start = 1150
highlight_end   = 2950
ax_hydrogen_max.axvspan(highlight_start, highlight_end, color='gray', alpha=0.3)
highlight_mid = (highlight_start + highlight_end) / 2
ax_hydrogen_max.text(
    highlight_mid, 0.1, "Cooling to room temperature",
    ha='center', va='bottom', fontsize=12, color='black', alpha=1,
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5', alpha=0)
)

real_time_text = ax_top.text(-22.5, 0.90, f'Time: {0} s', transform=plt.gca().transAxes, color='black', fontsize=20)

# ------------------ Find the last visible time within the xlim ----------------------
# Find the time value closest to the right end of the xlim (16266)
last_time_visible_1 = min(loaded_t_values_1, key=lambda x: abs(x - 2950))
last_value_visible_1 = max_hydrogen_1[loaded_t_values_1.index(last_time_visible_1)]

last_time_visible_2 = min(loaded_t_values_2, key=lambda x: abs(x - 2950))
last_value_visible_2 = max_hydrogen_2[loaded_t_values_2.index(last_time_visible_2)]

# ------------------ Add arrows and annotations for the visible last points ----------------------
# Arrow for file 1 (pointing down)
ax_hydrogen_max.annotate(
    f'{last_value_visible_1:.2f}%',
    xy=(last_time_visible_1, last_value_visible_1),
    xytext=(last_time_visible_1 - 180, last_value_visible_1 + 25),  # Adjust this for better visibility
    textcoords='data',
    arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2),  # Ensure arrow is red
    fontsize=12, ha='center', color='red'  # Text color is red
)

# Arrow for file 2 (pointing up)
ax_hydrogen_max.annotate(
    f'{last_value_visible_2:.2f}%',
    xy=(last_time_visible_2, last_value_visible_2),
    xytext=(last_time_visible_2 - 180, last_value_visible_2 - 45),  # Adjust this for better visibility
    textcoords='data',
    arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle="->", lw=2),  # Ensure arrow is black
    fontsize=12, ha='center', color='black'  # Text color is black
)

# ------------------ 8. Update logic: draw circles + text near them ---------------------
# We'll store references to circle-based annotations so we can remove them each frame
circle_patches  = []
circle_labels   = []


def update(frame):
    # Update images
    im_top.set_data(loaded_u_arrays_1[frame])
    im_bottom_left.set_data(loaded_h_arrays_1[frame])
    im_bottom_right.set_data(loaded_h_arrays_2[frame])
    real_time = loaded_t_values_1[frame]

    # Update the time text
    if real_time > 18000:
        real_time_text.set_text(f'Time: {int(real_time / 3600)} h')
    elif real_time > 900:
        real_time_text.set_text(f'Time: {int(real_time / 60)} min')
    else:
        real_time_text.set_text(f'Time: {int(real_time)} s')

    # Clear out old circle patches and labels
    global circle_patches, circle_labels
    for patch in circle_patches:
        patch.remove()
    for lbl in circle_labels:
        lbl.remove()
    circle_patches = []
    circle_labels  = []

    # ---------------- Max hydrogen for first array ---------------
    max_loc_1 = max_locations_1[frame]  # (row, col)
    circle_1 = plt.Circle(
        (max_loc_1[1], max_loc_1[0]),
        radius=6, color='white', fill=False, lw=2
    )
    ax_bottom_left.add_patch(circle_1)
    circle_patches.append(circle_1)

    # Label near circle 1
    offset_1 = (-17, 20)
    val_1 = loaded_h_arrays_1[frame][max_loc_1[0], max_loc_1[1]]
    ann_1 = ax_bottom_left.annotate(
        f'{val_1:.1f}%',
        xy=(max_loc_1[1], max_loc_1[0]),
        xycoords='data',
        xytext=(max_loc_1[1] + offset_1[0], max_loc_1[0] + offset_1[1]),
        textcoords='data',
        fontsize=12,
        color='white'
    )
    circle_labels.append(ann_1)

    # ---------------- Max hydrogen for second array --------------
    max_loc_2 = max_locations_2[frame]
    circle_2 = plt.Circle(
        (max_loc_2[1], max_loc_2[0]),
        radius=6, color='white', fill=False, lw=2
    )
    ax_bottom_right.add_patch(circle_2)
    circle_patches.append(circle_2)

    # Label near circle 2
    offset_2 = (-17, 20)
    val_2 = loaded_h_arrays_2[frame][max_loc_2[0], max_loc_2[1]]
    ann_2 = ax_bottom_right.annotate(
        f'{val_2:.1f}%',
        xy=(max_loc_2[1], max_loc_2[0]),
        xycoords='data',
        xytext=(max_loc_2[1] + offset_2[0], max_loc_2[0] + offset_2[1]),
        textcoords='data',
        fontsize=12,
        color='white'
    )
    circle_labels.append(ann_2)

    # ---------------- Update the existing heat annotations -------------
    # (if you still want them to follow the same point each frame)
    for ann, point in zip(heat_annotations, annotation_points['heat']):
        x, y = point['coords']
        ann.set_text(f'{loaded_u_arrays_1[frame][y, x]:.0f}°C')

    # Return updated artists (not strictly necessary unless blitting)
    return [im_top, im_bottom_left, im_bottom_right] + circle_patches + circle_labels + heat_annotations


# ------------------ 9. Save frames to MP4 with FFMpegWriter --------------------------------
print("Converting to .mp4 now. This may take some time. Please wait...")

writer = FFMpegWriter(fps=30, metadata=dict(artist='Denis Czeskleba'), bitrate=1800)

start_time = time.time()
with writer.saving(fig, r"02_Results\diffusion_comparison_animation.mp4", dpi=96):
    for frame in tqdm(range(len(loaded_u_arrays_1)), desc="Rendering Animation Frames"):
        update(frame)
        writer.grab_frame()

end_time = time.time()
print(f"Animation saved successfully! Time taken: {end_time - start_time:.2f} seconds")

# Close the figure
plt.close(fig)
