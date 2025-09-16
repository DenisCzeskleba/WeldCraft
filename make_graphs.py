import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import numpy as np

# Parameters and Configurations
file_name = 'diffusion_array.h5'  # The HDF5 file to load data from
target_times_to_display = [970, 1430, 1700, 1600000]  # Target times in seconds
graph_style = "Temp/Diffusion"  # Current options: Temp/Diffusion / Boundary

# Define annotation options
marker_style = dict(marker='o', markersize=6, markerfacecolor='white', markeredgecolor='white')
annotation_points = {
    'heat': [
        {'coords': (60, 11), 'text_offset': (0, 48)},
        {'coords': (90, 85), 'text_offset': (55, 3)}
    ],
    'diffusion': [
        {'coords': (90, 85), 'text_offset': (55, 3)},
        {'coords': (100, 10), 'text_offset': (55, 10)},
        {'coords': (90, 40), 'text_offset': (55, 3)}
    ]
}


# Function to create annotations
def create_annotations(ax, points, loaded_array, label):
    annotations = []
    for point in points:
        x, y = point['coords']
        x_offset, y_offset = point['text_offset']
        ax.scatter(x, y, s=marker_style['markersize'], c=marker_style['markerfacecolor'],
                   edgecolors=marker_style['markeredgecolor'])
        annotation = ax.annotate(f'{label}: {loaded_array[y, x]:.0f}', xy=(x, y), xycoords='data',
                                 xytext=(x + x_offset, y + y_offset), textcoords='data',
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='white'),
                                 fontsize=12, color='white')
        annotations.append(annotation)
    return annotations


# Function to retrieve specific step data and corresponding time
def get_step(simulation, target_time_step, file_name):
    # Load time step and time from the provided file
    with h5py.File(file_name, 'r') as hf:
        target_key1 = f'u_snapshot_{target_time_step:05d}'
        target_key2 = f'h_snapshot_{target_time_step:05d}'
        target_key3 = f't_snapshot_{target_time_step:05d}'
        loaded_u_array = loaded_h_array = loaded_t_value = None
        if target_key1 in hf:
            loaded_u_array = hf[target_key1][:]
        else:
            print(f'Dataset u for time step {target_time_step} not found in the file.')
        if target_key2 in hf:
            loaded_h_array = hf[target_key2][:]
        else:
            print(f'Dataset h for time step {target_time_step} not found in the file.')
        if target_key3 in hf:
            loaded_t_value = hf[target_key3][()]  # Read the saved time
        else:
            print(f'Dataset t for time step {target_time_step} not found in the file.')

    return loaded_u_array, loaded_h_array, loaded_t_value


# Function to get the closest time indices
def find_closest_time_steps(target_times, file_name):
    # Load all time steps from the file
    time_snapshots = []
    with h5py.File(file_name, 'r') as hf:
        for key in hf.keys():
            if key.startswith('t_snapshot_'):
                t_value = hf[key][()]  # Extract the scalar time value
                time_snapshots.append(t_value)

    # Convert to numpy array for easier indexing
    time_snapshots = np.array(time_snapshots)

    # Find closest time steps to the target times
    closest_steps = []
    for target_time in target_times:
        closest_index = np.abs(time_snapshots - target_time).argmin()
        closest_steps.append(closest_index)

    return closest_steps, time_snapshots


# Function to calculate total hydrogen and maximum concentration over all time steps
def calculate_hydrogen_stats_over_time(file_name):
    total_hydrogen = []
    max_hydrogen_concentration = []
    time_stamps = []
    with h5py.File(file_name, 'r') as hf:
        for key in hf.keys():
            if key.startswith('h_snapshot_'):
                h_array = hf[key][:]  # Load the hydrogen matrix
                total_h = np.sum(h_array[h_array >= 0])  # Sum non-negative values
                max_h = np.max(h_array[h_array >= 0])  # Find maximum hydrogen concentration
                total_hydrogen.append(total_h)
                max_hydrogen_concentration.append(max_h)

                # Corresponding time for this snapshot
                time_key = f't_snapshot_{key.split("_")[-1]}'
                if time_key in hf:
                    time_value = hf[time_key][()]
                    time_stamps.append(time_value)

    return time_stamps, total_hydrogen, max_hydrogen_concentration


# Main graphing code
fig = plt.figure(figsize=(16, 9))
t_cool = 160
t_room = 25

# Find the closest available time steps in the dataset
closest_time_steps, all_times = find_closest_time_steps(target_times_to_display, file_name)

for index, step in enumerate(closest_time_steps):

    if graph_style == "Temp/Diffusion":
        ax1 = fig.add_subplot(4, 2, 2 * index + 1)
        norm = mcolors.Normalize(vmin=t_room, vmax=400)
        cmap = plt.get_cmap('hot')
        cmap.set_under('0.85')
        u_array, h_array, current_time = get_step("u", step, file_name)
        im1 = ax1.imshow(u_array, cmap=cmap, norm=norm)  # , interpolation='none'

        # Create annotations for heat
        heat_annotations = create_annotations(ax1, annotation_points['heat'], u_array, 'Temp')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.05, 0.15, 0.03, 0.7])
        cbar_ax.set_xlabel('$T$ / K', labelpad=20)
        fig.colorbar(im1, cax=cbar_ax)

        # Second plot (Diffusion)
        ax2 = fig.add_subplot(4, 2, 2 * index + 2)
        norm = mcolors.Normalize(vmin=-0.1, vmax=100)
        cmap = plt.get_cmap('viridis')
        cmap.set_under('0.85')
        im2 = ax2.imshow(h_array, cmap=cmap, norm=norm)

        # Create annotations for diffusion
        diffusion_annotations = create_annotations(ax2, annotation_points['diffusion'], h_array, 'HD')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar_ax.set_xlabel('HD in ml/100g', labelpad=20)
        fig.colorbar(im2, cax=cbar_ax)

        # Calculate the position for the text title (horizontally centered between the two subplots)
        x_title = (ax1.get_position().xmax + ax2.get_position().xmin) / 2
        y_title = (ax1.get_position().ymax + ax1.get_position().ymin) / 2

        # Add the text title between the two subplots with the closest time
        fig.text(x_title, y_title, f'{current_time:.1f}s', fontsize=14, ha='center')

# Calculate and plot total hydrogen and max hydrogen concentration over all time steps
time_stamps, total_hydrogen, max_hydrogen_concentration = calculate_hydrogen_stats_over_time(file_name)

fig_total_hydrogen = plt.figure(figsize=(12, 6))

# First plot: Total Hydrogen Content
ax_hydrogen_total = fig_total_hydrogen.add_subplot(1, 2, 1)
ax_hydrogen_total.plot(time_stamps, total_hydrogen, linestyle='-', color='b')
ax_hydrogen_total.set_xlabel('Time (s)')
ax_hydrogen_total.set_ylabel('Total Hydrogen (ml/100g)')
ax_hydrogen_total.set_title('Total Hydrogen Content Over Time')
# ax_hydrogen_total.set_xscale('log')  # Set x-axis to logarithmic scale
# ax_hydrogen_total.set_yscale('log')  # Set x-axis to logarithmic scale

# Second plot: Maximum Hydrogen Concentration
ax_hydrogen_max = fig_total_hydrogen.add_subplot(1, 2, 2)
ax_hydrogen_max.plot(time_stamps, max_hydrogen_concentration, linestyle='-', color='r')
ax_hydrogen_max.set_xlabel('Time (s)')
ax_hydrogen_max.set_ylabel('Max Hydrogen Concentration (ml/100g)')
ax_hydrogen_max.set_title('Max Hydrogen Concentration Over Time')
# ax_hydrogen_max.set_xscale('log')  # Set x-axis to logarithmic scale
# ax_hydrogen_max.set_yscale('log')  # Set x-axis to logarithmic scale

plt.tight_layout()
plt.show()
