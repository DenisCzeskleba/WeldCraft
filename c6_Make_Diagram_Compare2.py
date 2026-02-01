import matplotlib.pyplot as plt
import h5py
import numpy as np


# File names for comparison (allowing up to 4 files) #-in up to 4
file_names = [
    r"02_Results\03_Batch-Executions\XXX.h5",  # Simulation 1
    r"02_Results\03_Batch-Executions\XXX.h5",  # Simulation 2
    # r"02_Results\03_Batch-Executions\XXX.h5",  # Simulation 3
    # r"02_Results\03_Batch-Executions\XXX.h5",  # Simulation 4
]
h_inside_values = {0: 0, 1: "Sievert Law 100", 2: 75, 3: 100}


# Ensure no more than 4 files are used
if len(file_names) > 4:
    raise ValueError("You can only compare up to 4 simulations.")

# Customize colors for the simulations (values between 0 and 1)
colors = [

    (0.2588, 0.4196, 0.6117),  # Custom blue for Simulation 1
    (0.5176, 0.0, 0.5176),  # Custom purple for Simulation 2
    (0.0, 0.0, 0.0),  # Black for Simulation 3
    (1, 0.0, 0.0),  # Red for Simulation 4

]


# Get the time for example to room temp
def get_time_to_rt_from_file(file_name):
    with h5py.File(file_name, 'r') as hf:
        return hf.attrs.get('total_time_to_rt', None)


rt_times = [get_time_to_rt_from_file(fn) for fn in file_names]
rt_times_valid = [t for t in rt_times if t is not None]

max_rt_time = max(rt_times_valid) if rt_times_valid else None

# Annotations for specific x-values (time steps) to indicate the RT point for each file
# annotation_x_values = [2950, 2950, 2950, 2950]  # Default for 4 files hardcoded
annotation_x_values = []  # alternativly get the data directly from the file, for example for RT
for fn in file_names:
    t_rt = get_time_to_rt_from_file(fn)
    annotation_x_values.append(t_rt)

# Dynamically adjust colors and annotation_x_values based on the number of files
colors = colors[:len(file_names)]  # Trim to match the number of simulations
annotation_x_values = annotation_x_values[:len(file_names)]  # Trim to match the number of simulations

# Set display option: 1 = Average Concentration Hydrogen, 2 = Max Concentration, 3 = Concentration at a Specific Point
display_option = 2  # Change this value to switch between options

# For Option 1, allow area to sum over (horizontal lines for now)
x_min, x_max = 90, 142

# For Option 3, set the specific point to track (x, y) coordinates
specific_point = (30, 160)  # Change this to the point you want to track

# Set max display time (0 = show full dataset, otherwise limit to the given time in seconds)
max_display_time = 0  # Change this to limit the displayed time

# If max_display_time == 0, you can still auto-limit to the *largest RT time* across all files
limit_to_max_rt_time = True  # True = x-axis ends at max(total_time_to_rt) across files
rt_margin_seconds = 60         # e.g. 60 to show 1 minute after RT


# Function to calculate hydrogen stats over time for a file, with max time filtering
def calculate_hydrogen_stats_over_time(file_name, max_time):
    avg_hydrogen_conc = []
    max_hydrogen_concentration = []
    specific_point_concentration = []
    time_stamps = []

    with h5py.File(file_name, 'r') as hf:
        for key in hf.keys():
            if key.startswith('h_snapshot_'):
                h_array = hf[key][:]  # Load the hydrogen matrix

                region = h_array[:, x_min:x_max]  # Select region of interest
                positive_values = region[region >= 0]  # Filter positive values only
                if positive_values.size > 0:  # Calculate average hydrogen concentration
                    avg_h = np.mean(positive_values)
                else:
                    avg_h = 0  # No positive values

                max_h = np.max(h_array[h_array >= 0])  # Find maximum hydrogen concentration
                point_conc = h_array[specific_point]  # Concentration at specific point

                # Append results
                avg_hydrogen_conc.append(avg_h)
                max_hydrogen_concentration.append(max_h)
                specific_point_concentration.append(point_conc)

                # Corresponding time for this snapshot
                time_key = f't_snapshot_{key.split("_")[-1]}'
                if time_key in hf:
                    time_value = hf[time_key][()]
                    if max_time == 0 or time_value <= max_time:  # Only load up to max_time
                        time_stamps.append(time_value)

    # Trim the arrays if we are filtering based on max_time
    if max_time > 0:
        max_index = len(time_stamps)
        avg_hydrogen_conc = avg_hydrogen_conc[:max_index]
        max_hydrogen_concentration = max_hydrogen_concentration[:max_index]
        specific_point_concentration = specific_point_concentration[:max_index]

    return time_stamps, avg_hydrogen_conc, max_hydrogen_concentration, specific_point_concentration


# Calculate stats for each simulation with optional max display time
simulation_data = []
for file_name in file_names:
    time_stamps, avg_hydrogen_conc, max_concentration, point_concentration = calculate_hydrogen_stats_over_time(file_name, max_display_time)
    simulation_data.append({
        'time': time_stamps,
        'avg_hydrogen_conc': avg_hydrogen_conc,
        'max_concentration': max_concentration,
        'point_concentration': point_concentration
    })

# Create a plot based on the display option
fig, ax = plt.subplots(figsize=(8, 6))

if display_option == 1:  # Average Concentration of Hydrogen
    for idx in range(len(file_names)):  # Loop over all simulations (files)
        ax.plot(simulation_data[idx]['time'], simulation_data[idx]['avg_hydrogen_conc'],
                label=f"H inside = {h_inside_values.get(idx, 'N/A')}%",
                linestyle='-',
                color=colors[idx])  # Use color from the dynamically adjusted list
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Average Hydrogen (% of 2.5 ml / 100g Fe (ISO3690)')
    ax.set_title('Average Hydrogen Content Comparison')

elif display_option == 2:  # Max Hydrogen Concentration
    for idx in range(len(file_names)):  # Loop over all simulations (files)
        ax.plot(simulation_data[idx]['time'], simulation_data[idx]['max_concentration'],
                label=f'$t_{{8/5}}$ = {10 if idx == 0 else 18}s, 20mm Blech, $D_{{min}}$',
                # label=f'{30 if idx == 0 else 60}mm, $t_{{8/5}}$ = {10 if idx == 0 else 18}s, $D_{{min}}$',
                linestyle='-',
                color=colors[idx])  # Use color from the dynamically adjusted list
    ax.set_xlabel('Time [s]', fontsize=16)
    ax.set_ylabel('Hydrogen Concentration [%]', fontsize=16)
    ax.set_title('Hydrogen Concentration (Maximum over Weld Area)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Add annotations for the specific x-values (adjust based on active files)
    for idx, annotation_x in enumerate(annotation_x_values):
        if annotation_x is None:
            continue
        closest_index = np.abs(np.array(simulation_data[idx]['time']) - annotation_x).argmin()
        annotation_y = simulation_data[idx]['max_concentration'][closest_index]

        # Annotate the graph with an arrow pointing to the value
        ax.annotate(f'{annotation_y:.0f}% (RT)', xy=(annotation_x, annotation_y),
                    xytext=(annotation_x, annotation_y + 6),  # Adjust text position as needed
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=14, ha='center')

elif display_option == 3:  # Concentration at a Specific Point
    for idx in range(len(file_names)):  # Loop over all simulations (files)
        ax.plot(simulation_data[idx]['time'], simulation_data[idx]['point_concentration'],
                label=f"H inside = {h_inside_values.get(idx, 'N/A')}%",
                linestyle='-',
                color=colors[idx])  # Use color from the dynamically adjusted list
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Hydrogen Concentration at mid point lower plate ((% of 2.5 ml / 100g Fe (ISO3690))')
    ax.set_title(f'Hydrogen Concentration Over Time' + "\n" + "(Weld line center and mid point of lower plate)")
    # ax.set_ylabel(f'Hydrogen Concentration at {specific_point} (ml/100g)')
    # ax.set_title(f'Hydrogen Concentration at {specific_point} Over Time')

# Add a legend and grid
ax.legend(fontsize=16)

ax.set_xlim(left=0)  # always start exactly at 0
if max_display_time != 0:
    ax.set_xlim(0, max_display_time)
elif limit_to_max_rt_time and (max_rt_time is not None):
    ax.set_xlim(0, max_rt_time + rt_margin_seconds)
# else: leave full dataset visible (matplotlib will auto-set right limit)

ax.set_ylim([0, 105])      # Example limit for y-axis (hydrogen concentration)

ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()