import h5py
import matplotlib.pyplot as plt
import numpy as np
from b4_functions import in_results

# Choose which curves to draw (1..5). Example: [2] draws only #2; [1,3,5] draws those; [] draws none.
PLOT_IDS = [1, 2, 3, 4, 5]

# Up to 5 curve definitions (you can leave unused ones as None)
CURVES = [
    None,  # index 0 unused so we can use 1-based IDs

    {  # 1
        "file_name": str(in_results("03_Batch-Executions", "XXX.h5")),
        "snapshot_time": 3853,
        "label": r"Mid Weld Process",  # r"$t_{8/5}$ = 10s, 20mm plate, $D_{min}$"
        "color": (0.0, 0.0, 0.0),  #(0.5176, 0.0, 0.5176) purple | (0.2588, 0.4196, 0.6117) blue | (0.0, 0.0, 0.0) black | (0.0, 0.2, 0.6) darker blue | (0.85, 0.0, 0.0) red
        "low_threshold": 0,
        "up_threshold": 0,
    },

    {  # 2
        "file_name": str(in_results("03_Batch-Executions", "XXX.h5")),
        "snapshot_time": 7211,
        "label": r"Last Weld Bead",
        "color": (0.85, 0.0, 0.0),
        "low_threshold": 0,
        "up_threshold": 0,
    },

    {  # 3
        "file_name": str(in_results("03_Batch-Executions", "XXX.h5")),
        "snapshot_time": 14933,
        "label": r"Room Temperature",
        "color": (0.0, 0.2, 0.6),
        "low_threshold": 0,
        "up_threshold": 0,
    },

    {  # 4
        "file_name": str(in_results("03_Batch-Executions", "XXX.h5")),
        "snapshot_time": 101333,
        "label": r"24h after Welding",
        "color": (0.2588, 0.4196, 0.6117),
        "low_threshold": 5,
        "up_threshold": 10,
    },

    {  # 5
        "file_name": str(in_results("03_Batch-Executions", "XXX.h5")),
        "snapshot_time": 619733,
        "label": r"168h after Welding",
        "color": (0.5176, 0.0, 0.5176),
        "low_threshold": 5,
        "up_threshold": 10,
    },

]

DY_MM = 0.5   # <-- set to your actual vertical grid spacing in mm

# --- Manual axis scaling switches ---
USE_MANUAL_XLIM = True
X_LIM = (0, 60)          # hydrogen [%] range, used if USE_MANUAL_XLIM=True

USE_MANUAL_YLIM = True
Y_LIM = (40, 0)          # y range; Remember that you flip here! if normalize=True this is typically 0..100


# --- Font size settings ---
FONTSIZE_TITLE  = 18
FONTSIZE_AXES   = 16   # both x + y labels
FONTSIZE_TICKS  = 16
FONTSIZE_LEGEND = 16

# Normalize option (set to True to normalize, False otherwise)
normalize = False  # Set to False to disable normalization


def plot_hydrogen_concentration(file_name, snapshot_time, ax, label, color, normalize_y=False, low_threshold=10, up_threshold=10):
    # Open the HDF5 file and access the datasets
    with h5py.File(file_name, 'r') as hf:
        # Find the dataset corresponding to the 't_snapshot' (time)
        t_datasets = [key for key in hf.keys() if key.startswith('t_snapshot')]

        # Extract the time values from the 't_snapshot' datasets
        t_values = []
        for dataset in t_datasets:
            t_data = hf[dataset][()]  # Load the time data
            t_values.append((t_data, dataset))  # Store the time and corresponding dataset name

    # Find the closest time to the requested snapshot time
    closest_time, closest_dataset = min(t_values, key=lambda x: abs(x[0] - snapshot_time))

    # Print the closest time and corresponding dataset
    print(f"Closest time to {snapshot_time}: {closest_time} seconds, Dataset: {closest_dataset}")

    # Open the HDF5 file again to retrieve the corresponding 'h_snapshot' data
    with h5py.File(file_name, 'r') as hf:
        # Use the corresponding 'h_snapshot' dataset
        h_data = hf[closest_dataset.replace('t_snapshot', 'h_snapshot')][:]  # Replace 't_snapshot' with 'h_snapshot' for the data

    # Get the middle column index
    middle_col = h_data.shape[1] // 2  # Get the column in the middle of the matrix

    # Second plot: Hydrogen concentration at the middle column, plot the row values
    row_values = h_data[:, middle_col]  # Get all rows in the middle column

    # Filter the row values to only include from min + threshold to max - threshold
    min_row = low_threshold
    max_row = len(h_data) - up_threshold
    filtered_row_values = row_values[min_row:max_row]  # Slice the row values based on the thresholds
    filtered_row_indices = np.arange(min_row, max_row)  # Corresponding row indices

    # Convert row indices to mm (bottom -> top) and optionally normalize
    filtered_y_mm = filtered_row_indices * DY_MM

    if normalize_y:
        # normalize to 0..100 % of the viewed height
        y0 = filtered_y_mm[0]
        y1 = filtered_y_mm[-1]
        if y1 > y0:
            filtered_y_plot = (filtered_y_mm - y0) / (y1 - y0) * 100.0
        else:
            filtered_y_plot = filtered_y_mm * 0.0
        ax.set_ylabel("Y [% of height]", fontsize=FONTSIZE_AXES)
        ax.set_ylim(0, 100)
    else:
        filtered_y_plot = filtered_y_mm
        ax.set_ylabel("Y [mm]", fontsize=FONTSIZE_AXES)
        ax.set_ylim(filtered_y_mm[0], filtered_y_mm[-1])

    # Plot the filtered row values on the given axis with the specified color
    ax.plot(filtered_row_values, filtered_y_plot, label=label, color=color)

    # Set labels and title
    ax.set_xlabel('Hydrogen Concentration [%]', fontsize=FONTSIZE_AXES)
    ax.set_title('Vertical distribution (168h)', fontsize=FONTSIZE_TITLE)

    # Add grid to the plot
    ax.grid(True)


# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))  # Create a single plot for both datasets

for curve_id in PLOT_IDS:
    if not (1 <= curve_id <= 5):
        raise ValueError(f"curve_id must be 1..5, got {curve_id}")

    cfg = CURVES[curve_id]
    if cfg is None:
        raise ValueError(f"CURVES[{curve_id}] is None but curve_id {curve_id} is in PLOT_IDS")

    plot_hydrogen_concentration(
        cfg["file_name"],
        cfg["snapshot_time"],
        ax,
        label=cfg["label"],
        color=cfg["color"],
        normalize_y=normalize,
        low_threshold=cfg.get("low_threshold", 10),
        up_threshold=cfg.get("up_threshold", 10),
    )

# --- Apply axis scaling (once, after plotting) ---
if USE_MANUAL_XLIM:
    ax.set_xlim(*X_LIM)

if USE_MANUAL_YLIM:
    ax.set_ylim(*Y_LIM)

# Add a legend

ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)
ax.legend(fontsize=FONTSIZE_LEGEND)


# Adjust layout and show the plot
plt.tight_layout()
plt.show()
