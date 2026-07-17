import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
P6_ROOT = CODE_DIR.parent
RESULTS_DIR = P6_ROOT / "02_Results"

# ---------------------- 1. Load Data ---------------------- #
def load_simulation_data(h5_filename):
    """Load saved simulation matrices and time steps from HDF5 file."""
    with h5py.File(h5_filename, 'r') as hf:
        matrices = hf["snapshots"][:]  # Load saved matrix snapshots
        saved_steps = hf.attrs["saved_steps"][:]  # Load saved time steps
        sink_source_thickness = hf.attrs["sink_source_thickness"]
    return matrices, saved_steps, sink_source_thickness

# ---------------------- 2. Compute Center of Mass (COM) ---------------------- #
def compute_com_in_zones(matrices, saved_steps, sink_source_thickness):
    """
    Computes the center of mass (COM) for two distinct regions:
    - Left side (excluding the sink and trap layer)
    - Right side (excluding the source and trap layer)
    """
    ny, nx = matrices.shape[1:]  # Get full matrix size
    mid_x = nx // 2  # Middle column (trap layer is here)
    trap_margin = 5  # Avoid central trap region

    left_zone = (slice(None), slice(sink_source_thickness, mid_x - trap_margin))
    right_zone = (slice(None), slice(mid_x + trap_margin, nx - sink_source_thickness))

    com_left = []
    com_right = []
    time_left = []
    time_right = []

    for step, (matrix, time) in tqdm(enumerate(zip(matrices, saved_steps)), total=len(matrices), desc="Computing COM for left and right zones"):
        # Left side COM
        y_positions_left, x_positions_left = np.where(matrix[left_zone] == 2)
        if len(x_positions_left) > 0:
            com_left_x = np.mean(x_positions_left) + sink_source_thickness  # Adjust for slice offset
            com_left_y = np.mean(y_positions_left)
            com_left.append((com_left_x, com_left_y))
            time_left.append(time)  # Save corresponding time step

        # Right side COM
        y_positions_right, x_positions_right = np.where(matrix[right_zone] == 2)
        if len(x_positions_right) > 0:
            com_right_x = np.mean(x_positions_right) + mid_x + trap_margin  # Adjust for slice offset
            com_right_y = np.mean(y_positions_right)
            com_right.append((com_right_x, com_right_y))
            time_right.append(time)  # Save corresponding time step

    # Convert to NumPy arrays
    com_left = np.array(com_left)
    com_right = np.array(com_right)
    time_left = np.array(time_left)
    time_right = np.array(time_right)

    return time_left, com_left, time_right, com_right

# ---------------------- 3. Compute Time-Resolved Diffusion Coefficient ---------------------- #
def compute_time_resolved_D(time_values, com_positions, window_size=100):
    """
    Computes the time-resolved diffusion coefficient using a sliding window approach.
    Only considers movement in the X-direction.
    """
    if len(com_positions) < window_size:
        print("Warning: Not enough data points for time-resolved diffusion calculation.")
        return np.array([]), np.array([])

    msd_values = []
    time_centers = []

    # Extract X-coordinates only
    x_positions = com_positions[:, 0]

    for i in range(len(x_positions) - window_size):
        t_window = time_values[i:i + window_size]
        x_window = x_positions[i:i + window_size]

        diffs = x_window - x_window[0]  # Only x-direction displacement
        msd = np.mean(diffs ** 2)  # Compute MSD

        time_centers.append(np.mean(t_window))
        msd_values.append(msd)

    time_centers = np.array(time_centers)
    msd_values = np.array(msd_values)

    # Estimate D_local from d(MSD)/dt using a linear fit in each window
    D_local = np.zeros_like(time_centers)
    for i in range(len(time_centers) - 1):
        dt = time_centers[i + 1] - time_centers[i]
        d_msd = msd_values[i + 1] - msd_values[i]
        D_local[i] = d_msd / (4 * dt) if dt > 0 else np.nan

    return time_centers, D_local

# ---------------------- 4. Plot Diffusion Coefficients Over Time ---------------------- #
def plot_diffusion_speed(time_left, D_left, time_right, D_right):
    """Plots time-resolved diffusion coefficients for both regions."""
    plt.figure(figsize=(8, 6))

    plt.plot(time_left, D_left, label="Left Zone $D_{local}$", color='blue')
    plt.plot(time_right, D_right, label="Right Zone $D_{local}$", color='red')

    plt.xlabel("Time Step")
    plt.ylabel("Time-Resolved Diffusion Coefficient $D_{local}$")
    plt.title("Time Evolution of Diffusion Coefficient")
    plt.legend()
    plt.grid()
    plt.show()


def compute_mean_displacement(time_values, com_positions):
    """
    Computes the mean step-to-step displacement in the X-direction.
    This provides insight into the 'speed' of diffusion even at equilibrium.
    """
    x_positions = com_positions[:, 0]  # Only use X-coordinate

    displacements = np.abs(np.diff(x_positions))  # Compute stepwise displacement
    time_intervals = np.diff(time_values)  # Compute time intervals

    mean_displacement = displacements / time_intervals  # Compute "speed"

    time_centers = (time_values[:-1] + time_values[1:]) / 2  # Midpoint times

    return time_centers, mean_displacement


def compute_variance_speed(time_values, com_positions, window_size=50):
    """
    Computes the rolling variance in X-direction to track movement persistence.
    """
    x_positions = com_positions[:, 0]  # X-coordinates only

    # Compute squared displacements
    squared_displacements = (x_positions[1:] - x_positions[:-1]) ** 2

    # Compute rolling variance
    rolling_variance = np.convolve(squared_displacements, np.ones(window_size) / window_size, mode='valid')

    # Compute corresponding time values for the rolling window
    time_centers = (time_values[:len(rolling_variance)] + time_values[1:len(rolling_variance)+1]) / 2

    return time_centers, rolling_variance



# ---------------------- 5. Main Execution ---------------------- #
file_name = RESULTS_DIR / "random_motion.h5"

print("\nLoading simulation data...")
matrices, saved_steps, sink_source_thickness = load_simulation_data(file_name)

print("\nComputing Center of Mass (COM) for both zones...")
time_left, com_left, time_right, com_right = compute_com_in_zones(matrices, saved_steps, sink_source_thickness)

print("\nComputing Time-Resolved Diffusion Coefficient...")
time_centers_left, D_local_left = compute_time_resolved_D(time_left, com_left, window_size=100)
time_centers_right, D_local_right = compute_time_resolved_D(time_right, com_right, window_size=100)

# 📊 Debugging: Print first few displacement values
print("First 10 COM displacements (Left Zone):", com_left[:10])
print("First 10 COM displacements (Right Zone):", com_right[:10])

# 📊 Debugging: Check actual displacement differences
displacement_diffs_left = com_left[1:, 0] - com_left[:-1, 0]
displacement_diffs_right = com_right[1:, 0] - com_right[:-1, 0]

print("First 10 displacement differences (Left Zone):", displacement_diffs_left[:10])
print("First 10 displacement differences (Right Zone):", displacement_diffs_right[:10])

# 📈 Also, plot the raw displacement of the center of mass over time
plt.figure(figsize=(8, 6))
plt.plot(time_left, com_left[:, 0], label="COM X (Left Zone)", color="blue")
plt.plot(time_right, com_right[:, 0], label="COM X (Right Zone)", color="red")
plt.xlabel("Time Step")
plt.ylabel("COM X Position")
plt.title("Center of Mass Movement Over Time")
plt.legend()
plt.grid()
plt.show()

print("\nPlotting Time-Resolved Diffusion Speed...")
plot_diffusion_speed(time_centers_left, D_local_left, time_centers_right, D_local_right)

print("\nComputing Variance-Based Displacement Speed...")
time_var_left, var_speed_left = compute_variance_speed(time_left, com_left, window_size=50)
time_var_right, var_speed_right = compute_variance_speed(time_right, com_right, window_size=50)

plt.figure(figsize=(8, 6))
plt.plot(time_var_left, var_speed_left, label="Left Zone Variance Speed", color='blue')
plt.plot(time_var_right, var_speed_right, label="Right Zone Variance Speed", color='red')
plt.xlabel("Time Step")
plt.ylabel("Variance of Displacement Per Step")
plt.title("Variance of COM Movement Over Time")
plt.legend()
plt.grid()
plt.show()



print("\nComputing Variance-Based Displacement Speed...")
time_var_left, var_speed_left = compute_variance_speed(time_left, com_left)
time_var_right, var_speed_right = compute_variance_speed(time_right, com_right)

plt.figure(figsize=(8, 6))
plt.plot(time_var_left, var_speed_left, label="Left Zone Variance Speed", color='blue')
plt.plot(time_var_right, var_speed_right, label="Right Zone Variance Speed", color='red')
plt.xlabel("Time Step")
plt.ylabel("Variance of Displacement Per Step")
plt.title("Variance of COM Movement Over Time")
plt.legend()
plt.grid()
plt.show()
