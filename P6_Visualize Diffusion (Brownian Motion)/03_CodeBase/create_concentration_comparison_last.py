import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
P6_ROOT = CODE_DIR.parent
RESULTS_DIR = P6_ROOT / "02_Results"

# ---------------------- 1. Load Data ---------------------- #
def load_last_snapshot(h5_filename):
    """Load the last saved matrix and time step from HDF5 file."""
    with h5py.File(h5_filename, 'r') as hf:
        matrices = hf["snapshots"][:]  # Load all matrices
        saved_steps = hf.attrs["saved_steps"][:]  # Load saved steps
    return matrices[-1], saved_steps[-1]  # Return the last entry

# ---------------------- 2. Compute Concentration Profile ---------------------- #
def compute_concentration_profile(matrix, smoothing_window=5, gaussian_sigma=1.5):
    """Compute and smooth the concentration profile from a matrix snapshot."""
    total_spots = np.sum(matrix > 0, axis=0)  # Count total active spots per column
    filled_spots = np.sum(matrix == 2, axis=0)  # Count filled spots per column

    # Compute ratio, avoiding division by zero
    concentration_profile = np.zeros_like(filled_spots, dtype=float)
    mask = total_spots > 0
    concentration_profile[mask] = filled_spots[mask] / total_spots[mask]

    # Expand profile by 5 columns to avoid edge dips
    expanded_profile = np.pad(concentration_profile, (5, 5), mode='edge')

    # Apply Gaussian smoothing
    smoothed_profile = np.convolve(expanded_profile, np.ones(smoothing_window) / smoothing_window, mode='same')
    smoothed_profile = gaussian_filter1d(smoothed_profile, sigma=gaussian_sigma, mode='reflect')

    # Remove padding to return to original size
    return smoothed_profile[5:-5]


# ---------------------- 3. Plot Final Concentration Profiles ---------------------- #
def plot_comparison(file1, file2):
    """Load, compute, and plot the final concentration profiles of two simulation runs."""
    # Load last snapshot and step from both files
    matrix1, step1 = load_last_snapshot(file1)
    matrix2, step2 = load_last_snapshot(file2)

    # Compute concentration profiles
    profile1 = compute_concentration_profile(matrix1)
    profile2 = compute_concentration_profile(matrix2)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(profile1)), profile1 * 100, label=f"{file1} (Step {step1})", color="blue")
    plt.plot(np.arange(len(profile2)), profile2 * 100, label=f"{file2} (Step {step2})", color="red")

    plt.xlabel("X Position")
    plt.ylabel("Concentration [%]")
    plt.title("Final Equilibrium Concentration Profiles")
    plt.legend()
    plt.grid()
    plt.show()


# ---------------------- 4. Main Execution ---------------------- #
if __name__ == "__main__":
    file1 = RESULTS_DIR / "gut mit trap layer_too long.h5"  # First simulation file
    file2 = RESULTS_DIR / "gut ohne trap layer.h5"  # Second simulation file

    plot_comparison(file1, file2)
