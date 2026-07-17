import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
P6_ROOT = CODE_DIR.parent
RESULTS_DIR = P6_ROOT / "02_Results"

# ---------------------- Load Data ---------------------- #
def load_data(h5_filename):
    with h5py.File(h5_filename, 'r') as hf:
        saved_steps = hf.attrs["saved_steps"][:]

        num_regions = len([key for key in hf.keys() if key.startswith("region_") and "mean_disp" in hf[key]])
        diffusion_data = {f"region_{i}": {} for i in range(num_regions)}

        for i in range(num_regions):
            for key in ["time", "mean_disp", "var_disp"]:
                if f"region_{i}/{key}" in hf:
                    diffusion_data[f"region_{i}"][key] = hf[f"region_{i}/{key}"][:]
                else:
                    diffusion_data[f"region_{i}"][key] = np.zeros(len(saved_steps))  # Fill with zeros if missing

    return saved_steps, diffusion_data

# File path
file_name = RESULTS_DIR / "random_motion.h5"

# Load data
saved_steps, diffusion_data = load_data(file_name)

# ---------------------- Print Data Overview ---------------------- #
print("\n🔍 **Diffusion Data Overview**")
for region, data in diffusion_data.items():
    time_data = data["time"]
    mean_disp = data["mean_disp"]
    var_disp = data["var_disp"]

    print(f"\n📌 {region}:")
    print(f"   - Time steps: {time_data[:10]} ...")  # Print first 10 time steps
    print(f"   - Mean displacement (min/max/mean): {np.min(mean_disp):.5f} / {np.max(mean_disp):.5f} / {np.mean(mean_disp):.5f}")
    print(f"   - Variance (min/max/mean): {np.min(var_disp):.5f} / {np.max(var_disp):.5f} / {np.mean(var_disp):.5f}")

# ---------------------- Plot the Diffusion Data ---------------------- #
plt.figure(figsize=(10, 6))

for region, data in diffusion_data.items():
    plt.plot(data["time"], data["mean_disp"], label=region)

plt.xlabel("Time Step")
plt.ylabel("Mean Displacement")
plt.title("Diffusion Data Overview")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
