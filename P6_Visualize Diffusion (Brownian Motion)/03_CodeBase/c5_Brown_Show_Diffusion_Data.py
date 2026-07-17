import matplotlib.pyplot as plt
import numpy as np

from b3_Brown_Functions import *


cfg = load_brown_config()
file_name = results_dir() / cfg.h5_filename

saved_steps, diffusion_data = load_diffusion_data(file_name)

print("\nDiffusion Data Overview")
for region, data in diffusion_data.items():
    time_data = data["time"]
    mean_disp = data["mean_disp"]
    var_disp = data["var_disp"]

    print(f"\n{region}:")
    print(f"   - Time steps: {time_data[:10]} ...")
    print(
        "   - Mean displacement (min/max/mean): "
        f"{np.min(mean_disp):.5f} / {np.max(mean_disp):.5f} / {np.mean(mean_disp):.5f}"
    )
    print(
        "   - Variance (min/max/mean): "
        f"{np.min(var_disp):.5f} / {np.max(var_disp):.5f} / {np.mean(var_disp):.5f}"
    )

plt.figure(figsize=(10, 6))

for region, data in diffusion_data.items():
    plt.plot(data["time"], data["mean_disp"], label=region)

plt.xlabel("Time Step")
plt.ylabel("Mean Displacement")
plt.title("Diffusion Data Overview")
plt.legend()
plt.grid(True)
plt.show()
