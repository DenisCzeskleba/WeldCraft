import matplotlib.pyplot as plt

from b3_Brown_Functions import *


def plot_diffusion_speed(time_left, D_left, time_right, D_right):
    plt.figure(figsize=(8, 6))
    plt.plot(time_left, D_left, label="Left Zone $D_{local}$", color="#0000FF")
    plt.plot(time_right, D_right, label="Right Zone $D_{local}$", color="#FF0000")
    plt.xlabel("Time Step")
    plt.ylabel("Time-Resolved Diffusion Coefficient $D_{local}$")
    plt.title("Time Evolution of Diffusion Coefficient")
    plt.legend()
    plt.grid()
    plt.show()


cfg = load_brown_config()
file_name = results_dir() / cfg.h5_filename

print("\nLoading simulation data...")
matrices, saved_steps, sink_source_thickness = load_simulation_data(file_name)

print("\nComputing Center of Mass (COM) for both zones...")
time_left, com_left, time_right, com_right = compute_com_in_zones(matrices, saved_steps, sink_source_thickness)

print("\nComputing Time-Resolved Diffusion Coefficient...")
time_centers_left, D_local_left = compute_time_resolved_D(time_left, com_left, window_size=100)
time_centers_right, D_local_right = compute_time_resolved_D(time_right, com_right, window_size=100)

print("First 10 COM displacements (Left Zone):", com_left[:10])
print("First 10 COM displacements (Right Zone):", com_right[:10])

if len(com_left) > 1:
    print("First 10 displacement differences (Left Zone):", com_left[1:11, 0] - com_left[:10, 0])
if len(com_right) > 1:
    print("First 10 displacement differences (Right Zone):", com_right[1:11, 0] - com_right[:10, 0])

plt.figure(figsize=(8, 6))
if len(com_left) > 0:
    plt.plot(time_left, com_left[:, 0], label="COM X (Left Zone)", color="#0000FF")
if len(com_right) > 0:
    plt.plot(time_right, com_right[:, 0], label="COM X (Right Zone)", color="#FF0000")
plt.xlabel("Time Step")
plt.ylabel("COM X Position")
plt.title("Center of Mass Movement Over Time")
plt.legend()
plt.grid()
plt.show()

print("\nPlotting Time-Resolved Diffusion Speed...")
plot_diffusion_speed(time_centers_left, D_local_left, time_centers_right, D_local_right)

print("\nComputing Variance-Based Displacement Speed...")
if len(com_left) > 1:
    time_var_left, var_speed_left = compute_variance_speed(time_left, com_left, window_size=50)
else:
    time_var_left, var_speed_left = [], []

if len(com_right) > 1:
    time_var_right, var_speed_right = compute_variance_speed(time_right, com_right, window_size=50)
else:
    time_var_right, var_speed_right = [], []

plt.figure(figsize=(8, 6))
plt.plot(time_var_left, var_speed_left, label="Left Zone Variance Speed", color="#0000FF")
plt.plot(time_var_right, var_speed_right, label="Right Zone Variance Speed", color="#FF0000")
plt.xlabel("Time Step")
plt.ylabel("Variance of Displacement Per Step")
plt.title("Variance of COM Movement Over Time")
plt.legend()
plt.grid()
plt.show()
