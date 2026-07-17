from __future__ import annotations

from pathlib import Path
import importlib

import h5py
import numpy as np
from numba import njit
from PIL import Image
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter1d


def load_brown_config():
    return importlib.import_module("b2_Brown_Config")


def get_config_value(param_name):
    cfg = load_brown_config()
    if hasattr(cfg, param_name):
        return getattr(cfg, param_name)
    raise ValueError(f"Parameter '{param_name}' not found in b2_Brown_Config.py")


def codebase_dir() -> Path:
    return Path(__file__).resolve().parent


def p6_root() -> Path:
    return codebase_dir().parent


def resources_dir() -> Path:
    return p6_root() / "01_Resources"


def results_dir() -> Path:
    return p6_root() / "02_Results"


def image_dir() -> Path:
    return resources_dir() / "Bilder"


def in_results(*parts: str) -> Path:
    return results_dir().joinpath(*parts)


def in_resources(*parts: str) -> Path:
    return resources_dir().joinpath(*parts)


def create_custom_matrix(x_func, y_func, num_possible_spots_a, num_possible_spots_b):
    matrix = np.zeros((y_func, x_func), dtype=int)

    indices_a = np.random.choice(x_func // 2 * y_func, num_possible_spots_a, replace=False)
    matrix[np.unravel_index(indices_a, (y_func, x_func // 2))] = 1

    indices_b = np.random.choice(x_func // 2 * y_func, num_possible_spots_b, replace=False)
    rows_b, cols_b = np.unravel_index(indices_b, (y_func, x_func // 2))
    cols_b += x_func // 2
    matrix[rows_b, cols_b] = 1

    return matrix


def create_matrix_from_image(image_path, max_sol_white, max_sol_black, show_plot=True):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    white_mask = img_array > 128
    black_mask = img_array <= 128

    num_white_ones = int(np.sum(white_mask) * max_sol_white)
    num_black_ones = int(np.sum(black_mask) * max_sol_black)

    matrix = np.zeros_like(img_array, dtype=int)

    if num_white_ones > 0:
        white_indices = np.argwhere(white_mask)
        selected_white = white_indices[np.random.choice(len(white_indices), num_white_ones, replace=False)]
        matrix[selected_white[:, 0], selected_white[:, 1]] = 1

    if num_black_ones > 0:
        black_indices = np.argwhere(black_mask)
        selected_black = black_indices[np.random.choice(len(black_indices), num_black_ones, replace=False)]
        matrix[selected_black[:, 0], selected_black[:, 1]] = 1

    if show_plot:
        import matplotlib.pyplot as plt
        import tkinter as tk
        from tkinter import messagebox

        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="gray", interpolation="nearest")
        plt.title("Generated Matrix from Image")
        plt.colorbar(label="Occupancy (1 = Filled)")
        plt.show()

        root = tk.Tk()
        root.withdraw()
        user_response = messagebox.askyesno("Matrix Verification", "Does the matrix look correct?")
        root.destroy()

        if not user_response:
            messagebox.showwarning("Aborting", "Check your image and parameters. Simulation will not continue.")
            raise SystemExit(1)

    return matrix


def define_concentration_to_halves(h_spots_matrix, concentration_a, concentration_b):
    half_x = h_spots_matrix.shape[1] // 2

    left_indices = np.where(h_spots_matrix[:, :half_x] == 1)
    num_changes_left = int(concentration_a / 100 * len(left_indices[0]))
    change_indices_left = np.random.choice(range(len(left_indices[0])), num_changes_left, replace=False)
    h_spots_matrix[left_indices[0][change_indices_left], left_indices[1][change_indices_left]] = 2

    right_indices = np.where(h_spots_matrix[:, half_x:] == 1)
    num_changes_right = int(concentration_b / 100 * len(right_indices[0]))
    change_indices_right = np.random.choice(range(len(right_indices[0])), num_changes_right, replace=False)
    h_spots_matrix[right_indices[0][change_indices_right], right_indices[1][change_indices_right] + half_x] = 2

    return h_spots_matrix


def define_concentration_sink_source(h_spots_matrix, pixel_thickness=1):
    h_spots_matrix[:, :pixel_thickness] = 1
    h_spots_matrix[:, -pixel_thickness:] = 2
    return h_spots_matrix


def clean_loners(clean_me, max_radius_to_jump):
    kernel_size = 2 * max_radius_to_jump + 1
    structure = np.ones((kernel_size, kernel_size))
    neighbor_count = ndi.convolve((clean_me > 0).astype(int), structure, mode="constant", cval=0)
    loners = (clean_me > 0) & (neighbor_count <= 1)

    num_loners = np.sum(loners)
    print(f"Number of loners to clean: {num_loners}")
    clean_me[loners] = 0

    return clean_me


def apply_spot(matrix, diameter=50, center_x=None, center_y=None):
    rows, cols = matrix.shape
    mid_y = rows // 2 if center_y is None else center_y
    mid_x = cols // 4 if center_x is None else center_x
    radius = diameter // 2

    y_start = max(0, mid_y - radius)
    y_end = min(rows, mid_y + radius)
    x_start = max(0, mid_x - radius)
    x_end = min(cols, mid_x + radius)

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if (x - mid_x) ** 2 + (y - mid_y) ** 2 < radius ** 2:
                matrix[y, x] = 1

    return matrix


def apply_layer(matrix, prob_matrix, width=10, movement_probability=0.2):
    rows, cols = matrix.shape
    mid_x = cols // 2

    left_bound = max(0, mid_x - width // 2)
    right_bound = min(cols, mid_x + width // 2)

    matrix[:, left_bound:right_bound] = 1
    prob_matrix[:, left_bound:right_bound] = movement_probability

    return matrix, prob_matrix


def create_region_mapping(nx, ny, sink_source_thickness, layer_width, num_subregions=3):
    mid_x = nx // 2
    left_start = sink_source_thickness
    left_end = mid_x - (layer_width // 2) - 1
    right_start = mid_x + (layer_width // 2)
    right_end = nx - sink_source_thickness

    left_sub_width = (left_end - left_start) // num_subregions
    right_sub_width = (right_end - right_start) // num_subregions

    region_map = -np.ones(nx, dtype=np.int32)

    for i in range(num_subregions):
        start_x = left_start + i * left_sub_width
        end_x = left_start + (i + 1) * left_sub_width
        region_map[start_x:end_x] = i

    for i in range(num_subregions):
        start_x = right_start + i * right_sub_width
        end_x = right_start + (i + 1) * right_sub_width
        region_map[start_x:end_x] = num_subregions + i

    region_map[left_end:right_start] = 2 * num_subregions

    return region_map, 2 * num_subregions + 1


def should_save_frame(step, save_every_steps):
    return step % save_every_steps == 0


@njit
def simulate_brownian_motion(matrix, random_values, nx, ny, rand_index, random_size, max_radius_to_jump,
                             movement_probability_matrix, sigma, sink_source_thickness, use_sink_source, region_map,
                             num_regions):
    new_matrix = np.copy(matrix)
    displacement_stats = np.zeros((num_regions, 3), dtype=np.float32)

    for j in range(ny):
        for i in range(nx):
            if matrix[j, i] == 2:
                rand_index = (rand_index + 3) % random_size
                rand_val_x = random_values[rand_index - 3]
                rand_val_y = random_values[rand_index - 2]
                rand_prob = random_values[rand_index - 1]

                move_x = int(rand_val_x * (2 * max_radius_to_jump + 1)) - max_radius_to_jump
                move_y = int(rand_val_y * (2 * max_radius_to_jump + 1)) - max_radius_to_jump

                new_j = j + move_y
                new_i = i + move_x

                if new_j == j and new_i == i:
                    continue

                if new_j < 0 or new_j >= ny or new_i < 0 or new_i >= nx:
                    continue

                distance = np.sqrt(move_x**2 + move_y**2)
                adjusted_probability = movement_probability_matrix[j, i] * np.exp(
                    -(distance**2) / (2 * sigma**2)
                )

                if matrix[new_j, new_i] == 1 and new_matrix[new_j, new_i] == 1 and rand_prob < adjusted_probability:
                    region_id = region_map[i]
                    if region_id >= 0:
                        displacement_stats[region_id, 0] += np.float32(abs(move_x))
                        displacement_stats[region_id, 1] += np.float32(move_x ** 2)
                        displacement_stats[region_id, 2] += np.float32(1)

                    new_matrix[j, i] = 1
                    new_matrix[new_j, new_i] = 2

    if use_sink_source:
        for j in range(ny):
            for i in range(sink_source_thickness):
                new_matrix[j, i] = 1

        for j in range(ny):
            for i in range(nx - sink_source_thickness, nx):
                if new_matrix[j, i] == 1:
                    new_matrix[j, i] = 2

    return new_matrix, rand_index, displacement_stats


def create_crystal_lattice_matrix(rows, cols, spacing, lattice_style="even"):
    matrix = np.zeros((rows, cols), dtype=int)

    if lattice_style == "even":
        matrix[spacing // 2::spacing, spacing // 2::spacing] = 1

    elif lattice_style == "prime":
        for row in range(0, rows, spacing):
            for col in range(0, cols, spacing):
                matrix[row, col] = 1
        for row in range(spacing // 2, rows, spacing):
            for col in range(spacing // 2, cols, spacing):
                matrix[row, col] = 1

    elif lattice_style == "diagonal":
        for start in range(0, rows, spacing):
            np.fill_diagonal(matrix[start:], 1)
            np.fill_diagonal(matrix[:, start:], 1)

    elif lattice_style == "border":
        matrix[0::spacing, 0] = 1
        matrix[0::spacing, -1] = 1
        matrix[0, 0::spacing] = 1
        matrix[-1, 0::spacing] = 1

    elif lattice_style == "checkerboard":
        for row in range(0, rows, spacing):
            for col in range(0, cols, spacing):
                if (row // spacing) % 2 == (col // spacing) % 2:
                    matrix[row:row + spacing, col:col + spacing] = 1

    elif lattice_style == "random":
        num_total_points = rows * cols
        num_points_to_place = int(num_total_points * (1 / (abs(spacing) + 1)))
        indices = np.random.choice(num_total_points, num_points_to_place, replace=False)
        matrix[np.unravel_index(indices, (rows, cols))] = 1

    elif lattice_style == "radial":
        center = (rows // 2, cols // 2)
        for r in range(spacing, min(center), spacing):
            angle_spacing = int(2 * np.pi * r / spacing)
            for angle in range(0, angle_spacing):
                x = int(center[0] + r * np.cos(angle * (2 * np.pi / angle_spacing)))
                y = int(center[1] + r * np.sin(angle * (2 * np.pi / angle_spacing)))
                matrix[y % rows, x % cols] = 1

    return matrix


def load_brownian_animation_data(h5_filename):
    with h5py.File(h5_filename, "r") as hf:
        matrices = hf["snapshots"][:]
        saved_steps = hf.attrs["saved_steps"][:]

        region_indices = []
        for key in hf.keys():
            if key.startswith("region_"):
                region_number = key.split("_", 1)[1]
                if region_number.isdigit():
                    region_indices.append(int(region_number))

        diffusion_data = {}
        for region_index in sorted(region_indices):
            group_name = f"region_{region_index}"
            group = hf[group_name]
            if "mean_disp" in group:
                mean_disp = group["mean_disp"][:]
            else:
                mean_disp = np.zeros(len(saved_steps), dtype=float)

            if len(mean_disp) < len(saved_steps):
                mean_disp = np.pad(mean_disp, (0, len(saved_steps) - len(mean_disp)), mode="constant")
            elif len(mean_disp) > len(saved_steps):
                mean_disp = mean_disp[:len(saved_steps)]

            diffusion_data[group_name] = mean_disp

    return matrices, saved_steps, diffusion_data


def load_last_snapshot(h5_filename):
    with h5py.File(h5_filename, "r") as hf:
        matrices = hf["snapshots"][:]
        saved_steps = hf.attrs["saved_steps"][:]
    return matrices[-1], saved_steps[-1]


def compute_concentration_profile(matrix, smoothing_window=5, gaussian_sigma=1.5):
    total_spots = np.sum(matrix > 0, axis=0)
    filled_spots = np.sum(matrix == 2, axis=0)

    concentration_profile = np.zeros_like(filled_spots, dtype=float)
    mask = total_spots > 0
    concentration_profile[mask] = filled_spots[mask] / total_spots[mask]

    expanded_profile = np.pad(concentration_profile, (5, 5), mode="edge")
    smoothed_profile = np.convolve(expanded_profile, np.ones(smoothing_window) / smoothing_window, mode="same")
    smoothed_profile = gaussian_filter1d(smoothed_profile, sigma=gaussian_sigma, mode="reflect")

    return smoothed_profile[5:-5]


def load_simulation_data(h5_filename):
    with h5py.File(h5_filename, "r") as hf:
        matrices = hf["snapshots"][:]
        saved_steps = hf.attrs["saved_steps"][:]
        sink_source_thickness = hf.attrs["sink_source_thickness"]
    return matrices, saved_steps, sink_source_thickness


def compute_com_in_zones(matrices, saved_steps, sink_source_thickness):
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    ny, nx = matrices.shape[1:]
    mid_x = nx // 2
    trap_margin = 5

    left_zone = (slice(None), slice(sink_source_thickness, mid_x - trap_margin))
    right_zone = (slice(None), slice(mid_x + trap_margin, nx - sink_source_thickness))

    com_left = []
    com_right = []
    time_left = []
    time_right = []

    iterator = enumerate(zip(matrices, saved_steps))
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(matrices), desc="Computing COM for left and right zones")

    for _, (matrix, time) in iterator:
        y_positions_left, x_positions_left = np.where(matrix[left_zone] == 2)
        if len(x_positions_left) > 0:
            com_left_x = np.mean(x_positions_left) + sink_source_thickness
            com_left_y = np.mean(y_positions_left)
            com_left.append((com_left_x, com_left_y))
            time_left.append(time)

        y_positions_right, x_positions_right = np.where(matrix[right_zone] == 2)
        if len(x_positions_right) > 0:
            com_right_x = np.mean(x_positions_right) + mid_x + trap_margin
            com_right_y = np.mean(y_positions_right)
            com_right.append((com_right_x, com_right_y))
            time_right.append(time)

    return np.array(time_left), np.array(com_left), np.array(time_right), np.array(com_right)


def compute_time_resolved_D(time_values, com_positions, window_size=100):
    if len(com_positions) < window_size:
        print("Warning: Not enough data points for time-resolved diffusion calculation.")
        return np.array([]), np.array([])

    msd_values = []
    time_centers = []
    x_positions = com_positions[:, 0]

    for i in range(len(x_positions) - window_size):
        t_window = time_values[i:i + window_size]
        x_window = x_positions[i:i + window_size]

        diffs = x_window - x_window[0]
        msd = np.mean(diffs ** 2)

        time_centers.append(np.mean(t_window))
        msd_values.append(msd)

    time_centers = np.array(time_centers)
    msd_values = np.array(msd_values)

    D_local = np.zeros_like(time_centers)
    for i in range(len(time_centers) - 1):
        dt = time_centers[i + 1] - time_centers[i]
        d_msd = msd_values[i + 1] - msd_values[i]
        D_local[i] = d_msd / (4 * dt) if dt > 0 else np.nan

    return time_centers, D_local


def compute_mean_displacement(time_values, com_positions):
    x_positions = com_positions[:, 0]
    displacements = np.abs(np.diff(x_positions))
    time_intervals = np.diff(time_values)
    mean_displacement = displacements / time_intervals
    time_centers = (time_values[:-1] + time_values[1:]) / 2

    return time_centers, mean_displacement


def compute_variance_speed(time_values, com_positions, window_size=50):
    x_positions = com_positions[:, 0]
    squared_displacements = (x_positions[1:] - x_positions[:-1]) ** 2
    rolling_variance = np.convolve(squared_displacements, np.ones(window_size) / window_size, mode="valid")
    time_centers = (time_values[:len(rolling_variance)] + time_values[1:len(rolling_variance) + 1]) / 2

    return time_centers, rolling_variance


def load_diffusion_data(h5_filename):
    with h5py.File(h5_filename, "r") as hf:
        saved_steps = hf.attrs["saved_steps"][:]

        region_indices = []
        for key in hf.keys():
            if key.startswith("region_"):
                region_number = key.split("_", 1)[1]
                if region_number.isdigit():
                    region_indices.append(int(region_number))

        diffusion_data = {f"region_{i}": {} for i in sorted(region_indices)}

        for region in diffusion_data:
            for key in ["time", "mean_disp", "var_disp"]:
                if f"{region}/{key}" in hf:
                    diffusion_data[region][key] = hf[f"{region}/{key}"][:]
                else:
                    diffusion_data[region][key] = np.zeros(len(saved_steps))

    return saved_steps, diffusion_data
