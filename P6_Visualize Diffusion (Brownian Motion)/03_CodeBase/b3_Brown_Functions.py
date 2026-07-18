from __future__ import annotations

import json
from pathlib import Path
import importlib
from fractions import Fraction

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


def _serialize_brown_config_value(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_serialize_brown_config_value(v) for v in list(value)]
    if isinstance(value, dict):
        return {str(k): _serialize_brown_config_value(v) for k, v in value.items()}
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return repr(value)


def brown_config_snapshot(runtime_values=None):
    cfg = load_brown_config()
    snapshot = {}

    for name in dir(cfg):
        if name.startswith("_"):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        snapshot[name] = _serialize_brown_config_value(value)

    snapshot["convention"] = "matrix rows: top->bottom, matrix columns: left->right"
    if runtime_values:
        snapshot.update({key: _serialize_brown_config_value(value) for key, value in runtime_values.items()})

    return snapshot


def write_brown_h5_metadata(hf, runtime_values=None):
    meta_group = hf["/meta"] if "/meta" in hf else hf.create_group("/meta")
    meta_group.attrs["brown_config_json"] = json.dumps(
        brown_config_snapshot(runtime_values),
        sort_keys=True,
        indent=2,
    )
    meta_group.attrs["brown_config_source"] = "b2_Brown_Config.py"


def load_brown_config_json(h5_filename, required=True):
    with h5py.File(h5_filename, "r") as hf:
        meta_group = hf.get("/meta")
        if meta_group is None:
            message = "Missing /meta group in HDF5 file; Brownian config metadata is unavailable."
            if required:
                raise RuntimeError(message)
            print(f"Warning: {message}")
            return None

        config_json = meta_group.attrs.get("brown_config_json")
        if config_json is None:
            message = "brown_config_json not found in /meta attrs; Brownian config metadata is unavailable."
            if required:
                raise RuntimeError(message)
            print(f"Warning: {message}")
            return None

        return json.loads(config_json)


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
    matrix = np.zeros((y_func, x_func), dtype=np.int8)

    indices_a = np.random.choice(x_func // 2 * y_func, num_possible_spots_a, replace=False)
    matrix[np.unravel_index(indices_a, (y_func, x_func // 2))] = 1

    indices_b = np.random.choice(x_func // 2 * y_func, num_possible_spots_b, replace=False)
    rows_b, cols_b = np.unravel_index(indices_b, (y_func, x_func // 2))
    cols_b += x_func // 2
    matrix[rows_b, cols_b] = 1

    return matrix


def _pick_lattice_candidates(candidate_matrix, row_slice, col_slice, num_to_pick):
    rows, cols = np.where(candidate_matrix[row_slice, col_slice] == 1)
    if len(rows) < num_to_pick:
        raise ValueError(f"Not enough lattice candidates: requested {num_to_pick}, found {len(rows)}")

    selected = np.random.choice(len(rows), num_to_pick, replace=False)
    row_offset = row_slice.start or 0
    col_offset = col_slice.start or 0

    return rows[selected] + row_offset, cols[selected] + col_offset


def create_lattice_matrix_for_halves(x_func, y_func, num_possible_spots_a, num_possible_spots_b,
                                     lattice_style="prime", start_spacing=5, min_spacing=1):
    spacing = start_spacing
    half_x = x_func // 2

    while spacing >= min_spacing:
        candidate_matrix = create_crystal_lattice_matrix(y_func, x_func, spacing, lattice_style=lattice_style)
        left_candidates = np.sum(candidate_matrix[:, :half_x] == 1)
        right_candidates = np.sum(candidate_matrix[:, half_x:] == 1)

        if left_candidates >= num_possible_spots_a and right_candidates >= num_possible_spots_b:
            matrix = np.zeros((y_func, x_func), dtype=np.int8)

            left_rows, left_cols = _pick_lattice_candidates(
                candidate_matrix,
                slice(None),
                slice(0, half_x),
                num_possible_spots_a,
            )
            right_rows, right_cols = _pick_lattice_candidates(
                candidate_matrix,
                slice(None),
                slice(half_x, x_func),
                num_possible_spots_b,
            )

            matrix[left_rows, left_cols] = 1
            matrix[right_rows, right_cols] = 1
            return matrix, spacing

        spacing -= 1

    raise ValueError(
        f"Lattice style '{lattice_style}' could not provide enough candidates "
        f"for max_sol_a/max_sol_b at spacing >= {min_spacing}."
    )


def create_matrix_from_image(image_path, max_sol_white, max_sol_black, show_plot=True):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    white_mask = img_array > 128
    black_mask = img_array <= 128

    num_white_ones = int(np.sum(white_mask) * max_sol_white)
    num_black_ones = int(np.sum(black_mask) * max_sol_black)

    matrix = np.zeros_like(img_array, dtype=np.int8)

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


def define_concentration_sink_source(h_spots_matrix, pixel_thickness=1, source_side="left"):
    if source_side == "left":
        h_spots_matrix[:, :pixel_thickness] = 2
        h_spots_matrix[:, -pixel_thickness:] = 1
    elif source_side == "right":
        h_spots_matrix[:, :pixel_thickness] = 1
        h_spots_matrix[:, -pixel_thickness:] = 2
    else:
        raise ValueError("source_side must be 'left' or 'right'")

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


def apply_layer(matrix, width=10):
    rows, cols = matrix.shape
    mid_x = cols // 2

    left_bound = max(0, mid_x - width // 2)
    right_bound = min(cols, mid_x + width // 2)

    matrix[:, left_bound:right_bound] = 1

    return matrix


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


def create_jump_probability_table(max_radius_to_jump, sigma):
    table_size = 2 * max_radius_to_jump + 1
    jump_probability_table = np.zeros((table_size, table_size), dtype=np.float32)

    for move_y in range(-max_radius_to_jump, max_radius_to_jump + 1):
        for move_x in range(-max_radius_to_jump, max_radius_to_jump + 1):
            distance_squared = move_x**2 + move_y**2
            jump_probability_table[move_y + max_radius_to_jump, move_x + max_radius_to_jump] = np.exp(
                -distance_squared / (2 * sigma**2)
            )

    return jump_probability_table


def create_active_site_arrays(matrix):
    active_y, active_x = np.where(matrix > 0)
    return active_y.astype(np.int32), active_x.astype(np.int32)


def create_hydrogen_site_arrays(matrix):
    hydrogen_y, hydrogen_x = np.where(matrix == 2)
    return hydrogen_y.astype(np.int32), hydrogen_x.astype(np.int32)


def create_forced_jump_lookup(matrix, max_radius_to_jump):
    site_y, site_x = create_active_site_arrays(matrix)
    site_lookup = np.full(matrix.shape, -1, dtype=np.int32)
    site_lookup[site_y, site_x] = np.arange(len(site_y), dtype=np.int32)

    max_target_count = (2 * max_radius_to_jump + 1) ** 2 - 1
    neighbor_site_ids = np.full((len(site_y), max_target_count), -1, dtype=np.int32)
    neighbor_counts = np.zeros(len(site_y), dtype=np.int32)

    rows, cols = matrix.shape
    for site_id in range(len(site_y)):
        y_pos = site_y[site_id]
        x_pos = site_x[site_id]
        count = 0

        y_start = max(0, y_pos - max_radius_to_jump)
        y_end = min(rows, y_pos + max_radius_to_jump + 1)
        x_start = max(0, x_pos - max_radius_to_jump)
        x_end = min(cols, x_pos + max_radius_to_jump + 1)

        for target_y in range(y_start, y_end):
            for target_x in range(x_start, x_end):
                if target_y == y_pos and target_x == x_pos:
                    continue

                target_site_id = site_lookup[target_y, target_x]
                if target_site_id >= 0:
                    neighbor_site_ids[site_id, count] = target_site_id
                    count += 1

        neighbor_counts[site_id] = count

    hydrogen_y, hydrogen_x = create_hydrogen_site_arrays(matrix)
    hydrogen_site_ids = site_lookup[hydrogen_y, hydrogen_x].astype(np.int32)
    site_states = matrix[site_y, site_x].astype(np.int8)

    return (
        site_y.astype(np.int32),
        site_x.astype(np.int32),
        neighbor_site_ids,
        neighbor_counts,
        site_states,
        hydrogen_site_ids,
    )


def create_matrix_from_site_states(matrix_shape, site_y, site_x, site_states):
    matrix = np.zeros(matrix_shape, dtype=np.int8)
    matrix[site_y, site_x] = site_states
    return matrix


@njit
def unit_interval_index(random_value, size):
    """Map a nominal [0, 1) value to a valid index, including rounded 1.0 inputs."""
    index = int(random_value * size)
    if index < 0:
        return 0
    if index >= size:
        return size - 1
    return index


def create_xoshiro256ss_state(seed):
    """Expand one integer seed into the nonzero 256-bit state used by xoshiro256**."""
    mask = (1 << 64) - 1
    value = int(seed) & mask
    state = np.empty(4, dtype=np.uint64)

    for index in range(4):
        value = (value + 0x9E3779B97F4A7C15) & mask
        mixed = value
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        state[index] = mixed ^ (mixed >> 31)

    return state


@njit
def rotate_left_uint64(value, shift):
    return (value << shift) | (value >> (64 - shift))


@njit
def xoshiro256ss_next(rng_state):
    """Return the next xoshiro256** uint64 and advance rng_state in place."""
    result = rotate_left_uint64(rng_state[1] * np.uint64(5), 7) * np.uint64(9)
    shifted = rng_state[1] << 17

    rng_state[2] ^= rng_state[0]
    rng_state[3] ^= rng_state[1]
    rng_state[1] ^= rng_state[2]
    rng_state[0] ^= rng_state[3]
    rng_state[2] ^= shifted
    rng_state[3] = rotate_left_uint64(rng_state[3], 45)

    return result


@njit
def sample_molecular_move(rng_state, move_range):
    """Draw exact-uniform x/y move indices plus a 24-bit acceptance variate from one word."""
    if move_range < 1 or move_range > (1 << 20):
        raise ValueError("move_range must be between 1 and 2**20")

    component_bits = np.uint64((1 << 20) - 1)
    component_limit = ((1 << 20) // move_range) * move_range

    while True:
        random_word = xoshiro256ss_next(rng_state)
        raw_x = random_word & component_bits
        raw_y = (random_word >> 20) & component_bits

        if raw_x < component_limit and raw_y < component_limit:
            move_x_index = int(raw_x % move_range)
            move_y_index = int(raw_y % move_range)
            random_probability = float(random_word >> 40) * (1.0 / (1 << 24))
            return move_x_index, move_y_index, random_probability


@njit
def simulate_brownian_motion(matrix, rng_state, active_y, active_x, nx, ny, max_radius_to_jump,
                             base_movement_probability, jump_probability_table, sink_source_thickness,
                             use_sink_source, source_on_left, region_map, num_regions, winner_source,
                             winner_priority, claim_epoch, touched_targets, epoch_id):
    new_matrix = np.copy(matrix)
    displacement_stats = np.zeros((num_regions, 3), dtype=np.float32)
    touched_count = 0
    move_range = 2 * max_radius_to_jump + 1

    for active_index in range(len(active_y)):
        j = active_y[active_index]
        i = active_x[active_index]

        if matrix[j, i] == 2:
            move_x_index, move_y_index, rand_prob = sample_molecular_move(rng_state, move_range)
            move_x = move_x_index - max_radius_to_jump
            move_y = move_y_index - max_radius_to_jump

            new_j = j + move_y
            new_i = i + move_x

            if new_j == j and new_i == i:
                continue

            if new_j < 0 or new_j >= ny or new_i < 0 or new_i >= nx:
                continue

            jump_probability = jump_probability_table[move_y + max_radius_to_jump, move_x + max_radius_to_jump]
            adjusted_probability = base_movement_probability * jump_probability

            if matrix[new_j, new_i] == 1 and rand_prob < adjusted_probability:
                target_flat = new_j * nx + new_i
                priority = xoshiro256ss_next(rng_state)

                if claim_epoch[target_flat] != epoch_id:
                    claim_epoch[target_flat] = epoch_id
                    winner_source[target_flat] = j * nx + i
                    winner_priority[target_flat] = priority
                    touched_targets[touched_count] = target_flat
                    touched_count += 1
                elif priority > winner_priority[target_flat]:
                    winner_source[target_flat] = j * nx + i
                    winner_priority[target_flat] = priority

    for touched_index in range(touched_count):
        target_flat = touched_targets[touched_index]
        source_flat = winner_source[target_flat]
        source_j = source_flat // nx
        source_i = source_flat % nx
        target_j = target_flat // nx
        target_i = target_flat % nx
        move_x = target_i - source_i

        region_id = region_map[source_i]
        if region_id >= 0:
            displacement_stats[region_id, 0] += np.float32(abs(move_x))
            displacement_stats[region_id, 1] += np.float32(move_x ** 2)
            displacement_stats[region_id, 2] += np.float32(1)

        new_matrix[source_j, source_i] = 1
        new_matrix[target_j, target_i] = 2

    if use_sink_source:
        if source_on_left:
            for j in range(ny):
                for i in range(sink_source_thickness):
                    if new_matrix[j, i] == 1:
                        new_matrix[j, i] = 2

            for j in range(ny):
                for i in range(nx - sink_source_thickness, nx):
                    new_matrix[j, i] = 1
        else:
            for j in range(ny):
                for i in range(sink_source_thickness):
                    new_matrix[j, i] = 1

            for j in range(ny):
                for i in range(nx - sink_source_thickness, nx):
                    if new_matrix[j, i] == 1:
                        new_matrix[j, i] = 2

    return new_matrix, displacement_stats


@njit
def gcd_int(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a


@njit
def simulate_brownian_motion_forced_jump_precomputed(site_states, random_values, hydrogen_site_ids, site_y, site_x,
                                                     neighbor_site_ids, neighbor_counts, rand_index, random_size,
                                                     region_map, num_regions):
    new_site_states = np.copy(site_states)
    displacement_stats = np.zeros((num_regions, 3), dtype=np.float32)
    valid_site_ids = np.zeros(neighbor_site_ids.shape[1], dtype=np.int32)
    hydrogen_count = len(hydrogen_site_ids)

    if hydrogen_count == 0:
        return new_site_states, rand_index, displacement_stats

    rand_index = (rand_index + 1) % random_size
    order_offset = unit_interval_index(random_values[rand_index - 1], hydrogen_count)

    order_stride = 1
    if hydrogen_count > 1:
        rand_index = (rand_index + 1) % random_size
        order_stride = unit_interval_index(random_values[rand_index - 1], hydrogen_count - 1) + 1
        while gcd_int(order_stride, hydrogen_count) != 1:
            order_stride += 1
            if order_stride >= hydrogen_count:
                order_stride = 1

    for order_index in range(hydrogen_count):
        hydrogen_index = (order_offset + order_index * order_stride) % hydrogen_count
        current_site_id = hydrogen_site_ids[hydrogen_index]
        j = site_y[current_site_id]
        i = site_x[current_site_id]

        if site_states[current_site_id] != 2:
            continue

        valid_count = 0
        for neighbor_index in range(neighbor_counts[current_site_id]):
            target_site_id = neighbor_site_ids[current_site_id, neighbor_index]

            if site_states[target_site_id] == 1 and new_site_states[target_site_id] == 1:
                valid_site_ids[valid_count] = target_site_id
                valid_count += 1

        if valid_count > 0:
            rand_index = (rand_index + 1) % random_size
            chosen_index = unit_interval_index(random_values[rand_index - 1], valid_count)

            target_site_id = valid_site_ids[chosen_index]
            new_j = site_y[target_site_id]
            new_i = site_x[target_site_id]
            move_x = new_i - i

            region_id = region_map[i]
            if region_id >= 0:
                displacement_stats[region_id, 0] += np.float32(abs(move_x))
                displacement_stats[region_id, 1] += np.float32(move_x ** 2)
                displacement_stats[region_id, 2] += np.float32(1)

            new_site_states[current_site_id] = 1
            new_site_states[target_site_id] = 2
            hydrogen_site_ids[hydrogen_index] = target_site_id

    return new_site_states, rand_index, displacement_stats


@njit
def simulate_brownian_motion_forced_jump(matrix, random_values, active_y, active_x, nx, ny, rand_index, random_size,
                                         max_radius_to_jump, sink_source_thickness, use_sink_source, source_on_left,
                                         region_map, num_regions):
    new_matrix = np.copy(matrix)
    displacement_stats = np.zeros((num_regions, 3), dtype=np.float32)
    max_target_count = (2 * max_radius_to_jump + 1) ** 2 - 1
    valid_move_x = np.zeros(max_target_count, dtype=np.int32)
    valid_move_y = np.zeros(max_target_count, dtype=np.int32)

    for active_index in range(len(active_y)):
        j = active_y[active_index]
        i = active_x[active_index]

        if matrix[j, i] == 2:
            valid_count = 0

            y_start = max(0, j - max_radius_to_jump)
            y_end = min(ny, j + max_radius_to_jump + 1)
            x_start = max(0, i - max_radius_to_jump)
            x_end = min(nx, i + max_radius_to_jump + 1)

            for target_y in range(y_start, y_end):
                for target_x in range(x_start, x_end):
                    if target_y == j and target_x == i:
                        continue

                    if matrix[target_y, target_x] == 1 and new_matrix[target_y, target_x] == 1:
                        valid_move_x[valid_count] = target_x - i
                        valid_move_y[valid_count] = target_y - j
                        valid_count += 1

            if valid_count > 0:
                rand_index = (rand_index + 1) % random_size
                chosen_index = unit_interval_index(random_values[rand_index - 1], valid_count)

                move_x = valid_move_x[chosen_index]
                move_y = valid_move_y[chosen_index]
                new_i = i + move_x
                new_j = j + move_y

                region_id = region_map[i]
                if region_id >= 0:
                    displacement_stats[region_id, 0] += np.float32(abs(move_x))
                    displacement_stats[region_id, 1] += np.float32(move_x ** 2)
                    displacement_stats[region_id, 2] += np.float32(1)

                new_matrix[j, i] = 1
                new_matrix[new_j, new_i] = 2

    if use_sink_source:
        if source_on_left:
            for j in range(ny):
                for i in range(sink_source_thickness):
                    if new_matrix[j, i] == 1:
                        new_matrix[j, i] = 2

            for j in range(ny):
                for i in range(nx - sink_source_thickness, nx):
                    new_matrix[j, i] = 1
        else:
            for j in range(ny):
                for i in range(sink_source_thickness):
                    new_matrix[j, i] = 1

            for j in range(ny):
                for i in range(nx - sink_source_thickness, nx):
                    if new_matrix[j, i] == 1:
                        new_matrix[j, i] = 2

    return new_matrix, rand_index, displacement_stats


def create_crystal_lattice_matrix(rows, cols, spacing, lattice_style="even"):
    matrix = np.zeros((rows, cols), dtype=np.int8)

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


def read_saved_steps(hf):
    if "saved_steps" in hf:
        return hf["saved_steps"][:]
    if "saved_steps" in hf.attrs:
        return hf.attrs["saved_steps"][:]
    raise RuntimeError("saved_steps not found in HDF5 file")


def load_brownian_animation_data(h5_filename, render_every_nth_frame=1):
    render_every_nth_frame = int(render_every_nth_frame)
    if render_every_nth_frame < 1:
        raise ValueError("render_every_nth_frame must be 1 or greater")

    frame_slice = slice(None, None, render_every_nth_frame)

    with h5py.File(h5_filename, "r") as hf:
        matrices = hf["snapshots"][frame_slice]
        saved_steps = read_saved_steps(hf)[frame_slice]

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
                mean_disp = group["mean_disp"][frame_slice]
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
        matrix = hf["snapshots"][-1]
        saved_steps = read_saved_steps(hf)
    return matrix, saved_steps[-1]


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
        saved_steps = read_saved_steps(hf)
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
        saved_steps = read_saved_steps(hf)

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
