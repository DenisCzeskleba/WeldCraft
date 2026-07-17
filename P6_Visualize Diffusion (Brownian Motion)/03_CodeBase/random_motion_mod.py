import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from tqdm import tqdm
import h5py
from numba import njit
import scipy.ndimage as ndi

# ---------------- SWITCHES ---------------- #
USE_SPOT = True          # Enable / disable the 100% solubility spot
USE_LAYER = False        # Enable / disable trap layer
USE_SINK_SOURCE = False  # Enable / disable left sink + right source

# ---------------- FUNCTIONS ---------------- #

def create_custom_matrix(x_func, y_func, num_possible_spots_a, num_possible_spots_b):
    matrix = np.zeros((y_func, x_func), dtype=int)

    # Left half
    indices_a = np.random.choice(x_func // 2 * y_func, num_possible_spots_a, replace=False)
    matrix[np.unravel_index(indices_a, (y_func, x_func // 2))] = 1

    # Right half
    indices_b = np.random.choice(x_func // 2 * y_func, num_possible_spots_b, replace=False)
    rows_b, cols_b = np.unravel_index(indices_b, (y_func, x_func // 2))
    cols_b += x_func // 2
    matrix[rows_b, cols_b] = 1

    return matrix

def define_concentration_to_halves(h_spots_matrix, concentration_a, concentration_b):
    half_x = h_spots_matrix.shape[1] // 2

    # Left half
    left_indices = np.where(h_spots_matrix[:, :half_x] == 1)
    num_changes_left = int(concentration_a / 100 * len(left_indices[0]))
    change_indices_left = np.random.choice(range(len(left_indices[0])), num_changes_left, replace=False)
    h_spots_matrix[left_indices[0][change_indices_left], left_indices[1][change_indices_left]] = 2

    # Right half
    right_indices = np.where(h_spots_matrix[:, half_x:] == 1)
    num_changes_right = int(concentration_b / 100 * len(right_indices[0]))
    change_indices_right = np.random.choice(range(len(right_indices[0])), num_changes_right, replace=False)
    h_spots_matrix[right_indices[0][change_indices_right], right_indices[1][change_indices_right] + half_x] = 2

    return h_spots_matrix

def clean_loners(clean_me, max_radius_to_jump):
    kernel_size = 2 * max_radius_to_jump + 1
    structure = np.ones((kernel_size, kernel_size))
    neighbor_count = ndi.convolve((clean_me > 0).astype(int), structure, mode='constant', cval=0)
    loners = (clean_me > 0) & (neighbor_count <= 1)
    num_loners = np.sum(loners)
    print(f"Number of loners cleaned: {num_loners}")
    clean_me[loners] = 0
    return clean_me

def apply_spot(matrix, diameter=50):
    rows, cols = matrix.shape
    mid_y = rows // 2
    mid_x = cols // 4
    radius = diameter // 2

    for y in range(mid_y - radius, mid_y + radius):
        for x in range(mid_x - radius, mid_x + radius):
            if (x - mid_x) ** 2 + (y - mid_y) ** 2 < radius ** 2:
                matrix[y, x] = 1
    return matrix

@njit
def simulate_brownian_motion(matrix, random_values, nx, ny, rand_index,
                             max_radius_to_jump, movement_probability_matrix,
                             sigma, sink_source_thickness, region_map, num_regions):

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
                adjusted_probability = movement_probability_matrix[j, i] * np.exp(- (distance**2) / (2 * sigma**2))

                if matrix[new_j, new_i] == 1 and new_matrix[new_j, new_i] == 1 and rand_prob < adjusted_probability:
                    region_id = region_map[i]
                    if region_id >= 0:
                        displacement_stats[region_id, 0] += np.float32(abs(move_x))
                        displacement_stats[region_id, 1] += np.float32(move_x ** 2)
                        displacement_stats[region_id, 2] += np.float32(1)

                    new_matrix[j, i] = 1
                    new_matrix[new_j, new_i] = 2

    if USE_SINK_SOURCE:
        for j in range(ny):
            for i in range(sink_source_thickness):
                new_matrix[j, i] = 1
        for j in range(ny):
            for i in range(nx - sink_source_thickness, nx):
                if new_matrix[j, i] == 1:
                    new_matrix[j, i] = 2

    return new_matrix, rand_index, displacement_stats

def measure_occupancy(matrix, region_mask, name="Region"):
    available = np.sum(region_mask)
    occupied = np.sum((matrix == 2) & region_mask)
    u = occupied / available if available > 0 else 0
    c = occupied / region_mask.size
    print(f"{name}: u = {u*100:.2f}% | c = {c*100:.2f}% | occupied = {occupied}/{available}")
    return u, c

# ---------------- SIMULATION SETTINGS ---------------- #
y = 200
x = 400
steps = 50000
max_radius_to_jump = 5
sigma = max_radius_to_jump / 3

S_left = 0.10
S_right = 0.05
num_possible_spots_a = int(y * (x // 2) * S_left)
num_possible_spots_b = int(y * (x // 2) * S_right)

h_spots_matrix = create_custom_matrix(x, y, num_possible_spots_a, num_possible_spots_b)
if USE_SPOT:
    h_spots_matrix = apply_spot(h_spots_matrix, diameter=50)

# 50% initial hydrogen fill on available spots
concentration_a = 50
concentration_b = 50
h_spots_matrix = define_concentration_to_halves(h_spots_matrix, concentration_a, concentration_b)

# Clean isolated dots
h_spots_matrix = clean_loners(h_spots_matrix, max_radius_to_jump)

# Random numbers precomputed
random_size = 10**7
random_values = np.random.rand(random_size).astype(np.float32)
rand_index = 0

movement_probability_matrix = np.ones((y, x), dtype=np.float32)

# ---------------- RUN SIMULATION ---------------- #
print("Starting simulation...")
for step in tqdm(range(steps)):
    h_spots_matrix, rand_index, _ = simulate_brownian_motion(
        h_spots_matrix, random_values, x, y, rand_index,
        max_radius_to_jump, movement_probability_matrix,
        sigma, sink_source_thickness=0,
        region_map=np.zeros(x, dtype=np.int32),
        num_regions=1
    )

print("Simulation complete.")

# ---------------- MEASUREMENT ---------------- #
# Define region masks
left_mask = np.zeros_like(h_spots_matrix, dtype=bool)
left_mask[:, :x//2] = h_spots_matrix[:, :x//2] > 0

right_mask = np.zeros_like(h_spots_matrix, dtype=bool)
right_mask[:, x//2:] = h_spots_matrix[:, x//2:] > 0

spot_mask = np.zeros_like(h_spots_matrix, dtype=bool)
if USE_SPOT:
    mid_y, mid_x = y // 2, x // 4
    radius = 25
    yy, xx = np.ogrid[:y, :x]
    spot_mask = ((xx - mid_x)**2 + (yy - mid_y)**2 <= radius**2)
    spot_mask &= h_spots_matrix > 0

# Measure occupancies
measure_occupancy(h_spots_matrix, left_mask & ~spot_mask, "Left Region (no spot)")
measure_occupancy(h_spots_matrix, right_mask, "Right Region")
if USE_SPOT:
    measure_occupancy(h_spots_matrix, spot_mask, "Spot Region")
