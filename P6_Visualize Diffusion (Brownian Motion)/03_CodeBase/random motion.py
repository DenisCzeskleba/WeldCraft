import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm  # Correct way to import colormaps
from tqdm import tqdm
import h5py
from numba import jit, njit
import scipy.ndimage as ndi
from matrix_from_image import create_matrix_from_image  # Import matrix_from_image.py
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
P6_ROOT = CODE_DIR.parent
RESOURCES_DIR = P6_ROOT / "01_Resources"
RESULTS_DIR = P6_ROOT / "02_Results"
IMAGE_DIR = RESOURCES_DIR / "Bilder"


def create_custom_matrix(x_func, y_func, num_possible_spots_a, num_possible_spots_b):

    # Create the matrix
    matrix = np.zeros((y_func, x_func), dtype=int)  # switch x, y for cartesian

    # Populate the left half
    indices_a = np.random.choice(x_func // 2 * y_func, num_possible_spots_a, replace=False)
    matrix[np.unravel_index(indices_a, (y_func, x_func // 2))] = 1

    # Populate the right half
    indices_b = np.random.choice(x_func // 2 * y_func, num_possible_spots_b, replace=False)
    rows_b, cols_b = np.unravel_index(indices_b, (y_func, x_func // 2))
    cols_b += x_func // 2  # Adjust column indices to reflect their position in the right half
    matrix[rows_b, cols_b] = 1  # Place 1s in the correct positions in the right half

    return matrix


def define_concentration_to_halves(h_spots_matrix, concentration_a, concentration_b):
    half_x = h_spots_matrix.shape[0]

    # Left half change
    left_indices = np.where(h_spots_matrix[:, :half_x] == 1)
    num_changes_left = int(concentration_a / 100 * len(left_indices[0]))
    change_indices_left = np.random.choice(range(len(left_indices[0])), num_changes_left, replace=False)
    h_spots_matrix[left_indices[0][change_indices_left], left_indices[1][change_indices_left]] = 2

    # Right half change
    right_indices = np.where(h_spots_matrix[:, half_x:] == 1)
    num_changes_right = int(concentration_b / 100 * len(right_indices[0]))
    change_indices_right = np.random.choice(range(len(right_indices[0])), num_changes_right, replace=False)
    h_spots_matrix[right_indices[0][change_indices_right], right_indices[1][change_indices_right] + half_x] = 2

    return h_spots_matrix


def define_concentration_sink_source(h_spots_matrix, pixel_thickness=1):
    # Apply concentration A to a thin layer on the very left
    h_spots_matrix[:, :pixel_thickness] = 1  # Set leftmost layer to 1 so it can have hydrogen but is "empty"

    # Apply concentration B to a thin layer on the very right
    h_spots_matrix[:, -pixel_thickness:] = 2  # Set rightmost layer to 2 so it can have hydrogen is is "full"

    return h_spots_matrix


def clean_loners(clean_me, max_radius_to_jump):  # Optimized version so we dont check EVERY to EVERY point
    # Create a kernel to count neighbors in a radius
    kernel_size = 2 * max_radius_to_jump + 1
    structure = np.ones((kernel_size, kernel_size))

    # Count how many neighbors each pixel has
    neighbor_count = ndi.convolve((clean_me > 0).astype(int), structure, mode='constant', cval=0)

    # Identify "loners" (spots that have no other neighbors in the radius)
    loners = (clean_me > 0) & (neighbor_count <= 1)

    # Count and remove the loners
    num_loners = np.sum(loners)
    print(f"Number of loners to clean: {num_loners}")
    clean_me[loners] = 0  # Set lonely spots to zero

    return clean_me


def apply_spot(matrix, diameter=50):
    rows, cols = matrix.shape

    # Calculate the midpoint of the left half
    mid_y = rows // 2
    mid_x = cols // 4  # Middle of the left half

    # Calculate the radius of the circle
    radius = diameter // 2

    # Apply changes in a circular pattern
    for y in range(mid_y - radius, mid_y + radius):
        for x in range(mid_x - radius, mid_x + radius):
            # Check if the point is inside the circle
            if (x - mid_x) ** 2 + (y - mid_y) ** 2 < radius ** 2:
                h_spots_matrix[y, x] = 1

    return h_spots_matrix


def apply_layer(matrix, prob_matrix, width=10):
    rows, cols = matrix.shape
    mid_x = cols // 2  # Middle column of the matrix

    left_bound = max(0, mid_x - width // 2)
    right_bound = min(cols, mid_x + width // 2)

    matrix[:, left_bound:right_bound] = 1  # Set all values in the layer to 1
    prob_matrix[:, left_bound:right_bound] = 0.2  # make jumps less likely

    return matrix, prob_matrix


# ---------------------- Define Region Mapping ---------------------- #
def create_region_mapping(nx, ny, sink_source_thickness, layer_width, num_subregions=3):
    """
    Creates a precomputed array where each x-index corresponds to a region.
    Avoids slow Python loops inside JIT functions.
    """
    mid_x = nx // 2  # Middle column index
    left_start = sink_source_thickness
    left_end = mid_x - (layer_width // 2) - 1
    right_start = mid_x + (layer_width // 2)
    right_end = nx - sink_source_thickness

    # Compute subregion sizes
    left_sub_width = (left_end - left_start) // num_subregions
    right_sub_width = (right_end - right_start) // num_subregions

    # Initialize the region mapping array (-1 = ignored area)
    region_map = -np.ones(nx, dtype=np.int32)

    # Assign left regions
    for i in range(num_subregions):
        start_x = left_start + i * left_sub_width
        end_x = left_start + (i + 1) * left_sub_width
        region_map[start_x:end_x] = i  # Assign region index

    # Assign right regions
    for i in range(num_subregions):
        start_x = right_start + i * right_sub_width
        end_x = right_start + (i + 1) * right_sub_width
        region_map[start_x:end_x] = num_subregions + i  # Offset index for right side

    # Assign middle layer (last index)
    region_map[left_end:right_start] = 2 * num_subregions

    return region_map, 2 * num_subregions + 1  # Total number of regions


# Function to control frame skipping (for saving)
def should_save_frame(step):
    """Determines whether a frame should be saved based on step count."""

    if step % 1000 == 0:
        return True  # Save every 200th
    return False  # Skip otherwise

    # if step < 100:
    #     return True  # Save first 100 frames
    # elif step < 1000 and step % 10 == 0:
    #     return True  # Save every 10th frame after 100
    # elif step % 100 == 0:
    #     return True  # Save every 100th frame after 1000
    # return False  # Skip otherwise

@njit
def simulate_brownian_motion(matrix, random_values, nx, ny, rand_index, max_radius_to_jump, movement_probability_matrix, sigma, sink_source_thickness, region_map, num_regions):

    new_matrix = np.copy(matrix)
    displacement_stats = np.zeros((num_regions, 3), dtype=np.float32)  # Columns: [x_sum, x_sq_sum, count]

    for j in range(ny):
        for i in range(nx):
            if matrix[j, i] == 2:

                # Increment rand_index once for all random values used
                rand_index = (rand_index + 3) % random_size

                # Get three random values
                rand_val_x = random_values[rand_index - 3]
                rand_val_y = random_values[rand_index - 2]
                rand_prob = random_values[rand_index - 1]  # Used for probability check

                # Generate integer jumps properly
                move_x = int(rand_val_x * (2 * max_radius_to_jump + 1)) - max_radius_to_jump
                move_y = int(rand_val_y * (2 * max_radius_to_jump + 1)) - max_radius_to_jump

                # Compute new position
                new_j = j + move_y
                new_i = i + move_x

                # Check if the new position is the same as the start position (no movement)
                if new_j == j and new_i == i:
                    continue  # Skip this iteration if no movement occurs

                # Check if the new position is within matrix bounds
                if new_j < 0 or new_j >= ny or new_i < 0 or new_i >= nx:
                    continue  # Skip this iteration if out of bounds

                # Compute Euclidean distance of the jump
                distance = np.sqrt(move_x**2 + move_y**2)

                # Gaussian decay based on distance
                adjusted_probability = movement_probability_matrix[j, i] * np.exp(- (distance**2) / (2 * sigma**2))

                # Check if movement is allowed
                if matrix[new_j, new_i] == 1 and new_matrix[new_j, new_i] == 1 and rand_prob < adjusted_probability:
                    # **FAST REGION TRACKING USING PRECOMPUTED ARRAY**
                    region_id = region_map[i]  # Get region index directly
                    if region_id >= 0:  # If valid region
                        displacement_stats[region_id, 0] += np.float32(abs(move_x))  # Sum of displacements
                        displacement_stats[region_id, 1] += np.float32(move_x ** 2)  # Sum of squared displacements
                        displacement_stats[region_id, 2] += np.float32(1)  # Count of movements

                    # Move molecule
                    new_matrix[j, i] = 1  # Free original position
                    new_matrix[new_j, new_i] = 2  # Move molecule

    if True:  # This is sort of like a boundary condition you can use if you use sinks/sources
        # Step 5: Apply boundary conditions correctly
        # (sink)
        for j in range(ny):
            for i in range(sink_source_thickness):  # Leftmost columns (sink)
                new_matrix[j, i] = 1  # Sink stays at 1

        # (source)
        for j in range(ny):
            for i in range(nx - sink_source_thickness, nx):  # Rightmost columns (source)
                if new_matrix[j, i] == 1:  # Only overwrite empty spots
                    new_matrix[j, i] = 2  # Fill with molecules

    return new_matrix, rand_index, displacement_stats


# Make the initial Matrix
y = 250  # Height (y)
x = 500  # Width (x)
# x = 2 * y  # Width (x)
steps = 40000
max_radius_to_jump = 10

# Precompute random numbers
random_size = 10 ** 7  # Number of precomputed random numbers
random_values = np.random.rand(random_size).astype(np.float32)
rand_index = 0  # Start index for random values
sigma = max_radius_to_jump / 3  # Ensure max distance is ~3σ

# Generate TIFF-like movement probability matrix
tiff_like_matrix = np.ones((y, x)) * 125
movement_probability_matrix = 1.0 - (tiff_like_matrix / 255.0)  # Scale to 0-1 range

image_path = IMAGE_DIR / "Gef\u00fcge_array.tiff"
# Define concentrations for white and black regions
max_sol_white = Fraction(40, 100)  # x% concentration in white areas
max_sol_black = Fraction(2, 100)  # y% concentration in black areas

h5_filename = RESULTS_DIR / "random_motion.h5"
max_ram_mb = 500  # Adjustable (use 1000 for 1GB)
# Count how many frames will be saved
num_saved_frames = sum(should_save_frame(step) for step in range(steps))
print(f"Total frames to be saved: {num_saved_frames}, approx. "
      f"{int((num_saved_frames * (y * x * 4) / (1024 ** 2)) * 1.15)} MB")

# Apply max solubility
max_sol_a = Fraction(10, 100)  # (left side) change to any fraction, right now its in %
max_sol_b = Fraction(5, 100)  # (right side) change to any fraction, right now its in %
num_possible_spots_a = int(y * y * max_sol_a)  # (left side) actual amount of spots
num_possible_spots_b = int(y * y * max_sol_b)  # (right side) actual amount of spots

print("Creating initial matrix")
# Random lattice
h_spots_matrix = create_custom_matrix(x, y, num_possible_spots_a, num_possible_spots_b)
# From Picture
# h_spots_matrix = create_matrix_from_image(image_path, max_sol_white, max_sol_black)

# Apply the current concentration - looks funny if its the same as above e.i 20/60 and 20/60 for both
concentration_a = 50
concentration_b = 50
sink_source_thickness = 10  # should be the same as jump range to avoid jams on the sink side

# Define regions
num_subregions = 1  # Number of parts to split left/right into
layer_width = 19  # Width of the middle trap layer
# Generate the region map
region_map, num_regions = create_region_mapping(x, y, sink_source_thickness, layer_width, num_subregions)

print("Applying concentration")
# This gives each half the initial concentration
# h_spots_matrix = define_concentration_to_halves(h_spots_matrix, concentration_a, concentration_b)
# This applies a thin layer on the left and right side
h_spots_matrix = define_concentration_sink_source(h_spots_matrix, sink_source_thickness)

print("Adding Specials")
# add something?
h_spots_matrix = apply_spot(h_spots_matrix)  # add a spot
# h_spots_matrix, movement_probability_matrix = apply_layer(h_spots_matrix, movement_probability_matrix)  # add a layer (trap?)

print("Cleaning Loners")
# Clean some loners
clean_loners(h_spots_matrix, max_radius_to_jump)

if True:  # Change to False if you want to keep the file or mess with the filename
    if h5_filename.exists():  # Delete old files
        h5_filename.unlink()

# Matrix dimensions
height, width = h_spots_matrix.shape
# Compute snapshot size in MB
snapshot_size_mb = (height * width * 4) / (1024 ** 2)  # Convert bytes to MB (4 bytes per int32)
# Compute optimal batch size (rounded down)
optimal_batch_size = max(1, int(max_ram_mb / snapshot_size_mb))

print(f"Matrix size: {height}x{width}, Snapshot size: {snapshot_size_mb:.2f} MB")
print(f"Using batch size of {optimal_batch_size} frames (max {max_ram_mb}MB RAM usage)")

# Storage for per-region displacement data
region_displacement_data = {f"region_{i}": {"time": [], "mean_disp": [], "var_disp": []} for i in range(num_regions)}
print(f"Initial matrix unique values: {np.unique(h_spots_matrix)}")

print("Region Map Summary:")
unique_regions, counts = np.unique(region_map, return_counts=True)
for region, count in zip(unique_regions, counts):
    print(f"Region {region}: {count} columns assigned")


with h5py.File(h5_filename, 'w') as hf:
    # Create dataset for snapshots
    dset = hf.create_dataset("snapshots", shape=(num_saved_frames, height, width), dtype=int, chunks=True)

    # Predefine datasets for displacement data
    displacement_dsets = {}
    for i in range(num_regions):
        displacement_dsets[f"time_{i}"] = hf.create_dataset(f"region_{i}/time", shape=(num_saved_frames,), dtype=int)
        displacement_dsets[f"mean_disp_{i}"] = hf.create_dataset(f"region_{i}/mean_disp", shape=(num_saved_frames,), dtype=np.float32)
        displacement_dsets[f"var_disp_{i}"] = hf.create_dataset(f"region_{i}/var_disp", shape=(num_saved_frames,), dtype=np.float32)

    # Initialize buffers
    save_counter = 0
    buffer = np.empty((optimal_batch_size, height, width), dtype=int)
    buffer_index = 0

    disp_buffer = np.zeros((optimal_batch_size, num_regions, 3), dtype=np.float32)  # [x_sum, x_sq_sum, count]
    disp_buffer_index = 0

    # Initialize previous matrix for tracking changes
    previous_matrix = h_spots_matrix.copy()

    for step in tqdm(range(steps)):
        h_spots_matrix, rand_index, disp_stats = simulate_brownian_motion(
            h_spots_matrix, random_values, x, y, rand_index, max_radius_to_jump, movement_probability_matrix,
            sigma, sink_source_thickness, region_map, num_regions
        )

        if should_save_frame(step):

            # print(f"Step {step}: disp_stats (first 3 regions):")
            # for i in range(min(6, num_regions)):  # Print only first 3 to avoid spamming
            #     print(f"  Region {i}: {disp_stats[i]}")

            buffer[buffer_index] = h_spots_matrix
            disp_buffer[disp_buffer_index] = disp_stats

            buffer_index += 1
            disp_buffer_index += 1

            # Write to disk when buffer is full
            if buffer_index == optimal_batch_size:
                dset[save_counter:save_counter + buffer_index] = buffer[:buffer_index]

                for i in range(num_regions):
                    displacement_dsets[f"time_{i}"][save_counter:save_counter + buffer_index] = np.arange(
                        save_counter, save_counter + buffer_index
                    )
                    displacement_dsets[f"mean_disp_{i}"][save_counter:save_counter + buffer_index] = (
                        disp_buffer[:buffer_index, i, 0] / np.maximum(disp_buffer[:buffer_index, i, 2], 1)
                    )
                    displacement_dsets[f"var_disp_{i}"][save_counter:save_counter + buffer_index] = (
                        (disp_buffer[:buffer_index, i, 1] / np.maximum(disp_buffer[:buffer_index, i, 2], 1)) -
                        (disp_buffer[:buffer_index, i, 0] / np.maximum(disp_buffer[:buffer_index, i, 2], 1))**2
                    )

                save_counter += buffer_index
                buffer_index = 0
                disp_buffer_index = 0

    # Final buffer flush to ensure all data is saved
    if buffer_index > 0:
        print(f"Final flush: {buffer_index} remaining frames written to HDF5.")
        dset[save_counter:save_counter + buffer_index] = buffer[:buffer_index]

        for i in range(num_regions):
            displacement_dsets[f"time_{i}"][save_counter:save_counter + buffer_index] = np.arange(
                save_counter, save_counter + buffer_index
            )
            displacement_dsets[f"mean_disp_{i}"][save_counter:save_counter + buffer_index] = (
                disp_buffer[:buffer_index, i, 0] / np.maximum(disp_buffer[:buffer_index, i, 2], 1)
            )
            displacement_dsets[f"var_disp_{i}"][save_counter:save_counter + buffer_index] = (
                (disp_buffer[:buffer_index, i, 1] / np.maximum(disp_buffer[:buffer_index, i, 2], 1)) -
                (disp_buffer[:buffer_index, i, 0] / np.maximum(disp_buffer[:buffer_index, i, 2], 1))**2
            )

    # Save metadata
    hf.attrs["max_radius_to_jump"] = max_radius_to_jump
    hf.attrs["matrix_shape"] = h_spots_matrix.shape
    hf.attrs["sink_source_thickness"] = sink_source_thickness

    # Save saved steps for animation script
    saved_steps = np.array([step for step in range(steps) if should_save_frame(step)])
    hf.attrs["saved_steps"] = saved_steps
