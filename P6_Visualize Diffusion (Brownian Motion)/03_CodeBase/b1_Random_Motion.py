import h5py
import numpy as np
from tqdm import tqdm

from b3_Brown_Functions import *


cfg = load_brown_config()


# Make the initial Matrix
y = cfg.y
x = cfg.x
steps = cfg.steps
max_radius_to_jump = cfg.max_radius_to_jump

# Precompute random numbers
random_size = cfg.random_size
random_values = np.random.rand(random_size).astype(np.float32)
rand_index = 0
sigma = max_radius_to_jump / 3

# Generate TIFF-like movement probability matrix
tiff_like_matrix = np.ones((y, x)) * cfg.tiff_like_value
movement_probability_matrix = 1.0 - (tiff_like_matrix / 255.0)

h5_filename = results_dir() / cfg.h5_filename
num_saved_frames = sum(should_save_frame(step, cfg.save_every_steps) for step in range(steps))
print(f"Total frames to be saved: {num_saved_frames}, approx. "
      f"{int((num_saved_frames * (y * x * 4) / (1024 ** 2)) * 1.15)} MB")

print("Creating initial matrix")
if cfg.USE_IMAGE_MATRIX:
    h_spots_matrix = create_matrix_from_image(
        image_dir() / cfg.image_name,
        cfg.max_sol_white,
        cfg.max_sol_black,
        show_plot=cfg.show_image_matrix_plot,
    )
else:
    num_possible_spots_a = int(y * (x // 2) * cfg.max_sol_a)
    num_possible_spots_b = int(y * (x // 2) * cfg.max_sol_b)
    h_spots_matrix = create_custom_matrix(x, y, num_possible_spots_a, num_possible_spots_b)

sink_source_thickness = cfg.SINK_SOURCE_THICKNESS if cfg.USE_SINK_SOURCE else 0
region_map, num_regions = create_region_mapping(
    x,
    y,
    sink_source_thickness,
    cfg.TRAP_LAYER_WIDTH,
    cfg.num_subregions,
)

print("Applying concentration")
if cfg.USE_SINK_SOURCE:
    h_spots_matrix = define_concentration_sink_source(h_spots_matrix, sink_source_thickness)
else:
    h_spots_matrix = define_concentration_to_halves(
        h_spots_matrix,
        cfg.concentration_a,
        cfg.concentration_b,
    )

print("Adding Specials")
if cfg.USE_SPOT:
    h_spots_matrix = apply_spot(
        h_spots_matrix,
        diameter=cfg.SPOT_DIAMETER,
        center_x=cfg.SPOT_CENTER_X,
        center_y=cfg.SPOT_CENTER_Y,
    )

if cfg.USE_TRAP_LAYER:
    h_spots_matrix, movement_probability_matrix = apply_layer(
        h_spots_matrix,
        movement_probability_matrix,
        width=cfg.TRAP_LAYER_WIDTH,
        movement_probability=cfg.TRAP_LAYER_MOVEMENT_PROBABILITY,
    )

print("Cleaning Loners")
clean_loners(h_spots_matrix, max_radius_to_jump)

if cfg.delete_old_h5 and h5_filename.exists():
    h5_filename.unlink()

height, width = h_spots_matrix.shape
snapshot_size_mb = (height * width * 4) / (1024 ** 2)
optimal_batch_size = max(1, int(cfg.max_ram_mb / snapshot_size_mb))

print(f"Matrix size: {height}x{width}, Snapshot size: {snapshot_size_mb:.2f} MB")
print(f"Using batch size of {optimal_batch_size} frames (max {cfg.max_ram_mb}MB RAM usage)")
print(f"Initial matrix unique values: {np.unique(h_spots_matrix)}")

print("Region Map Summary:")
unique_regions, counts = np.unique(region_map, return_counts=True)
for region, count in zip(unique_regions, counts):
    print(f"Region {region}: {count} columns assigned")

with h5py.File(h5_filename, "w") as hf:
    dset = hf.create_dataset("snapshots", shape=(num_saved_frames, height, width), dtype=int, chunks=True)

    displacement_dsets = {}
    for i in range(num_regions):
        displacement_dsets[f"time_{i}"] = hf.create_dataset(f"region_{i}/time", shape=(num_saved_frames,), dtype=int)
        displacement_dsets[f"mean_disp_{i}"] = hf.create_dataset(
            f"region_{i}/mean_disp",
            shape=(num_saved_frames,),
            dtype=np.float32,
        )
        displacement_dsets[f"var_disp_{i}"] = hf.create_dataset(
            f"region_{i}/var_disp",
            shape=(num_saved_frames,),
            dtype=np.float32,
        )

    save_counter = 0
    buffer = np.empty((optimal_batch_size, height, width), dtype=int)
    buffer_index = 0

    disp_buffer = np.zeros((optimal_batch_size, num_regions, 3), dtype=np.float32)
    disp_buffer_index = 0

    for step in tqdm(range(steps)):
        h_spots_matrix, rand_index, disp_stats = simulate_brownian_motion(
            h_spots_matrix,
            random_values,
            x,
            y,
            rand_index,
            random_size,
            max_radius_to_jump,
            movement_probability_matrix,
            sigma,
            sink_source_thickness,
            cfg.USE_SINK_SOURCE,
            region_map,
            num_regions,
        )

        if should_save_frame(step, cfg.save_every_steps):
            buffer[buffer_index] = h_spots_matrix
            disp_buffer[disp_buffer_index] = disp_stats

            buffer_index += 1
            disp_buffer_index += 1

            if buffer_index == optimal_batch_size:
                dset[save_counter:save_counter + buffer_index] = buffer[:buffer_index]

                for i in range(num_regions):
                    displacement_dsets[f"time_{i}"][save_counter:save_counter + buffer_index] = np.arange(
                        save_counter,
                        save_counter + buffer_index,
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

    if buffer_index > 0:
        print(f"Final flush: {buffer_index} remaining frames written to HDF5.")
        dset[save_counter:save_counter + buffer_index] = buffer[:buffer_index]

        for i in range(num_regions):
            displacement_dsets[f"time_{i}"][save_counter:save_counter + buffer_index] = np.arange(
                save_counter,
                save_counter + buffer_index,
            )
            displacement_dsets[f"mean_disp_{i}"][save_counter:save_counter + buffer_index] = (
                disp_buffer[:buffer_index, i, 0] / np.maximum(disp_buffer[:buffer_index, i, 2], 1)
            )
            displacement_dsets[f"var_disp_{i}"][save_counter:save_counter + buffer_index] = (
                (disp_buffer[:buffer_index, i, 1] / np.maximum(disp_buffer[:buffer_index, i, 2], 1)) -
                (disp_buffer[:buffer_index, i, 0] / np.maximum(disp_buffer[:buffer_index, i, 2], 1))**2
            )

    hf.attrs["max_radius_to_jump"] = max_radius_to_jump
    hf.attrs["matrix_shape"] = h_spots_matrix.shape
    hf.attrs["sink_source_thickness"] = sink_source_thickness
    hf.attrs["saved_steps"] = np.array([step for step in range(steps) if should_save_frame(step, cfg.save_every_steps)])
