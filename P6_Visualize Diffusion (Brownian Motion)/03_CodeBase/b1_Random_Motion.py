import h5py
import numpy as np
from tqdm import tqdm

from b3_Brown_Functions import *


cfg = load_brown_config()
if cfg.SOURCE_SIDE not in ("left", "right"):
    raise ValueError("SOURCE_SIDE must be 'left' or 'right'")
if cfg.MATRIX_SOURCE not in ("random", "image", "lattice"):
    raise ValueError("MATRIX_SOURCE must be 'random', 'image', or 'lattice'")
if cfg.simulation_mode not in ("molecular_wiggle", "forced_jump"):
    raise ValueError("simulation_mode must be 'molecular_wiggle' or 'forced_jump'")


# Make the initial Matrix
y = cfg.y
x = cfg.x
steps = cfg.steps
max_radius_to_jump = cfg.max_radius_to_jump
print(f"Simulation mode: {cfg.simulation_mode}")

# Precompute random numbers
random_size = cfg.random_size
random_values = np.random.rand(random_size).astype(np.float32)
rand_index = 0
if cfg.simulation_mode == "molecular_wiggle":
    sigma = max_radius_to_jump / 3
    jump_probability_table = create_jump_probability_table(max_radius_to_jump, sigma)
else:
    jump_probability_table = np.empty((0, 0), dtype=np.float32)

h5_filename = results_dir() / cfg.h5_filename
saved_steps = np.arange(0, steps, cfg.save_every_steps, dtype=np.int64)
num_saved_frames = len(saved_steps)
print(f"Total frames to be saved: {num_saved_frames}, approx. "
      f"{int((num_saved_frames * (y * x * np.dtype(np.int8).itemsize) / (1024 ** 2)) * 1.15)} MB")

print("Creating initial matrix")
lattice_spacing_used = None
if cfg.MATRIX_SOURCE == "image":
    h_spots_matrix = create_matrix_from_image(
        image_dir() / cfg.image_name,
        cfg.max_sol_white,
        cfg.max_sol_black,
        show_plot=cfg.show_image_matrix_plot,
    )
else:
    num_possible_spots_a = int(y * (x // 2) * cfg.max_sol_a)
    num_possible_spots_b = int(y * (x // 2) * cfg.max_sol_b)
    if cfg.MATRIX_SOURCE == "random":
        h_spots_matrix = create_custom_matrix(x, y, num_possible_spots_a, num_possible_spots_b)
    else:
        h_spots_matrix, lattice_spacing_used = create_lattice_matrix_for_halves(
            x,
            y,
            num_possible_spots_a,
            num_possible_spots_b,
            lattice_style=cfg.LATTICE_STYLE,
            start_spacing=cfg.LATTICE_START_SPACING,
            min_spacing=cfg.LATTICE_MIN_SPACING,
        )
        print(f"Lattice style: {cfg.LATTICE_STYLE}, spacing used: {lattice_spacing_used}")

sink_source_thickness = cfg.SINK_SOURCE_THICKNESS if cfg.USE_SINK_SOURCE else 0
region_map, num_regions = create_region_mapping(
    x,
    y,
    sink_source_thickness,
    cfg.TRAP_LAYER_WIDTH,
    cfg.num_subregions,
)

print("Applying concentration")
h_spots_matrix = define_concentration_to_halves(
    h_spots_matrix,
    cfg.concentration_a,
    cfg.concentration_b,
)
if cfg.USE_SINK_SOURCE:
    h_spots_matrix = define_concentration_sink_source(
        h_spots_matrix,
        sink_source_thickness,
        source_side=cfg.SOURCE_SIDE,
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
    h_spots_matrix = apply_layer(
        h_spots_matrix,
        width=cfg.TRAP_LAYER_WIDTH,
    )

print("Cleaning Loners")
clean_loners(h_spots_matrix, max_radius_to_jump)

if cfg.delete_old_h5 and h5_filename.exists():
    h5_filename.unlink()

height, width = h_spots_matrix.shape
active_y, active_x = create_active_site_arrays(h_spots_matrix)
use_forced_jump_precomputed_lane = cfg.simulation_mode == "forced_jump" and not cfg.USE_SINK_SOURCE
if use_forced_jump_precomputed_lane:
    site_y, site_x, neighbor_site_ids, neighbor_counts, site_states, hydrogen_site_ids = create_forced_jump_lookup(
        h_spots_matrix,
        max_radius_to_jump,
    )
    average_forced_jump_target_count = float(np.mean(neighbor_counts)) if len(neighbor_counts) else 0.0
else:
    site_y = np.empty(0, dtype=np.int32)
    site_x = np.empty(0, dtype=np.int32)
    neighbor_site_ids = np.empty((0, 4), dtype=np.int32)
    neighbor_counts = np.empty(0, dtype=np.int32)
    site_states = np.empty(0, dtype=np.int8)
    hydrogen_site_ids = np.empty(0, dtype=np.int32)
    average_forced_jump_target_count = None
snapshot_size_mb = (height * width * np.dtype(np.int8).itemsize) / (1024 ** 2)
optimal_batch_size = max(1, int(cfg.max_ram_mb / snapshot_size_mb))

print(f"Matrix size: {height}x{width}, Snapshot size: {snapshot_size_mb:.2f} MB")
print(f"Active sites scanned per step: {len(active_y)} of {height * width}")
if cfg.simulation_mode == "forced_jump":
    if use_forced_jump_precomputed_lane:
        print(
            "Forced jump compact precomputed lane: "
            f"{len(hydrogen_site_ids)} hydrogen atoms, {len(site_y)} possible sites, "
            f"{average_forced_jump_target_count:.1f} average targets/site"
        )
    else:
        print("Forced jump safe lane: scanning active sites because USE_SINK_SOURCE changes hydrogen count")
print(f"Using batch size of {optimal_batch_size} frames (max {cfg.max_ram_mb}MB RAM usage)")
print(f"Initial matrix unique values: {np.unique(h_spots_matrix)}")

print("Region Map Summary:")
unique_regions, counts = np.unique(region_map, return_counts=True)
for region, count in zip(unique_regions, counts):
    print(f"Region {region}: {count} columns assigned")

with h5py.File(h5_filename, "w") as hf:
    write_brown_h5_metadata(
        hf,
        runtime_values={
            "actual_matrix_shape": h_spots_matrix.shape,
            "num_saved_frames": num_saved_frames,
            "saved_steps_first": int(saved_steps[0]) if len(saved_steps) else None,
            "saved_steps_last": int(saved_steps[-1]) if len(saved_steps) else None,
            "sink_source_thickness_used": sink_source_thickness,
            "num_regions": num_regions,
            "active_site_count": len(active_y),
            "hydrogen_site_count": len(hydrogen_site_ids) if use_forced_jump_precomputed_lane else None,
            "forced_jump_precomputed_lane_used": use_forced_jump_precomputed_lane,
            "forced_jump_average_target_count": average_forced_jump_target_count,
            "lattice_spacing_used": lattice_spacing_used,
            "simulation_mode_used": cfg.simulation_mode,
        },
    )

    dset = hf.create_dataset("snapshots", shape=(num_saved_frames, height, width), dtype=np.int8, chunks=True)
    hf.create_dataset("saved_steps", data=saved_steps, dtype=np.int64)

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
    buffer = np.empty((optimal_batch_size, height, width), dtype=np.int8)
    buffer_index = 0

    disp_buffer = np.zeros((optimal_batch_size, num_regions, 3), dtype=np.float32)
    disp_buffer_index = 0

    for step in tqdm(range(steps)):
        if cfg.simulation_mode == "molecular_wiggle":
            h_spots_matrix, rand_index, disp_stats = simulate_brownian_motion(
                h_spots_matrix,
                random_values,
                active_y,
                active_x,
                x,
                y,
                rand_index,
                random_size,
                max_radius_to_jump,
                cfg.base_movement_probability,
                jump_probability_table,
                sink_source_thickness,
                cfg.USE_SINK_SOURCE,
                cfg.SOURCE_SIDE == "left",
                region_map,
                num_regions,
            )
        elif use_forced_jump_precomputed_lane:
            site_states, rand_index, disp_stats = simulate_brownian_motion_forced_jump_precomputed(
                site_states,
                random_values,
                hydrogen_site_ids,
                site_y,
                site_x,
                neighbor_site_ids,
                neighbor_counts,
                rand_index,
                random_size,
                region_map,
                num_regions,
            )
        else:
            h_spots_matrix, rand_index, disp_stats = simulate_brownian_motion_forced_jump(
                h_spots_matrix,
                random_values,
                active_y,
                active_x,
                x,
                y,
                rand_index,
                random_size,
                max_radius_to_jump,
                sink_source_thickness,
                cfg.USE_SINK_SOURCE,
                cfg.SOURCE_SIDE == "left",
                region_map,
                num_regions,
            )

        if should_save_frame(step, cfg.save_every_steps):
            if use_forced_jump_precomputed_lane:
                buffer[buffer_index] = create_matrix_from_site_states(
                    (height, width),
                    site_y,
                    site_x,
                    site_states,
                )
            else:
                buffer[buffer_index] = h_spots_matrix
            disp_buffer[disp_buffer_index] = disp_stats

            buffer_index += 1
            disp_buffer_index += 1

            if buffer_index == optimal_batch_size:
                dset[save_counter:save_counter + buffer_index] = buffer[:buffer_index]

                for i in range(num_regions):
                    displacement_dsets[f"time_{i}"][save_counter:save_counter + buffer_index] = (
                        saved_steps[save_counter:save_counter + buffer_index]
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
            displacement_dsets[f"time_{i}"][save_counter:save_counter + buffer_index] = (
                saved_steps[save_counter:save_counter + buffer_index]
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
