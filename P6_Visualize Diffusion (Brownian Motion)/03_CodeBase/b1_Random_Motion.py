import h5py
import numpy as np
from tqdm import tqdm

from b3_Brown_Functions import *


cfg = load_brown_config()
wiggle_modes = ("molecular_wiggle", "random_sequential_wiggle")
if cfg.SOURCE_SIDE not in ("left", "right"):
    raise ValueError("SOURCE_SIDE must be 'left' or 'right'")
if cfg.MATRIX_SOURCE not in ("random", "image", "lattice"):
    raise ValueError("MATRIX_SOURCE must be 'random', 'image', or 'lattice'")
if cfg.simulation_mode not in (*wiggle_modes, "forced_jump"):
    raise ValueError(
        "simulation_mode must be 'molecular_wiggle', 'random_sequential_wiggle', or 'forced_jump'"
    )


# Make the initial Matrix
y = cfg.y
x = cfg.x
steps = cfg.steps
max_radius_to_jump = cfg.max_radius_to_jump
print(f"Simulation mode: {cfg.simulation_mode}")

# Initialize reproducible random state. None selects a fresh seed that is saved in the HDF5 metadata.
configured_random_seed = getattr(cfg, "random_seed", None)
if configured_random_seed is None:
    random_seed_used = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
else:
    random_seed_used = int(configured_random_seed) & ((1 << 64) - 1)

np.random.seed(random_seed_used & 0xFFFFFFFF)
random_size = cfg.random_size
rand_index = 0
if cfg.simulation_mode in wiggle_modes:
    random_values = np.empty(0, dtype=np.float32)
    molecular_rng_state = create_xoshiro256ss_state(random_seed_used)
    sigma = max_radius_to_jump / 3
    jump_probability_table = create_jump_probability_table(max_radius_to_jump, sigma)
    random_algorithm_used = "xoshiro256**"
else:
    random_values = np.random.default_rng(random_seed_used).random(random_size, dtype=np.float32)
    molecular_rng_state = np.empty(0, dtype=np.uint64)
    jump_probability_table = np.empty((0, 0), dtype=np.float32)
    random_algorithm_used = "precomputed PCG64 float32 buffer"

print(f"Random algorithm: {random_algorithm_used}, seed: {random_seed_used}")

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

height, width = h_spots_matrix.shape
sink_source_thickness = cfg.SINK_SOURCE_THICKNESS if cfg.USE_SINK_SOURCE else 0
region_map, num_regions = create_region_mapping(
    width,
    height,
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

if cfg.USE_SPOT:
    print("Adding spot")
    h_spots_matrix = apply_spot(
        h_spots_matrix,
        diameter=cfg.SPOT_DIAMETER,
        center_x=cfg.SPOT_CENTER_X,
        center_y=cfg.SPOT_CENTER_Y,
    )

if cfg.USE_TRAP_LAYER:
    print("Adding trap layer")
    h_spots_matrix = apply_layer(
        h_spots_matrix,
        width=cfg.TRAP_LAYER_WIDTH,
    )

print("Cleaning Loners")
clean_loners(h_spots_matrix, max_radius_to_jump)

if cfg.delete_old_h5 and h5_filename.exists():
    h5_filename.unlink()

active_y, active_x = create_active_site_arrays(h_spots_matrix)
if cfg.simulation_mode == "molecular_wiggle":
    winner_source = np.empty(height * width, dtype=np.int32)
    winner_priority = np.empty(height * width, dtype=np.uint64)
    claim_epoch = np.zeros(height * width, dtype=np.int32)
    touched_targets = np.empty(len(active_y), dtype=np.int32)
else:
    winner_source = np.empty(0, dtype=np.int32)
    winner_priority = np.empty(0, dtype=np.uint64)
    claim_epoch = np.empty(0, dtype=np.int32)
    touched_targets = np.empty(0, dtype=np.int32)
use_forced_jump_precomputed_lane = cfg.simulation_mode == "forced_jump" and not cfg.USE_SINK_SOURCE
use_random_sequential_lane = cfg.simulation_mode == "random_sequential_wiggle"
random_sequential_hydrogen_count = 0
random_sequential_transition_count = 0
random_sequential_lookup_mb = 0.0
transition_offsets = np.empty(0, dtype=np.int64)
transition_targets = np.empty(0, dtype=np.int32)
transition_cdf = np.empty(0, dtype=np.float32)
transition_totals = np.empty(0, dtype=np.float32)
source_site_flags = np.empty(0, dtype=np.uint8)
sink_site_flags = np.empty(0, dtype=np.uint8)

if use_random_sequential_lane:
    (
        site_y,
        site_x,
        site_states,
        hydrogen_site_ids,
        random_sequential_hydrogen_count,
        transition_offsets,
        transition_targets,
        transition_cdf,
        transition_totals,
        source_site_flags,
        sink_site_flags,
    ) = create_random_sequential_wiggle_lookup(
        h_spots_matrix,
        max_radius_to_jump,
        cfg.base_movement_probability,
        jump_probability_table,
        sink_source_thickness,
        cfg.USE_SINK_SOURCE,
        cfg.SOURCE_SIDE == "left",
    )
    neighbor_site_ids = np.empty((0, 4), dtype=np.int32)
    neighbor_counts = np.empty(0, dtype=np.int32)
    random_sequential_transition_count = len(transition_targets)
    random_sequential_lookup_mb = sum(
        array.nbytes for array in (
            site_y,
            site_x,
            site_states,
            hydrogen_site_ids,
            transition_offsets,
            transition_targets,
            transition_cdf,
            transition_totals,
            source_site_flags,
            sink_site_flags,
        )
    ) / (1024 ** 2)
    average_forced_jump_target_count = None
elif use_forced_jump_precomputed_lane:
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
if cfg.simulation_mode == "molecular_wiggle":
    print(f"Active sites scanned per step: {len(active_y)} of {height * width}")
elif use_random_sequential_lane:
    average_targets = random_sequential_transition_count / max(len(site_y), 1)
    print(
        "Random sequential compact lane: "
        f"{random_sequential_hydrogen_count} hydrogen atoms, {len(site_y)} possible sites, "
        f"{average_targets:.1f} average targets/site, {random_sequential_lookup_mb:.1f} MB lookup"
    )
if cfg.simulation_mode == "forced_jump":
    if use_forced_jump_precomputed_lane:
        print(
            "Forced jump compact precomputed lane: "
            f"{len(hydrogen_site_ids)} hydrogen atoms, {len(site_y)} possible sites, "
            f"{average_forced_jump_target_count:.1f} average targets/site"
        )
    else:
        print("Forced jump safe lane: scanning active sites because USE_SINK_SOURCE changes hydrogen count")
if not use_random_sequential_lane:
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
            "hydrogen_site_count": (
                random_sequential_hydrogen_count
                if use_random_sequential_lane
                else len(hydrogen_site_ids) if use_forced_jump_precomputed_lane else None
            ),
            "forced_jump_precomputed_lane_used": use_forced_jump_precomputed_lane,
            "forced_jump_average_target_count": average_forced_jump_target_count,
            "random_sequential_compact_lane_used": use_random_sequential_lane,
            "random_sequential_transition_count": random_sequential_transition_count,
            "random_sequential_lookup_mb": random_sequential_lookup_mb,
            "step_definition": (
                "hydrogen_count_at_step_start random selections with replacement"
                if use_random_sequential_lane
                else "one global molecular update sweep"
            ),
            "lattice_spacing_used": lattice_spacing_used,
            "simulation_mode_used": cfg.simulation_mode,
            "random_algorithm_used": random_algorithm_used,
            "random_seed_used": random_seed_used,
            "initialization_random_algorithm_used": "NumPy legacy MT19937",
            "molecular_seed_expander_used": "SplitMix64" if cfg.simulation_mode in wiggle_modes else None,
            "molecular_sampler_used": (
                "precomputed exact marginal CDF with 32-bit selection"
                if use_random_sequential_lane
                else "20/20/24-bit rejection sampler v1" if cfg.simulation_mode == "molecular_wiggle" else None
            ),
        },
    )
    hf["meta"].attrs["random_seed_used_uint64"] = np.uint64(random_seed_used)

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

    if use_random_sequential_lane:
        hydrogen_count_dset = hf.create_dataset(
            "hydrogen_count",
            shape=(num_saved_frames,),
            dtype=np.int64,
        )
        wiggle_attempt_count_dset = hf.create_dataset(
            "wiggle_attempt_count",
            shape=(num_saved_frames,),
            dtype=np.int64,
        )
        previous_saved_step = 0
        random_sequential_attempt_count = 0

        with tqdm(total=steps, desc="Random sequential wiggle steps") as progress:
            for frame_index, saved_step in enumerate(saved_steps):
                steps_to_run = 0 if frame_index == 0 else int(saved_step - previous_saved_step)
                if steps_to_run > 0:
                    (
                        random_sequential_hydrogen_count,
                        disp_stats,
                        completed_attempts,
                    ) = simulate_random_sequential_wiggle_steps(
                        site_states,
                        hydrogen_site_ids,
                        random_sequential_hydrogen_count,
                        transition_offsets,
                        transition_targets,
                        transition_cdf,
                        transition_totals,
                        source_site_flags,
                        sink_site_flags,
                        site_x,
                        region_map,
                        num_regions,
                        molecular_rng_state,
                        steps_to_run,
                    )
                    random_sequential_attempt_count += completed_attempts
                    progress.update(steps_to_run)
                else:
                    disp_stats = np.zeros((num_regions, 3), dtype=np.float64)

                dset[frame_index] = create_matrix_from_site_states(
                    (height, width),
                    site_y,
                    site_x,
                    site_states,
                )
                hydrogen_count_dset[frame_index] = random_sequential_hydrogen_count
                wiggle_attempt_count_dset[frame_index] = random_sequential_attempt_count

                for i in range(num_regions):
                    movement_count = max(disp_stats[i, 2], 1)
                    mean_displacement = disp_stats[i, 0] / movement_count
                    displacement_dsets[f"time_{i}"][frame_index] = saved_step
                    displacement_dsets[f"mean_disp_{i}"][frame_index] = mean_displacement
                    displacement_dsets[f"var_disp_{i}"][frame_index] = (
                        (disp_stats[i, 1] / movement_count) - mean_displacement ** 2
                    )

                previous_saved_step = int(saved_step)

            remaining_steps = steps - previous_saved_step
            if remaining_steps > 0:
                (
                    random_sequential_hydrogen_count,
                    _,
                    completed_attempts,
                ) = simulate_random_sequential_wiggle_steps(
                    site_states,
                    hydrogen_site_ids,
                    random_sequential_hydrogen_count,
                    transition_offsets,
                    transition_targets,
                    transition_cdf,
                    transition_totals,
                    source_site_flags,
                    sink_site_flags,
                    site_x,
                    region_map,
                    num_regions,
                    molecular_rng_state,
                    remaining_steps,
                )
                random_sequential_attempt_count += completed_attempts
                progress.update(remaining_steps)
    else:
        save_counter = 0
        buffer = np.empty((optimal_batch_size, height, width), dtype=np.int8)
        buffer_index = 0

        disp_buffer = np.zeros((optimal_batch_size, num_regions, 3), dtype=np.float32)
        disp_buffer_index = 0

        for step in tqdm(range(steps)):
            if cfg.simulation_mode == "molecular_wiggle":
                h_spots_matrix, disp_stats = simulate_brownian_motion(
                    h_spots_matrix,
                    molecular_rng_state,
                    active_y,
                    active_x,
                    width,
                    height,
                    max_radius_to_jump,
                    cfg.base_movement_probability,
                    jump_probability_table,
                    sink_source_thickness,
                    cfg.USE_SINK_SOURCE,
                    cfg.SOURCE_SIDE == "left",
                    region_map,
                    num_regions,
                    winner_source,
                    winner_priority,
                    claim_epoch,
                    touched_targets,
                    step + 1,
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
                    width,
                    height,
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
    if cfg.simulation_mode in wiggle_modes:
        hf["meta"].attrs["molecular_rng_state_after_run"] = molecular_rng_state
    if use_random_sequential_lane:
        hf["meta"].attrs["hydrogen_count_after_run"] = random_sequential_hydrogen_count
        hf["meta"].attrs["wiggle_attempt_count_after_run"] = random_sequential_attempt_count
