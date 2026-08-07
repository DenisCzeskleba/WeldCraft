import os
import sys
import json
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from b3_Brown_Functions import *
from b4_Brown_Checkpoint import (
    build_saved_steps,
    load_resume_state,
    resolve_resume_path,
    restore_compact_hydrogen_order,
    restore_ordered_hydrogen_site_ids,
    validate_resume_config,
    write_exact_checkpoint,
    write_final_checkpoint,
)


cfg = load_brown_config()
gui_run = os.environ.get("WELDCRAFT_P6_GUI_RUN") == "1"
gui_stop_file = os.environ.get("WELDCRAFT_P6_STOP_FILE", "").strip()
gui_settings_json = os.environ.get("WELDCRAFT_P6_SETTINGS_JSON", "").strip()
if gui_settings_json:
    gui_settings = json.loads(gui_settings_json)
    for fraction_name in ("max_sol_a", "max_sol_b", "max_sol_spot", "max_sol_trap_layer"):
        if fraction_name in gui_settings:
            gui_settings[fraction_name] = Fraction(gui_settings[fraction_name])
    for setting_name, setting_value in gui_settings.items():
        setattr(cfg, setting_name, setting_value)


def gui_stop_requested():
    return bool(gui_stop_file and Path(gui_stop_file).exists())


def report_gui_progress(completed, total, committed_frames, message):
    if not gui_run:
        return
    fraction = 1.0 if total <= 0 else min(1.0, max(0.0, completed / total))
    print(
        f"P6_GUI_PROGRESS|{fraction:.12f}|{int(completed)}|{int(committed_frames)}|{message}",
        flush=True,
    )
compact_wiggle_modes = ("random_sequential_wiggle", "event_driven_wiggle")
wiggle_modes = ("molecular_wiggle", *compact_wiggle_modes)
if cfg.SOURCE_SIDE not in ("left", "right"):
    raise ValueError("SOURCE_SIDE must be 'left' or 'right'")
if cfg.MATRIX_SOURCE not in ("random", "image", "lattice"):
    raise ValueError("MATRIX_SOURCE must be 'random', 'image', or 'lattice'")
if cfg.simulation_mode not in (*wiggle_modes, "forced_jump"):
    raise ValueError(
        "simulation_mode must be 'molecular_wiggle', 'random_sequential_wiggle', "
        "'event_driven_wiggle', or 'forced_jump'"
    )
if cfg.base_movement_probability < 0 or cfg.base_movement_probability > 1:
    raise ValueError("base_movement_probability must be between 0 and 1")


# A resumed run treats cfg.steps as additional work and writes a new HDF5 segment.
steps = int(cfg.steps)
if steps < 0:
    raise ValueError("steps must be non-negative")
max_radius_to_jump = cfg.max_radius_to_jump
print(f"Simulation mode: {cfg.simulation_mode}")
if cfg.simulation_mode == "forced_jump":
    print(
        "DEPRECATED simulation mode: forced_jump is retained for old comparisons only "
        "and ignores AREA_CHARACTERISTICS."
    )

results_directory = results_dir()
h5_filename = results_directory / cfg.h5_filename
resume_path = resolve_resume_path(getattr(cfg, "RESUME_FROM_H5", None), results_directory)
resume_state = None
if resume_path is not None:
    if resume_path == h5_filename.resolve():
        raise ValueError(
            "RESUME_FROM_H5 and h5_filename resolve to the same file. "
            "Continuation must be written to a new HDF5 file."
        )
    resume_state = load_resume_state(resume_path)
    validate_resume_config(resume_state["source_config"], brown_config_snapshot())
    if resume_state["mode"] != cfg.simulation_mode:
        raise RuntimeError(
            "Resume source simulation mode does not match the current configuration: "
            f"{resume_state['mode']!r} != {cfg.simulation_mode!r}"
        )

segment_start_step = int(resume_state["step"]) if resume_state is not None else 0
segment_end_step = segment_start_step + steps
saved_steps = build_saved_steps(
    segment_start_step,
    steps,
    cfg.save_every_steps,
)
num_saved_frames = len(saved_steps)

# Exact checkpoints restore the running generator. Statistical continuation from
# legacy/interrupted files intentionally starts a fresh saved seed.
configured_random_seed = getattr(cfg, "random_seed", None)
if resume_state is not None and resume_state["exact"]:
    if resume_state["random_seed_used"] is None:
        raise RuntimeError("Exact checkpoint is missing random_seed_used_uint64.")
    random_seed_used = int(resume_state["random_seed_used"])
elif configured_random_seed is None:
    random_seed_used = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
else:
    random_seed_used = int(configured_random_seed) & ((1 << 64) - 1)

np.random.seed(random_seed_used & 0xFFFFFFFF)
random_size = cfg.random_size
rand_index = (
    int(resume_state["forced_jump_rand_index"])
    if resume_state is not None
    and resume_state["exact"]
    and resume_state["forced_jump_rand_index"] is not None
    else 0
)
if cfg.simulation_mode in wiggle_modes:
    random_values = np.empty(0, dtype=np.float32)
    if resume_state is not None and resume_state["exact"]:
        if resume_state["rng_state"] is None or len(resume_state["rng_state"]) != 4:
            raise RuntimeError("Exact wiggle checkpoint is missing its four-word RNG state.")
        molecular_rng_state = resume_state["rng_state"].copy()
    else:
        molecular_rng_state = create_xoshiro256ss_state(random_seed_used)
    sigma = max_radius_to_jump / 3
    jump_probability_table = create_jump_probability_table(max_radius_to_jump, sigma)
    uniformization_site_bound = create_wiggle_uniformization_bound(
        max_radius_to_jump,
        jump_probability_table,
    )
    random_algorithm_used = "xoshiro256**"
else:
    random_values = np.random.default_rng(random_seed_used).random(random_size, dtype=np.float32)
    molecular_rng_state = np.empty(0, dtype=np.uint64)
    jump_probability_table = np.empty((0, 0), dtype=np.float32)
    uniformization_site_bound = 0.0
    random_algorithm_used = "precomputed PCG64 float32 buffer"

print(f"Random algorithm: {random_algorithm_used}, seed: {random_seed_used}")
lattice_spacing_used = None
sink_source_thickness = cfg.SINK_SOURCE_THICKNESS if cfg.USE_SINK_SOURCE else 0
if resume_state is not None:
    h_spots_matrix = resume_state["matrix"].copy()
    resume_kind = "exact" if resume_state["exact"] else "statistical"
    print(
        f"Continuing {resume_kind} state from {resume_path}\n"
        f"  source snapshot index: {resume_state['snapshot_index']}\n"
        f"  starting global step: {segment_start_step:,}\n"
        f"  additional steps: {steps:,}\n"
        f"  ending global step: {segment_end_step:,}"
    )
    if resume_state["legacy_step_adjusted"]:
        print("  adjusted legacy loop-based snapshot label by +1 step")
else:
    resume_kind = "new"
    y = cfg.y
    x = cfg.x
    print("Creating initial matrix")
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

    print("Applying concentration")
    if cfg.USE_INITIAL_CONCENTRATION_PROFILE:
        concentration_profile_a = validate_concentration_profile(
            cfg.concentration_profile_a,
            "concentration_profile_a",
        )
        concentration_profile_b = validate_concentration_profile(
            cfg.concentration_profile_b,
            "concentration_profile_b",
        )
        print(
            "Applying left-to-right initial concentration profiles: "
            f"A {concentration_profile_a[0]:g}% -> {concentration_profile_a[1]:g}%, "
            f"B {concentration_profile_b[0]:g}% -> {concentration_profile_b[1]:g}%"
        )
        h_spots_matrix = define_linear_concentration_profiles_to_halves(
            h_spots_matrix,
            concentration_profile_a,
            concentration_profile_b,
        )
    else:
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

    if cfg.USE_TRAP_LAYER:
        print(
            f"Adding trap layer at x={cfg.TRAP_LAYER_CENTER_X}, "
            f"{float(cfg.max_sol_trap_layer) * 100:g}% max. solubility, "
            f"{cfg.concentration_trap_layer}% initial concentration"
        )
        h_spots_matrix = apply_layer(
            h_spots_matrix,
            width=cfg.TRAP_LAYER_WIDTH,
            center_x=cfg.TRAP_LAYER_CENTER_X,
            max_solubility=cfg.max_sol_trap_layer,
            concentration=cfg.concentration_trap_layer,
        )

    if cfg.USE_SPOT:
        print(
            f"Adding spot with {float(cfg.max_sol_spot) * 100:g}% max. solubility "
            f"and {cfg.concentration_spot}% initial concentration"
        )
        h_spots_matrix = apply_spot(
            h_spots_matrix,
            diameter=cfg.SPOT_DIAMETER,
            center_x=cfg.SPOT_CENTER_X,
            center_y=cfg.SPOT_CENTER_Y,
            concentration=cfg.concentration_spot,
            max_solubility=cfg.max_sol_spot,
        )

    print("Cleaning Loners")
    clean_loners(h_spots_matrix, max_radius_to_jump)

height, width = h_spots_matrix.shape
print(
    f"Total frames to be saved: {num_saved_frames}, approx. "
    f"{int((num_saved_frames * (height * width * np.dtype(np.int8).itemsize) / (1024 ** 2)) * 1.15)} MB"
)
region_map, num_regions = create_region_mapping(
    width,
    height,
    sink_source_thickness,
    cfg.TRAP_LAYER_WIDTH,
    cfg.num_subregions,
    layer_center_x=cfg.TRAP_LAYER_CENTER_X,
)
region_widths = np.asarray(
    [np.count_nonzero(region_map == region_index) for region_index in range(num_regions)],
    dtype=np.float64,
)
region_centers_x = np.asarray(
    [
        np.mean(np.flatnonzero(region_map == region_index))
        for region_index in range(num_regions)
    ],
    dtype=np.float64,
)
if np.any(region_widths <= 0) or np.any(~np.isfinite(region_centers_x)):
    raise RuntimeError("Transport regions must each contain at least one matrix x-column.")

characteristic_map = create_area_characteristic_map(
    h_spots_matrix.shape,
    use_spot=cfg.USE_SPOT,
    spot_diameter=cfg.SPOT_DIAMETER,
    spot_center_x=cfg.SPOT_CENTER_X,
    spot_center_y=cfg.SPOT_CENTER_Y,
    use_trap_layer=cfg.USE_TRAP_LAYER,
    trap_layer_width=cfg.TRAP_LAYER_WIDTH,
    trap_layer_center_x=cfg.TRAP_LAYER_CENTER_X,
)
area_affinities, area_mobilities, characteristic_transition_multipliers = (
    create_area_transition_model(cfg.AREA_CHARACTERISTICS)
)
print("Area characteristics:")
for area_id, area_name in enumerate(AREA_CHARACTERISTIC_NAMES):
    print(
        f"  {area_name}: affinity={area_affinities[area_id]:g}, "
        f"mobility={area_mobilities[area_id]:g}"
    )

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
use_event_driven_lane = cfg.simulation_mode == "event_driven_wiggle"
use_compact_wiggle_lane = cfg.simulation_mode in compact_wiggle_modes
event_pending_wait_steps = (
    int(resume_state["event_pending_wait_steps"])
    if resume_state is not None and resume_state["exact"] and use_event_driven_lane
    else 0
)
compact_hydrogen_count = 0
compact_transition_count = 0
compact_lookup_mb = 0.0
transition_offsets = np.empty(0, dtype=np.int64)
transition_targets = np.empty(0, dtype=np.int32)
transition_cdf = np.empty(0, dtype=np.float32)
transition_totals = np.empty(0, dtype=np.float32)
hydrogen_transition_totals = np.empty(0, dtype=np.float32)
hydrogen_probability_tree = np.empty(0, dtype=np.float64)
total_transition_weight = 0.0
max_transition_total = 0.0
source_site_flags = np.empty(0, dtype=np.uint8)
sink_site_flags = np.empty(0, dtype=np.uint8)

if use_compact_wiggle_lane:
    (
        site_y,
        site_x,
        site_states,
        hydrogen_site_ids,
        hydrogen_transition_totals,
        compact_hydrogen_count,
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
        characteristic_map,
        characteristic_transition_multipliers,
    )
    if resume_state is not None and resume_state["exact"]:
        saved_order = resume_state["ordered_hydrogen_site_ids"]
        if saved_order is None:
            raise RuntimeError("Exact compact checkpoint is missing hydrogen ordering.")
        compact_hydrogen_count = restore_compact_hydrogen_order(
            site_states,
            hydrogen_site_ids,
            hydrogen_transition_totals,
            transition_totals,
            saved_order,
        )
    neighbor_site_ids = np.empty((0, 4), dtype=np.int32)
    neighbor_counts = np.empty(0, dtype=np.int32)
    if use_event_driven_lane:
        hydrogen_probability_tree, total_transition_weight = (
            create_hydrogen_probability_fenwick_tree(
                hydrogen_transition_totals,
                compact_hydrogen_count,
            )
        )
        if resume_state is not None and resume_state["exact"]:
            saved_tree = resume_state["event_fenwick_tree"]
            saved_total = resume_state["event_total_transition_weight"]
            if saved_tree is None or saved_tree.shape != hydrogen_probability_tree.shape:
                raise RuntimeError(
                    "Exact event-driven checkpoint Fenwick tree has the wrong shape."
                )
            if saved_total is None or not np.isfinite(saved_total):
                raise RuntimeError(
                    "Exact event-driven checkpoint transition weight is invalid."
                )
            hydrogen_probability_tree[:] = saved_tree
            total_transition_weight = float(saved_total)
    compact_transition_count = len(transition_targets)
    max_transition_total = (
        float(np.max(transition_totals)) if len(transition_totals) else 0.0
    )
    compact_lookup_mb = sum(
        array.nbytes for array in (
            site_y,
            site_x,
            site_states,
            hydrogen_site_ids,
            hydrogen_transition_totals,
            transition_offsets,
            transition_targets,
            transition_cdf,
            transition_totals,
            source_site_flags,
            sink_site_flags,
            hydrogen_probability_tree,
        )
    ) / (1024 ** 2)
    average_forced_jump_target_count = None
elif use_forced_jump_precomputed_lane:
    site_y, site_x, neighbor_site_ids, neighbor_counts, site_states, hydrogen_site_ids = create_forced_jump_lookup(
        h_spots_matrix,
        max_radius_to_jump,
    )
    if resume_state is not None and resume_state["exact"]:
        saved_order = resume_state["ordered_hydrogen_site_ids"]
        if saved_order is None:
            raise RuntimeError(
                "Exact forced-jump checkpoint is missing hydrogen ordering."
            )
        restore_ordered_hydrogen_site_ids(
            site_states,
            hydrogen_site_ids,
            saved_order,
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

print(f"Matrix size: {height}x{width}, Snapshot size: {snapshot_size_mb:.2f} MB")
if cfg.simulation_mode == "molecular_wiggle":
    print(f"Active sites scanned per step: {len(active_y)} of {height * width}")
elif use_random_sequential_lane:
    average_targets = compact_transition_count / max(len(site_y), 1)
    print(
        "Random sequential compact lane: "
        f"{compact_hydrogen_count} hydrogen atoms, {len(site_y)} possible sites, "
        f"{average_targets:.1f} average targets/site, {compact_lookup_mb:.1f} MB lookup"
    )
elif use_event_driven_lane:
    average_targets = compact_transition_count / max(len(site_y), 1)
    initial_uniformized_proposal_fraction = (
        total_transition_weight / (compact_hydrogen_count * uniformization_site_bound)
        if compact_hydrogen_count > 0 and uniformization_site_bound > 0
        else 0.0
    )
    print(
        "Event-driven compact lane: "
        f"{compact_hydrogen_count} hydrogen atoms, {len(site_y)} possible sites, "
        f"{average_targets:.1f} average targets/site, {compact_lookup_mb:.1f} MB lookup, "
        f"{initial_uniformized_proposal_fraction:.1%} initial non-null proposals"
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
print(f"Initial matrix unique values: {np.unique(h_spots_matrix)}")

print("Region Map Summary:")
unique_regions, counts = np.unique(region_map, return_counts=True)
for region, count in zip(unique_regions, counts):
    print(f"Region {region}: {count} columns assigned")

if use_random_sequential_lane:
    step_definition = "hydrogen_count_at_step_start random selections with replacement"
    molecular_sampler_used = "precomputed exact marginal CDF with 32-bit selection"
elif use_event_driven_lane:
    step_definition = "one uniformized wiggle opportunity; null waiting is skipped geometrically"
    molecular_sampler_used = (
        "uniformized Fenwick rate selection with geometric null-wait skipping"
    )
elif cfg.simulation_mode == "molecular_wiggle":
    step_definition = "one synchronous global molecular update sweep"
    molecular_sampler_used = "20/20/24-bit rejection sampler v1"
else:
    step_definition = "one forced-jump update sweep"
    molecular_sampler_used = None


def run_compact_wiggle_work(
    hydrogen_count,
    transition_weight,
    work_count,
    pending_wait_steps,
    interface_crossing_deltas,
    reservoir_event_counts,
):
    if use_random_sequential_lane:
        hydrogen_count, displacement_stats, wiggle_attempt_count = (
            simulate_random_sequential_wiggle_steps(
                site_states,
                hydrogen_site_ids,
                hydrogen_count,
                hydrogen_transition_totals,
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
                work_count,
                interface_crossing_deltas,
                reservoir_event_counts,
            )
        )
        return (
            hydrogen_count,
            transition_weight,
            displacement_stats,
            work_count,
            wiggle_attempt_count,
            pending_wait_steps,
        )

    return simulate_event_driven_wiggle_events(
        site_states,
        hydrogen_site_ids,
        hydrogen_count,
        hydrogen_transition_totals,
        hydrogen_probability_tree,
        transition_weight,
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
        uniformization_site_bound,
        work_count,
        pending_wait_steps,
        interface_crossing_deltas,
        reservoir_event_counts,
    )


with h5py.File(h5_filename, "w") as hf:
    write_brown_h5_metadata(
        hf,
        runtime_values={
            "actual_matrix_shape": h_spots_matrix.shape,
            "num_saved_frames": num_saved_frames,
            "saved_steps_first": int(saved_steps[0]) if len(saved_steps) else None,
            "saved_steps_last": int(saved_steps[-1]) if len(saved_steps) else None,
            "segment_start_step": segment_start_step,
            "segment_additional_steps": steps,
            "segment_end_step": segment_end_step,
            "resume_kind": resume_kind,
            "resumed_from": str(resume_path) if resume_path is not None else None,
            "resumed_source_snapshot_index": (
                resume_state["snapshot_index"] if resume_state is not None else None
            ),
            "sink_source_thickness_used": sink_source_thickness,
            "num_regions": num_regions,
            "active_site_count": len(active_y),
            "hydrogen_site_count": (
                compact_hydrogen_count
                if use_compact_wiggle_lane
                else len(hydrogen_site_ids) if use_forced_jump_precomputed_lane else None
            ),
            "forced_jump_precomputed_lane_used": use_forced_jump_precomputed_lane,
            "forced_jump_average_target_count": average_forced_jump_target_count,
            "random_sequential_compact_lane_used": use_random_sequential_lane,
            "event_driven_compact_lane_used": use_event_driven_lane,
            "compact_wiggle_transition_count": compact_transition_count,
            "compact_wiggle_lookup_mb": compact_lookup_mb,
            "step_definition": step_definition,
            "lattice_spacing_used": lattice_spacing_used,
            "simulation_mode_used": cfg.simulation_mode,
            "random_algorithm_used": random_algorithm_used,
            "random_seed_used": random_seed_used,
            "initialization_random_algorithm_used": "NumPy legacy MT19937",
            "molecular_seed_expander_used": "SplitMix64" if cfg.simulation_mode in wiggle_modes else None,
            "molecular_sampler_used": molecular_sampler_used,
            "area_characteristic_names": AREA_CHARACTERISTIC_NAMES,
            "area_affinities_used": area_affinities,
            "area_mobilities_used": area_mobilities,
            "area_transition_rule": "symmetric geometric-mean mobility times Metropolis affinity acceptance",
            "area_transition_multipliers": characteristic_transition_multipliers,
            "forced_jump_deprecated": cfg.simulation_mode == "forced_jump",
            "event_driven_uniformization_used": use_event_driven_lane,
            "event_driven_max_site_transition_total": (
                max_transition_total if use_event_driven_lane else None
            ),
            "event_driven_uniformization_site_bound": (
                uniformization_site_bound if use_event_driven_lane else None
            ),
        },
    )
    hf["meta"].attrs["random_seed_used_uint64"] = np.uint64(random_seed_used)
    hf.attrs["run_status"] = "in_progress"
    hf.attrs["frames_written"] = np.int64(0)

    dset = hf.create_dataset(
        "snapshots",
        shape=(num_saved_frames, height, width),
        dtype=np.int8,
        chunks=(1, height, width),
    )
    hf.create_dataset("saved_steps", data=saved_steps, dtype=np.int64)

    if resume_state is not None:
        provenance_group = hf.create_group("resume")
        provenance_group.attrs["source_path"] = str(resume_path)
        provenance_group.attrs["source_snapshot_index"] = np.int64(
            resume_state["snapshot_index"]
        )
        provenance_group.attrs["source_step"] = np.int64(segment_start_step)
        provenance_group.attrs["kind"] = resume_kind
        provenance_group.attrs["legacy_step_adjusted"] = bool(
            resume_state["legacy_step_adjusted"]
        )

    transport_group = hf.create_group("transport")
    transport_group.attrs["schema"] = "interface_crossings_v2"
    transport_group.attrs["description"] = (
        "Signed particle crossings through every vertical x-interface during each "
        "saved interval, with source insertions and sink removals recorded separately."
    )
    interface_net_crossings_dset = transport_group.create_dataset(
        "interface_net_crossings",
        shape=(num_saved_frames, max(width - 1, 0)),
        dtype=np.int64,
    )
    transport_group.create_dataset(
        "interface_x",
        data=np.arange(max(width - 1, 0), dtype=np.float64) + 0.5,
    )
    source_insertion_count_dset = transport_group.create_dataset(
        "source_insertion_count",
        shape=(num_saved_frames,),
        dtype=np.int64,
    )
    sink_removal_count_dset = transport_group.create_dataset(
        "sink_removal_count",
        shape=(num_saved_frames,),
        dtype=np.int64,
    )
    net_x_displacement_dset = transport_group.create_dataset(
        "net_x_displacement",
        shape=(num_saved_frames, num_regions),
        dtype=np.float64,
    )
    net_x_displacement_dset.attrs["description"] = (
        "Legacy signed hop displacement grouped by the hop's source region; "
        "this is not a local cross-sectional flux."
    )
    accepted_move_count_dset = transport_group.create_dataset(
        "accepted_move_count",
        shape=(num_saved_frames, num_regions),
        dtype=np.float64,
    )
    accepted_move_count_dset.attrs["description"] = (
        "Accepted hops grouped by source region; useful as a local activity diagnostic."
    )
    interval_steps_dset = transport_group.create_dataset(
        "interval_steps",
        shape=(num_saved_frames,),
        dtype=np.int64,
    )
    transport_group.create_dataset("region_widths", data=region_widths)
    transport_group.create_dataset("region_centers_x", data=region_centers_x)

    if use_compact_wiggle_lane:
        hydrogen_count_dset = hf.create_dataset(
            "hydrogen_count",
            shape=(num_saved_frames,),
            dtype=np.int64,
        )
        progress_count_name = (
            "wiggle_attempt_count"
            if use_random_sequential_lane
            else "proposal_event_count"
        )
        progress_count_dset = hf.create_dataset(
            progress_count_name,
            shape=(num_saved_frames,),
            dtype=np.int64,
        )
        previous_saved_step = segment_start_step
        compact_progress_count = 0
        progress_description = (
            "Random sequential wiggle steps"
            if use_random_sequential_lane
            else "Event-driven uniformized steps"
        )
        cancelled = False

        with tqdm(total=steps, desc=progress_description, disable=gui_run) as progress:
            for frame_index, saved_step in enumerate(saved_steps):
                steps_to_run = int(saved_step - previous_saved_step)
                interface_crossing_deltas = np.zeros(width, dtype=np.int64)
                reservoir_event_counts = np.zeros(2, dtype=np.int64)
                if steps_to_run > 0:
                    disp_stats = np.zeros((num_regions, 3), dtype=np.float64)
                    completed_in_interval = 0
                    chunk_size = min(
                        1_000_000,
                        max(10_000, int(cfg.save_every_steps) // 100),
                    )
                    while completed_in_interval < steps_to_run:
                        work_count = min(
                            chunk_size,
                            steps_to_run - completed_in_interval,
                        )
                        (
                            compact_hydrogen_count,
                            total_transition_weight,
                            chunk_stats,
                            completed_work,
                            completed_activity_count,
                            event_pending_wait_steps,
                        ) = run_compact_wiggle_work(
                            compact_hydrogen_count,
                            total_transition_weight,
                            work_count,
                            event_pending_wait_steps,
                            interface_crossing_deltas,
                            reservoir_event_counts,
                        )
                        if use_event_driven_lane and completed_work != work_count:
                            raise RuntimeError(
                                "event_driven_wiggle failed to complete its uniformized steps"
                            )
                        completed_in_interval += completed_work
                        compact_progress_count += completed_activity_count
                        disp_stats += chunk_stats
                        progress.update(completed_work)
                        completed_total = (
                            previous_saved_step
                            - segment_start_step
                            + completed_in_interval
                        )
                        report_gui_progress(
                            completed_total,
                            steps,
                            frame_index,
                            f"Simulating step {segment_start_step + completed_total:,}",
                        )
                        if gui_stop_requested():
                            cancelled = True
                            break
                    if cancelled:
                        break
                else:
                    disp_stats = np.zeros((num_regions, 3), dtype=np.float64)

                dset[frame_index] = create_matrix_from_site_states(
                    (height, width),
                    site_y,
                    site_x,
                    site_states,
                )
                hydrogen_count_dset[frame_index] = compact_hydrogen_count
                progress_count_dset[frame_index] = compact_progress_count

                net_x_displacement_dset[frame_index] = disp_stats[:, 0]
                accepted_move_count_dset[frame_index] = disp_stats[:, 2]
                interface_net_crossings_dset[frame_index] = np.cumsum(
                    interface_crossing_deltas,
                    dtype=np.int64,
                )[:-1]
                source_insertion_count_dset[frame_index] = reservoir_event_counts[0]
                sink_removal_count_dset[frame_index] = reservoir_event_counts[1]
                interval_steps_dset[frame_index] = steps_to_run

                previous_saved_step = int(saved_step)
                hf.attrs["frames_written"] = np.int64(frame_index + 1)
                write_exact_checkpoint(
                    hf,
                    step=int(saved_step),
                    snapshot_index=frame_index,
                    simulation_mode=cfg.simulation_mode,
                    random_seed_used=random_seed_used,
                    rng_state=molecular_rng_state,
                    ordered_hydrogen_site_ids=(
                        hydrogen_site_ids[:compact_hydrogen_count].copy()
                    ),
                    event_pending_wait_steps=(
                        event_pending_wait_steps if use_event_driven_lane else None
                    ),
                    event_fenwick_tree=(
                        hydrogen_probability_tree if use_event_driven_lane else None
                    ),
                    event_total_transition_weight=(
                        total_transition_weight if use_event_driven_lane else None
                    ),
                )
                hf.flush()
                report_gui_progress(
                    int(saved_step) - segment_start_step,
                    steps,
                    frame_index + 1,
                    f"Committed frame {frame_index + 1}/{num_saved_frames} at step {int(saved_step):,}",
                )
        if cancelled:
            hf.attrs["run_status"] = "cancelled"
            hf.flush()
            committed_frames = int(hf.attrs.get("frames_written", 0))
            committed_step = (
                int(hf["saved_steps"][committed_frames - 1])
                if committed_frames
                else segment_start_step
            )
            report_gui_progress(
                committed_step - segment_start_step,
                steps,
                committed_frames,
                f"Cancelled; retained exact checkpoint at step {committed_step:,}",
            )
            print(
                f"P6_GUI_CANCELLED: retained {committed_frames} frame(s) through step {committed_step}",
                flush=True,
            )
            raise SystemExit(2)
    else:
        completed_step = segment_start_step
        molecular_epoch = 0
        with tqdm(total=steps, desc="Simulation steps", disable=gui_run) as progress:
            for frame_index, saved_step in enumerate(saved_steps):
                steps_to_run = int(saved_step - completed_step)
                interval_transport_stats = np.zeros(
                    (num_regions, 3),
                    dtype=np.float64,
                )
                interface_crossing_deltas = np.zeros(width, dtype=np.int64)
                reservoir_event_counts = np.zeros(2, dtype=np.int64)

                for _ in range(steps_to_run):
                    if cfg.simulation_mode == "molecular_wiggle":
                        molecular_epoch += 1
                        if molecular_epoch > np.iinfo(np.int32).max:
                            claim_epoch.fill(0)
                            molecular_epoch = 1
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
                            characteristic_map,
                            characteristic_transition_multipliers,
                            sink_source_thickness,
                            cfg.USE_SINK_SOURCE,
                            cfg.SOURCE_SIDE == "left",
                            region_map,
                            num_regions,
                            winner_source,
                            winner_priority,
                            claim_epoch,
                            touched_targets,
                            molecular_epoch,
                            interface_crossing_deltas,
                            reservoir_event_counts,
                        )
                    elif use_forced_jump_precomputed_lane:
                        (
                            site_states,
                            rand_index,
                            disp_stats,
                        ) = simulate_brownian_motion_forced_jump_precomputed(
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
                            interface_crossing_deltas,
                        )
                    else:
                        (
                            h_spots_matrix,
                            rand_index,
                            disp_stats,
                        ) = simulate_brownian_motion_forced_jump(
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
                            interface_crossing_deltas,
                            reservoir_event_counts,
                        )
                    interval_transport_stats += disp_stats

                if use_forced_jump_precomputed_lane:
                    dset[frame_index] = create_matrix_from_site_states(
                        (height, width),
                        site_y,
                        site_x,
                        site_states,
                    )
                else:
                    dset[frame_index] = h_spots_matrix
                net_x_displacement_dset[frame_index] = interval_transport_stats[:, 0]
                accepted_move_count_dset[frame_index] = interval_transport_stats[:, 2]
                interface_net_crossings_dset[frame_index] = np.cumsum(
                    interface_crossing_deltas,
                    dtype=np.int64,
                )[:-1]
                source_insertion_count_dset[frame_index] = reservoir_event_counts[0]
                sink_removal_count_dset[frame_index] = reservoir_event_counts[1]
                interval_steps_dset[frame_index] = steps_to_run

                completed_step = int(saved_step)
                progress.update(steps_to_run)
                hf.attrs["frames_written"] = np.int64(frame_index + 1)
                hf.flush()

    hf.attrs["max_radius_to_jump"] = max_radius_to_jump
    hf.attrs["matrix_shape"] = h_spots_matrix.shape
    hf.attrs["sink_source_thickness"] = sink_source_thickness
    if cfg.simulation_mode in wiggle_modes:
        hf["meta"].attrs["molecular_rng_state_after_run"] = molecular_rng_state
    if use_compact_wiggle_lane:
        hf["meta"].attrs["hydrogen_count_after_run"] = compact_hydrogen_count
        if use_random_sequential_lane:
            hf["meta"].attrs["wiggle_attempt_count_after_run"] = compact_progress_count
        else:
            hf["meta"].attrs["proposal_event_count_after_run"] = compact_progress_count
            hf["meta"].attrs["uniformized_step_count_after_run"] = steps
            hf["meta"].attrs["total_transition_weight_after_run"] = total_transition_weight

    if use_compact_wiggle_lane:
        checkpoint_hydrogen_order = hydrogen_site_ids[:compact_hydrogen_count].copy()
    elif use_forced_jump_precomputed_lane:
        checkpoint_hydrogen_order = hydrogen_site_ids.copy()
    else:
        checkpoint_hydrogen_order = None

    write_final_checkpoint(
        hf,
        step=segment_end_step,
        snapshot_index=num_saved_frames - 1,
        simulation_mode=cfg.simulation_mode,
        random_seed_used=random_seed_used,
        rng_state=molecular_rng_state if cfg.simulation_mode in wiggle_modes else None,
        ordered_hydrogen_site_ids=checkpoint_hydrogen_order,
        forced_jump_rand_index=rand_index if cfg.simulation_mode == "forced_jump" else None,
        event_pending_wait_steps=(
            event_pending_wait_steps if use_event_driven_lane else None
        ),
        event_fenwick_tree=(
            hydrogen_probability_tree if use_event_driven_lane else None
        ),
        event_total_transition_weight=(
            total_transition_weight if use_event_driven_lane else None
        ),
    )
