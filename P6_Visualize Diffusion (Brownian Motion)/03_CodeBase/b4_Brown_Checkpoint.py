from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


CHECKPOINT_SCHEMA = "brownian_exact_restart_v1"

# These settings affect motion after the starting matrix has been loaded. Initial
# matrix-generation settings are intentionally absent because their result is
# already contained in the snapshot.
RESUME_DYNAMICS_KEYS = (
    "simulation_mode",
    "max_radius_to_jump",
    "base_movement_probability",
    "USE_SINK_SOURCE",
    "SINK_SOURCE_THICKNESS",
    "SOURCE_SIDE",
    "USE_SPOT",
    "SPOT_DIAMETER",
    "SPOT_CENTER_X",
    "SPOT_CENTER_Y",
    "USE_TRAP_LAYER",
    "TRAP_LAYER_CENTER_X",
    "TRAP_LAYER_WIDTH",
    "AREA_CHARACTERISTICS",
)


def build_saved_steps(start_step: int, additional_steps: int, save_every_steps: int) -> np.ndarray:
    """Return segment-relative save targets, including both endpoints exactly once."""
    start_step = int(start_step)
    additional_steps = int(additional_steps)
    save_every_steps = int(save_every_steps)
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if additional_steps < 0:
        raise ValueError("steps must be non-negative")
    if save_every_steps <= 0:
        raise ValueError("save_every_steps must be positive")

    end_step = start_step + additional_steps
    targets = np.arange(start_step, end_step, save_every_steps, dtype=np.int64)
    if len(targets) == 0 or int(targets[-1]) != end_step:
        targets = np.append(targets, np.int64(end_step))
    return targets


def resolve_resume_path(value, results_directory: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = results_directory / path
    return path.resolve()


def _load_config_json(hf: h5py.File) -> dict | None:
    meta = hf.get("meta")
    if meta is None:
        return None
    encoded = meta.attrs.get("brown_config_json")
    if encoded is None:
        return None
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    return json.loads(encoded)


def validate_resume_config(source_config: dict | None, current_config: dict) -> None:
    if source_config is None:
        raise RuntimeError(
            "The resume source has no Brownian config metadata, so its dynamics "
            "cannot be validated safely."
        )

    differences = []
    for key in RESUME_DYNAMICS_KEYS:
        if key not in source_config:
            differences.append(f"{key}: missing from source")
            continue
        if key not in current_config:
            differences.append(f"{key}: missing from current config")
            continue
        if source_config[key] != current_config[key]:
            differences.append(
                f"{key}: source={source_config[key]!r}, current={current_config[key]!r}"
            )

    if differences:
        details = "\n  - ".join(differences)
        raise RuntimeError(
            "Resume aborted because dynamics settings differ from the source HDF5:\n"
            f"  - {details}"
        )


def _snapshot_has_structure(snapshots: h5py.Dataset, index: int) -> bool:
    # Every usable simulation matrix contains possible sites (values 1 or 2).
    # Unwritten preallocated HDF5 frames contain only the default fill value 0.
    return bool(np.any(snapshots[index] != 0))


def find_last_valid_snapshot_index(hf: h5py.File) -> int:
    if "snapshots" not in hf or "saved_steps" not in hf:
        raise RuntimeError("Resume source must contain /snapshots and /saved_steps.")

    snapshots = hf["snapshots"]
    saved_steps = hf["saved_steps"]
    frame_count = min(len(snapshots), len(saved_steps))
    if frame_count == 0:
        raise RuntimeError("Resume source contains no snapshot frames.")

    recorded_count = hf.attrs.get("frames_written")
    if recorded_count is not None:
        recorded_count = int(recorded_count)
        if recorded_count < 1 or recorded_count > frame_count:
            raise RuntimeError(
                f"Invalid frames_written={recorded_count} for {frame_count} allocated frames."
            )
        return recorded_count - 1

    last_index = frame_count - 1
    if _snapshot_has_structure(snapshots, last_index):
        return last_index
    if not _snapshot_has_structure(snapshots, 0):
        raise RuntimeError("The first snapshot is unwritten or contains no possible sites.")

    # Legacy interrupted files preallocated their full datasets. Their written
    # frames form one contiguous prefix, so a binary search avoids reading a
    # potentially enormous file frame by frame.
    low = 0
    high = last_index
    while low + 1 < high:
        middle = (low + high) // 2
        if _snapshot_has_structure(snapshots, middle):
            low = middle
        else:
            high = middle
    return low


def load_resume_state(source_path: Path) -> dict:
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Resume source does not exist: {source_path}")

    with h5py.File(source_path, "r") as hf:
        source_config = _load_config_json(hf)
        checkpoint = hf.get("checkpoint")
        exact = bool(
            checkpoint is not None
            and checkpoint.attrs.get("schema") == CHECKPOINT_SCHEMA
            and bool(checkpoint.attrs.get("complete", False))
            and hf.attrs.get("run_status") == "complete"
        )

        if exact:
            snapshot_index = int(checkpoint.attrs["snapshot_index"])
            if snapshot_index < 0 or snapshot_index >= len(hf["snapshots"]):
                raise RuntimeError("Checkpoint snapshot_index is outside /snapshots.")
            step = int(checkpoint.attrs["step"])
            if int(hf["saved_steps"][snapshot_index]) != step:
                raise RuntimeError("Checkpoint step does not match its saved snapshot.")
        else:
            snapshot_index = find_last_valid_snapshot_index(hf)
            step = int(hf["saved_steps"][snapshot_index])

        matrix = hf["snapshots"][snapshot_index].astype(np.int8)
        mode = (
            str(checkpoint.attrs["simulation_mode"])
            if exact
            else str((source_config or {}).get("simulation_mode_used")
                     or (source_config or {}).get("simulation_mode", ""))
        )

        legacy_step_adjusted = False
        if not exact and mode in ("molecular_wiggle", "forced_jump"):
            # Before endpoint checkpoints were introduced, these two loop-based
            # modes saved after executing the zero-based loop index but labelled
            # the frame with that index.
            step += 1
            legacy_step_adjusted = True

        rng_state = None
        ordered_hydrogen_site_ids = None
        random_seed_used = None
        forced_jump_rand_index = None
        event_pending_wait_steps = 0
        event_fenwick_tree = None
        event_total_transition_weight = None
        if exact:
            if "rng_state" in checkpoint:
                rng_state = checkpoint["rng_state"][:].astype(np.uint64)
            if "ordered_hydrogen_site_ids" in checkpoint:
                ordered_hydrogen_site_ids = checkpoint[
                    "ordered_hydrogen_site_ids"
                ][:].astype(np.int32)
            if "random_seed_used_uint64" in checkpoint.attrs:
                random_seed_used = int(checkpoint.attrs["random_seed_used_uint64"])
            if "forced_jump_rand_index" in checkpoint.attrs:
                forced_jump_rand_index = int(checkpoint.attrs["forced_jump_rand_index"])
            if mode == "event_driven_wiggle":
                if "event_pending_wait_steps" not in checkpoint.attrs:
                    raise RuntimeError(
                        "Exact event-driven checkpoint is missing its pending wait state."
                    )
                event_pending_wait_steps = int(
                    checkpoint.attrs["event_pending_wait_steps"]
                )
                if (
                    "event_fenwick_tree" not in checkpoint
                    or "event_total_transition_weight" not in checkpoint.attrs
                ):
                    raise RuntimeError(
                        "Exact event-driven checkpoint is missing its Fenwick state."
                    )
                event_fenwick_tree = checkpoint["event_fenwick_tree"][:].astype(
                    np.float64
                )
                event_total_transition_weight = float(
                    checkpoint.attrs["event_total_transition_weight"]
                )

        return {
            "source_path": source_path,
            "source_config": source_config,
            "matrix": matrix,
            "step": step,
            "snapshot_index": snapshot_index,
            "exact": exact,
            "mode": mode,
            "rng_state": rng_state,
            "ordered_hydrogen_site_ids": ordered_hydrogen_site_ids,
            "random_seed_used": random_seed_used,
            "forced_jump_rand_index": forced_jump_rand_index,
            "event_pending_wait_steps": event_pending_wait_steps,
            "event_fenwick_tree": event_fenwick_tree,
            "event_total_transition_weight": event_total_transition_weight,
            "legacy_step_adjusted": legacy_step_adjusted,
        }


def restore_compact_hydrogen_order(
    site_states: np.ndarray,
    hydrogen_site_ids: np.ndarray,
    hydrogen_transition_totals: np.ndarray,
    transition_totals: np.ndarray,
    saved_hydrogen_site_ids: np.ndarray,
) -> int:
    saved = np.asarray(saved_hydrogen_site_ids, dtype=np.int32)
    occupied = np.flatnonzero(site_states == 2).astype(np.int32)
    if len(saved) != len(occupied):
        raise RuntimeError(
            "Checkpoint hydrogen ordering length does not match the final matrix."
        )
    if len(saved):
        if np.min(saved) < 0 or np.max(saved) >= len(site_states):
            raise RuntimeError("Checkpoint hydrogen ordering contains an invalid site ID.")
        if len(np.unique(saved)) != len(saved) or not np.array_equal(
            np.sort(saved), occupied
        ):
            raise RuntimeError(
                "Checkpoint hydrogen ordering does not describe the occupied matrix sites."
            )
        hydrogen_site_ids[:len(saved)] = saved
        hydrogen_transition_totals[:len(saved)] = transition_totals[saved]
    return len(saved)


def restore_ordered_hydrogen_site_ids(
    site_states: np.ndarray,
    hydrogen_site_ids: np.ndarray,
    saved_hydrogen_site_ids: np.ndarray,
) -> None:
    saved = np.asarray(saved_hydrogen_site_ids, dtype=np.int32)
    occupied = np.flatnonzero(site_states == 2).astype(np.int32)
    if len(saved) != len(hydrogen_site_ids) or len(saved) != len(occupied):
        raise RuntimeError(
            "Checkpoint hydrogen ordering length does not match the final matrix."
        )
    if len(saved) and (
        np.min(saved) < 0
        or np.max(saved) >= len(site_states)
        or len(np.unique(saved)) != len(saved)
        or not np.array_equal(np.sort(saved), occupied)
    ):
        raise RuntimeError(
            "Checkpoint hydrogen ordering does not describe the occupied matrix sites."
        )
    hydrogen_site_ids[:] = saved


def write_final_checkpoint(
    hf: h5py.File,
    *,
    step: int,
    snapshot_index: int,
    simulation_mode: str,
    random_seed_used: int,
    rng_state: np.ndarray | None = None,
    ordered_hydrogen_site_ids: np.ndarray | None = None,
    forced_jump_rand_index: int | None = None,
    event_pending_wait_steps: int | None = None,
    event_fenwick_tree: np.ndarray | None = None,
    event_total_transition_weight: float | None = None,
) -> None:
    checkpoint = hf.create_group("checkpoint")
    checkpoint.attrs["schema"] = CHECKPOINT_SCHEMA
    checkpoint.attrs["step"] = np.int64(step)
    checkpoint.attrs["snapshot_index"] = np.int64(snapshot_index)
    checkpoint.attrs["simulation_mode"] = simulation_mode
    checkpoint.attrs["random_seed_used_uint64"] = np.uint64(random_seed_used)

    if rng_state is not None:
        checkpoint.create_dataset("rng_state", data=np.asarray(rng_state, dtype=np.uint64))
    if ordered_hydrogen_site_ids is not None:
        checkpoint.create_dataset(
            "ordered_hydrogen_site_ids",
            data=np.asarray(ordered_hydrogen_site_ids, dtype=np.int32),
        )
    if forced_jump_rand_index is not None:
        checkpoint.attrs["forced_jump_rand_index"] = np.int64(forced_jump_rand_index)
    if event_pending_wait_steps is not None:
        checkpoint.attrs["event_pending_wait_steps"] = np.int64(
            event_pending_wait_steps
        )
    if event_fenwick_tree is not None:
        checkpoint.create_dataset(
            "event_fenwick_tree",
            data=np.asarray(event_fenwick_tree, dtype=np.float64),
        )
    if event_total_transition_weight is not None:
        checkpoint.attrs["event_total_transition_weight"] = np.float64(
            event_total_transition_weight
        )

    # A reader only treats the checkpoint as exact after these markers have been
    # flushed. Interrupted files therefore fall back to statistical continuation.
    hf.flush()
    checkpoint.attrs["complete"] = True
    hf.attrs["run_status"] = "complete"
    hf.flush()
