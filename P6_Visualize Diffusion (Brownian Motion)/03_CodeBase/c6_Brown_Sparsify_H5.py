"""
Create a smaller Brownian-motion HDF5 file by keeping every Nth saved snapshot.

This is meant for very large simulation outputs. It does not rerun the
simulation; it copies only selected saved frames and matching time/speed arrays
into a new HDF5 file that the animation and diagram scripts can load faster.

Metadata is never rewritten here. If the input H5 has /meta, it is copied
unchanged. If the input H5 has no /meta, the sparse output also has no /meta.
"""

from pathlib import Path

import numpy as np
import h5py

from b3_Brown_Functions import read_saved_steps, results_dir

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------- Input / Output ---------------------- #
INPUT_H5_FILENAME = "random_motion.h5"
OUTPUT_H5_FILENAME = "random_motion_sparse.h5"
KEEP_EVERY_NTH_FRAME = 25  # Keep every Nth saved frame; 1000 turns 40000 saved frames into about 40 frames.
OVERWRITE_OUTPUT = False


# ---------------------- Copy Behavior ---------------------- #
SNAPSHOT_COPY_BATCH_SIZE = 10  # Number of selected snapshots to copy per batch; lower uses less RAM.


def resolve_results_file(filename):
    path = Path(filename)
    if path.is_absolute():
        return path
    return results_dir() / filename


def copy_attrs(source_obj, target_obj):
    for key, value in source_obj.attrs.items():
        target_obj.attrs[key] = value


def create_dataset_like(target_group, name, data, source_dataset):
    data_ndim = getattr(data, "ndim", 0)
    target_group.create_dataset(
        name,
        data=data,
        dtype=source_dataset.dtype,
        chunks=True if data_ndim > 0 else None,
    )


def copy_frame_aligned_dataset(source_dataset, target_group, name, frame_indices, frame_count):
    if source_dataset.shape and source_dataset.shape[0] == frame_count:
        data = source_dataset[frame_indices]
    else:
        data = source_dataset[()]

    create_dataset_like(target_group, name, data, source_dataset)
    copy_attrs(source_dataset, target_group[name])


def copy_group_datasets(source_group, target_group, frame_indices, frame_count):
    copy_attrs(source_group, target_group)

    for name, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            copy_frame_aligned_dataset(item, target_group, name, frame_indices, frame_count)
        elif isinstance(item, h5py.Group):
            nested_target = target_group.create_group(name)
            copy_group_datasets(item, nested_target, frame_indices, frame_count)


def copy_snapshots(source_snapshots, target_hf, frame_indices):
    output_shape = (len(frame_indices),) + source_snapshots.shape[1:]
    target_snapshots = target_hf.create_dataset(
        "snapshots",
        shape=output_shape,
        dtype=source_snapshots.dtype,
        chunks=True,
    )
    copy_attrs(source_snapshots, target_snapshots)

    iterator = range(0, len(frame_indices), SNAPSHOT_COPY_BATCH_SIZE)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Copying selected snapshots")

    for output_start in iterator:
        output_end = min(output_start + SNAPSHOT_COPY_BATCH_SIZE, len(frame_indices))
        batch_indices = frame_indices[output_start:output_end]
        target_snapshots[output_start:output_end] = source_snapshots[batch_indices]


def copy_metadata_unchanged(source_hf, target_hf):
    if "meta" in source_hf:
        source_hf.copy(source_hf["meta"], target_hf, name="meta")


def sparsify_h5(input_path, output_path, keep_every_nth_frame):
    keep_every_nth_frame = int(keep_every_nth_frame)
    if keep_every_nth_frame < 1:
        raise ValueError("KEEP_EVERY_NTH_FRAME must be 1 or greater")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output H5 must be different from input H5")
    if not input_path.exists():
        raise FileNotFoundError(f"Input H5 file does not exist: {input_path}")
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Expected output directory does not exist: {output_path.parent}")
    if output_path.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(f"Output H5 already exists. Set OVERWRITE_OUTPUT = True to replace it: {output_path}")
        output_path.unlink()

    with h5py.File(input_path, "r") as source_hf:
        if "snapshots" not in source_hf:
            raise RuntimeError(f"No 'snapshots' dataset found in {input_path}")

        source_snapshots = source_hf["snapshots"]
        frame_count = source_snapshots.shape[0]
        frame_indices = np.arange(0, frame_count, keep_every_nth_frame, dtype=np.int64)
        if len(frame_indices) == 0:
            raise RuntimeError("No frames selected. Use a smaller KEEP_EVERY_NTH_FRAME value.")

        saved_steps = read_saved_steps(source_hf)
        sparse_saved_steps = saved_steps[frame_indices]

        with h5py.File(output_path, "w") as target_hf:
            copy_attrs(source_hf, target_hf)
            copy_metadata_unchanged(source_hf, target_hf)
            copy_snapshots(source_snapshots, target_hf, frame_indices)
            target_hf.create_dataset("saved_steps", data=sparse_saved_steps, dtype=saved_steps.dtype)

            for name, item in source_hf.items():
                if name in ("snapshots", "saved_steps", "meta"):
                    continue

                if isinstance(item, h5py.Group):
                    target_group = target_hf.create_group(name)
                    copy_group_datasets(item, target_group, frame_indices, frame_count)
                elif isinstance(item, h5py.Dataset):
                    copy_frame_aligned_dataset(item, target_hf, name, frame_indices, frame_count)

    print(f"Input H5: {input_path}")
    print(f"Output H5: {output_path}")
    print(f"Kept {len(frame_indices)} of {frame_count} saved frames.")
    print(f"Saved steps: {int(sparse_saved_steps[0])} -> {int(sparse_saved_steps[-1])}")
    print("Done.")


def main():
    input_path = resolve_results_file(INPUT_H5_FILENAME)
    output_path = resolve_results_file(OUTPUT_H5_FILENAME)
    sparsify_h5(input_path, output_path, KEEP_EVERY_NTH_FRAME)


if __name__ == "__main__":
    main()
