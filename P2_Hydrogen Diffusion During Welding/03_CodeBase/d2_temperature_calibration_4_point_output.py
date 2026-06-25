"""
Export temperature-vs-time at 4 fixed points from simulation HDF5 to CSV.
"""

import csv
import h5py
import numpy as np
import io
import contextlib

# Keep imports quiet in case config prints are triggered elsewhere
with contextlib.redirect_stdout(io.StringIO()):
    from b4_functions import in_results, load_param_config_json


# ------------------------------- User settings -------------------------------

# Source HDF5 from simulation
INPUT_H5 = str(in_results("00_diffusion_array.h5"))

# Output CSV in 02_Results/06_Temperature
OUTPUT_CSV = str(in_results("06_Temperature", "temperature_4_points.csv", mkdir=True))

# 4 hardcoded points in mm: (x_mm, y_mm)
POINTS_MM = [
    ("P1", 55.0, 6.0),
    ("P2", 50.0, 6.0),
    ("P3", 35.0, 6.0),
    ("P4", 10.0, 6.0),
]


# ------------------------------- Helpers -------------------------------------

def mm_to_idx(x_mm, y_mm, dx, dy, nx, ny):
    x_idx = int(round(x_mm / dx))
    y_idx = int(round(y_mm / dy))
    if not (0 <= x_idx < nx and 0 <= y_idx < ny):
        raise ValueError(
            f"Point ({x_mm:.3f}, {y_mm:.3f}) mm -> ({x_idx}, {y_idx}) is outside array shape ({ny}, {nx})."
        )
    return x_idx, y_idx


def load_snapshots(file_name):
    u_arrays = []
    t_values = []

    with h5py.File(file_name, "r") as hf:
        for key in sorted(hf.keys()):
            if key.startswith("u_snapshot_"):
                u_arrays.append(hf[key][:])
            elif key.startswith("t_snapshot_"):
                t_values.append(float(hf[key][()]))

    if not u_arrays or not t_values:
        raise ValueError("No u_snapshot_* or t_snapshot_* datasets found.")

    if len(u_arrays) != len(t_values):
        raise ValueError("Mismatch between number of temperature and time snapshots.")

    return u_arrays, np.array(t_values, dtype=float)


# ------------------------------- Main ----------------------------------------

def main():
    param_cfg = load_param_config_json(INPUT_H5)
    dx = float(param_cfg["dx"])
    dy = float(param_cfg["dy"])

    u_arrays, times_s = load_snapshots(INPUT_H5)
    ny, nx = u_arrays[0].shape

    points_idx = []
    for name, x_mm, y_mm in POINTS_MM:
        x_idx, y_idx = mm_to_idx(x_mm, y_mm, dx, dy, nx, ny)
        points_idx.append((name, x_idx, y_idx))

    header = ["time_s"] + [name for name, _, _ in points_idx]

    rows = []
    for i, u in enumerate(u_arrays):
        row = [times_s[i]]
        for _, x_idx, y_idx in points_idx:
            row.append(float(u[y_idx, x_idx]))
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
