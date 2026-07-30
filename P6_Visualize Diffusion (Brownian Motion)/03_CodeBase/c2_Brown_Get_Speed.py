"""
Estimate and plot signed hydrogen transport through the Brownian lattice.

This replaces the old centre-of-mass "diffusion speed" calculation.  A centre
of mass is dominated by finite-particle jitter, and differentiating its MSD in
short moving windows does not measure directional transport.

Two data paths are supported:

1. New HDF5 files can contain ``/transport/net_x_displacement``.  This is the
   signed molecular x-displacement accumulated between saved frames.  Dividing
   it by elapsed steps and sampled x-width gives the spatially averaged net
   crossing flux directly.
   Publication-optimized files may keep a full independent transport timeline
   in ``/transport/saved_steps`` while retaining fewer full matrix snapshots.
2. Legacy files contain only occupancy snapshots.  Their steady through-flow
   is not observable from mass balance alone, so an effective diffusivity is
   fitted from the transient relaxation of coarse concentration profiles.
   Signed flux is then estimated with Fick's first law,
   ``J = -D_eff * site_density * dc/dx``.

Flux units are hydrogen particles per simulation step across a vertical
cross-section.  Positive flux points toward increasing matrix x.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dst, idst
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

from b3_Brown_Functions import in_results, load_brown_config_json, read_saved_steps, results_dir


# ---------------------- Input ---------------------- #
INPUT_H5_FILENAME = "Diss Case 1 Most Simple.h5"
FRAME_STRIDE = 1
SELECTED_FRAME = -1


# ---------------------- Estimator ---------------------- #
# Leave as None to prefer recorded transport and otherwise fit D_eff from the
# transient profiles.  Set a positive value to impose D_eff in cell^2/step.
EFFECTIVE_DIFFUSIVITY = None

# Binning is chosen automatically to put roughly this many available lattice
# sites into every x-bin.  This suppresses occupancy noise without erasing the
# macroscopic concentration gradient.
TARGET_AVAILABLE_SITES_PER_BIN = 1500
MIN_X_BINS = 24
MAX_X_BINS = 100

# Smoothing is applied only after D_eff has been fitted.  It stabilizes the
# displayed instantaneous flux; it does not improve the fit artificially.
SPATIAL_SMOOTH_SIGMA_BINS = 1.0
TEMPORAL_SMOOTH_SIGMA_FRAMES = 2.0

# Multi-lag profile prediction is substantially less noise-sensitive than
# differentiating adjacent frames.
FIT_LAGS_IN_FRAMES = (2, 5, 10, 20)
FIT_FRAME_FRACTION = 0.35
MIN_FIT_IMPROVEMENT = 0.10
DIFFUSIVITY_BOUNDS = (1e-8, 1e4)


# ---------------------- Output ---------------------- #
SHOW_PLOT = True
SAVE_PNG = False
SAVE_PDF = False
SAVE_SVG = False
OUTPUT_BASENAME = "brownian_net_flux"
SAVE_DPI = 300


def resolve_h5_path(filename=INPUT_H5_FILENAME):
    path = Path(filename)
    return path if path.is_absolute() else in_results(filename)


def normalize_frame_index(frame_index, frame_count):
    normalized = frame_count + frame_index if frame_index < 0 else frame_index
    if normalized < 0 or normalized >= frame_count:
        raise IndexError(
            f"Frame index {frame_index} is outside the available range "
            f"-{frame_count}..-1 or 0..{frame_count - 1}."
        )
    return normalized


def _metadata_int(metadata, *names, default=0):
    for name in names:
        if name in metadata:
            return int(metadata[name])
    return int(default)


def _choose_bin_edges(first_snapshot):
    rows, columns = first_snapshot.shape
    sites_per_column = np.sum(first_snapshot > 0, axis=0)
    mean_sites_per_column = max(float(np.mean(sites_per_column)), 1.0)
    preferred_width = max(
        1,
        int(round(TARGET_AVAILABLE_SITES_PER_BIN / mean_sites_per_column)),
    )
    preferred_bins = int(round(columns / preferred_width))
    bin_count = min(MAX_X_BINS, max(MIN_X_BINS, preferred_bins))
    bin_count = min(bin_count, columns)

    edges = np.rint(np.linspace(0, columns, bin_count + 1)).astype(int)
    edges[0] = 0
    edges[-1] = columns
    if np.any(np.diff(edges) <= 0):
        raise RuntimeError("Could not construct non-empty concentration bins.")
    return edges


def _sum_columns_by_bin(values, edges):
    return np.add.reduceat(values, edges[:-1])[: len(edges) - 1]


def load_binned_concentration(h5_path, frame_stride=1):
    frame_stride = int(frame_stride)
    if frame_stride < 1:
        raise ValueError("frame_stride must be 1 or greater")

    metadata = load_brown_config_json(h5_path, required=True)
    with h5py.File(h5_path, "r") as hf:
        if "snapshots" not in hf:
            raise RuntimeError(f"No 'snapshots' dataset found in {h5_path}")

        snapshots = hf["snapshots"]
        all_steps = np.asarray(read_saved_steps(hf), dtype=np.float64)
        if len(all_steps) != snapshots.shape[0]:
            raise RuntimeError(
                "The root saved_steps dataset must align with the matrix snapshots."
            )

        transport_group = hf.get("transport")
        required_transport_datasets = {
            "net_x_displacement",
            "interval_steps",
            "region_widths",
            "region_centers_x",
        }
        has_recorded_transport = (
            transport_group is not None
            and required_transport_datasets.issubset(transport_group.keys())
        )
        has_independent_transport_timeline = (
            has_recorded_transport and "saved_steps" in transport_group
        )

        frame_indices = np.arange(0, snapshots.shape[0], frame_stride, dtype=int)
        saved_steps = all_steps[frame_indices]
        minimum_snapshot_count = 1 if has_independent_transport_timeline else 2
        if len(saved_steps) < minimum_snapshot_count:
            raise RuntimeError(
                f"At least {minimum_snapshot_count} saved snapshot"
                f"{'s are' if minimum_snapshot_count != 1 else ' is'} "
                "required for transport analysis."
            )
        if np.any(np.diff(saved_steps) <= 0):
            raise RuntimeError("Saved simulation steps must be strictly increasing.")

        first_snapshot = snapshots[int(frame_indices[0])]
        edges = _choose_bin_edges(first_snapshot)
        available_by_column = np.sum(first_snapshot > 0, axis=0, dtype=np.int64)
        available_by_bin = _sum_columns_by_bin(available_by_column, edges).astype(np.float64)
        if np.any(available_by_bin <= 0):
            raise RuntimeError(
                "At least one x-bin has no available lattice sites; "
                "the one-dimensional flux estimator is not applicable."
            )

        hydrogen_by_bin = np.empty((len(frame_indices), len(edges) - 1), dtype=np.float64)
        for output_index, frame_index in enumerate(frame_indices):
            hydrogen_by_column = np.sum(
                snapshots[int(frame_index)] == 2,
                axis=0,
                dtype=np.int64,
            )
            hydrogen_by_bin[output_index] = _sum_columns_by_bin(
                hydrogen_by_column,
                edges,
            )

        recorded_transport = None
        if has_recorded_transport:
            all_displacement = transport_group["net_x_displacement"][:]
            all_interval_steps = transport_group["interval_steps"][:]

            if has_independent_transport_timeline:
                all_transport_steps = np.asarray(
                    transport_group["saved_steps"][:],
                    dtype=np.float64,
                )
                transport_frame_count = len(all_transport_steps)
                if (
                    len(all_displacement) != transport_frame_count
                    or len(all_interval_steps) != transport_frame_count
                ):
                    raise RuntimeError(
                        "Independent recorded-transport datasets have "
                        "incompatible timeline lengths."
                    )
                if np.any(np.diff(all_transport_steps) <= 0):
                    raise RuntimeError(
                        "Independent recorded-transport steps must be "
                        "strictly increasing."
                    )

                transport_indices = np.arange(
                    0,
                    transport_frame_count,
                    frame_stride,
                    dtype=int,
                )
                transport_time = all_transport_steps[transport_indices]
                if frame_stride == 1:
                    displacement = np.asarray(all_displacement, dtype=np.float64)
                    interval_steps = np.asarray(all_interval_steps, dtype=np.int64)
                else:
                    displacement = np.zeros(
                        (len(transport_indices), all_displacement.shape[1]),
                        dtype=np.float64,
                    )
                    interval_steps = np.zeros(len(transport_indices), dtype=np.int64)
                    previous_frame = -1
                    for output_index, transport_index in enumerate(transport_indices):
                        interval_slice = slice(
                            previous_frame + 1,
                            int(transport_index) + 1,
                        )
                        displacement[output_index] = np.sum(
                            all_displacement[interval_slice],
                            axis=0,
                        )
                        interval_steps[output_index] = np.sum(
                            all_interval_steps[interval_slice],
                        )
                        previous_frame = int(transport_index)
            else:
                transport_time = saved_steps
                displacement = np.zeros(
                    (len(frame_indices), all_displacement.shape[1]),
                    dtype=np.float64,
                )
                interval_steps = np.zeros(len(frame_indices), dtype=np.int64)
                previous_frame = -1
                for output_index, frame_index in enumerate(frame_indices):
                    interval_slice = slice(previous_frame + 1, int(frame_index) + 1)
                    displacement[output_index] = np.sum(
                        all_displacement[interval_slice],
                        axis=0,
                    )
                    interval_steps[output_index] = np.sum(
                        all_interval_steps[interval_slice],
                    )
                    previous_frame = int(frame_index)
            recorded_transport = {
                "time": transport_time,
                "net_x_displacement": displacement,
                "interval_steps": interval_steps,
                "region_widths": transport_group["region_widths"][:],
                "region_centers_x": transport_group["region_centers_x"][:],
            }

        if recorded_transport is None and len(saved_steps) < 3:
            raise RuntimeError(
                "At least three saved snapshots are required when recorded "
                "transport data is unavailable."
            )

    widths = np.diff(edges).astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    concentration = hydrogen_by_bin / available_by_bin

    return {
        "metadata": metadata,
        "frame_indices": frame_indices,
        "time": saved_steps,
        "x": centers,
        "bin_edges": edges,
        "bin_widths": widths,
        "available_sites": available_by_bin,
        "site_density": available_by_bin / widths,
        "concentration": concentration,
        "recorded_transport": recorded_transport,
    }


def _bulk_bin_slice(profile_data):
    metadata = profile_data["metadata"]
    edges = profile_data["bin_edges"]
    matrix_width = int(edges[-1])
    thickness = _metadata_int(
        metadata,
        "sink_source_thickness_used",
        "SINK_SOURCE_THICKNESS",
        default=0,
    )

    centers = profile_data["x"]
    bulk_mask = (centers >= thickness) & (centers < matrix_width - thickness)
    bulk_indices = np.flatnonzero(bulk_mask)
    if len(bulk_indices) < 8:
        bulk_indices = np.arange(len(centers))
    if len(bulk_indices) < 8:
        raise RuntimeError("At least eight x-bins are required to estimate net flux.")
    return slice(int(bulk_indices[0]), int(bulk_indices[-1]) + 1)


def _predict_diffusion(profile, left_value, right_value, decay):
    point_count = len(profile)
    baseline = np.linspace(left_value, right_value, point_count)
    interior_offset = profile[1:-1] - baseline[1:-1]
    coefficients = dst(interior_offset, type=1, norm="ortho")
    prediction = baseline.copy()
    prediction[1:-1] += idst(coefficients * decay, type=1, norm="ortho")
    return prediction


def _fit_one_lag(profiles, times, dx, lag, pair_stop):
    interior_count = profiles.shape[1] - 2
    mode_numbers = np.arange(1, interior_count + 1, dtype=np.float64)
    laplacian_eigenvalues = (
        4
        * np.sin(np.pi * mode_numbers / (2 * (interior_count + 1))) ** 2
        / dx**2
    )
    pair_indices = np.arange(0, pair_stop - lag, max(1, lag // 2), dtype=int)
    if len(pair_indices) < 2:
        return None

    def mean_squared_error(log_diffusivity):
        diffusivity = float(np.exp(log_diffusivity))
        squared_error = 0.0
        value_count = 0
        for start in pair_indices:
            stop = start + lag
            delta_time = times[stop] - times[start]
            decay = np.exp(-diffusivity * laplacian_eigenvalues * delta_time)
            left_value = 0.5 * (profiles[start, 0] + profiles[stop, 0])
            right_value = 0.5 * (profiles[start, -1] + profiles[stop, -1])
            prediction = _predict_diffusion(
                profiles[start],
                left_value,
                right_value,
                decay,
            )
            residual = prediction[1:-1] - profiles[stop, 1:-1]
            squared_error += float(np.dot(residual, residual))
            value_count += len(residual)
        return squared_error / max(value_count, 1)

    lower, upper = DIFFUSIVITY_BOUNDS
    optimum = minimize_scalar(
        mean_squared_error,
        bounds=(np.log(lower), np.log(upper)),
        method="bounded",
        options={"xatol": 1e-5},
    )
    fitted_diffusivity = float(np.exp(optimum.x))
    no_diffusion_error = mean_squared_error(np.log(lower))
    improvement = (
        1 - float(optimum.fun) / no_diffusion_error
        if no_diffusion_error > 0
        else 0.0
    )
    return fitted_diffusivity, improvement, float(optimum.fun), len(pair_indices)


def fit_effective_diffusivity(profile_data):
    bulk_slice = _bulk_bin_slice(profile_data)
    profiles = profile_data["concentration"][:, bulk_slice]
    times = profile_data["time"]
    dx = float(np.mean(profile_data["bin_widths"][bulk_slice]))

    frame_count = len(times)
    pair_stop = min(
        frame_count,
        max(8, int(np.ceil(frame_count * FIT_FRAME_FRACTION))),
    )
    valid_lags = sorted(
        {
            int(lag)
            for lag in FIT_LAGS_IN_FRAMES
            if int(lag) >= 1 and int(lag) < pair_stop - 1
        }
    )
    if not valid_lags:
        valid_lags = [max(1, pair_stop // 4)]

    fits = []
    for lag in valid_lags:
        result = _fit_one_lag(profiles, times, dx, lag, pair_stop)
        if result is not None:
            diffusivity, improvement, error, pair_count = result
            fits.append(
                {
                    "lag": lag,
                    "diffusivity": diffusivity,
                    "improvement": improvement,
                    "error": error,
                    "pair_count": pair_count,
                }
            )

    accepted = [fit for fit in fits if fit["improvement"] >= MIN_FIT_IMPROVEMENT]
    if not accepted:
        details = ", ".join(
            f"lag {fit['lag']}: {100 * fit['improvement']:.1f}%"
            for fit in fits
        )
        raise RuntimeError(
            "The snapshots do not contain a resolvable diffusion transient, so "
            "D_eff and steady flux cannot be inferred reliably. Set "
            "EFFECTIVE_DIFFUSIVITY manually or analyze a run with recorded "
            f"transport data. Fit improvements: {details or 'none'}."
        )

    estimates = np.asarray([fit["diffusivity"] for fit in accepted])
    diffusivity = float(np.median(estimates))
    median_absolute_deviation = float(np.median(np.abs(estimates - diffusivity)))
    relative_spread = (
        1.4826 * median_absolute_deviation / diffusivity
        if diffusivity > 0
        else np.inf
    )
    return {
        "effective_diffusivity": diffusivity,
        "relative_fit_spread": relative_spread,
        "fit_improvement": float(np.median([fit["improvement"] for fit in accepted])),
        "lag_fits": fits,
    }


def _smooth_time_series(values, sigma):
    values = np.asarray(values, dtype=np.float64)
    if sigma <= 0 or len(values) < 3:
        return values.copy()
    return gaussian_filter(values, sigma=float(sigma), mode="nearest")


def _analysis_from_recorded_transport(profile_data):
    recorded = profile_data["recorded_transport"]
    transport_time = np.asarray(recorded["time"], dtype=np.float64)
    displacement = np.asarray(recorded["net_x_displacement"], dtype=np.float64)
    interval_steps = np.asarray(recorded["interval_steps"], dtype=np.float64)
    region_widths = np.asarray(recorded["region_widths"], dtype=np.float64)

    if displacement.ndim != 2 or displacement.shape[1] != len(region_widths):
        raise RuntimeError("Recorded transport datasets have incompatible shapes.")
    if len(transport_time) != len(displacement) or len(interval_steps) != len(displacement):
        raise RuntimeError("Recorded transport timeline and interval arrays do not align.")

    denominator = interval_steps[:, None] * region_widths[None, :]
    region_flux = np.full_like(displacement, np.nan, dtype=np.float64)
    np.divide(displacement, denominator, out=region_flux, where=denominator > 0)

    total_width = float(np.sum(region_widths))
    total_displacement = np.sum(displacement, axis=1)
    net_flux = np.full(len(interval_steps), np.nan, dtype=np.float64)
    np.divide(
        total_displacement,
        interval_steps * total_width,
        out=net_flux,
        where=interval_steps > 0,
    )

    for index in range(region_flux.shape[1]):
        finite = np.isfinite(region_flux[:, index])
        if np.any(finite):
            region_flux[finite, index] = _smooth_time_series(
                region_flux[finite, index],
                TEMPORAL_SMOOTH_SIGMA_FRAMES,
            )
    finite = np.isfinite(net_flux)
    if np.any(finite):
        net_flux[finite] = _smooth_time_series(
            net_flux[finite],
            TEMPORAL_SMOOTH_SIGMA_FRAMES,
        )

    flux_low = np.full(len(region_flux), np.nan, dtype=np.float64)
    flux_high = np.full(len(region_flux), np.nan, dtype=np.float64)
    for frame_index, values in enumerate(region_flux):
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            flux_low[frame_index] = np.quantile(finite_values, 0.10)
            flux_high[frame_index] = np.quantile(finite_values, 0.90)

    return {
        "method": "recorded signed molecular crossings",
        "method_key": "recorded",
        "time": transport_time,
        "concentration_time": profile_data["time"],
        "x": np.asarray(recorded["region_centers_x"], dtype=np.float64),
        "concentration_x": profile_data["x"],
        "concentration": profile_data["concentration"],
        "flux_profile": region_flux,
        "net_flux": net_flux,
        "flux_low": flux_low,
        "flux_high": flux_high,
        "effective_diffusivity": np.nan,
        "relative_fit_spread": np.nan,
        "fit_improvement": np.nan,
        "lag_fits": [],
    }


def _analysis_from_profiles(profile_data, imposed_diffusivity=None):
    if imposed_diffusivity is not None:
        diffusivity = float(imposed_diffusivity)
        if not np.isfinite(diffusivity) or diffusivity <= 0:
            raise ValueError("EFFECTIVE_DIFFUSIVITY must be a positive finite value.")
        fit = {
            "effective_diffusivity": diffusivity,
            "relative_fit_spread": np.nan,
            "fit_improvement": np.nan,
            "lag_fits": [],
        }
        method = "Fick flux using imposed effective diffusivity"
        method_key = "fick_imposed"
    else:
        fit = fit_effective_diffusivity(profile_data)
        diffusivity = fit["effective_diffusivity"]
        method = "Fick flux from fitted concentration relaxation"
        method_key = "fick_fitted"

    concentration = gaussian_filter(
        profile_data["concentration"],
        sigma=(TEMPORAL_SMOOTH_SIGMA_FRAMES, SPATIAL_SMOOTH_SIGMA_BINS),
        mode="nearest",
    )
    concentration_gradient = np.gradient(
        concentration,
        profile_data["x"],
        axis=1,
    )
    site_density = gaussian_filter(
        profile_data["site_density"],
        sigma=SPATIAL_SMOOTH_SIGMA_BINS,
        mode="nearest",
    )
    flux_profile = -diffusivity * site_density[None, :] * concentration_gradient

    bulk_slice = _bulk_bin_slice(profile_data)
    bulk_flux = flux_profile[:, bulk_slice]
    net_flux = np.median(bulk_flux, axis=1)
    flux_low = np.quantile(bulk_flux, 0.10, axis=1)
    flux_high = np.quantile(bulk_flux, 0.90, axis=1)

    return {
        "method": method,
        "method_key": method_key,
        "time": profile_data["time"],
        "concentration_time": profile_data["time"],
        "x": profile_data["x"],
        "concentration_x": profile_data["x"],
        "concentration": concentration,
        "flux_profile": flux_profile,
        "net_flux": net_flux,
        "flux_low": flux_low,
        "flux_high": flux_high,
        **fit,
    }


def analyze_transport(
    h5_path,
    frame_stride=1,
    effective_diffusivity=EFFECTIVE_DIFFUSIVITY,
):
    h5_path = Path(h5_path)
    profile_data = load_binned_concentration(h5_path, frame_stride=frame_stride)
    if profile_data["recorded_transport"] is not None:
        analysis = _analysis_from_recorded_transport(profile_data)
    else:
        analysis = _analysis_from_profiles(
            profile_data,
            imposed_diffusivity=effective_diffusivity,
        )
    analysis["h5_path"] = h5_path
    analysis["metadata"] = profile_data["metadata"]
    analysis["frame_indices"] = profile_data["frame_indices"]
    return analysis


def print_analysis_summary(analysis):
    finite_flux = analysis["net_flux"][np.isfinite(analysis["net_flux"])]
    print(f"Loaded: {analysis['h5_path']}")
    print(f"Transport method: {analysis['method']}")
    if np.isfinite(analysis["effective_diffusivity"]):
        print(
            "Effective diffusivity: "
            f"{analysis['effective_diffusivity']:.6g} cell^2/step"
        )
        if np.isfinite(analysis["relative_fit_spread"]):
            print(
                "Multi-lag relative fit spread: "
                f"{100 * analysis['relative_fit_spread']:.1f}%"
            )
        if np.isfinite(analysis["fit_improvement"]):
            print(
                "Median improvement over a no-diffusion prediction: "
                f"{100 * analysis['fit_improvement']:.1f}%"
            )
    if len(finite_flux):
        tail_start = max(0, int(0.75 * len(finite_flux)))
        print(
            "Late-time median net flux: "
            f"{np.median(finite_flux[tail_start:]):.6g} "
            "H particles/step (+x is positive)"
        )


def create_transport_figure(analysis, selected_frame=SELECTED_FRAME):
    frame_index = normalize_frame_index(selected_frame, len(analysis["time"]))
    time = analysis["time"]
    concentration_time = analysis.get("concentration_time", time)

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.15), hspace=0.34, wspace=0.28)
    concentration_axis = fig.add_subplot(grid[0, 0])
    flux_profile_axis = fig.add_subplot(grid[0, 1])
    flux_time_axis = fig.add_subplot(grid[1, :])

    concentration_frame_count = len(analysis["concentration"])
    if concentration_frame_count == 0 or len(concentration_time) != concentration_frame_count:
        raise RuntimeError("Concentration profiles and their timeline do not align.")
    selected_concentration_index = int(
        np.searchsorted(
            concentration_time,
            time[frame_index],
            side="right",
        )
        - 1
    )
    selected_concentration_index = min(
        concentration_frame_count - 1,
        max(0, selected_concentration_index),
    )
    profile_indices = sorted(
        {
            0,
            max(0, selected_concentration_index // 4),
            max(0, selected_concentration_index // 2),
            selected_concentration_index,
        }
    )
    for index in profile_indices:
        concentration_axis.plot(
            analysis["concentration_x"],
            100 * analysis["concentration"][index],
            label=f"step {int(concentration_time[index]):,}",
        )
    concentration_axis.set_title("Noise-reduced concentration profiles")
    concentration_axis.set_xlabel("Matrix x")
    concentration_axis.set_ylabel("Occupied available sites (%)")
    concentration_axis.set_ylim(-2, 102)
    concentration_axis.grid(alpha=0.25)
    concentration_axis.legend(fontsize=8)

    flux_profile_axis.plot(
        analysis["x"],
        analysis["flux_profile"][frame_index],
        color="#7A1FA2",
        linewidth=1.8,
    )
    flux_profile_axis.axhline(0, color="#303030", linewidth=0.8)
    flux_profile_axis.set_title(f"Flux profile at step {int(time[frame_index]):,}")
    flux_profile_axis.set_xlabel("Matrix x")
    flux_profile_axis.set_ylabel("Net flux (H/step)")
    flux_profile_axis.grid(alpha=0.25)

    flux_time_axis.fill_between(
        time,
        analysis["flux_low"],
        analysis["flux_high"],
        color="#B39DDB",
        alpha=0.35,
        label="10th–90th spatial percentile",
    )
    flux_time_axis.plot(
        time,
        analysis["net_flux"],
        color="#4A148C",
        linewidth=1.8,
        label="Spatial median net flux",
    )
    flux_time_axis.axhline(0, color="#303030", linewidth=0.8)
    flux_time_axis.axvline(time[frame_index], color="#303030", linestyle="--", linewidth=0.9)
    flux_time_axis.set_title("Directional transport through the lattice")
    flux_time_axis.set_xlabel("Simulation step")
    flux_time_axis.set_ylabel("Net flux (H particles/step)")
    flux_time_axis.grid(alpha=0.25)
    flux_time_axis.legend()

    detail = analysis["method"]
    if np.isfinite(analysis["effective_diffusivity"]):
        detail += f"; D_eff = {analysis['effective_diffusivity']:.4g} cell²/step"
    fig.suptitle(f"Brownian net transport\n{detail}", fontsize=14)
    return fig


def save_outputs(fig, output_dir):
    if SAVE_PNG:
        path = output_dir / f"{OUTPUT_BASENAME}.png"
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        print(f"Saved PNG: {path}")
    if SAVE_PDF:
        path = output_dir / f"{OUTPUT_BASENAME}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved PDF: {path}")
    if SAVE_SVG:
        path = output_dir / f"{OUTPUT_BASENAME}.svg"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved SVG: {path}")


def main():
    h5_path = resolve_h5_path()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    analysis = analyze_transport(h5_path, frame_stride=FRAME_STRIDE)
    print_analysis_summary(analysis)
    fig = create_transport_figure(analysis)
    save_outputs(fig, results_dir())

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
