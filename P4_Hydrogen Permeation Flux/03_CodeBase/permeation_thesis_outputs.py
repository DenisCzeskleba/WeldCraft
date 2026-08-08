"""Generate corrected thesis tables and trapping figure for the ideal lattice reference.

This is intentionally separate from the historical P4 response atlas.  The
atlas has a slow apparent/reference diffusivity for compatibility with its
existing results; the thesis comparison requested here uses an explicit ideal
lattice reference of 6e-3 mm^2/s and keeps the physical detrapping calibration
unchanged.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from permeation_cases import (
    MODULE_ROOT,
    _base_config,
    _binding_energy_to_half_time_ref,
    _trap_config,
    load_settings,
)
from permeation_diagrams import build_figure
from permeation_model import SimulationResult, simulate_case
from permeation_persistence import load_atlas_hdf5, save_atlas_hdf5
from permeation_cases import SurfaceHistory


OUTPUT_DIR = MODULE_ROOT / "02_Results"
LENGTH_MM = 0.5
LATTICE_DIFFUSIVITY_MM2_S = 6.0e-3
END_TIME_REF = 125.0
FIGURE_WINDOW_REF = 20.0
LEGACY_COMPARISON_WINDOW_REF = 1.25
N_NODES = 201
N_OUTPUT = 2001

SURFACE_RATIOS = (1.0, 0.9, 0.8, 0.6)
TRAP_STRENGTH_ENERGIES = (0.0, 20.0, 22.0, 25.0, 27.5, 30.0, 32.0, 35.0, 40.0, math.inf)
TRAP_DENSITIES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0, 2.0, 5.0, 10.0)
TABLE13_ENERGIES = (10.0, 20.0, 30.0, 32.0, 33.0, 35.0, 37.5, 40.0, 45.0, 50.0, 60.0, 100.0)
FIGURE_STRENGTH_ENERGIES = (20.0, 25.0, 27.0, 30.0, 32.0, 35.0, 37.0, 40.0, 50.0)
FIGURE_DENSITIES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


def _base(settings: Mapping[str, object]):
    return _base_config(
        settings,
        end_time_ref=END_TIME_REF,
        n_nodes=N_NODES,
        n_output=N_OUTPUT,
        reference_length_mm=LENGTH_MM,
        reference_diffusivity_mm2_s=LATTICE_DIFFUSIVITY_MM2_S,
    )


def _method_values(result: SimulationResult) -> Dict[str, float]:
    """Evaluate the three thesis methods from one normalized transient."""

    config = result.config
    time_s = result.time_seconds
    steady = config.steady_flux_common_reference
    normalized = result.outlet_flux_common / steady

    threshold = 0.0963
    indices = np.flatnonzero(normalized >= threshold)
    if not indices.size:
        raise RuntimeError(f"Breakthrough threshold was not reached for {config.label!r}.")
    index = int(indices[0])
    if index == 0:
        breakthrough_s = float(time_s[0])
    else:
        fraction = (threshold - normalized[index - 1]) / (
            normalized[index] - normalized[index - 1]
        )
        breakthrough_s = float(
            time_s[index - 1] + fraction * (time_s[index] - time_s[index - 1])
        )

    # The thesis uses the standardized flux-threshold construction described
    # in Eq. (4.27)--(4.30), not the cumulative-integral intercept stored in
    # P4's general-purpose ``time_lag`` metric.  The exact analytical level
    # for the present notation is J/J_max = 0.617.
    lag_indices = np.flatnonzero(normalized >= 0.617)
    if not lag_indices.size:
        raise RuntimeError(f"Time-lag threshold was not reached for {config.label!r}.")
    lag_index = int(lag_indices[0])
    if lag_index == 0:
        lag_s = float(time_s[0])
    else:
        fraction = (0.617 - normalized[lag_index - 1]) / (
            normalized[lag_index] - normalized[lag_index - 1]
        )
        lag_s = float(
            time_s[lag_index - 1] + fraction * (time_s[lag_index] - time_s[lag_index - 1])
        )
    if breakthrough_s <= 0.0 or lag_s <= 0.0:
        raise RuntimeError(f"Invalid characteristic time for {config.label!r}.")

    # Fit a short local cubic around the steepest point to avoid making the
    # inflection-point slope depend on one finite-difference interval.
    slope = np.gradient(normalized, time_s)
    inflection_index = int(np.argmax(slope))
    half_window = 4
    lower = max(0, inflection_index - half_window)
    upper = min(time_s.size, inflection_index + half_window + 1)
    local_time = time_s[lower:upper] - time_s[inflection_index]
    local_flux = normalized[lower:upper]
    polynomial = np.polyfit(local_time, local_flux, 3)
    inflection_slope = float(polynomial[-2])
    if inflection_slope <= 0.0:
        raise RuntimeError(f"Invalid inflection slope for {config.label!r}.")

    return {
        "D_b": LENGTH_MM**2 / (15.3 * breakthrough_s),
        "D_lag": LENGTH_MM**2 / (6.0 * lag_s),
        "D_IP": 0.168878 * LENGTH_MM**2 * inflection_slope,
    }


def _surface_results(settings: Mapping[str, object]) -> Dict[str, SimulationResult]:
    base = _base(settings)
    reference = simulate_case(base.with_changes(label="surface reference"))
    onset = 0.5 * reference.metrics["t50"]
    time_constant = reference.metrics["t50"]
    results: Dict[str, SimulationResult] = {}
    for ratio in SURFACE_RATIOS:
        history = SurfaceHistory(
            base_concentration=1.0,
            delta_concentration=ratio - 1.0,
            onset_time_ref=onset,
            time_constant_ref=time_constant,
            transition_mode="exponential",
        )
        results[f"surface:{ratio:g}"] = simulate_case(
            base.with_changes(
                label=f"Surface ratio {ratio:g}",
                surface=history,
            )
        )
    return results


def _trap_results(
    settings: Mapping[str, object],
) -> Tuple[
    Dict[str, SimulationResult],
    Dict[str, SimulationResult],
    Dict[str, SimulationResult],
]:
    base = _base(settings)
    strength_table = _strength_results(settings, TRAP_STRENGTH_ENERGIES, 0.5)
    strength_table["0"] = _strength_results(settings, (0.0,), 0.0)["0"]
    strength_figure = _strength_results(settings, FIGURE_STRENGTH_ENERGIES, 1.0)
    strength_figure["inf"] = _strength_results(settings, (math.inf,), 1.0)["inf"]
    density: Dict[str, SimulationResult] = {}

    for capacity in set(TRAP_DENSITIES) | set(FIGURE_DENSITIES):
        half_time = _binding_energy_to_half_time_ref(settings, base, 30.0)
        key = f"{capacity:g}"
        density[key] = simulate_case(
            base.with_changes(
                label=f"N_T/C_ref = {capacity:g}",
                traps=_trap_config(settings, capacity, half_time),
            )
        )
    return strength_table, strength_figure, density


def _strength_results(
    settings: Mapping[str, object],
    energies: Iterable[float],
    capacity: float,
) -> Dict[str, SimulationResult]:
    base = _base(settings)
    results: Dict[str, SimulationResult] = {}
    for energy in energies:
        half_time = _binding_energy_to_half_time_ref(settings, base, energy)
        key = "inf" if math.isinf(energy) else f"{energy:g}"
        actual_capacity = 0.0 if energy == 0.0 else capacity
        results[key] = simulate_case(
            base.with_changes(
                label=("infinite retention" if math.isinf(energy) else f"{energy:g} kJ/mol"),
                traps=_trap_config(settings, actual_capacity, half_time),
            )
        )
    return results


def _row(table: str, parameter: str, values: Mapping[str, float]) -> Dict[str, object]:
    return {
        "table": table,
        "parameter": parameter,
        "D_b_mm2_s": values["D_b"],
        "D_lag_mm2_s": values["D_lag"],
        "D_IP_mm2_s": values["D_IP"],
    }


def _table13_rows(settings: Mapping[str, object]) -> list[Dict[str, object]]:
    """Return physical detrapping half-times for Table 13."""

    base = _base(settings)
    rows: list[Dict[str, object]] = []
    for energy in TABLE13_ENERGIES:
        half_time_seconds = (
            _binding_energy_to_half_time_ref(settings, base, energy)
            * base.tau_ref_seconds
        )
        rows.append(
            {
                "E_B_kJ_mol": energy,
                "half_time_seconds": half_time_seconds,
                "half_time_minutes": half_time_seconds / 60.0,
                "half_time_hours": half_time_seconds / 3600.0,
                "half_time_days": half_time_seconds / 86400.0,
                "half_time_years": half_time_seconds / (365.25 * 86400.0),
            }
        )
    return rows


def _make_rows(
    surface: Mapping[str, SimulationResult],
    strength_table: Mapping[str, SimulationResult],
    density: Mapping[str, SimulationResult],
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for ratio in SURFACE_RATIOS:
        rows.append(_row("Table 12", f"surface ratio {ratio:g}", _method_values(surface[f"surface:{ratio:g}"])))
    for energy in TRAP_STRENGTH_ENERGIES:
        key = "inf" if math.isinf(energy) else f"{energy:g}"
        rows.append(_row("Table 15", "infinity" if math.isinf(energy) else f"{energy:g} kJ/mol", _method_values(strength_table[key])))
    for capacity in TRAP_DENSITIES:
        key = f"{capacity:g}"
        rows.append(_row("Table 16", f"N_T/C_ref = {capacity:g}", _method_values(density[key])))
    return rows


def _write_tables(rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    csv_path = OUTPUT_DIR / "hydrogen_permeation_flux_thesis_DL6e-3_tables.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rounded_csv_path = OUTPUT_DIR / "hydrogen_permeation_flux_thesis_DL6e-3_tables_rounded_1dp.csv"
    rounded_fields = [
        "table",
        "parameter",
        "D_b_x10^-3_mm2_s",
        "D_lag_x10^-3_mm2_s",
        "D_IP_x10^-3_mm2_s",
    ]
    with rounded_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rounded_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "table": row["table"],
                    "parameter": row["parameter"],
                    "D_b_x10^-3_mm2_s": f"{float(row['D_b_mm2_s']) * 1e3:.1f}",
                    "D_lag_x10^-3_mm2_s": f"{float(row['D_lag_mm2_s']) * 1e3:.1f}",
                    "D_IP_x10^-3_mm2_s": f"{float(row['D_IP_mm2_s']) * 1e3:.1f}",
                }
            )

    by_table: Dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_table.setdefault(str(row["table"]), []).append(row)
    markdown = [
        "# Corrected thesis trapping tables",
        "",
        "Calibration: `L = 0.5 mm`, `D_L = 6.0e-3 mm^2/s`, `T = 20 °C`, "
        "`p0 = 5.0e3 s^-1`, `E_D = 4.5 kJ/mol`, and `N_T/C_ref` as listed.",
        "The tabulated coefficients are apparent coefficients obtained from the idealized transients.",
        "Values are shown in `10^-3 mm^2/s`.",
        "",
    ]
    markdown = [line.replace("Â°C", "degC") for line in markdown]
    markdown = [line.replace("\u00c2\u00b0C", "degC") for line in markdown]
    markdown = [line.replace("\u00b0C", "degC") for line in markdown]
    for table_name in ("Table 12", "Table 15", "Table 16"):
        markdown.extend([
            f"## {table_name}",
            "",
            "| Case | D_b | D_lag | D_IP |",
            "|---|---:|---:|---:|",
        ])
        for row in by_table[table_name]:
            markdown.append(
                f"| {row['parameter']} | {float(row['D_b_mm2_s']) * 1e3:.4f} | "
                f"{float(row['D_lag_mm2_s']) * 1e3:.4f} | {float(row['D_IP_mm2_s']) * 1e3:.4f} |"
            )
        markdown.append("")
    table13 = _table13_rows(load_settings())
    markdown.extend([
        "## Table 13",
        "",
        "Equivalent detrapping half-times from the 20 °C Arrhenius calibration. "
        "These physical times do not change when `D_L` is changed; only their "
        "value in units of `tau_L` changes.",
        "",
        "| $E_B$ [kJ/mol] | Half-time $t_{0.5}$ |",
        "|---:|---:|",
    ])
    for row in table13:
        energy = float(row["E_B_kJ_mol"])
        seconds = float(row["half_time_seconds"])
        if seconds < 1.0:
            display = f"{seconds:.2f} s"
        elif seconds < 60.0:
            display = f"{seconds:.1f} s"
        elif seconds < 12.0 * 3600.0:
            minutes = seconds / 60.0
            display = f"{minutes:.0f} min" if minutes >= 100.0 else f"{minutes:.1f} min"
        elif seconds < 365.25 * 86400.0:
            display = f"{seconds / 86400.0:.1f} d" if seconds >= 7.0 * 86400.0 else f"{seconds / 3600.0:.1f} h"
        elif seconds < 100.0 * 365.25 * 86400.0:
            display = f"{seconds / (365.25 * 86400.0):.1f} y"
        else:
            display = f"{seconds / (365.25 * 86400.0):,.0f} y"
        markdown.append(f"| {energy:g} | {display} |")
    markdown.append("")
    markdown = [line.replace("\u00b0C", "degC") for line in markdown]
    (OUTPUT_DIR / "hydrogen_permeation_flux_thesis_DL6e-3_tables.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    table13_csv = OUTPUT_DIR / "hydrogen_permeation_flux_thesis_DL6e-3_table13.csv"
    with table13_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table13[0]))
        writer.writeheader()
        writer.writerows(table13)


def _write_legacy_rescaled_figure() -> None:
    """Render the old dimensionless curves with the corrected physical x-axis.

    This is deliberately a comparison figure, not a new trapping simulation:
    the old normalized response curves are retained and only their time axis
    is converted from the old D_app reference to the new D_L reference.
    """

    legacy_path = OUTPUT_DIR / "hydrogen_permeation_flux.h5"
    if not legacy_path.exists():
        return
    legacy, _ = load_atlas_hdf5(legacy_path)
    release_keys = [
        f"trap_release:{energy:g}" for energy in FIGURE_STRENGTH_ENERGIES
    ] + ["trap_release:inf"]
    capacity_keys = [f"trap_capacity:{capacity:g}" for capacity in FIGURE_DENSITIES]
    keys = [*release_keys, *capacity_keys]
    if any(key not in legacy for key in keys):
        return
    rescaled_results = {
        key: legacy[key]
        for key in keys
    }
    rescaled_results = {
        key: type(result)(
            config=result.config.with_changes(
                reference_diffusivity_mm2_s=LATTICE_DIFFUSIVITY_MM2_S
            ),
            x_ref=result.x_ref,
            time_ref=result.time_ref,
            mobile_concentration=result.mobile_concentration,
            trap_occupancy=result.trap_occupancy,
            outlet_flux_common=result.outlet_flux_common,
            inlet_concentration=result.inlet_concentration,
            total_hydrogen=result.total_hydrogen,
            initial_profile=result.initial_profile,
            prefill_age_time_ref=result.prefill_age_time_ref,
            internal_steps=result.internal_steps,
            metrics=result.metrics,
        )
        for key, result in rescaled_results.items()
    }
    figure = build_figure(
        rescaled_results,
        "trapping",
        normalization="common_reference",
        time_axis="minutes",
        comparison_window_ref=LEGACY_COMPARISON_WINDOW_REF,
        style={"formats": ["png", "svg"], "dpi": 300},
    )
    figure.text(
        0.5,
        -0.012,
        r"Legacy dimensionless curves; x-axis rescaled only; "
        r"$L=0.5$ mm; $D_L=6.0\times10^{-3}$ mm$^2$/s; "
        r"$\tau_L=0.694$ min; window $=1.25\tau_L=0.868$ min",
        ha="center",
        fontsize=7.0,
    )
    for suffix in ("png", "svg"):
        for stem in (
            "hydrogen_permeation_flux_trapping_DL6e-3",
            "hydrogen_permeation_flux_trapping_rescaled_DL6e-3_same_shape",
        ):
            destination = OUTPUT_DIR / f"{stem}.{suffix}"
            figure.savefig(destination, dpi=300, format=suffix, bbox_inches="tight")


def main() -> None:
    settings = load_settings()
    surface = _surface_results(settings)
    hdf5_path = OUTPUT_DIR / "hydrogen_permeation_flux_thesis_DL6e-3_trapping.h5"
    if hdf5_path.exists():
        stored, metadata = load_atlas_hdf5(hdf5_path)
        stored_diffusivity = float(metadata.get("reference_diffusivity_mm2_s", float("nan")))
        stored_capacity = float(metadata.get("figure_strength_capacity_ratio", float("nan")))
        stored_case_capacity = float(
            stored.get("trap_release:20", next(iter(stored.values()))).config.traps.capacity_ratio
        )
        if (
            math.isclose(stored_diffusivity, LATTICE_DIFFUSIVITY_MM2_S)
            and math.isclose(stored_capacity, 1.0)
            and math.isclose(stored_case_capacity, 1.0)
        ):
            strength_figure = {
                key.removeprefix("trap_release:"): value
                for key, value in stored.items()
                if key.startswith("trap_release:")
            }
            density = {
                key.removeprefix("trap_capacity:"): value
                for key, value in stored.items()
                if key.startswith("trap_capacity:")
            }
            missing_densities = [
                capacity for capacity in TRAP_DENSITIES if f"{capacity:g}" not in density
            ]
            if missing_densities:
                base = _base(settings)
                half_time = _binding_energy_to_half_time_ref(settings, base, 30.0)
                for capacity in missing_densities:
                    density[f"{capacity:g}"] = simulate_case(
                        base.with_changes(
                            label=f"N_T/C_ref = {capacity:g}",
                            traps=_trap_config(settings, capacity, half_time),
                        )
                    )
            strength_table = _strength_results(settings, TRAP_STRENGTH_ENERGIES, 0.5)
            strength_table["0"] = density["0"]
        else:
            strength_table, strength_figure, density = _trap_results(settings)
    else:
        strength_table, strength_figure, density = _trap_results(settings)
    if "0" not in strength_table and "0" in density:
        strength_table["0"] = density["0"]
    rows = _make_rows(surface, strength_table, density)
    _write_tables(rows)

    figure_results: Dict[str, SimulationResult] = {}
    for energy in FIGURE_STRENGTH_ENERGIES:
        key = f"{energy:g}"
        figure_results[f"trap_release:{key}"] = strength_figure[key]
    figure_results["trap_release:inf"] = strength_figure["inf"]
    for capacity in FIGURE_DENSITIES:
        key = f"{capacity:g}"
        figure_results[f"trap_capacity:{key}"] = density[key]

    metadata = {
        "format": "P4 thesis correction output",
        "reference_diffusivity_mm2_s": LATTICE_DIFFUSIVITY_MM2_S,
        "reference_length_mm": LENGTH_MM,
        "comparison_window_ref": END_TIME_REF,
        "figure_strength_capacity_ratio": 1.0,
        "description": "Corrected trapping figure and tables using ideal lattice diffusivity.",
    }
    save_atlas_hdf5(hdf5_path, figure_results, metadata)

    figure = build_figure(
        figure_results,
        "trapping",
        normalization="common_reference",
        time_axis="minutes",
        comparison_window_ref=FIGURE_WINDOW_REF,
        style={"formats": ["png", "svg"], "dpi": 300},
    )
    figure.text(
        0.5,
        -0.012,
        r"$L=0.5$ mm; $D_L=6.0\times10^{-3}$ mm$^2$/s; "
        r"$\tau_L=L^2/D_L=0.694$ min; $T=20\,^{\circ}$C",
        ha="center",
        fontsize=7.2,
    )
    for suffix in ("png", "svg"):
        destination = OUTPUT_DIR / f"hydrogen_permeation_flux_trapping_DL6e-3_physical.{suffix}"
        figure.savefig(destination, dpi=300, format=suffix, bbox_inches="tight")
    _write_legacy_rescaled_figure()
    print(f"Saved corrected tables and trapping figure to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
