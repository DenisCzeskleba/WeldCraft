"""Black-and-white response-atlas diagrams for P4."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from permeation_model import SimulationCancelled, SimulationError, SimulationResult


LINE_STYLES: Sequence[Tuple[object, str]] = (
    ("-", ""),
    ("--", "o"),
    ("-.", "s"),
    ((0, (1, 1)), "^"),
    ((0, (5, 2, 1, 2)), "D"),
    ((0, (3, 1, 1, 1, 1, 1)), "v"),
)


# The solver stores every 1 kJ/mol case from 20 to 100. Edit this list to
# change only the curves displayed in the overview strength panel; the
# trap-free reference represents the effectively zero-energy response here.
TRAPPING_STRENGTH_DISPLAY_ENERGIES_KJ_MOL = [20.0, 25.0, 27.0, 30.0, 32.0, 35.0, 37.0, 40.0, 50.0]


THESIS_STYLE = {
    "font.family": "serif",
    "font.size": 9.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "0.78",
    "grid.linewidth": 0.5,
    "grid.linestyle": ":",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def apply_thesis_style() -> None:
    """Apply the shipped style globally for legacy direct renderer callers."""

    plt.rcParams.update(THESIS_STYLE)


def _apply_publication_style(figure, style: Mapping[str, object] | None) -> None:
    values = dict(style or {})
    figure_scale = float(values.get("figure_scale", 1.0))
    font_scale = float(values.get("font_scale", 1.0))
    line_scale = float(values.get("line_width_scale", 1.0))
    marker_scale = float(values.get("marker_scale", 1.0))
    grid_visible = bool(values.get("grid_visible", True))
    grid_style = str(values.get("grid_style", ":"))
    legend_mode = str(values.get("legend_mode", "original"))

    if figure_scale != 1.0:
        width, height = figure.get_size_inches()
        figure.set_size_inches(width * figure_scale, height * figure_scale, forward=True)
    for text in figure.findobj(match=lambda artist: hasattr(artist, "get_fontsize")):
        try:
            text.set_fontsize(float(text.get_fontsize()) * font_scale)
        except (TypeError, ValueError):
            pass
    for axis in figure.axes:
        axis.grid(grid_visible, linestyle=grid_style)
        for line in axis.lines:
            line.set_linewidth(line.get_linewidth() * line_scale)
            line.set_markersize(line.get_markersize() * marker_scale)
        legend = axis.get_legend()
        if legend is None:
            continue
        if legend_mode == "hidden":
            legend.remove()
        elif legend_mode == "best":
            legend.set_loc("best")
        elif legend_mode == "outside":
            legend.set_loc("upper left")
            legend.set_bbox_to_anchor((1.02, 1.0))

    title = getattr(figure, "_suptitle", None)
    if title is not None:
        title.set_visible(bool(values.get("show_title", True)))
        override = str(values.get("title_override", "")).strip()
        if override:
            title.set_text(override)


def _select(results: Mapping[str, SimulationResult], prefix: str) -> List[Tuple[str, SimulationResult]]:
    return [(key, value) for key, value in results.items() if key.startswith(prefix)]


def _time_values(result: SimulationResult, time_axis: str) -> np.ndarray:
    if time_axis == "reference":
        return result.time_ref
    if time_axis == "fo":
        return result.fourier_number
    if time_axis == "seconds":
        return result.time_seconds
    if time_axis == "minutes":
        return result.time_minutes
    raise ValueError(f"Unknown time axis: {time_axis}")


def _time_label(time_axis: str) -> str:
    return {
        "reference": r"Normalized time, $t^*=tD_{\mathrm{ref}}/L_{\mathrm{ref}}^2$",
        "fo": r"Fourier number, $Fo=Dt/L^2$",
        "seconds": "Time, t [s]",
        "minutes": "Time, t [min]",
    }[time_axis]


def _reference_time_on_axis(
    result: SimulationResult, time_ref: float, time_axis: str
) -> float:
    if time_axis == "reference":
        return float(time_ref)
    if time_axis == "seconds":
        return float(time_ref * result.config.tau_ref_seconds)
    if time_axis == "minutes":
        return float(time_ref * result.config.tau_ref_seconds / 60.0)
    raise ValueError("A common reference-time window is not defined on a per-curve Fo axis.")


def _flux_label(normalization: str) -> str:
    return {
        "common_reference": r"Outlet flux, $J/J_{\mathrm{ref}}$",
        "per_curve": r"Outlet flux, $J/J_{\mathrm{ss}}$",
        "physical": r"Outlet flux, $J$ [mol mm$^{-2}$ s$^{-1}$]",
    }[normalization]


def _plot_family(
    axis,
    cases: Sequence[Tuple[str, SimulationResult]],
    normalization: str,
    time_axis: str,
    label_getter,
    marker_stride: int = 38,
) -> None:
    for index, (_, result) in enumerate(cases):
        line_style, marker = LINE_STYLES[index % len(LINE_STYLES)]
        x = _time_values(result, time_axis)
        y = result.flux(normalization)
        axis.plot(
            x,
            y,
            color="black",
            linestyle=line_style,
            linewidth=1.25,
            marker=marker or None,
            markevery=max(1, marker_stride + index * 3),
            markersize=3.1,
            markerfacecolor="white",
            label=label_getter(result),
        )
    axis.grid(True)
    axis.set_xlabel(_time_label(time_axis))
    axis.set_ylabel(_flux_label(normalization))


RESPONSE_LEVELS = (0.30, 0.50, 0.75, 0.90)


def _crossing_x(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Return the first rising crossing of ``level`` by linear interpolation."""

    indices = np.flatnonzero(y >= level)
    if not len(indices):
        return float("nan")
    index = int(indices[0])
    if index == 0 or y[index] == y[index - 1]:
        return float(x[index])
    fraction = (level - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def _active_x_limit(
    cases: Sequence[Tuple[str, SimulationResult]],
    time_axis: str,
    final_fraction: float = 0.97,
) -> float:
    """Crop a family shortly after its slowest relevant rising transient."""

    crossings: List[float] = []
    maximum_available = 0.0
    for _, result in cases:
        x = _time_values(result, time_axis)
        final = result.config.steady_flux_common_reference
        maximum_available = max(maximum_available, float(x[-1]))
        if final > 0.0:
            crossing = _crossing_x(x, result.outlet_flux_common, final_fraction * final)
            if np.isfinite(crossing):
                crossings.append(crossing)
    if not crossings:
        return maximum_available
    return min(maximum_available, max(crossings) * 1.12)


def _add_response_guides(axis, x_limit: float, qualifier: str = "") -> None:
    for level in RESPONSE_LEVELS:
        axis.axhline(level, color="0.68", linewidth=0.65, linestyle=":", zorder=0)
        axis.text(
            x_limit * 0.995,
            level,
            f"{int(level * 100)}%{qualifier}",
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="0.28",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
        )


def _label_at_fraction(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    text: str,
    fraction: float,
    final_value: float,
    maximum_x: float | None = None,
    fallback_x: float | None = None,
) -> None:
    position = _crossing_x(x, y, fraction * final_value)
    if maximum_x is not None and (
        not np.isfinite(position) or position > maximum_x
    ):
        position = fallback_x if fallback_x is not None else maximum_x
    if not np.isfinite(position):
        return
    value = float(np.interp(position, x, y))
    axis.annotate(
        text,
        (position, value),
        xytext=(4, 1),
        textcoords="offset points",
        fontsize=7.0,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.45, "alpha": 0.88},
    )


def _plot_direct_response_family(
    axis,
    cases: Sequence[Tuple[str, SimulationResult]],
    time_axis: str,
    value_getter,
    normalized_per_curve: bool = True,
    comparison_window_ref: float | None = None,
    skip_label_indices: set[int] | None = None,
    solid_indices: set[int] | None = None,
) -> float:
    if comparison_window_ref is not None and time_axis != "fo":
        x_limit = _reference_time_on_axis(
            cases[0][1], comparison_window_ref, time_axis
        )
    else:
        x_limit = _active_x_limit(cases, time_axis)
    label_fractions = np.linspace(0.24, 0.80, max(1, len(cases)))
    for index, ((_, result), label_fraction) in enumerate(zip(cases, label_fractions)):
        x = _time_values(result, time_axis)
        final = result.config.steady_flux_common_reference
        y = result.outlet_flux_common / final if normalized_per_curve else result.outlet_flux_common
        line_style = "-" if solid_indices and index in solid_indices else LINE_STYLES[index % len(LINE_STYLES)][0]
        axis.plot(
            x,
            y,
            color="black",
            linestyle=line_style,
            linewidth=1.25,
        )
        displayed_final = 1.0 if normalized_per_curve else final
        if skip_label_indices is None or index not in skip_label_indices:
            _label_at_fraction(
                axis,
                x,
                y,
                value_getter(result),
                float(label_fraction),
                displayed_final,
                maximum_x=0.91 * x_limit,
                fallback_x=x_limit
                * (0.62 + 0.24 * index / max(1, len(cases) - 1)),
            )
    axis.set_xlim(0.0, x_limit)
    axis.grid(True, axis="x")
    return x_limit


def render_ideal(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    d_cases = sorted(
        _select(results, "ideal:D:"),
        key=lambda item: item[1].config.diffusivity_ratio,
    )
    l_cases = sorted(
        _select(results, "ideal:L:"),
        key=lambda item: item[1].config.length_ratio,
    )
    s_cases = sorted(
        _select(results, "ideal:S:"),
        key=lambda item: item[1].config.solubility_ratio,
    )
    reference = min(
        d_cases, key=lambda item: abs(item[1].config.diffusivity_ratio - 1.0)
    )[1]
    reference_length = reference.config.reference_length_mm
    reference_diffusivity = reference.config.reference_diffusivity_mm2_s
    tau_minutes = reference.config.tau_ref_seconds / 60.0

    for axis, cases, title, values in (
        (
            axes[0, 0],
            d_cases,
            r"Diffusivity ratio $D/D_{\mathrm{ref}}$",
            lambda r: f"{r.config.diffusivity_ratio:g}",
        ),
        (
            axes[0, 1],
            l_cases,
            r"Thickness ratio $L/L_{\mathrm{ref}}$",
            lambda r: f"{r.config.length_ratio:g}",
        ),
    ):
        x_limit = _plot_direct_response_family(
            axis,
            cases,
            time_axis,
            values,
            comparison_window_ref=comparison_window_ref,
        )
        _add_response_guides(axis, x_limit)
        axis.set_ylim(-0.03, 1.06)
        axis.set_xlabel(_time_label(time_axis))
        axis.set_ylabel(r"Normalized response, $J/J_{\mathrm{ss}}$")
        axis.set_title(title + " (values on curves)")

    solubility_axis = axes[1, 0]
    s_limit = _plot_direct_response_family(
        solubility_axis,
        s_cases,
        time_axis,
        lambda r: f"{r.config.solubility_ratio:g}",
        normalized_per_curve=False,
    )
    _add_response_guides(solubility_axis, s_limit, r" of $J_{\mathrm{ref}}$")
    maximum_solubility = max(
        result.config.solubility_ratio for _, result in s_cases
    )
    solubility_axis.set_ylim(-0.03, maximum_solubility * 1.05)
    solubility_axis.set_xlabel(_time_label(time_axis))
    solubility_axis.set_ylabel(r"Common-reference flux, $J/J_{\mathrm{ref}}$")
    solubility_axis.set_title(
        r"Solubility ratio $S/S_{\mathrm{ref}}$ (values on curves)"
    )
    solubility_axis.text(
        0.03,
        0.95,
        "At fixed activity: flux scale changes;\nrise shape does not.",
        transform=solubility_axis.transAxes,
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
    )

    profile_axis = axes[1, 1]
    position = reference.x_ref / reference.config.length_ratio
    normalized_flux = reference.flux("per_curve")
    profile_levels = (0.10, 0.30, 0.50, 0.75, 0.90)
    label_positions = (0.22, 0.34, 0.46, 0.58, 0.70)
    for index, (level, label_position) in enumerate(
        zip(profile_levels, label_positions)
    ):
        crossing_index = int(np.flatnonzero(normalized_flux >= level)[0])
        if crossing_index == 0:
            profile = reference.mobile_concentration[0]
        else:
            lower_flux = normalized_flux[crossing_index - 1]
            upper_flux = normalized_flux[crossing_index]
            fraction = (level - lower_flux) / (upper_flux - lower_flux)
            profile = (
                (1.0 - fraction)
                * reference.mobile_concentration[crossing_index - 1]
                + fraction * reference.mobile_concentration[crossing_index]
            )
        line_style = LINE_STYLES[index % len(LINE_STYLES)][0]
        profile_axis.plot(
            position,
            profile,
            color="black",
            linestyle=line_style,
            linewidth=1.25,
        )
        label_y = float(np.interp(label_position, position, profile))
        profile_axis.text(
            label_position,
            label_y,
            f"{int(level * 100)}%",
            ha="left",
            va="bottom",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.88},
        )
    steady_profile = 1.0 - position
    profile_axis.plot(
        position,
        steady_profile,
        color="0.45",
        linestyle=(0, (2, 2)),
        linewidth=1.0,
    )
    profile_axis.text(
        0.76,
        0.25,
        "steady",
        color="0.35",
        fontsize=7.0,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
    )
    profile_axis.set_xlim(0.0, 1.0)
    profile_axis.set_ylim(-0.03, 1.04)
    profile_axis.set_xlabel(r"Position through membrane, $x/L$")
    profile_axis.set_ylabel(r"Mobile concentration, $C/C_{\mathrm{in}}$")
    profile_axis.set_title(r"Profiles when outlet flux reaches $J/J_{\mathrm{ss}}$")
    profile_axis.grid(True)

    fig.suptitle("Ideal influences on the rising permeation transient", fontsize=12)
    fig.text(
        0.5,
        -0.012,
        rf"Reference specimen: $L_{{\mathrm{{ref}}}}={reference_length:g}$ mm, "
        rf"$D_{{\mathrm{{ref}}}}={reference_diffusivity / 1.0e-5:g}\times10^{{-5}}$ mm$^2$/s, "
        rf"$\tau_{{\mathrm{{ref}}}}=L_{{\mathrm{{ref}}}}^2/D_{{\mathrm{{ref}}}}={tau_minutes:.1f}$ min.",
        ha="center",
        fontsize=7.3,
    )
    return fig


def render_surface(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "surface:"),
        key=lambda item: item[1].config.surface.final_concentration,
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.6, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.0, 2.0]},
        constrained_layout=True,
    )
    top, absolute_axis, normalized_axis = axes
    for index, (_, result) in enumerate(cases):
        line_style = LINE_STYLES[index % len(LINE_STYLES)][0]
        x = _time_values(result, time_axis)
        ratio = result.config.surface.final_concentration
        top.plot(
            x,
            result.inlet_concentration,
            color="black",
            linestyle=line_style,
            linewidth=1.2,
        )
        absolute_axis.plot(
            x,
            result.outlet_flux_common,
            color="black",
            linestyle=line_style,
            linewidth=1.25,
        )
        peak_flux = float(np.max(result.outlet_flux_common))
        normalized_axis.plot(
            x,
            result.outlet_flux_common / peak_flux,
            color="black",
            linestyle=line_style,
            linewidth=1.25,
        )
    if comparison_window_ref is not None and time_axis != "fo":
        x_limit = _reference_time_on_axis(
            cases[0][1], comparison_window_ref, time_axis
        )
    else:
        x_limit = _active_x_limit(cases, time_axis)
    reference_result = cases[0][1]
    onset_ref = reference_result.config.surface.onset_time_ref
    onset = (
        onset_ref
        if time_axis == "fo"
        else _reference_time_on_axis(reference_result, onset_ref, time_axis)
    )
    tau_surface_minutes = (
        reference_result.config.surface.time_constant_ref
        * reference_result.config.tau_ref_seconds
        / 60.0
    )
    onset_minutes = onset_ref * reference_result.config.tau_ref_seconds / 60.0
    for index, (_, result) in enumerate(cases):
        ratio = result.config.surface.final_concentration
        top.text(
            x_limit * 0.99,
            ratio,
            f"{ratio:g}",
            ha="right",
            va="bottom",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
        )
        x = _time_values(result, time_axis)
        label_x = x_limit * 0.88
        label_y = float(np.interp(label_x, x, result.outlet_flux_common))
        absolute_axis.text(
            label_x,
            label_y,
            f"{ratio:g}",
            ha="left",
            va="bottom",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
        peak_flux = float(np.max(result.outlet_flux_common))
        normalized_flux = result.outlet_flux_common / peak_flux
        normalized_label_x = x_limit * (
            0.53 + 0.39 * index / max(1, len(cases) - 1)
        )
        normalized_label_y = float(
            np.interp(normalized_label_x, x, normalized_flux)
        )
        normalized_axis.text(
            normalized_label_x,
            normalized_label_y,
            f"{ratio:g}",
            ha="left",
            va="bottom",
            fontsize=6.8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
    for axis in axes:
        axis.set_xlim(0.0, x_limit)
        axis.axvline(onset, color="0.35", linewidth=0.8, linestyle="--")
    top.annotate(
        f"change begins at {onset_minutes:.1f} min",
        xy=(onset, 1.01),
        xytext=(12, -32),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7.0,
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.7},
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )
    top.set_ylabel(r"Effective $C_{\mathrm{in}}/C_{\mathrm{ref}}$")
    top.set_ylim(0.25, 1.15)
    top.set_title(
        r"Progressive entry-side subsurface concentration; asymptotic "
        r"$C_{\mathrm{in},\infty}/C_{\mathrm{in},0}$ (values on lines)"
    )
    top.grid(True, axis="x")
    absolute_axis.set_ylabel(r"Outlet flux, $J/J_{\mathrm{ref}}$")
    absolute_axis.set_ylim(-0.03, 1.15)
    absolute_axis.set_title("Absolute effect at the outlet")
    absolute_axis.grid(True, axis="x")
    _add_response_guides(absolute_axis, x_limit)
    absolute_axis.text(
        0.98,
        0.06,
        "Lower level: progressive entry attenuation, e.g. oxidation / passivation\n"
        "Higher level: progressive entry enhancement, e.g. activation or improved current stabilization",
        transform=absolute_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.9},
    )
    normalized_axis.axhline(1.0, color="0.55", linewidth=0.7, linestyle=":")
    normalized_axis.set_xlabel(_time_label(time_axis))
    normalized_axis.set_ylabel(r"Individually normalized, $J/\max_t(J)$")
    normalized_axis.set_ylim(-0.03, 1.06)
    normalized_axis.set_title("Would individual-maximum normalization hide the degradation?")
    normalized_axis.grid(True, axis="x")
    normalized_axis.text(
        0.02,
        0.06,
        "A boundary that was lower but constant from t=0 would collapse here.\n"
        "Separation, overshoot, or decline indicates change during the run.",
        transform=normalized_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.9},
    )
    fig.suptitle("Progressive entry-condition change during breakthrough", fontsize=12)
    fig.text(
        0.5,
        -0.012,
        rf"Bounded exponential change with $\tau_s={tau_surface_minutes:.1f}$ min; the boundary is an effective "
        "subsurface concentration, not a resolved electrochemical interface.",
        ha="center",
        fontsize=7.2,
    )
    return fig


def _draw_well(
    axis,
    depth: float,
    width: float,
    label: str,
    x_offset: float = 0.0,
    linestyle: object = "-",
) -> None:
    x = np.linspace(-1.0, 1.0, 160)
    y = -depth * np.exp(-((x - x_offset) / width) ** 2)
    axis.plot(x, y, color="black", linewidth=1.1, linestyle=linestyle, label=label)


def _render_trapping_legacy(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "trap:"),
        key=lambda item: (
            item[1].config.traps.capacity_ratio > 0.0,
            item[1].config.traps.capacity_ratio,
            item[1].config.traps.release_half_time_ref,
        ),
    )
    fig = plt.figure(figsize=(8.0, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1.0])
    flux_axis = fig.add_subplot(grid[:, :2])
    release_axis = fig.add_subplot(grid[0, 2])
    capacity_axis = fig.add_subplot(grid[1, 2])
    _plot_family(flux_axis, cases, normalization, time_axis, lambda r: r.config.label, marker_stride=44)
    flux_axis.set_title("Downstream permeation response")
    flux_axis.legend(fontsize=7.4, loc="lower right")
    flux_axis.annotate(
        "slower release / deeper conceptual well",
        xy=(0.93, 0.23),
        xytext=(0.50, 0.23),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=7.4,
    )

    _draw_well(release_axis, 0.35, 0.42, "shallow / fast release", linestyle="--")
    _draw_well(release_axis, 0.85, 0.42, "deep / slow release")
    release_axis.axhline(0.0, color="black", linewidth=0.6)
    release_axis.set_title("Retention symbol", fontsize=9)
    release_axis.set_xticks([])
    release_axis.set_yticks([])
    release_axis.legend(fontsize=6.5, loc="lower center")

    x = np.linspace(-1.0, 1.0, 260)
    capacity_axis.plot(
        x,
        -0.62 * np.exp(-((x + 0.65) / 0.17) ** 2),
        color="black",
        linewidth=1.0,
    )
    for offset in (0.10, 0.45, 0.80):
        capacity_axis.plot(
            x,
            -0.62 * np.exp(-((x - offset) / 0.09) ** 2),
            color="black",
            linewidth=1.0,
        )
    capacity_axis.axhline(0.0, color="black", linewidth=0.6)
    capacity_axis.annotate(
        "more available\ntrap storage",
        xy=(0.87, 0.78),
        xytext=(0.12, 0.78),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=7.2,
        va="center",
    )
    capacity_axis.set_title("Capacity symbol", fontsize=9)
    capacity_axis.set_xticks([])
    capacity_axis.set_yticks([])
    capacity_axis.set_ylim(-0.72, 0.06)
    capacity_axis.text(
        0.5,
        0.015,
        "symbols only—not literal geometry",
        transform=capacity_axis.transAxes,
        ha="center",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )
    fig.suptitle("McNabb–Foster kinetic trapping trends", fontsize=12)
    return fig


def _draw_trapping_explanation(axis) -> None:
    """Draw the compact lattice/trap schematic above the response plates."""

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    axis.text(0.03, 0.78, "Perfect lattice / no traps", fontsize=7.0, va="center")
    axis.annotate(
        "",
        xy=(0.43, 0.52),
        xytext=(0.18, 0.52),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "black"},
    )
    axis.scatter([0.21, 0.30, 0.39], [0.52, 0.52, 0.52], s=11, color="black")
    axis.text(0.21, 0.34, "mobile H", fontsize=6.7, ha="center")
    axis.text(0.43, 0.34, "all mobile H remains mobile", fontsize=6.7, ha="center")

    axis.text(0.56, 0.78, "Traps added", fontsize=7.0, va="center")
    axis.annotate(
        "",
        xy=(0.94, 0.52),
        xytext=(0.64, 0.52),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "black"},
    )
    positions = (0.67, 0.79, 0.91)
    axis.scatter(positions, [0.52] * len(positions), s=11, color="black")
    for position in positions:
        axis.plot(
            [position - 0.035, position, position + 0.035],
            [0.19, 0.13, 0.19],
            color="black",
            linewidth=0.9,
        )
    axis.annotate(
        "",
        xy=(0.67, 0.22),
        xytext=(0.67, 0.49),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "black"},
    )
    axis.annotate(
        "",
        xy=(0.91, 0.49),
        xytext=(0.91, 0.22),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "black"},
    )
    axis.text(0.67, 0.02, "capture", fontsize=6.7, ha="center")
    axis.text(0.79, 0.22, r"$\tau_{0.5}$", fontsize=6.7, ha="center")
    axis.text(0.91, 0.02, "release", fontsize=6.7, ha="center")


def _plot_early_trapping_zoom(axis, cases, time_axis: str) -> None:
    """Plot the early response where capture and release first separate curves."""

    for index, (_, result) in enumerate(cases):
        line_style = LINE_STYLES[index % len(LINE_STYLES)][0]
        x = _time_values(result, time_axis)
        final = result.config.steady_flux_common_reference
        axis.plot(
            x,
            result.outlet_flux_common / final,
            color="black",
            linestyle=line_style,
            linewidth=1.15,
        )
    axis.set_xlim(0.0, 15.0)
    axis.set_ylim(0.0, 0.20)
    axis.set_xlabel(_time_label(time_axis))
    axis.set_ylabel(r"Normalized response, $J/J_{\mathrm{ss}}$")
    axis.set_title("Early-time zoom: 0–15 min and 0–20% of steady response", fontsize=8.8)
    axis.grid(True, linestyle=":")
    axis.axhline(0.10, color="0.65", linewidth=0.6, linestyle=":")
    axis.text(
        0.99,
        0.93,
        "Same line meanings as above; solid = no traps",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9},
    )


def render_trapping(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    capacity_cases = sorted(
        _select(results, "trap_capacity:"),
        key=lambda item: item[1].config.traps.capacity_ratio,
    )
    release_by_key = dict(_select(results, "trap_release:"))
    display_keys = [
        f"trap_release:{energy:g}"
        for energy in TRAPPING_STRENGTH_DISPLAY_ENERGIES_KJ_MOL
    ]
    display_keys.append("trap_release:inf")
    release_cases = [
        (key, release_by_key[key]) for key in display_keys if key in release_by_key
    ]
    trap_free = capacity_cases[0]
    strength_cases = [trap_free, *release_cases]
    strength_capacity = (
        release_cases[0][1].config.traps.capacity_ratio if release_cases else float("nan")
    )
    strength_value = lambda result: (
        "no traps"
        if result.config.traps.capacity_ratio == 0.0
        else r"$\infty$"
        if np.isinf(result.config.traps.release_half_time_ref)
        else result.config.label
    )
    fig = plt.figure(figsize=(8.8, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.62, 2.55])
    explanation_axis = fig.add_subplot(grid[0, :])
    strength_axis = fig.add_subplot(grid[1, 0])
    capacity_axis = fig.add_subplot(grid[1, 1])
    _draw_trapping_explanation(explanation_axis)

    thirty_case = next(
        (item for item in release_cases if item[0] == "trap_release:30"),
        None,
    )
    if thirty_case is not None:
        zero_result = trap_free[1]
        thirty_result = thirty_case[1]
        zero_x = _time_values(zero_result, time_axis)
        zero_y = zero_result.outlet_flux_common / zero_result.config.steady_flux_common_reference
        thirty_x = _time_values(thirty_result, time_axis)
        thirty_y = thirty_result.outlet_flux_common / thirty_result.config.steady_flux_common_reference
        if np.allclose(zero_x, thirty_x):
            thirty_y_on_zero_grid = thirty_y
        else:
            thirty_y_on_zero_grid = np.interp(zero_x, thirty_x, thirty_y)
        strength_axis.fill_between(
            zero_x,
            zero_y,
            thirty_y_on_zero_grid,
            color="0.88",
            alpha=0.75,
            zorder=0,
        )

    panels = (
        (
            strength_axis,
            strength_cases,
            strength_value,
            r"Trap strength $E_B$ [kJ/mol]",
            None,
        ),
        (
            capacity_axis,
            capacity_cases,
            lambda result: f"{result.config.traps.capacity_ratio:g}",
            r"Trap density $N_T/C_{\mathrm{ref}}$",
            r"$E_b = 30\ \mathrm{kJ/mol}$",
        ),
    )
    for panel_index, (axis, cases, values, title, fixed_note) in enumerate(panels):
        x_limit = _plot_direct_response_family(
            axis,
            cases,
            time_axis,
            values,
            comparison_window_ref=comparison_window_ref,
            skip_label_indices={0} if panel_index == 0 else None,
            solid_indices={len(cases) - 1} if panel_index == 0 else None,
        )
        if panel_index == 0:
            trap_free_result = strength_cases[0][1]
            trap_free_x = _time_values(trap_free_result, time_axis)
            trap_free_y = trap_free_result.outlet_flux_common / trap_free_result.config.steady_flux_common_reference
            no_traps_x = _crossing_x(trap_free_x, trap_free_y, 0.90)
            if np.isfinite(no_traps_x):
                axis.annotate(
                    "no traps",
                    xy=(no_traps_x, 0.90),
                    xytext=(no_traps_x + 2.0, 0.96),
                    textcoords="data",
                    arrowprops={"arrowstyle": "-", "color": "0.25", "linewidth": 0.6},
                    fontsize=7.0,
                    ha="left",
                    va="bottom",
                )
            axis.text(
                0.10,
                0.50,
                "reversible",
                transform=axis.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.0,
                color="0.25",
            )
            axis.text(
                0.03,
                0.95,
                rf"$N_T/C_{{\mathrm{{ref}}}} = {strength_capacity:g}$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
            )
        axis.set_ylim(-0.03, 1.06)
        axis.set_xlabel(_time_label(time_axis))
        axis.set_ylabel(r"Normalized response, $J/J_{\mathrm{ss}}$")
        axis.set_title(title, fontsize=9.2)
        if fixed_note is not None:
            axis.text(
                0.03,
                0.96,
                fixed_note,
                transform=axis.transAxes,
                va="top",
                fontsize=6.8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9},
            )
    return fig


def render_prefill(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "prefill:"),
        key=lambda item: item[1].config.prefill.enabled,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), constrained_layout=True)
    profile_axis, flux_axis = axes
    for index, (_, result) in enumerate(cases):
        line_style, marker = LINE_STYLES[index % len(LINE_STYLES)]
        profile_axis.plot(
            result.x_ref / result.config.length_ratio,
            result.initial_profile,
            color="black",
            linestyle=line_style,
            linewidth=1.35,
            label=result.config.label,
        )
        flux_axis.plot(
            _time_values(result, time_axis),
            result.flux(normalization),
            color="black",
            linestyle=line_style,
            marker=marker or None,
            markevery=45,
            markerfacecolor="white",
            markersize=3.3,
            label=result.config.label,
        )
    aged = next((result for key, result in cases if key.endswith("aged")), None)
    profile_axis.set_xlabel(r"Position, $x/L$")
    profile_axis.set_ylabel(r"Initial $C/C_{\mathrm{ref}}$")
    profile_axis.set_title("Profile after free effusion")
    profile_axis.grid(True)
    profile_axis.legend(fontsize=7.5)
    if aged is not None:
        profile_axis.text(
            0.5,
            0.05,
            f"20% uniform prefill aged to a 10% centre peak\n"
            f"age = {aged.prefill_age_time_ref:.3g} τref",
            transform=profile_axis.transAxes,
            ha="center",
            fontsize=7.0,
        )
    flux_axis.set_xlabel(_time_label(time_axis))
    flux_axis.set_ylabel(_flux_label(normalization))
    flux_axis.set_title("Subsequent permeation response")
    flux_axis.grid(True)
    flux_axis.legend(fontsize=7.5)
    fig.suptitle("Residual hydrogen after specimen preparation", fontsize=12)
    return fig


def render_annex_model(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    reference = next(iter(results.values()))
    config = reference.config
    tau_minutes = config.tau_ref_seconds / 60.0
    fig, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    axis.set_axis_off()
    membrane = Rectangle(
        (0.16, 0.48),
        0.42,
        0.30,
        transform=axis.transAxes,
        facecolor="0.94",
        edgecolor="black",
        linewidth=1.2,
        hatch="///",
    )
    axis.add_patch(membrane)
    axis.text(0.37, 0.64, "one-dimensional membrane", transform=axis.transAxes, ha="center", fontsize=10)
    axis.text(0.16, 0.42, r"$x=0$", transform=axis.transAxes, ha="center")
    axis.text(0.58, 0.42, r"$x=L$", transform=axis.transAxes, ha="center")
    axis.text(0.08, 0.70, r"effective entry condition", transform=axis.transAxes, ha="center", fontsize=8)
    axis.text(0.08, 0.61, r"$C_L(0,t)=C_{\mathrm{in}}(t)$", transform=axis.transAxes, ha="center")
    axis.text(0.68, 0.70, "perfect sink", transform=axis.transAxes, ha="center", fontsize=8)
    axis.text(0.68, 0.61, r"$C_L(L,t)=0$", transform=axis.transAxes, ha="center")
    axis.add_patch(
        FancyArrowPatch(
            (0.02, 0.63),
            (0.15, 0.63),
            transform=axis.transAxes,
            arrowstyle="->",
            mutation_scale=12,
            color="black",
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.59, 0.63),
            (0.76, 0.63),
            transform=axis.transAxes,
            arrowstyle="->",
            mutation_scale=12,
            color="black",
        )
    )
    axis.text(0.75, 0.66, r"$J_{\mathrm{out}}$", transform=axis.transAxes, ha="left")
    axis.text(
        0.18,
        0.28,
        r"$\frac{\partial C_L}{\partial t}"
        r"=D_L\frac{\partial^2 C_L}{\partial x^2}-N_T\frac{\partial\theta}{\partial t}$",
        transform=axis.transAxes,
        ha="center",
        fontsize=11,
    )
    axis.text(
        0.50,
        0.28,
        r"$\frac{\partial\theta}{\partial t}"
        r"=k_t C_L(1-\theta)-k_d\theta$",
        transform=axis.transAxes,
        ha="center",
        fontsize=11,
    )
    axis.text(
        0.82,
        0.28,
        r"$J_{\mathrm{out}}=-D_L\left.\frac{\partial C_L}{\partial x}\right|_{x=L}$",
        transform=axis.transAxes,
        ha="center",
        fontsize=11,
    )
    axis.text(
        0.50,
        0.08,
        rf"Reference system: $L_{{\mathrm{{ref}}}}={config.reference_length_mm:g}$ mm, "
        rf"$D_{{\mathrm{{ref}}}}={config.reference_diffusivity_mm2_s / 1.0e-5:g}\times10^{{-5}}$ mm$^2$/s, "
        rf"$\tau_{{\mathrm{{ref}}}}=L_{{\mathrm{{ref}}}}^2/D_{{\mathrm{{ref}}}}={tau_minutes:.1f}$ min.",
        transform=axis.transAxes,
        ha="center",
        fontsize=8.2,
    )
    fig.suptitle("Numerical membrane model and reported outlet quantity", fontsize=12)
    return fig


def render_annex_trap_capacity(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "trap_capacity:"),
        key=lambda item: item[1].config.traps.capacity_ratio,
    )
    fig, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    x_limit = _plot_direct_response_family(
        axis,
        cases,
        time_axis,
        lambda result: f"{result.config.traps.capacity_ratio:g}",
        comparison_window_ref=comparison_window_ref,
    )
    _add_response_guides(axis, x_limit)
    half_time_minutes = cases[0][1].config.traps.release_half_time_ref * cases[0][1].config.tau_ref_seconds / 60.0
    axis.set_ylim(-0.03, 1.06)
    axis.set_xlabel(_time_label(time_axis))
    axis.set_ylabel(r"Normalized outlet flux, $J/J_{\mathrm{ss}}$")
    axis.set_title(r"Trap-capacity ratio $N_T/C_{\mathrm{ref}}$ (values on curves)")
    axis.text(
        0.03,
        0.96,
        rf"McNabb-Foster kinetics; fixed detrapping half-time = {half_time_minutes:.1f} min",
        transform=axis.transAxes,
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
    )
    fig.suptitle("Effect of reversible trap-storage capacity", fontsize=12)
    return fig


def render_annex_trap_release(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "trap_release:"),
        key=lambda item: item[1].config.traps.release_half_time_ref,
    )
    fig, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    x_limit = _plot_direct_response_family(
        axis,
        cases,
        time_axis,
        lambda result: f"{result.config.traps.release_half_time_ref * result.config.tau_ref_seconds / 60.0:.1f}",
        comparison_window_ref=comparison_window_ref,
    )
    _add_response_guides(axis, x_limit)
    capacity = cases[0][1].config.traps.capacity_ratio
    axis.set_ylim(-0.03, 1.06)
    axis.set_xlabel(_time_label(time_axis))
    axis.set_ylabel(r"Normalized outlet flux, $J/J_{\mathrm{ss}}$")
    axis.set_title(r"Detrapping half-time $t_{1/2,\mathrm{det}}$ [min] (values on curves)")
    axis.text(
        0.03,
        0.96,
        rf"McNabb-Foster kinetics; fixed trap capacity $N_T/C_{{\mathrm{{ref}}}}={capacity:g}$",
        transform=axis.transAxes,
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
    )
    fig.suptitle("Effect of the reversible-trap release rate", fontsize=12)
    return fig


def _interpolate_field_at_flux(
    result: SimulationResult, field: np.ndarray, level: float
) -> Tuple[np.ndarray, float]:
    response = result.flux("per_curve")
    indices = np.flatnonzero(response >= level)
    if not len(indices):
        raise ValueError(f"Flux level {level:g} is unavailable for {result.config.label}.")
    index = int(indices[0])
    if index == 0:
        return field[0].copy(), float(result.time_minutes[0])
    lower = response[index - 1]
    upper = response[index]
    fraction = (level - lower) / (upper - lower)
    profile = (1.0 - fraction) * field[index - 1] + fraction * field[index]
    time_minutes = (1.0 - fraction) * result.time_minutes[index - 1] + fraction * result.time_minutes[index]
    return profile, float(time_minutes)


def render_annex_trap_profiles(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    candidates = _select(results, "trap_capacity:")
    result = min(
        candidates,
        key=lambda item: abs(item[1].config.traps.capacity_ratio - 1.0),
    )[1]
    position = result.x_ref / result.config.length_ratio
    levels = (0.10, 0.50, 0.90)
    fig, (mobile_axis, trapped_axis) = plt.subplots(
        1, 2, figsize=(8.0, 4.2), constrained_layout=True
    )
    for index, level in enumerate(levels):
        mobile, time_minutes = _interpolate_field_at_flux(
            result, result.mobile_concentration, level
        )
        occupancy, _ = _interpolate_field_at_flux(
            result, result.trap_occupancy, level
        )
        trapped = result.config.traps.capacity_ratio * occupancy
        style = LINE_STYLES[index][0]
        label = f"{int(level * 100)}% ({time_minutes:.1f} min)"
        mobile_axis.plot(position, mobile, color="black", linestyle=style, linewidth=1.3)
        trapped_axis.plot(position, trapped, color="black", linestyle=style, linewidth=1.3)
        mobile_x = 0.30 + 0.13 * index
        trapped_x = 0.34 + 0.15 * index
        mobile_axis.text(
            mobile_x,
            float(np.interp(mobile_x, position, mobile)),
            label,
            fontsize=7.0,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
        trapped_axis.text(
            trapped_x,
            float(np.interp(trapped_x, position, trapped)),
            label,
            fontsize=7.0,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
    for axis in (mobile_axis, trapped_axis):
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(r"Position through membrane, $x/L$")
        axis.grid(True)
    mobile_axis.set_ylabel(r"Mobile concentration, $C_L/C_{\mathrm{ref}}$")
    trapped_axis.set_ylabel(r"Trapped concentration, $C_T/C_{\mathrm{ref}}=N_T\theta/C_{\mathrm{ref}}$")
    mobile_axis.set_title("Mobile lattice hydrogen")
    trapped_axis.set_title("Occupied reversible traps")
    fig.suptitle(
        r"Internal profiles for $N_T/C_{\mathrm{ref}}=1$ and $t_{1/2,\mathrm{det}}=34.7$ min",
        fontsize=12,
    )
    return fig


def _prefill_cases(
    results: Mapping[str, SimulationResult]
) -> List[Tuple[str, SimulationResult]]:
    return sorted(
        _select(results, "prefill:"),
        key=lambda item: (
            item[1].config.prefill.enabled,
            item[1].config.prefill.target_center_fraction,
        ),
    )


def render_annex_prefill_aging(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = [item for item in _prefill_cases(results) if item[1].config.prefill.enabled]
    fig, (profile_axis, age_axis) = plt.subplots(
        1, 2, figsize=(8.0, 4.2), gridspec_kw={"width_ratios": [1.55, 1.0]}, constrained_layout=True
    )
    position = cases[0][1].x_ref / cases[0][1].config.length_ratio
    initial_fraction = cases[0][1].config.prefill.initial_fraction
    profile_axis.plot(
        position,
        np.full_like(position, initial_fraction),
        color="0.45",
        linestyle=(0, (2, 2)),
        linewidth=1.0,
    )
    profile_axis.text(0.70, initial_fraction, "initial uniform 20%", color="0.35", fontsize=7.0, va="bottom")
    targets = []
    ages = []
    for index, (_, result) in enumerate(cases):
        target = result.config.prefill.target_center_fraction
        age_minutes = result.prefill_age_time_ref * result.config.tau_ref_seconds / 60.0
        targets.append(target * 100.0)
        ages.append(age_minutes)
        profile_axis.plot(
            position,
            result.initial_profile,
            color="black",
            linestyle=LINE_STYLES[index % len(LINE_STYLES)][0],
            linewidth=1.3,
        )
        profile_axis.text(
            0.50,
            target,
            f"{target * 100:g}% centre; {age_minutes:.1f} min",
            fontsize=7.0,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
    profile_axis.set_xlim(0.0, 1.0)
    profile_axis.set_ylim(-0.005, initial_fraction * 1.10)
    profile_axis.set_xlabel(r"Position through membrane, $x/L$")
    profile_axis.set_ylabel(r"Residual mobile concentration, $C/C_{\mathrm{ref}}$")
    profile_axis.set_title("Profiles retained after free effusion")
    profile_axis.grid(True)
    age_axis.plot(targets, ages, color="black", marker="o", markerfacecolor="white", linewidth=1.2)
    for target, age in zip(targets, ages):
        age_axis.text(target, age, f" {age:.1f}", fontsize=7.0, va="bottom")
    age_axis.set_xlabel(r"Retained centre concentration [% of $C_{\mathrm{ref}}$]")
    age_axis.set_ylabel("Required free-effusion time [min]")
    age_axis.set_title("Preparation history")
    age_axis.grid(True)
    fig.suptitle("Diffusion-generated residual-hydrogen profiles before charging", fontsize=12)
    return fig


def render_annex_prefill_response(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = _prefill_cases(results)
    fig, (raw_axis, corrected_axis) = plt.subplots(1, 2, figsize=(8.2, 4.3), constrained_layout=True)
    reference = cases[0][1]
    window_ref = 0.65
    x_limit = _reference_time_on_axis(reference, window_ref, time_axis) if time_axis != "fo" else _active_x_limit(cases, time_axis)
    corrected_minimum = 0.0
    for index, (_, result) in enumerate(cases):
        x = _time_values(result, time_axis)
        flux = result.outlet_flux_common
        initial = float(flux[0])
        corrected = (flux - initial) / (result.config.steady_flux_common_reference - initial)
        corrected_minimum = min(corrected_minimum, float(np.min(corrected)))
        style = LINE_STYLES[index % len(LINE_STYLES)][0]
        raw_axis.plot(x, flux, color="black", linestyle=style, linewidth=1.25)
        corrected_axis.plot(x, corrected, color="black", linestyle=style, linewidth=1.25)
        label = "empty" if not result.config.prefill.enabled else f"{result.config.prefill.target_center_fraction * 100:g}%"
        label_x = x_limit * (0.13 + 0.055 * index)
        raw_axis.text(
            label_x,
            float(np.interp(label_x, x, flux)),
            label,
            fontsize=7.0,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
        corrected_axis.text(
            label_x,
            float(np.interp(label_x, x, corrected)),
            label,
            fontsize=7.0,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.88},
        )
    for axis in (raw_axis, corrected_axis):
        axis.set_xlim(0.0, x_limit)
        axis.set_xlabel(_time_label(time_axis))
        axis.grid(True)
    raw_axis.set_ylim(-0.03, 1.06)
    raw_axis.set_ylabel(r"Measured outlet flux, $J/J_{\mathrm{ref}}$")
    raw_axis.set_title("Raw signal, including residual effusion")
    corrected_axis.axhline(1.0, color="0.55", linestyle=":", linewidth=0.7)
    corrected_axis.set_ylim(min(-0.08, corrected_minimum * 1.08), 1.06)
    corrected_axis.set_ylabel(r"Baseline-corrected response, $(J-J_0)/(J_{\mathrm{ss}}-J_0)$")
    corrected_axis.set_title("Same signals after initial-baseline correction")
    fig.suptitle("Effect of residual hydrogen on the subsequent permeation transient", fontsize=12)
    fig.text(
        0.5,
        -0.012,
        "Curve values denote the retained centre concentration before charging.",
        ha="center",
        fontsize=7.3,
    )
    return fig


def _render_annex_trap_flux_pair(
    cases: Sequence[Tuple[str, SimulationResult]],
    time_axis: str,
    title: str,
    parameter_title: str,
    value_getter,
    fixed_condition: str,
    comparison_window_ref: float | None,
):
    """Render only the observable flux and its individual normalization."""

    fig, (common_axis, normalized_axis) = plt.subplots(
        1, 2, figsize=(8.2, 4.3), constrained_layout=True
    )
    if comparison_window_ref is not None and time_axis != "fo":
        x_limit = _reference_time_on_axis(
            cases[0][1], comparison_window_ref, time_axis
        )
    else:
        x_limit = _active_x_limit(cases, time_axis)
    label_fractions = np.linspace(0.24, 0.80, len(cases))
    for index, ((_, result), label_fraction) in enumerate(
        zip(cases, label_fractions)
    ):
        x = _time_values(result, time_axis)
        common_flux = result.outlet_flux_common
        individual_flux = common_flux / float(np.max(common_flux))
        style = LINE_STYLES[index % len(LINE_STYLES)][0]
        for axis, values in (
            (common_axis, common_flux),
            (normalized_axis, individual_flux),
        ):
            axis.plot(x, values, color="black", linestyle=style, linewidth=1.25)
            _label_at_fraction(
                axis,
                x,
                values,
                value_getter(result),
                float(label_fraction),
                1.0,
                maximum_x=0.90 * x_limit,
                fallback_x=x_limit
                * (0.60 + 0.27 * index / max(1, len(cases) - 1)),
            )
    for axis in (common_axis, normalized_axis):
        axis.set_xlim(0.0, x_limit)
        axis.set_ylim(-0.03, 1.06)
        axis.set_xlabel(_time_label(time_axis))
        axis.grid(True, axis="x")
        _add_response_guides(axis, x_limit)
    common_axis.set_ylabel(r"Outlet flux, $J/J_{\mathrm{ref}}$")
    common_axis.set_title("Common-reference flux")
    normalized_axis.set_ylabel(r"Individually normalized flux, $J/\max_t(J)$")
    normalized_axis.set_title("Same curves normalized individually")
    fig.suptitle(title, fontsize=12)
    fig.supxlabel(
        f"{parameter_title}; {fixed_condition}",
        fontsize=7.3,
    )
    return fig


def render_annex_trap_capacity_flux(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "trap_capacity:"),
        key=lambda item: item[1].config.traps.capacity_ratio,
    )
    half_time = (
        cases[0][1].config.traps.release_half_time_ref
        * cases[0][1].config.tau_ref_seconds
        / 60.0
    )
    return _render_annex_trap_flux_pair(
        cases,
        time_axis,
        "Permeation-flux response to reversible trap capacity",
        r"Values on curves: $N_T/C_{\mathrm{ref}}$",
        lambda result: f"{result.config.traps.capacity_ratio:g}",
        f"fixed detrapping half-time = {half_time:.1f} min",
        comparison_window_ref,
    )


def render_annex_trap_release_flux(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "trap_release:"),
        key=lambda item: item[1].config.traps.release_half_time_ref,
    )
    capacity = cases[0][1].config.traps.capacity_ratio
    return _render_annex_trap_flux_pair(
        cases,
        time_axis,
        "Permeation-flux response to reversible-trap release rate",
        r"Values on curves: detrapping half-time $t_{1/2,\mathrm{det}}$ [min]",
        lambda result: (
            f"{result.config.traps.release_half_time_ref * result.config.tau_ref_seconds / 60.0:.1f}"
        ),
        rf"fixed trap capacity $N_T/C_{{\mathrm{{ref}}}}={capacity:g}$",
        comparison_window_ref,
    )


def render_annex_trap_capture_flux(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(
        _select(results, "trap_capture:"),
        key=lambda item: item[1].config.traps.capture_rate_ref,
    )
    capacity = cases[0][1].config.traps.capacity_ratio
    half_time = (
        cases[0][1].config.traps.release_half_time_ref
        * cases[0][1].config.tau_ref_seconds
        / 60.0
    )
    return _render_annex_trap_flux_pair(
        cases,
        time_axis,
        "Permeation-flux response to the trap-capture rate",
        r"Values on curves: $k_tC_{\mathrm{ref}}\tau_{\mathrm{ref}}$",
        lambda result: f"{result.config.traps.capture_rate_ref:g}",
        rf"fixed $N_T/C_{{\mathrm{{ref}}}}={capacity:g}$ and detrapping half-time = {half_time:.1f} min",
        comparison_window_ref,
    )


def render_annex_trap_combined_flux(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = sorted(_select(results, "trap_combined:"), key=lambda item: item[0])
    return _render_annex_trap_flux_pair(
        cases,
        time_axis,
        "Combined influence of trap capacity and release rate on permeation flux",
        "Curve labels describe the selected capacity / release-rate combination",
        lambda result: result.config.label,
        r"few: $N_T/C_{\mathrm{ref}}=0.5$; many: 2.0; fast: $t_{1/2}=6.9$ min; slow: 138.9 min",
        comparison_window_ref,
    )


def _annex_prefill_window(
    cases: Sequence[Tuple[str, SimulationResult]],
    time_axis: str,
    comparison_window_ref: float | None,
) -> float:
    window_ref = 0.55
    if comparison_window_ref is not None:
        window_ref = min(window_ref, comparison_window_ref)
    if time_axis == "fo":
        return _active_x_limit(cases, time_axis)
    return _reference_time_on_axis(cases[0][1], window_ref, time_axis)


def _prefill_curve_label(result: SimulationResult) -> str:
    if not result.config.prefill.enabled:
        return "empty"
    return f"{100.0 * result.config.prefill.target_center_fraction:g}%"


def render_annex_prefill_flux(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = _prefill_cases(results)
    fig, (early_axis, full_axis) = plt.subplots(
        1, 2, figsize=(8.2, 4.3), constrained_layout=True
    )
    x_limit = _annex_prefill_window(cases, time_axis, comparison_window_ref)
    early_limit = 0.48 * x_limit
    label_fractions = np.linspace(0.12, 0.58, len(cases))
    for index, ((_, result), label_fraction) in enumerate(
        zip(cases, label_fractions)
    ):
        x = _time_values(result, time_axis)
        flux = result.outlet_flux_common
        style = LINE_STYLES[index % len(LINE_STYLES)][0]
        early_axis.plot(x, flux, color="black", linestyle=style, linewidth=1.25)
        full_axis.plot(x, flux, color="black", linestyle=style, linewidth=1.25)
        _label_at_fraction(
            full_axis,
            x,
            flux,
            _prefill_curve_label(result),
            float(label_fraction),
            1.0,
            maximum_x=0.88 * x_limit,
            fallback_x=x_limit * (0.58 + 0.30 * index / max(1, len(cases) - 1)),
        )
    for axis, limit in ((early_axis, early_limit), (full_axis, x_limit)):
        axis.set_xlim(0.0, limit)
        axis.set_ylim(-0.03, 1.06)
        axis.set_xlabel(_time_label(time_axis))
        axis.grid(True)
    early_axis.set_ylabel(r"Outlet flux, $J/J_{\mathrm{ref}}$")
    early_axis.set_title("Early-time detail")
    full_axis.set_ylabel(r"Outlet flux, $J/J_{\mathrm{ref}}$")
    full_axis.set_title("Complete rising transient")
    fig.suptitle("Measured flux with residual hydrogen present before charging", fontsize=12)
    fig.supxlabel(
        r"Values on curves: retained centre concentration $C(x=L/2,t=0)/C_{\mathrm{ref}}$; no baseline subtraction",
        fontsize=7.3,
    )
    return fig


def render_annex_prefill_normalized(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    cases = _prefill_cases(results)
    fig, (maximum_axis, corrected_axis) = plt.subplots(
        1, 2, figsize=(8.2, 4.3), constrained_layout=True
    )
    x_limit = _annex_prefill_window(cases, time_axis, comparison_window_ref)
    corrected_minimum = 0.0
    label_fractions = np.linspace(0.18, 0.72, len(cases))
    for index, ((_, result), label_fraction) in enumerate(
        zip(cases, label_fractions)
    ):
        x = _time_values(result, time_axis)
        flux = result.outlet_flux_common
        maximum_normalized = flux / float(np.max(flux))
        initial = float(flux[0])
        corrected = (flux - initial) / (
            result.config.steady_flux_common_reference - initial
        )
        corrected_minimum = min(corrected_minimum, float(np.min(corrected)))
        style = LINE_STYLES[index % len(LINE_STYLES)][0]
        maximum_axis.plot(
            x, maximum_normalized, color="black", linestyle=style, linewidth=1.25
        )
        corrected_axis.plot(
            x, corrected, color="black", linestyle=style, linewidth=1.25
        )
        for axis, values in (
            (maximum_axis, maximum_normalized),
            (corrected_axis, corrected),
        ):
            _label_at_fraction(
                axis,
                x,
                values,
                _prefill_curve_label(result),
                float(label_fraction),
                1.0,
                maximum_x=0.88 * x_limit,
                fallback_x=x_limit
                * (0.58 + 0.30 * index / max(1, len(cases) - 1)),
            )
    for axis in (maximum_axis, corrected_axis):
        axis.set_xlim(0.0, x_limit)
        axis.set_xlabel(_time_label(time_axis))
        axis.grid(True)
        axis.axhline(1.0, color="0.55", linestyle=":", linewidth=0.7)
    maximum_axis.set_ylim(-0.03, 1.06)
    maximum_axis.set_ylabel(r"Individually normalized flux, $J/\max_t(J)$")
    maximum_axis.set_title("Individual-maximum normalization")
    corrected_axis.set_ylim(min(-0.08, 1.08 * corrected_minimum), 1.06)
    corrected_axis.set_ylabel(
        r"Baseline-corrected flux, $(J-J_0)/(J_{\mathrm{ss}}-J_0)$"
    )
    corrected_axis.set_title("Initial-baseline subtraction and normalization")
    fig.suptitle("How normalization changes residual-hydrogen signatures", fontsize=12)
    fig.supxlabel(
        r"Values on curves: retained centre concentration $C(x=L/2,t=0)/C_{\mathrm{ref}}$",
        fontsize=7.3,
    )
    return fig


def _metric_label(metric: str) -> str:
    return {
        "t10": r"$t_{10}/\tau_{\mathrm{ref}}$",
        "t50": r"$t_{50}/\tau_{\mathrm{ref}}$",
        "t90": r"$t_{90}/\tau_{\mathrm{ref}}$",
        "time_lag": r"Time lag $/\tau_{\mathrm{ref}}$",
        "peak_flux": r"Peak $J/J_{\mathrm{ref}}$",
        "final_flux": r"Final $J/J_{\mathrm{ref}}$",
        "overshoot": r"Overshoot $/J_{\mathrm{ref}}$",
    }.get(metric, metric)


def render_response_map(results: Mapping[str, SimulationResult], metric: str):
    cases = _select(results, "map:")
    if not cases:
        raise ValueError("Response-map cases are missing.")
    capacities = sorted({result.config.traps.capacity_ratio for _, result in cases})
    half_times = sorted({result.config.traps.release_half_time_ref for _, result in cases})
    values = np.full((len(capacities), len(half_times)), np.nan)
    for _, result in cases:
        row = capacities.index(result.config.traps.capacity_ratio)
        column = half_times.index(result.config.traps.release_half_time_ref)
        values[row, column] = result.metrics.get(metric, np.nan)
    if not np.any(np.isfinite(values)):
        raise ValueError(f"Metric '{metric}' is unavailable for the response map.")

    x, y = np.meshgrid(half_times, capacities)
    fig, (map_axis, flux_axis) = plt.subplots(
        1,
        2,
        figsize=(9.6, 4.7),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )
    finite = values[np.isfinite(values)]
    if np.nanmax(finite) - np.nanmin(finite) < 1.0e-12:
        levels = np.array([finite[0]])
    else:
        levels = np.linspace(np.nanmin(finite), np.nanmax(finite), 7)
    contour = map_axis.contour(
        x,
        y,
        np.ma.masked_invalid(values),
        levels=levels,
        colors="black",
        linewidths=1.0,
    )
    map_axis.clabel(
        contour,
        inline=True,
        fontsize=7.1,
        fmt=lambda value: f"{metric}={value:.3g}",
    )
    map_axis.scatter(
        x,
        y,
        facecolors="white",
        edgecolors="black",
        s=24,
        zorder=3,
    )

    corner_parameters = (
        (capacities[0], half_times[0]),
        (capacities[0], half_times[-1]),
        (capacities[-1], half_times[0]),
        (capacities[-1], half_times[-1]),
    )
    corner_labels = ("A", "B", "C", "D")
    selected_cases = []
    label_offsets = ((6, 6), (-14, 6), (6, -13), (-14, -13))
    for letter, (capacity, half_time), label_offset in zip(
        corner_labels, corner_parameters, label_offsets
    ):
        selected = next(
            result
            for _, result in cases
            if result.config.traps.capacity_ratio == capacity
            and result.config.traps.release_half_time_ref == half_time
        )
        selected_cases.append((letter, selected))
        map_axis.scatter(
            [half_time],
            [capacity],
            facecolors="black",
            edgecolors="black",
            s=34,
            zorder=4,
        )
        map_axis.annotate(
            letter,
            (half_time, capacity),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
        )

    map_axis.set_xscale("log")
    map_axis.set_xlabel(
        "Trap release half-time\n"
        r"$t_{1/2,\mathrm{det}}/\tau_{\mathrm{ref}}$  (right = slower release)"
    )
    map_axis.set_ylabel(
        "Normalized trap capacity\n"
        r"$N_T/C_{\mathrm{ref}}$  (up = more storage)"
    )
    if metric in {"t10", "t50", "t90"}:
        map_axis.set_title(
            rf"Equal $t_{{{metric[1:]}}}$ lines: same {metric[1:]}%-flux arrival time"
        )
    else:
        map_axis.set_title(f"Equal {_metric_label(metric)} lines")
    map_axis.grid(True, which="both")
    line_key = (
        rf"○ = numerical simulation     — = equal $t_{{{metric[1:]}}}$ line"
        if metric in {"t10", "t50", "t90"}
        else f"○ = numerical simulation     — = equal {_metric_label(metric)} line"
    )
    map_axis.text(
        0.50,
        0.02,
        line_key,
        transform=map_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )

    reference_candidates = [
        result
        for key, result in results.items()
        if key.startswith("map_reference:")
        or (key.startswith("trap:") and result.config.traps.capacity_ratio == 0.0)
    ]
    if reference_candidates:
        reference = reference_candidates[0]
        flux_axis.plot(
            reference.time_ref,
            reference.outlet_flux_common,
            color="black",
            linewidth=1.5,
            label="Trap-free reference",
        )
    qualitative_labels = {
        "A": "low capacity, fast release",
        "B": "low capacity, slow release",
        "C": "high capacity, fast release",
        "D": "high capacity, slow release",
    }
    for index, (letter, result) in enumerate(selected_cases, start=1):
        line_style, marker = LINE_STYLES[index % len(LINE_STYLES)]
        flux_axis.plot(
            result.time_ref,
            result.outlet_flux_common,
            color="black",
            linestyle=line_style,
            marker=marker or None,
            markerfacecolor="white",
            markersize=3.0,
            markevery=48 + index * 4,
            linewidth=1.2,
            label=f"{letter}: {qualitative_labels[letter]}",
        )
    flux_axis.axhline(0.5, color="0.45", linestyle=":", linewidth=0.9)
    flux_axis.text(
        0.98,
        0.51,
        r"50% of ideal steady flux",
        transform=flux_axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7.0,
    )
    flux_axis.set_xlabel(r"Reference time, $t/\tau_{\mathrm{ref}}$")
    flux_axis.set_ylabel(r"Outlet flux, $J/J_{\mathrm{ref}}$")
    flux_axis.set_title("Flux curves at A–D")
    flux_axis.grid(True)
    flux_axis.legend(fontsize=6.8, loc="lower right")
    fig.suptitle("How trap capacity and release time alter the permeation flux", fontsize=12)
    return fig


def render_overview(
    results: Mapping[str, SimulationResult],
    normalization: str,
    time_axis: str,
    comparison_window_ref: float | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    panels = (
        (axes[0, 0], sorted(_select(results, "ideal:D:"), key=lambda item: item[1].config.diffusivity_ratio), "Diffusivity", lambda r: f"D/Dref={r.config.diffusivity_ratio:g}"),
        (axes[0, 1], sorted(_select(results, "surface:"), key=lambda item: item[1].config.surface.delta_concentration), "Entry-surface history", lambda r: r.config.label),
        (axes[1, 0], sorted(_select(results, "trap_capacity:"), key=lambda item: item[1].config.traps.capacity_ratio), "Kinetic trapping", lambda r: r.config.label),
        (axes[1, 1], sorted(_select(results, "prefill:"), key=lambda item: item[1].config.prefill.enabled), "Aged prefilling", lambda r: r.config.label),
    )
    for axis, cases, title, labels in panels:
        _plot_family(axis, cases, normalization, time_axis, labels, marker_stride=52)
        axis.set_title(title)
        if cases:
            axis.legend(fontsize=6.7)
    fig.suptitle("Hydrogen permeation flux response", fontsize=13)
    return fig


RENDERERS = {
    "overview": render_overview,
    "ideal": render_ideal,
    "surface": render_surface,
    "trapping": render_trapping,
    "prefill": render_prefill,
    "annex_model": render_annex_model,
    "annex_trap_capacity": render_annex_trap_capacity,
    "annex_trap_release": render_annex_trap_release,
    "annex_trap_profiles": render_annex_trap_profiles,
    "annex_prefill_aging": render_annex_prefill_aging,
    "annex_prefill_response": render_annex_prefill_response,
    "1.1_trap_capacity_flux": render_annex_trap_capacity_flux,
    "1.2_trap_release_flux": render_annex_trap_release_flux,
    "1.3_trap_capture_flux": render_annex_trap_capture_flux,
    "1.4_combined_trap_flux": render_annex_trap_combined_flux,
    "2.1_residual_hydrogen_flux": render_annex_prefill_flux,
    "2.2_residual_hydrogen_normalized_flux": render_annex_prefill_normalized,
}


def build_figure(
    results: Mapping[str, SimulationResult],
    figure_name: str,
    normalization: str = "common_reference",
    time_axis: str = "reference",
    response_metric: str = "t50",
    comparison_window_ref: float | None = None,
    style: Mapping[str, object] | None = None,
):
    """Build one P4 figure for both the CLI exporter and Qt preview."""

    with plt.rc_context(THESIS_STYLE):
        if figure_name == "response_map":
            figure = render_response_map(results, response_metric)
        else:
            renderer = RENDERERS.get(figure_name)
            if renderer is None:
                raise ValueError(f"Unknown figure type: {figure_name}")
            figure = renderer(
                results,
                normalization,
                time_axis,
                comparison_window_ref,
            )
        _apply_publication_style(figure, style)
    return figure


def render_figures(
    results: Mapping[str, SimulationResult],
    figure_names: Iterable[str],
    output_directory: Path | str,
    result_name: str,
    normalization: str = "common_reference",
    time_axis: str = "reference",
    response_metric: str = "t50",
    comparison_window_ref: float | None = None,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
    show: bool = False,
    style: Mapping[str, object] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_flag=None,
) -> List[Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    figure_paths: List[Path] = []
    open_figures = []
    names = list(figure_names)
    total = len(names)
    for index, figure_name in enumerate(names):
        if cancel_flag is not None and cancel_flag[0] != 0:
            raise SimulationCancelled("Figure rendering cancelled.")
        if progress_callback:
            progress_callback(index, total, f"Rendering {figure_name}")
        figure = build_figure(
            results,
            figure_name,
            normalization=normalization,
            time_axis=time_axis,
            response_metric=response_metric,
            comparison_window_ref=comparison_window_ref,
            style=style,
        )
        for extension in formats:
            if cancel_flag is not None and cancel_flag[0] != 0:
                plt.close(figure)
                raise SimulationCancelled("Figure rendering cancelled.")
            extension = str(extension).lower().lstrip(".")
            if extension not in {"pdf", "svg", "png"}:
                raise ValueError(f"Unsupported figure format: {extension}")
            path = destination / f"{result_name}_{figure_name}.{extension}"
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            figure_paths.append(path)
        open_figures.append(figure)
        if not show:
            plt.close(figure)
        if progress_callback:
            progress_callback(index + 1, total, f"Rendered {figure_name}")
    if show:
        plt.show()
        for figure in open_figures:
            plt.close(figure)
    return figure_paths
