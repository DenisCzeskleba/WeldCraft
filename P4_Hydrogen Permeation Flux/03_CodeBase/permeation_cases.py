"""Configuration loading and standard P4 response-atlas case families."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import runpy
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from permeation_model import (
    PrefillConfig,
    SimulationConfig,
    SimulationResult,
    SurfaceHistory,
    TrapConfig,
    SimulationCancelled,
    simulate_case,
    validate_config,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = MODULE_ROOT / "01_Resources"
DEFAULT_CONFIG_PATH = RESOURCE_DIR / "config_default.py"
RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "config.py"
PRESET_DIR = RESOURCE_DIR / "Diagram_Presets"

ProgressCallback = Callable[[int, int, str], None]
CaseRunner = Callable[[str, SimulationConfig], SimulationResult]


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _read_python_config(path: Path) -> Dict[str, Any]:
    namespace = runpy.run_path(str(path))
    values = namespace.get("CONFIG", namespace.get("DEFAULT_CONFIG"))
    if not isinstance(values, dict):
        raise ValueError(f"Configuration {path} must define CONFIG or DEFAULT_CONFIG as a dictionary.")
    return deepcopy(values)


def load_settings(config_path: Path | str | None = None) -> Dict[str, Any]:
    settings = _read_python_config(DEFAULT_CONFIG_PATH)
    chosen_path: Path | None
    if config_path is not None:
        chosen_path = Path(config_path)
    elif RUNTIME_CONFIG_PATH.exists():
        chosen_path = RUNTIME_CONFIG_PATH
    else:
        chosen_path = None
    if chosen_path is not None and chosen_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        _deep_merge(settings, _read_python_config(chosen_path))
    return validate_settings(settings)


def _number_list(values: Any, name: str, *, positive: bool = True) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    converted = [float(value) for value in values]
    if positive and any(value <= 0.0 for value in converted):
        raise ValueError(f"Every value in {name} must be positive.")
    return converted


def validate_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a normalized, fully validated P4 settings dictionary."""

    checked = deepcopy(dict(settings))
    required = {"simulation", "ideal", "surface", "trapping", "prefill", "diagram"}
    missing = sorted(required.difference(checked))
    if missing:
        raise ValueError(f"Missing P4 configuration section(s): {', '.join(missing)}")

    simulation = checked["simulation"]
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be a dictionary.")
    simulation["n_nodes"] = int(simulation["n_nodes"])
    simulation["n_output"] = int(simulation["n_output"])
    simulation["max_internal_steps"] = int(simulation["max_internal_steps"])
    for name in (
        "end_time_ref", "diffusion_safety", "reaction_safety",
        "reference_length_mm", "reference_diffusivity_mm2_s",
    ):
        simulation[name] = float(simulation[name])
    concentration = simulation.get("reference_concentration_mol_mm3")
    simulation["reference_concentration_mol_mm3"] = (
        None if concentration in (None, "") else float(concentration)
    )
    validate_config(SimulationConfig(**simulation))
    if simulation["reference_length_mm"] <= 0.0 or simulation["reference_diffusivity_mm2_s"] <= 0.0:
        raise ValueError("Reference length and diffusivity must be positive.")
    if simulation["reference_concentration_mol_mm3"] is not None and simulation["reference_concentration_mol_mm3"] <= 0.0:
        raise ValueError("Reference concentration must be positive when configured.")

    ideal = checked["ideal"]
    for name in ("diffusivity_ratios", "length_ratios", "solubility_ratios"):
        ideal[name] = _number_list(ideal[name], f"ideal.{name}")

    surface = checked["surface"]
    surface["onset_fraction_of_ideal_t50"] = float(surface["onset_fraction_of_ideal_t50"])
    surface["time_constant_fraction_of_ideal_t50"] = float(surface["time_constant_fraction_of_ideal_t50"])
    surface["entry_concentration_ratios"] = _number_list(
        surface["entry_concentration_ratios"], "surface.entry_concentration_ratios"
    )
    if surface["onset_fraction_of_ideal_t50"] < 0.0 or surface["time_constant_fraction_of_ideal_t50"] <= 0.0:
        raise ValueError("Surface onset must be non-negative and its time constant positive.")

    trapping = checked["trapping"]
    scalar_names = (
        "end_time_ref", "capture_rate_ref", "capture_sweep_capacity_ratio",
        "capture_sweep_release_half_time_ref", "capacity_sweep_release_half_time_ref",
        "release_sweep_capacity_ratio", "capacity_sweep_binding_energy_kJ_mol",
        "binding_energy_temperature_K", "detrapping_prefactor_s_inv",
        "lattice_activation_energy_kJ_mol",
    )
    for name in scalar_names:
        trapping[name] = float(trapping[name])
        if trapping[name] <= 0.0 and name != "capture_sweep_capacity_ratio":
            raise ValueError(f"trapping.{name} must be positive.")
    for name in (
        "capture_rate_refs", "capacity_ratios", "release_half_times_ref",
        "binding_energy_kJ_mol", "map_capacity_ratios", "map_release_half_times_ref",
    ):
        trapping[name] = _number_list(
            trapping[name], f"trapping.{name}",
            positive=name not in {"capacity_ratios", "binding_energy_kJ_mol"},
        )
    if any(value < 0.0 for value in trapping["capacity_ratios"]):
        raise ValueError("Trap capacity ratios cannot be negative.")
    if any(value < 0.0 for value in trapping["binding_energy_kJ_mol"]):
        raise ValueError("Binding energies cannot be negative.")
    combined = trapping.get("combined_cases")
    if not isinstance(combined, list) or not combined:
        raise ValueError("trapping.combined_cases must be a non-empty list.")
    normalized_combined = []
    for index, item in enumerate(combined):
        if not isinstance(item, Mapping):
            raise ValueError(f"trapping.combined_cases[{index}] must be a dictionary.")
        value = {
            "label": str(item["label"]),
            "capacity_ratio": float(item["capacity_ratio"]),
            "release_half_time_ref": float(item["release_half_time_ref"]),
        }
        if value["capacity_ratio"] < 0.0 or value["release_half_time_ref"] <= 0.0:
            raise ValueError("Combined trap capacities must be non-negative and half-times positive.")
        normalized_combined.append(value)
    trapping["combined_cases"] = normalized_combined

    prefill = checked["prefill"]
    for name in ("initial_fraction", "target_center_fraction", "maximum_age_time_ref"):
        prefill[name] = float(prefill[name])
    prefill["target_center_fractions"] = _number_list(
        prefill["target_center_fractions"], "prefill.target_center_fractions"
    )
    targets = [prefill["target_center_fraction"], *prefill["target_center_fractions"]]
    if not (0.0 < prefill["initial_fraction"] <= 1.0):
        raise ValueError("Prefill initial fraction must lie between zero and one.")
    if any(not 0.0 < value < prefill["initial_fraction"] for value in targets):
        raise ValueError("Every prefill target must lie between zero and the initial fraction.")
    if prefill["maximum_age_time_ref"] <= 0.0:
        raise ValueError("Maximum prefill ageing time must be positive.")

    diagram = checked["diagram"]
    if diagram["normalization"] not in {"common_reference", "per_curve", "physical"}:
        raise ValueError("Unknown diagram normalization.")
    if diagram["time_axis"] not in {"reference", "fo", "seconds", "minutes"}:
        raise ValueError("Unknown diagram time axis.")
    if diagram["response_metric"] not in {"t10", "t50", "t90", "time_lag", "peak_flux", "final_flux", "overshoot"}:
        raise ValueError("Unknown response-map metric.")
    diagram["comparison_window_ref"] = float(diagram["comparison_window_ref"])
    diagram["dpi"] = int(diagram["dpi"])
    diagram["formats"] = [str(value).lower().lstrip(".") for value in diagram.get("formats", [])]
    if any(value not in {"png", "pdf", "svg"} for value in diagram["formats"]):
        raise ValueError("Diagram formats must be png, pdf, or svg.")
    if diagram["comparison_window_ref"] <= 0.0 or not 50 <= diagram["dpi"] <= 2400:
        raise ValueError("Comparison window must be positive and DPI must be between 50 and 2400.")
    defaults = {
        "figure_scale": 1.0, "font_scale": 1.0, "line_width_scale": 1.0,
        "marker_scale": 1.0, "grid_visible": True, "grid_style": ":",
        "legend_mode": "original", "show_title": True, "title_override": "",
    }
    for name, value in defaults.items():
        diagram.setdefault(name, value)
    for name in ("figure_scale", "font_scale", "line_width_scale", "marker_scale"):
        diagram[name] = float(diagram[name])
        if not 0.25 <= diagram[name] <= 4.0:
            raise ValueError(f"diagram.{name} must lie between 0.25 and 4.0.")
    diagram["grid_visible"] = bool(diagram["grid_visible"])
    diagram["show_title"] = bool(diagram["show_title"])
    diagram["title_override"] = str(diagram["title_override"])
    if diagram["grid_style"] not in {":", "--", "-", "-."}:
        raise ValueError("Unknown grid style.")
    if diagram["legend_mode"] not in {"original", "best", "outside", "hidden"}:
        raise ValueError("Unknown legend mode.")
    if diagram["normalization"] == "physical" and simulation["reference_concentration_mol_mm3"] is None:
        raise ValueError("Physical flux normalization requires a reference concentration.")
    return checked


def list_presets() -> Dict[str, Dict[str, Any]]:
    presets: Dict[str, Dict[str, Any]] = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        values = json.loads(path.read_text(encoding="utf-8"))
        name = str(values.get("name", path.stem))
        presets[name] = values
    return presets


def load_preset(name: str) -> Dict[str, Any]:
    presets = list_presets()
    if name not in presets:
        available = ", ".join(presets)
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return deepcopy(presets[name])


def _base_config(settings: Mapping[str, Any], **changes: Any) -> SimulationConfig:
    simulation = dict(settings["simulation"])
    simulation.update(changes)
    return SimulationConfig(**simulation)


def _build_ideal_cases(settings: Mapping[str, Any], run_case: CaseRunner) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings)
    for ratio in settings["ideal"]["diffusivity_ratios"]:
        config = base.with_changes(label=f"D/Dref = {ratio:g}", diffusivity_ratio=float(ratio))
        results[f"ideal:D:{ratio:g}"] = run_case(f"Ideal diffusivity: {ratio:g}", config)
    for ratio in settings["ideal"]["length_ratios"]:
        config = base.with_changes(label=f"L/Lref = {ratio:g}", length_ratio=float(ratio))
        results[f"ideal:L:{ratio:g}"] = run_case(f"Ideal length: {ratio:g}", config)
    for ratio in settings["ideal"]["solubility_ratios"]:
        config = base.with_changes(label=f"S/Sref = {ratio:g}", solubility_ratio=float(ratio))
        results[f"ideal:S:{ratio:g}"] = run_case(f"Ideal solubility: {ratio:g}", config)
    return results


def _build_surface_cases(settings: Mapping[str, Any], run_case: CaseRunner) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings)
    surface_settings = settings["surface"]
    reference = run_case("Surface reference", base.with_changes(label="Unchanged entry condition"))
    onset = (
        float(surface_settings["onset_fraction_of_ideal_t50"])
        * reference.metrics["t50"]
    )
    time_constant = (
        float(surface_settings["time_constant_fraction_of_ideal_t50"])
        * reference.metrics["t50"]
    )
    for ratio in surface_settings["entry_concentration_ratios"]:
        ratio = float(ratio)
        history = SurfaceHistory(
            base_concentration=1.0,
            delta_concentration=ratio - 1.0,
            onset_time_ref=onset,
            time_constant_ref=time_constant,
            transition_mode="exponential",
        )
        results[f"surface:ratio={ratio:g}"] = run_case(
            f"Entry concentration ratio: {ratio:g}",
            base.with_changes(
                label=f"C_entry,2/C_entry,1 = {ratio:g}", surface=history
            )
        )
    return results


def _trap_config(
    settings: Mapping[str, Any],
    capacity: float,
    half_time: float,
    capture_rate: float | None = None,
) -> TrapConfig:
    return TrapConfig(
        enabled=capacity > 0.0,
        capacity_ratio=float(capacity),
        capture_rate_ref=float(
            settings["trapping"]["capture_rate_ref"]
            if capture_rate is None
            else capture_rate
        ),
        release_half_time_ref=float(half_time),
    )


def _binding_energy_to_half_time_ref(
    settings: Mapping[str, Any],
    base: SimulationConfig,
    binding_energy_kJ_mol: float,
) -> float:
    """Convert an equivalent binding energy into the solver's release half-time."""

    if math.isinf(binding_energy_kJ_mol):
        return math.inf
    trap_settings = settings["trapping"]
    gas_constant = 8.314462618  # J/(mol K)
    temperature = float(trap_settings["binding_energy_temperature_K"])
    prefactor = float(trap_settings["detrapping_prefactor_s_inv"])
    lattice_barrier = float(trap_settings["lattice_activation_energy_kJ_mol"])
    release_rate_s_inv = prefactor * math.exp(
        -((lattice_barrier + binding_energy_kJ_mol) * 1000.0)
        / (gas_constant * temperature)
    )
    return math.log(2.0) / (release_rate_s_inv * base.tau_ref_seconds)


def _build_trap_cases(settings: Mapping[str, Any], run_case: CaseRunner) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings, end_time_ref=float(settings["trapping"]["end_time_ref"]))
    trap_settings = settings["trapping"]
    fixed_half_time = _binding_energy_to_half_time_ref(
        settings,
        base,
        float(trap_settings["capacity_sweep_binding_energy_kJ_mol"]),
    )
    for capacity in trap_settings["capacity_ratios"]:
        capacity = float(capacity)
        config = base.with_changes(
            label=f"N_T/C_ref = {capacity:g}",
            traps=_trap_config(settings, capacity, fixed_half_time),
        )
        results[f"trap_capacity:{capacity:g}"] = run_case(f"Trap capacity: {capacity:g}", config)
    fixed_capacity = float(trap_settings["release_sweep_capacity_ratio"])
    binding_energies = [float(value) for value in trap_settings["binding_energy_kJ_mol"]]
    for binding_energy in [*binding_energies, math.inf]:
        half_time = _binding_energy_to_half_time_ref(settings, base, binding_energy)
        energy_label = "∞ (no detrapping)" if math.isinf(binding_energy) else f"{binding_energy:g} kJ/mol"
        config = base.with_changes(
            label=energy_label,
            traps=_trap_config(settings, fixed_capacity, half_time),
        )
        key = "inf" if math.isinf(binding_energy) else f"{binding_energy:g}"
        results[f"trap_release:{key}"] = run_case(
            f"Equivalent binding energy: {energy_label}", config
        )
    capture_capacity = float(trap_settings["capture_sweep_capacity_ratio"])
    capture_half_time = float(trap_settings["capture_sweep_release_half_time_ref"])
    for capture_rate in trap_settings["capture_rate_refs"]:
        capture_rate = float(capture_rate)
        config = base.with_changes(
            label=f"k_capture*C_ref*tau_ref = {capture_rate:g}",
            traps=_trap_config(
                settings,
                capture_capacity,
                capture_half_time,
                capture_rate=capture_rate,
            ),
        )
        results[f"trap_capture:{capture_rate:g}"] = run_case(f"Trap capture rate: {capture_rate:g}", config)
    for index, values in enumerate(trap_settings["combined_cases"]):
        capacity = float(values["capacity_ratio"])
        half_time = float(values["release_half_time_ref"])
        label = str(values["label"])
        config = base.with_changes(
            label=label,
            traps=_trap_config(settings, capacity, half_time),
        )
        results[f"trap_combined:{index}"] = run_case(f"Combined trap case: {label}", config)
    return results


def _build_prefill_cases(settings: Mapping[str, Any], run_case: CaseRunner) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings)
    results["prefill:empty"] = run_case("Empty prefill reference", base.with_changes(label="Initially empty"))
    values = settings["prefill"]
    targets = values.get(
        "target_center_fractions", [values["target_center_fraction"]]
    )
    for target in targets:
        target = float(target)
        prefill = PrefillConfig(
            enabled=True,
            initial_fraction=float(values["initial_fraction"]),
            target_center_fraction=target,
            maximum_age_time_ref=float(values["maximum_age_time_ref"]),
        )
        results[f"prefill:center={target:g}"] = run_case(
            f"Prefill centre target: {target:g}",
            base.with_changes(
                label=f"Residual centre C/Cref = {target:g}", prefill=prefill
            )
        )
    return results


def _build_response_map_cases(settings: Mapping[str, Any], run_case: CaseRunner) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings, end_time_ref=float(settings["trapping"]["end_time_ref"]))
    results["map_reference:no_traps"] = run_case(
        "Response-map reference",
        base.with_changes(label="Trap-free reference")
    )
    capacities = settings["trapping"]["map_capacity_ratios"]
    half_times = settings["trapping"]["map_release_half_times_ref"]
    for capacity in capacities:
        for half_time in half_times:
            capacity = float(capacity)
            half_time = float(half_time)
            config = base.with_changes(
                label=f"capacity={capacity:g}, release half-time={half_time:g}",
                traps=_trap_config(settings, capacity, half_time),
            )
            key = f"map:cap={capacity:g}:half={half_time:g}"
            results[key] = run_case(
                f"Response map: capacity {capacity:g}, half-time {half_time:g}", config
            )
    return results


def requested_case_families(figure_names: Iterable[str]) -> tuple[list[str], set[str]]:
    figures = list(dict.fromkeys(figure_names))
    requested = set(figures)
    if "overview" in requested:
        requested.update({"ideal", "surface", "trapping"})
    if requested.intersection(
        {
            "1.1_trap_capacity_flux",
            "1.2_trap_release_flux",
            "1.3_trap_capture_flux",
            "1.4_combined_trap_flux",
        }
    ):
        requested.add("trapping")
    if requested.intersection(
        {"2.1_residual_hydrogen_flux", "2.2_residual_hydrogen_normalized_flux"}
    ):
        requested.add("prefill")
    return figures, requested


def estimate_case_count(settings: Mapping[str, Any], figure_names: Iterable[str]) -> int:
    """Return the exact number of solver calls, including hidden references."""

    checked = validate_settings(settings)
    _figures, requested = requested_case_families(figure_names)
    count = 0
    if "ideal" in requested:
        count += sum(len(checked["ideal"][name]) for name in (
            "diffusivity_ratios", "length_ratios", "solubility_ratios"
        ))
    if "surface" in requested:
        count += 1 + len(checked["surface"]["entry_concentration_ratios"])
    if "trapping" in requested:
        count += len(checked["trapping"]["capacity_ratios"])
        count += len(checked["trapping"]["binding_energy_kJ_mol"]) + 1
        count += len(checked["trapping"]["capture_rate_refs"])
        count += len(checked["trapping"]["combined_cases"])
    if "prefill" in requested:
        count += 1 + len(checked["prefill"]["target_center_fractions"])
    if "response_map" in requested:
        count += 1 + (
            len(checked["trapping"]["map_capacity_ratios"])
            * len(checked["trapping"]["map_release_half_times_ref"])
        )
    return count


def build_atlas_cases(
    settings: Mapping[str, Any],
    figure_names: Iterable[str],
    progress_callback: Optional[ProgressCallback] = None,
    cancel_flag=None,
) -> Tuple[Dict[str, SimulationResult], Dict[str, Any]]:
    checked = validate_settings(settings)
    figures, requested = requested_case_families(figure_names)
    total = estimate_case_count(checked, figures)
    completed = 0

    def run_case(message: str, config: SimulationConfig) -> SimulationResult:
        nonlocal completed
        if cancel_flag is not None and cancel_flag[0] != 0:
            raise SimulationCancelled("Simulation cancelled.")
        if progress_callback:
            progress_callback(completed, total, message)
        result = simulate_case(config, cancel_flag=cancel_flag)
        completed += 1
        if progress_callback:
            progress_callback(completed, total, message)
        return result

    results: Dict[str, SimulationResult] = {}
    if "ideal" in requested:
        results.update(_build_ideal_cases(checked, run_case))
    if "surface" in requested:
        results.update(_build_surface_cases(checked, run_case))
    if "trapping" in requested:
        results.update(_build_trap_cases(checked, run_case))
    if "prefill" in requested:
        results.update(_build_prefill_cases(checked, run_case))
    if "response_map" in requested:
        results.update(_build_response_map_cases(checked, run_case))

    metadata = {
        "figures": figures,
        "diagram": deepcopy(checked["diagram"]),
        "case_count": len(results),
    }
    return results, metadata
