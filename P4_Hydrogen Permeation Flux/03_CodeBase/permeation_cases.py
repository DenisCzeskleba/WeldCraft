"""Configuration loading and standard P4 response-atlas case families."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
from typing import Any, Dict, Iterable, Mapping, Tuple

from permeation_model import (
    PrefillConfig,
    SimulationConfig,
    SimulationResult,
    SurfaceHistory,
    TrapConfig,
    simulate_case,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = MODULE_ROOT / "01_Resources"
DEFAULT_CONFIG_PATH = RESOURCE_DIR / "config_default.py"
RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "config.py"
PRESET_DIR = RESOURCE_DIR / "Diagram_Presets"


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
    return settings


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


def _build_ideal_cases(settings: Mapping[str, Any]) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings)
    for ratio in settings["ideal"]["diffusivity_ratios"]:
        config = base.with_changes(label=f"D/Dref = {ratio:g}", diffusivity_ratio=float(ratio))
        results[f"ideal:D:{ratio:g}"] = simulate_case(config)
    for ratio in settings["ideal"]["length_ratios"]:
        config = base.with_changes(label=f"L/Lref = {ratio:g}", length_ratio=float(ratio))
        results[f"ideal:L:{ratio:g}"] = simulate_case(config)
    for ratio in settings["ideal"]["solubility_ratios"]:
        config = base.with_changes(label=f"S/Sref = {ratio:g}", solubility_ratio=float(ratio))
        results[f"ideal:S:{ratio:g}"] = simulate_case(config)
    return results


def _build_surface_cases(settings: Mapping[str, Any]) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings)
    surface_settings = settings["surface"]
    reference = simulate_case(base.with_changes(label="Unchanged entry condition"))
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
        results[f"surface:ratio={ratio:g}"] = simulate_case(
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


def _build_trap_cases(settings: Mapping[str, Any]) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings, end_time_ref=float(settings["trapping"]["end_time_ref"]))
    trap_settings = settings["trapping"]
    fixed_half_time = float(trap_settings["capacity_sweep_release_half_time_ref"])
    for capacity in trap_settings["capacity_ratios"]:
        capacity = float(capacity)
        config = base.with_changes(
            label=f"N_T/C_ref = {capacity:g}",
            traps=_trap_config(settings, capacity, fixed_half_time),
        )
        results[f"trap_capacity:{capacity:g}"] = simulate_case(config)
    fixed_capacity = float(trap_settings["release_sweep_capacity_ratio"])
    for half_time in trap_settings["release_half_times_ref"]:
        half_time = float(half_time)
        config = base.with_changes(
            label=f"t_half,det/tau_ref = {half_time:g}",
            traps=_trap_config(settings, fixed_capacity, half_time),
        )
        results[f"trap_release:{half_time:g}"] = simulate_case(config)
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
        results[f"trap_capture:{capture_rate:g}"] = simulate_case(config)
    for index, values in enumerate(trap_settings["combined_cases"]):
        capacity = float(values["capacity_ratio"])
        half_time = float(values["release_half_time_ref"])
        label = str(values["label"])
        config = base.with_changes(
            label=label,
            traps=_trap_config(settings, capacity, half_time),
        )
        results[f"trap_combined:{index}"] = simulate_case(config)
    return results


def _build_prefill_cases(settings: Mapping[str, Any]) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings)
    results["prefill:empty"] = simulate_case(base.with_changes(label="Initially empty"))
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
        results[f"prefill:center={target:g}"] = simulate_case(
            base.with_changes(
                label=f"Residual centre C/Cref = {target:g}", prefill=prefill
            )
        )
    return results


def _build_response_map_cases(settings: Mapping[str, Any]) -> Dict[str, SimulationResult]:
    results: Dict[str, SimulationResult] = {}
    base = _base_config(settings, end_time_ref=float(settings["trapping"]["end_time_ref"]))
    results["map_reference:no_traps"] = simulate_case(
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
            results[key] = simulate_case(config)
    return results


def build_atlas_cases(
    settings: Mapping[str, Any],
    figure_names: Iterable[str],
) -> Tuple[Dict[str, SimulationResult], Dict[str, Any]]:
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

    results: Dict[str, SimulationResult] = {}
    if "ideal" in requested:
        results.update(_build_ideal_cases(settings))
    if "surface" in requested:
        results.update(_build_surface_cases(settings))
    if "trapping" in requested:
        results.update(_build_trap_cases(settings))
    if "prefill" in requested:
        results.update(_build_prefill_cases(settings))
    if "response_map" in requested:
        results.update(_build_response_map_cases(settings))

    metadata = {
        "figures": figures,
        "diagram": deepcopy(settings["diagram"]),
        "case_count": len(results),
    }
    return results, metadata
