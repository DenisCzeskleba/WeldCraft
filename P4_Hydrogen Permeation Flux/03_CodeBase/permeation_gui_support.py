"""Persistent GUI state, validation, and transactional P4 run services."""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pprint
import re
import runpy
from typing import Any, Callable, Mapping
import uuid

import numpy as np

from permeation_cases import (
    DEFAULT_CONFIG_PATH,
    RUNTIME_CONFIG_PATH,
    build_atlas_cases,
    list_presets,
    load_preset,
    load_settings,
    validate_settings,
)
from permeation_diagrams import render_figures
from permeation_model import SimulationCancelled
from permeation_persistence import load_atlas_hdf5, save_atlas_hdf5


RESULT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
PROFILE_NAME_PATTERN = re.compile(r"[^\x00-\x1f\\/:*?\"<>|]{1,80}\Z")
MANAGED_ASSIGNMENTS = ("CONFIG", "GUI_STATE", "USER_PROFILES")

DEFAULT_GUI_STATE = {
    "preset": "overview",
    "result_name": "hydrogen_permeation_flux",
    "formats": ["png"],
    "last_result_path": "",
    "selected_tab": 0,
    "advanced_visible": False,
    "splitter_sizes": [430, 1100],
    "window_geometry": "",
    "selected_figure": "",
}


def normalize_gui_state(values: Mapping[str, Any] | None) -> dict[str, Any]:
    state = deepcopy(DEFAULT_GUI_STATE)
    if values:
        state.update(deepcopy(dict(values)))
    presets = list_presets()
    if state["preset"] not in presets:
        state["preset"] = "overview"
    state["result_name"] = validate_result_name(str(state["result_name"]))
    state["formats"] = [
        str(value).lower() for value in state.get("formats", [])
        if str(value).lower() in {"png", "pdf", "svg"}
    ]
    state["last_result_path"] = str(state.get("last_result_path", ""))
    state["selected_tab"] = max(0, int(state.get("selected_tab", 0)))
    state["advanced_visible"] = bool(state.get("advanced_visible", False))
    sizes = state.get("splitter_sizes", [1100, 430])
    state["splitter_sizes"] = [int(value) for value in sizes[:2]] if isinstance(sizes, list) else [430, 1100]
    if len(state["splitter_sizes"]) != 2:
        state["splitter_sizes"] = [430, 1100]
    state["window_geometry"] = str(state.get("window_geometry", ""))
    state["selected_figure"] = str(state.get("selected_figure", ""))
    return state


def validate_result_name(value: str) -> str:
    value = value.strip()
    if not RESULT_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "Result name must be a plain filename stem containing letters, numbers, '.', '_' or '-'."
        )
    return value


def validate_profile_name(value: str) -> str:
    value = value.strip()
    if not PROFILE_NAME_PATTERN.fullmatch(value):
        raise ValueError("Profile name must contain 1-80 ordinary filename-safe characters.")
    if value.casefold() in {name.casefold() for name in list_presets()}:
        raise ValueError("User profiles cannot use a shipped preset name.")
    return value


def validate_profiles(values: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if values is None:
        return profiles
    if not isinstance(values, Mapping):
        raise ValueError("USER_PROFILES must be a dictionary.")
    for raw_name, raw_settings in values.items():
        name = validate_profile_name(str(raw_name))
        profile = validate_settings(raw_settings)
        # Output choices belong to the working GUI state, not reusable profiles.
        profile["diagram"]["formats"] = []
        profiles[name] = profile
    return profiles


def load_runtime_state(config_path: Path = RUNTIME_CONFIG_PATH):
    """Load merged settings plus GUI-only state and named profiles."""

    path = Path(config_path)
    settings = load_settings(path if path.exists() else None)
    if not path.exists():
        return settings, normalize_gui_state(None), {}
    namespace = runpy.run_path(str(path))
    return (
        settings,
        normalize_gui_state(namespace.get("GUI_STATE")),
        validate_profiles(namespace.get("USER_PROFILES")),
    )


def _assignment_nodes(source: str) -> dict[str, ast.AST]:
    tree = ast.parse(source or "")
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in MANAGED_ASSIGNMENTS:
                found[target.id] = node
    return found


def _managed_source(source: str, assignments: Mapping[str, Any]) -> str:
    source = source or (
        '"""Persistent user settings for P4 Hydrogen Permeation Flux.\n\n'
        "Generated and validated by the GUI; direct CLI runs use CONFIG.\n"
        '"""\n\n'
    )
    nodes = _assignment_nodes(source)
    replacements = []
    for name, value in assignments.items():
        rendered = f"{name} = {pprint.pformat(value, sort_dicts=False, width=110)}"
        node = nodes.get(name)
        if node is None:
            continue
        lines = source.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))
        start = starts[node.lineno - 1] + node.col_offset
        end = starts[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, rendered))
    for start, end, rendered in sorted(replacements, reverse=True):
        source = source[:start] + rendered + source[end:]
    missing = [name for name in assignments if name not in nodes]
    if missing:
        source = source.rstrip() + "\n\n# GUI-managed values\n"
        for name in missing:
            source += f"{name} = {pprint.pformat(assignments[name], sort_dicts=False, width=110)}\n\n"
    return source.rstrip() + "\n"


def write_runtime_state(
    settings: Mapping[str, Any],
    gui_state: Mapping[str, Any],
    profiles: Mapping[str, Any],
    config_path: Path = RUNTIME_CONFIG_PATH,
):
    """Validate and atomically replace the managed runtime assignments."""

    checked = validate_settings(settings)
    state = normalize_gui_state(gui_state)
    checked_profiles = validate_profiles(profiles)
    path = Path(config_path)
    source = path.read_text(encoding="utf-8") if path.exists() else ""
    rendered = _managed_source(
        source,
        {"CONFIG": checked, "GUI_STATE": state, "USER_PROFILES": checked_profiles},
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        namespace = runpy.run_path(str(temporary))
        validate_settings(namespace["CONFIG"])
        normalize_gui_state(namespace["GUI_STATE"])
        validate_profiles(namespace["USER_PROFILES"])
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return checked, state, checked_profiles


def ensure_runtime_state(config_path: Path = RUNTIME_CONFIG_PATH):
    path = Path(config_path)
    if path.exists():
        return load_runtime_state(path)
    settings = load_settings(DEFAULT_CONFIG_PATH)
    return write_runtime_state(settings, DEFAULT_GUI_STATE, {}, path)


def recover_runtime_defaults(config_path: Path = RUNTIME_CONFIG_PATH):
    """Replace an unreadable runtime file after explicit user confirmation."""

    path = Path(config_path)
    settings = load_settings(DEFAULT_CONFIG_PATH)
    state = normalize_gui_state(None)
    profiles = {}
    rendered = _managed_source(
        "",
        {"CONFIG": settings, "GUI_STATE": state, "USER_PROFILES": profiles},
    )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        runpy.run_path(str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return settings, state, profiles


def restore_defaults(
    gui_state: Mapping[str, Any],
    profiles: Mapping[str, Any],
    config_path: Path = RUNTIME_CONFIG_PATH,
):
    defaults = load_settings(DEFAULT_CONFIG_PATH)
    state = normalize_gui_state(gui_state)
    state["preset"] = "overview"
    state["formats"] = ["png"]
    return write_runtime_state(defaults, state, profiles, config_path)


def create_profile(name: str, settings: Mapping[str, Any], profiles: Mapping[str, Any]):
    checked_name = validate_profile_name(name)
    updated = deepcopy(dict(profiles))
    if any(existing.casefold() == checked_name.casefold() for existing in updated):
        raise ValueError(f"A profile named '{checked_name}' already exists.")
    updated[checked_name] = validate_settings(settings)
    return validate_profiles(updated)


def rename_profile(old_name: str, new_name: str, profiles: Mapping[str, Any]):
    if old_name not in profiles:
        raise ValueError(f"Unknown user profile: {old_name}")
    checked_name = validate_profile_name(new_name)
    if any(name != old_name and name.casefold() == checked_name.casefold() for name in profiles):
        raise ValueError(f"A profile named '{checked_name}' already exists.")
    updated = {}
    for name, values in profiles.items():
        updated[checked_name if name == old_name else name] = deepcopy(values)
    return validate_profiles(updated)


def scientific_settings_hash(settings: Mapping[str, Any]) -> str:
    checked = validate_settings(settings)
    scientific = {key: value for key, value in checked.items() if key != "diagram"}
    payload = json.dumps(scientific, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def result_matches_settings(metadata: Mapping[str, Any], settings: Mapping[str, Any]):
    stored = metadata.get("scientific_settings_hash")
    if not stored:
        return None
    return str(stored) == scientific_settings_hash(settings)


def expected_output_paths(
    output_directory: Path,
    result_name: str,
    figures: list[str],
    formats: list[str],
) -> list[Path]:
    destination = Path(output_directory)
    name = validate_result_name(result_name)
    paths = [destination / f"{name}.h5"]
    for figure in figures:
        for extension in formats:
            paths.append(destination / f"{name}_{figure}.{extension}")
    return paths


def existing_output_paths(*args, **kwargs) -> list[Path]:
    return [path for path in expected_output_paths(*args, **kwargs) if path.exists()]


def _check_cancel(cancel_flag) -> None:
    if cancel_flag is not None and cancel_flag[0] != 0:
        raise SimulationCancelled("P4 operation cancelled.")


def _commit_files(temp_to_final: Mapping[Path, Path]) -> None:
    token = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for final in temp_to_final.values():
            if final.exists():
                backup = final.with_name(f".{final.name}.{token}.bak")
                os.replace(final, backup)
                backups[final] = backup
        for temporary, final in temp_to_final.items():
            os.replace(temporary, final)
            committed.append(final)
    except Exception:
        for final in committed:
            if final.exists():
                final.unlink()
        for final, backup in backups.items():
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def build_result_metadata(settings, preset_name, figures, result_name):
    return {
        "preset": preset_name,
        "figures": list(figures),
        "result_name": result_name,
        "settings_snapshot": deepcopy(settings),
        "scientific_settings_hash": scientific_settings_hash(settings),
        "diagram": deepcopy(settings["diagram"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "application": "WeldCraft P4 Hydrogen Permeation Flux",
    }


def run_atlas_job(
    settings: Mapping[str, Any],
    preset_name: str,
    result_name: str,
    formats: list[str],
    output_directory: Path,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_flag=None,
):
    """Run, save, render, and atomically publish one complete GUI job."""

    checked = validate_settings(settings)
    preset = load_preset(preset_name)
    figures = list(preset["figures"])
    name = validate_result_name(result_name)
    formats = [str(value).lower() for value in formats]
    if any(value not in {"png", "pdf", "svg"} for value in formats):
        raise ValueError("Output formats must be PNG, PDF, or SVG.")
    destination = Path(output_directory)
    if not destination.is_dir():
        raise FileNotFoundError(f"P4 result directory does not exist: {destination}")
    token = uuid.uuid4().hex
    temporary_stem = f".{name}.{token}.tmp"
    temp_h5 = destination / f"{temporary_stem}.h5"
    temp_paths: list[Path] = [temp_h5]

    def report(fraction, message):
        if progress_callback:
            progress_callback(float(fraction), str(message))

    try:
        report(0.0, "Preparing case plan")
        results, metadata = build_atlas_cases(
            checked,
            figures,
            progress_callback=lambda done, total, message: report(
                0.78 * done / max(1, total), message
            ),
            cancel_flag=cancel_flag,
        )
        _check_cancel(cancel_flag)
        metadata.update(build_result_metadata(checked, preset_name, figures, name))
        metadata["case_count"] = len(results)
        report(0.80, "Saving reusable HDF5 result")
        save_atlas_hdf5(temp_h5, results, metadata)
        if formats:
            report(0.84, "Rendering selected figure formats")
            rendered = render_figures(
                results,
                figures,
                destination,
                temporary_stem,
                normalization=checked["diagram"]["normalization"],
                time_axis=checked["diagram"]["time_axis"],
                response_metric=checked["diagram"]["response_metric"],
                comparison_window_ref=checked["diagram"]["comparison_window_ref"],
                formats=formats,
                dpi=checked["diagram"]["dpi"],
                style=checked["diagram"],
                progress_callback=lambda done, total, message: report(
                    0.84 + 0.14 * done / max(1, total), message
                ),
                cancel_flag=cancel_flag,
            )
            temp_paths.extend(rendered)
        _check_cancel(cancel_flag)
        final_paths = expected_output_paths(destination, name, figures, formats)
        mapping = {temp_h5: final_paths[0]}
        for temporary in temp_paths[1:]:
            suffix = temporary.name[len(temporary_stem):]
            mapping[temporary] = destination / f"{name}{suffix}"
        report(0.99, "Publishing completed outputs")
        _commit_files(mapping)
        report(1.0, "P4 run complete")
        return {
            "hdf5_path": final_paths[0],
            "figure_paths": final_paths[1:],
            "figures": figures,
            "case_count": len(results),
        }
    finally:
        for path in temp_paths:
            if path.exists():
                path.unlink()
        for path in destination.glob(f"{temporary_stem}_*"):
            if path.is_file():
                path.unlink()


def export_loaded_results(
    results,
    metadata: Mapping[str, Any],
    settings: Mapping[str, Any],
    result_name: str,
    formats: list[str],
    output_directory: Path,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_flag=None,
):
    checked = validate_settings(settings)
    figures = list(metadata.get("figures") or ["overview"])
    name = validate_result_name(result_name)
    destination = Path(output_directory)
    token = uuid.uuid4().hex
    temporary_stem = f".{name}.{token}.tmp"
    paths = []
    try:
        paths = render_figures(
            results,
            figures,
            destination,
            temporary_stem,
            normalization=checked["diagram"]["normalization"],
            time_axis=checked["diagram"]["time_axis"],
            response_metric=checked["diagram"]["response_metric"],
            comparison_window_ref=checked["diagram"]["comparison_window_ref"],
            formats=formats,
            dpi=checked["diagram"]["dpi"],
            style=checked["diagram"],
            progress_callback=lambda done, total, message: progress_callback(
                done / max(1, total), message
            ) if progress_callback else None,
            cancel_flag=cancel_flag,
        )
        _check_cancel(cancel_flag)
        mapping = {}
        for temporary in paths:
            suffix = temporary.name[len(temporary_stem):]
            mapping[temporary] = destination / f"{name}{suffix}"
        _commit_files(mapping)
        return list(mapping.values())
    finally:
        for path in paths:
            if path.exists():
                path.unlink()
        for path in destination.glob(f"{temporary_stem}_*"):
            if path.is_file():
                path.unlink()


def load_result(path: Path):
    results, metadata = load_atlas_hdf5(path)
    return results, metadata
