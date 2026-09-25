"""P2 GUI configuration, validation, defaults, and material presets.

The normal developer workflow continues to use ``b2_Simulation_Settings.py``
directly. The GUI only updates an allow-listed set of assignments and always
validates a complete candidate before an atomic replace.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import pprint
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "01_Resources"
RESULTS_DIR = PROJECT_DIR / "02_Results"
PARAM_CONFIG_PATH = CODE_DIR / "b2_Simulation_Settings.py"
DEFAULT_PARAM_PATH = RESOURCE_DIR / "b2_simulation_settings_default.py"
DEGREE_C = "\N{DEGREE SIGN}C"


class P2ConfigError(ValueError):
    """Raised when a working, default, or temporary P2 configuration is invalid."""


# Only these top-level assignments in b2 may be changed by the GUI.
PARAMETER_FIELDS = {
    "simulation_type": ("Simulation", "Joint type", "choice", False),
    "thermal_diffusion_calibration": ("Simulation", "Thermal-only calibration", "bool", False),
    "diffusion_scheme": ("Simulation", "Hydrogen diffusion scheme", "choice", True),
    "add_bead_mode": ("Simulation", "Bead timing mode", "choice", True),
    "dx": ("Mesh", "Mesh spacing [mm]", "float", False),
    "dy": ("Mesh", "Vertical mesh spacing [mm] (locked)", "float", True),
    "no_of_weld_beads": ("Weld beads", "Number of weld beads", "int", False),
    "bead_height": ("Weld beads", "Bead height [mm]", "float", False),
    "bead_width": ("Weld beads", "Bead width [mm]", "float", False),
    "bead_scales": ("Weld beads", "Lap/ISO bead scales", "pairs", True),
    "time_before_first_weld": ("Timing", "Pre-weld time [s]", "float", False),
    "time_for_weld_bead": ("Timing", "Time between beads [s]", "float", False),
    "time_after_last_weld": ("Timing", "Post-weld hold [s]", "float", False),
    "time_heat_hold": ("Timing", "Bead heat hold [s]", "float", False),
    "time_cooling_to_rt": ("Timing", "Cooling-to-room time [s]", "float", False),
    "time_diffusion_at_rt": ("Timing", "Room-temperature diffusion [s]", "float", False),
    "safety_factor": ("Numerics", "Stability safety factor", "float", True),
    "use_big_dt_override": ("Numerics", "Limit room-temperature timestep", "bool", True),
    "big_dt_override": ("Numerics", "Maximum room-temperature dt [s]", "float", True),
    "guess_adaptive_stable_dt": ("Numerics", "Experimental adaptive dt", "bool", True),
    "t_cool": ("Thermal", f"Interpass temperature [{DEGREE_C}]", "float", False),
    "t_hot": ("Thermal", f"New bead temperature [{DEGREE_C}]", "float", False),
    "t_room": ("Thermal", f"Room temperature [{DEGREE_C}]", "float", False),
    "haz_creation_temperature": ("Thermal", f"HAZ threshold [{DEGREE_C}]", "float", True),
    "haz_creation_check_time_window": ("Thermal", "HAZ check window [s]", "float", True),
    "t_conv_air": ("Thermal", "Air heat-transfer coefficient", "float", True),
    "t_conv_cu": ("Thermal", "Copper heat-transfer coefficient", "float", True),
    "hydro_weld_metal": ("Hydrogen", "New-bead hydrogen [%]", "float", False),
    "h_cont_initial": ("Hydrogen", "Initial hydrogen [%]", "float", False),
    "reference_from_iso3690": ("Hydrogen", "ISO 3690 reference", "float", True),
    "pipe_line_inner_hydrogen": ("Hydrogen", "Lap-joint inner boundary", "choice", False),
    "h_on_the_inside": ("Hydrogen", "Constant inner hydrogen [%]", "float", False),
    "t_conv_h2": ("Hydrogen", "Hydrogen-side heat transfer (unused)", "float", True),
    "file_name": ("Saving and animation", "HDF5 result path", "path", False),
    "animation_name": ("Saving and animation", "MP4 output path", "path", False),
    "include_animation_after_run": ("Saving and animation", "Render animation after simulation", "bool", False),
    "s_per_frame_part1": ("Saving and animation", "Main-phase save interval [s]", "float", False),
    "animation_frame_stride": ("Saving and animation", "Animation frame stride", "int", False),
    "use_sparse_saving_in_just_diffusion": ("Saving and animation", "Sparse RT saving", "bool", True),
    "s_per_frame_just_diffusion_sparse": ("Saving and animation", "Sparse RT save interval [s]", "float", True),
    "precalc_grid_step": ("Material models", f"Material lookup step [{DEGREE_C}]", "float", True),
    "debug_bead_plots": ("Experimental", "Blocking bead debug plots", "bool", True),
}

MATERIAL_FIELDS = {
    "microstructure_thermal_diff": "D",
    "microstructure_hydrogen_diff": "D_H",
    "microstructure_solubility": "S",
}

GEOMETRY_FIELDS = {
    "butt joint": {
        "le": "Left plate width",
        "ri": "Right plate width",
        "we": "Weld width",
        "th": "Plate thickness",
        "su_h": "Support height",
        "su_w": "Support width",
        "fr_ab": "Space above",
        "fr_be": "Space below",
    },
    "lap joint": {
        "le": "Upper plate length",
        "we": "Interface gap width",
        "th": "Upper plate thickness",
        "su_h": "Lower plate thickness",
        "fr_le": "Interface gap length",
        "fr_ri": "Right extension",
        "fr_ab": "Space above",
        "fr_be": "Space below",
    },
    "iso3690": {
        "le": "Sample width",
        "th": "Sample thickness",
        "fr_le": "Space left",
        "fr_ri": "Space right",
        "fr_ab": "Space above",
        "fr_be": "Space below",
    },
}


PARAMETER_TOOLTIPS = {
    "simulation_type": "Selects the geometry and matching weld-bead logic.",
    "thermal_diffusion_calibration": "Runs the thermal model only and skips the hydrogen-only phase.",
    "diffusion_scheme": "Hydrogen solver: 0 simplified Fick, 1 flux-conservative, 2 chemical-potential driven.",
    "add_bead_mode": "Weld-bead scheduling method. Only regular intervals are currently implemented.",
    "dx": "Square mesh spacing in mm, e.g. 0.5. Smaller values increase runtime strongly.",
    "dy": "Locked to the main mesh spacing because the current P2 solver requires square cells.",
    "no_of_weld_beads": "Positive whole number. Lap joints support up to 4; ISO 3690 supports up to 3.",
    "bead_height": "Weld-bead height in mm. Prefer a multiple of dx for exact mesh alignment.",
    "bead_width": "Weld-bead width in mm. Prefer a multiple of dx for exact mesh alignment.",
    "bead_scales": "List of positive (width, height) scale pairs used by lap and ISO 3690 beads.",
    "time_before_first_weld": "Simulated time in seconds before the first bead is added.",
    "time_for_weld_bead": "Positive time in seconds between consecutive weld beads.",
    "time_after_last_weld": "Hold time in seconds after the final bead before forced cooling.",
    "time_heat_hold": "Positive time in seconds for which a new bead is held at its initial temperature.",
    "time_cooling_to_rt": "Forced cooling duration in seconds. Zero skips this phase.",
    "time_diffusion_at_rt": "Hydrogen-diffusion duration at room temperature in seconds.",
    "safety_factor": "Multiplier for the calculated stable timestep. Must be greater than zero.",
    "use_big_dt_override": "Limits the automatically calculated room-temperature timestep.",
    "big_dt_override": "Maximum room-temperature timestep in seconds when the limit is enabled.",
    "guess_adaptive_stable_dt": "Experimental attempt to enlarge the timestep during a run.",
    "t_cool": "Interpass and initial plate temperature in C.",
    "t_hot": "Initial temperature of newly added weld metal in C.",
    "t_room": "Ambient and final cooling temperature in C.",
    "haz_creation_temperature": "Base-metal cells above this temperature become HAZ.",
    "haz_creation_check_time_window": "Seconds after bead insertion during which HAZ creation is checked.",
    "t_conv_air": "Thermal boundary coefficient for surfaces exposed to air.",
    "t_conv_cu": "Thermal boundary coefficient for contact with the copper support.",
    "hydro_weld_metal": "Hydrogen concentration assigned to newly added weld metal.",
    "h_cont_initial": "Initial hydrogen concentration in the base material.",
    "reference_from_iso3690": "Reference concentration used for the variable lap-joint hydrogen boundary. Must be positive.",
    "pipe_line_inner_hydrogen": "Constant concentration or temperature-dependent Sieverts-law boundary.",
    "h_on_the_inside": "Inner-boundary concentration used when constant mode is selected.",
    "t_conv_h2": "Reserved hydrogen-side thermal coefficient; currently unused by the solver.",
    "file_name": "HDF5 result path. Must end in .h5 or .hdf5.",
    "animation_name": "Animation output path. Must end in .mp4.",
    "include_animation_after_run": "Render the standard MP4 after a successful GUI-launched simulation.",
    "s_per_frame_part1": "Simulation seconds between saved frames during welding and cooling.",
    "animation_frame_stride": "Positive whole number; renders every nth saved frame.",
    "use_sparse_saving_in_just_diffusion": "Uses a separate, larger save interval during room-temperature diffusion.",
    "s_per_frame_just_diffusion_sparse": "Simulation seconds between room-temperature frames in sparse mode.",
    "precalc_grid_step": "Temperature increment in C for material-property lookup tables.",
    "debug_bead_plots": "Shows blocking diagnostic plots when a bead is inserted.",
}


GEOMETRY_TOOLTIPS = {
    ("butt joint", "le"): "Left plate width in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("butt joint", "ri"): "Right plate width in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("butt joint", "we"): "Weld width in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("butt joint", "th"): "Plate thickness in mm, e.g. 20.0. Prefer a multiple of the mesh spacing for exact alignment.",
    ("butt joint", "su_h"): "Backing-support height in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("butt joint", "su_w"): "Backing-support width in mm. It should normally cover the weld opening.",
    ("lap joint", "le"): "Upper-plate length in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("lap joint", "we"): "Interface-gap width in mm. Zero creates no gap; otherwise prefer a multiple of the mesh spacing.",
    ("lap joint", "th"): "Upper-plate thickness in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("lap joint", "su_h"): "Lower-plate thickness in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("lap joint", "fr_le"): "Length of the separated interface from the common left edge. It should be smaller than the upper-plate length so that the plates still contact.",
    ("iso3690", "le"): "ISO 3690 sample width in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    ("iso3690", "th"): "ISO 3690 sample thickness in mm. Prefer a multiple of the mesh spacing for exact alignment.",
}


def geometry_tooltip(joint: str, name: str, label: str) -> str:
    return GEOMETRY_TOOLTIPS.get(
        (joint, name),
        f"{label.split(',')[0]} in mm. Prefer a multiple of the mesh spacing for exact alignment.",
    )


def _path_on_sys_path(path: Path):
    class _PathContext:
        def __enter__(self):
            self.inserted = str(path) not in sys.path
            if self.inserted:
                sys.path.insert(0, str(path))

        def __exit__(self, *_args):
            if self.inserted:
                try:
                    sys.path.remove(str(path))
                except ValueError:
                    pass

    return _PathContext()


def _run_python_source(source: str, filename: Path) -> dict[str, Any]:
    namespace = {"__file__": str(filename), "__name__": "p2_runtime_config"}
    try:
        with _path_on_sys_path(CODE_DIR), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, str(filename), "exec"), namespace)
    except Exception as exc:
        raise P2ConfigError(f"Could not evaluate {filename.name}: {exc}") from exc
    return namespace


def _run_python_file(path: Path) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P2ConfigError(f"Could not read {path}: {exc}") from exc
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise P2ConfigError(f"{path.name} has invalid Python syntax at line {exc.lineno}: {exc.msg}") from exc
    return _run_python_source(source, path)


def _export_namespace(namespace: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in namespace.items():
        if name.startswith("_") or callable(value) or isinstance(value, ModuleType):
            continue
        if name in {"np", "Path"}:
            continue
        result[name] = deepcopy(value)
    return result


def _ast_assignment_values(source: str, filename: Path, names: set[str], namespace=None) -> dict[str, Any]:
    """Read named top-level assignments through their AST nodes.

    Most GUI fields are literals. A few path assignments intentionally call a
    local helper, so those values fall back to the already evaluated namespace
    after the AST has identified the exact allow-listed assignment.
    """
    try:
        tree = ast.parse(source, filename=str(filename))
    except SyntaxError as exc:
        raise P2ConfigError(
            f"{filename.name} has invalid Python syntax at line {exc.lineno}: {exc.msg}"
        ) from exc
    values = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_names = [target.id for target in targets if isinstance(target, ast.Name)]
        for name in set(target_names) & names:
            try:
                values[name] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if namespace is None or name not in namespace:
                    raise P2ConfigError(f"Could not resolve assignment {name} in {filename.name}")
                values[name] = deepcopy(namespace[name])
    missing = sorted(names - set(values))
    if missing:
        raise P2ConfigError(f"Missing GUI-owned assignment(s) in {filename.name}: {', '.join(missing)}")
    return values


def load_parameter_values(path: Path = PARAM_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P2ConfigError(f"Could not read {path}: {exc}") from exc
    namespace = _run_python_source(source, path)
    result = _export_namespace(namespace)
    result.update(_ast_assignment_values(source, path, set(PARAMETER_FIELDS) | set(MATERIAL_FIELDS), namespace))
    return result


def _joint_branch_bodies(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    """Return the bodies of the simulation_type if/elif geometry block."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        bodies = {}
        current = node
        while isinstance(current, ast.If):
            test = current.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "simulation_type"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
            ):
                joint = test.comparators[0].value
                if joint in GEOMETRY_FIELDS:
                    bodies[joint] = current.body
            current = current.orelse[0] if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If) else None
        if set(bodies) == set(GEOMETRY_FIELDS):
            return bodies
    raise P2ConfigError("The settings file must contain the butt, lap, and iso3690 geometry branches")


def _geometry_from_source(source: str, path: Path) -> dict[str, dict[str, float]]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise P2ConfigError(f"{path.name} has invalid Python syntax at line {exc.lineno}: {exc.msg}") from exc
    geometry = {}
    for joint, body in _joint_branch_bodies(tree).items():
        expected = set(GEOMETRY_FIELDS[joint])
        values = {}
        for node in body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in expected:
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError) as exc:
                        raise P2ConfigError(
                            f"Geometry setting {joint}.{target.id} must be a literal number"
                        ) from exc
        missing = sorted(expected - set(values))
        if missing:
            raise P2ConfigError(f"{joint} geometry is missing: {', '.join(missing)}")
        geometry[joint] = values
    return geometry


def load_geometry_settings(path: Path = PARAM_CONFIG_PATH) -> dict[str, dict[str, float]]:
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P2ConfigError(f"Could not read {path}: {exc}") from exc
    return _geometry_from_source(source, path)


def seed_missing_working_files(
    parameter_path: Path = PARAM_CONFIG_PATH,
    default_parameter_path: Path = DEFAULT_PARAM_PATH,
):
    """Seed an absent settings file; never replace an existing manual edit."""
    target = Path(parameter_path)
    if target.exists():
        return []
    recover_full_file(target, Path(default_parameter_path))
    return [target]


def _value_text(value: Any) -> str:
    if isinstance(value, str) and "\n" in value:
        escaped = value.replace('"""', '\\"\\"\\"')
        return f'"""{escaped}"""'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return pprint.pformat(value, width=110, sort_dicts=False)


def _updated_source(source: str, changes: Mapping[str, Any], *, allowed: set[str]) -> str:
    disallowed = sorted(set(changes) - allowed)
    if disallowed:
        raise P2ConfigError("Unsupported configuration field(s): " + ", ".join(disallowed))
    tree = ast.parse(source)
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    replacements = []
    found = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        matched = [name for name in names if name in changes]
        if not matched:
            continue
        if len(matched) != 1:
            raise P2ConfigError(f"Cannot safely update chained assignment: {', '.join(matched)}")
        name = matched[0]
        value_node = node.value
        start = line_offsets[value_node.lineno - 1] + value_node.col_offset
        end = line_offsets[value_node.end_lineno - 1] + value_node.end_col_offset
        replacements.append((start, end, _value_text(changes[name])))
        found.add(name)
    missing = sorted(set(changes) - found)
    if missing:
        raise P2ConfigError("Configuration assignment(s) not found: " + ", ".join(missing))
    updated = source
    for start, end, replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    ast.parse(updated)
    return updated


def _updated_geometry_source(
    source: str,
    changes: Mapping[str, Mapping[str, Any]],
    *,
    path: Path = PARAM_CONFIG_PATH,
) -> str:
    """Replace literal geometry assignments inside the appropriate joint branch."""
    if not changes:
        return source
    unknown_joints = sorted(set(changes) - set(GEOMETRY_FIELDS))
    if unknown_joints:
        raise P2ConfigError("Unknown geometry profile(s): " + ", ".join(unknown_joints))
    tree = ast.parse(source, filename=str(path))
    bodies = _joint_branch_bodies(tree)
    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    replacements = []
    for joint, requested in changes.items():
        unknown = sorted(set(requested) - set(GEOMETRY_FIELDS[joint]))
        if unknown:
            raise P2ConfigError(f"Unknown {joint} geometry value(s): {', '.join(unknown)}")
        found = set()
        for node in bodies[joint]:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            matched = [name for name in names if name in requested]
            if not matched:
                continue
            if len(matched) != 1:
                raise P2ConfigError(f"Cannot safely update chained geometry assignment: {', '.join(matched)}")
            name = matched[0]
            value_node = node.value
            start = line_offsets[value_node.lineno - 1] + value_node.col_offset
            end = line_offsets[value_node.end_lineno - 1] + value_node.end_col_offset
            replacements.append((start, end, _value_text(requested[name])))
            found.add(name)
        missing = sorted(set(requested) - found)
        if missing:
            raise P2ConfigError(f"{joint} geometry assignment(s) not found: {', '.join(missing)}")
    updated = source
    for start, end, replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    ast.parse(updated, filename=str(path))
    return updated


def _atomic_replace(path: Path, source: str) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp",
            dir=str(path.parent), delete=False,
        ) as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        ast.parse(source, filename=str(path))
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise


def _candidate_configuration(
    parameter_changes: Mapping[str, Any] | None = None,
    geometry_changes: Mapping[str, Mapping[str, Any]] | None = None,
    path: Path = PARAM_CONFIG_PATH,
):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if parameter_changes:
        source = _updated_source(
            source, parameter_changes,
            allowed=set(PARAMETER_FIELDS) | set(MATERIAL_FIELDS),
        )
    if geometry_changes:
        source = _updated_geometry_source(source, geometry_changes, path=path)
    namespace = _run_python_source(source, path)
    parameters = _export_namespace(namespace)
    parameters.update(
        _ast_assignment_values(source, path, set(PARAMETER_FIELDS) | set(MATERIAL_FIELDS), namespace)
    )
    geometry = _geometry_from_source(source, path)
    parameters, geometry = validate_configuration(parameters, geometry)
    return source, parameters, geometry


def candidate_parameter_values(changes: Mapping[str, Any] | None = None, path: Path = PARAM_CONFIG_PATH):
    _source, parameters, _geometry = _candidate_configuration(changes, None, path)
    return parameters


def candidate_configuration(
    parameter_changes: Mapping[str, Any] | None = None,
    geometry_changes: Mapping[str, Mapping[str, Any]] | None = None,
    path: Path = PARAM_CONFIG_PATH,
):
    """Return one fully resolved, validated in-memory settings candidate."""
    _source, parameters, geometry = _candidate_configuration(parameter_changes, geometry_changes, path)
    return parameters, geometry


def write_parameter_values(changes: Mapping[str, Any], path: Path = PARAM_CONFIG_PATH):
    source, parameters, _geometry = _candidate_configuration(changes, None, path)
    _atomic_replace(Path(path), source)
    return parameters


def candidate_geometry_settings(
    changes: Mapping[str, Mapping[str, Any]] | None = None,
    path: Path = PARAM_CONFIG_PATH,
):
    geometry = load_geometry_settings(path)
    for joint, values in (changes or {}).items():
        if joint not in GEOMETRY_FIELDS:
            raise P2ConfigError(f"Unknown geometry profile: {joint}")
        unknown = sorted(set(values) - set(GEOMETRY_FIELDS[joint]))
        if unknown:
            raise P2ConfigError(f"Unknown {joint} geometry value(s): {', '.join(unknown)}")
        geometry[joint].update(values)
    return geometry


def write_geometry_settings(changes: Mapping[str, Mapping[str, Any]], path: Path = PARAM_CONFIG_PATH):
    source, _parameters, geometry = _candidate_configuration(None, changes, path)
    _atomic_replace(Path(path), source)
    return geometry


def write_configuration(
    parameter_changes: Mapping[str, Any],
    geometry_changes: Mapping[str, Mapping[str, Any]],
    path: Path = PARAM_CONFIG_PATH,
):
    """Validate and atomically write parameter and geometry changes together."""
    source, parameters, geometry = _candidate_configuration(parameter_changes, geometry_changes, path)
    _atomic_replace(Path(path), source)
    return parameters, geometry


def _validate_material_spec(spec: str, materials, variable: str, t_min: float, t_max: float):
    from b3_Functions import build_lookup_table, build_temperature_grid, parse_spec_v2

    parsed = parse_spec_v2(spec, variable)
    missing = [name for name in materials if name not in parsed]
    if missing:
        raise P2ConfigError(f"{variable}: missing material block(s): {', '.join(missing)}")
    for material in materials:
        ranges = sorted(parsed[material], key=lambda item: item[0])
        if not ranges:
            raise P2ConfigError(f"{variable}: {material} has no ranges")
        previous = ranges[0][0]
        for lower, upper, _expr in ranges:
            if upper <= lower:
                raise P2ConfigError(f"{variable}: invalid range for {material}: {lower}, {upper}")
            if lower != previous:
                raise P2ConfigError(f"{variable}: gap or overlap before {lower:g} {DEGREE_C} for {material}")
            previous = upper
        if ranges[0][0] > t_min or ranges[-1][1] < t_max:
            raise P2ConfigError(
                f"{variable}: {material} does not cover the expected {t_min:g} to {t_max:g} {DEGREE_C} range"
            )
    grid = build_temperature_grid(t_min, t_max, max(0.5, min(5.0, (t_max - t_min) / 500.0)))
    try:
        table = build_lookup_table(spec, materials, grid, variable)
    except Exception as exc:
        raise P2ConfigError(f"{variable}: could not evaluate formula: {exc}") from exc
    if not np.isfinite(table).all():
        raise P2ConfigError(f"{variable}: formula produced NaN or infinity")
    if np.any(table < 0.0):
        raise P2ConfigError(f"{variable}: formula produced a negative value")


def validate_parameters(values: Mapping[str, Any]) -> dict[str, Any]:
    checked = deepcopy(dict(values))
    required = set(PARAMETER_FIELDS) | set(MATERIAL_FIELDS) | {"microstructures"}
    missing = sorted(required - set(checked))
    if missing:
        raise P2ConfigError("Missing parameter(s): " + ", ".join(missing))
    if checked["simulation_type"] not in GEOMETRY_FIELDS:
        raise P2ConfigError("simulation_type must be butt joint, lap joint, or iso3690")
    if int(checked["diffusion_scheme"]) not in {0, 1, 2}:
        raise P2ConfigError("diffusion_scheme must be 0, 1, or 2")
    if checked["add_bead_mode"] != "regular_intervals":
        raise P2ConfigError("interpass_temperature_controlled is not implemented; use regular_intervals")
    if checked["pipe_line_inner_hydrogen"] not in {"constant", "variable"}:
        raise P2ConfigError("pipe_line_inner_hydrogen must be constant or variable")

    for name, (_category, _label, kind, _advanced) in PARAMETER_FIELDS.items():
        if kind not in {"float", "int"}:
            continue
        try:
            numeric = float(checked[name])
        except (TypeError, ValueError) as exc:
            raise P2ConfigError(f"{name} must be numeric") from exc
        if not math.isfinite(numeric):
            raise P2ConfigError(f"{name} must be finite")
        if kind == "int" and not numeric.is_integer():
            raise P2ConfigError(f"{name} must be a whole number")

    positive = {
        "dx", "dy", "no_of_weld_beads", "bead_height", "bead_width", "safety_factor",
        "reference_from_iso3690", "animation_frame_stride", "precalc_grid_step",
        "time_for_weld_bead", "time_heat_hold",
    }
    for name in positive:
        if float(checked[name]) <= 0:
            raise P2ConfigError(f"{name} must be greater than zero")
    if checked["use_big_dt_override"] and float(checked["big_dt_override"]) <= 0:
        raise P2ConfigError("big_dt_override must be greater than zero while its override is enabled")
    nonnegative = {
        "time_before_first_weld", "time_after_last_weld", "time_cooling_to_rt", "time_diffusion_at_rt",
    }
    for name in nonnegative:
        if float(checked[name]) < 0:
            raise P2ConfigError(f"{name} cannot be negative")
    if not math.isclose(float(checked["dx"]), float(checked["dy"]), rel_tol=0.0, abs_tol=1e-12):
        raise P2ConfigError("P2 currently requires dx and dy to be equal")
    beads = int(checked["no_of_weld_beads"])
    if checked["simulation_type"] == "lap joint" and beads > 4:
        raise P2ConfigError("Lap-joint simulations support at most 4 weld beads")
    if checked["simulation_type"] == "iso3690" and beads > 3:
        raise P2ConfigError("ISO 3690 simulations support at most 3 weld beads")
    scales = checked["bead_scales"]
    if not isinstance(scales, (list, tuple)) or not scales:
        raise P2ConfigError("bead_scales must contain at least one (x, y) pair")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or min(map(float, pair)) <= 0 for pair in scales):
        raise P2ConfigError("Each bead_scales entry must contain two positive numbers")
    for name, suffix in (("file_name", (".h5", ".hdf5")), ("animation_name", (".mp4",))):
        value = str(checked[name]).strip()
        if not value or not value.lower().endswith(suffix):
            raise P2ConfigError(f"{name} must end in {' or '.join(suffix)}")
    materials = list(checked["microstructures"])
    t_min = float(checked.get("precalc_min_temp", min(checked["t_room"], checked["t_cool"], checked["t_hot"])))
    t_max = float(checked.get("precalc_max_temp", max(checked["t_room"], checked["t_cool"], checked["t_hot"])))
    for field, variable in MATERIAL_FIELDS.items():
        _validate_material_spec(str(checked[field]), materials, variable, t_min, t_max)
    return checked


def validate_geometry(values: Mapping[str, Any], parameters: Mapping[str, Any] | None = None):
    geometry = deepcopy(dict(values))
    if set(geometry) != set(GEOMETRY_FIELDS):
        missing = sorted(set(GEOMETRY_FIELDS) - set(geometry))
        extra = sorted(set(geometry) - set(GEOMETRY_FIELDS))
        raise P2ConfigError(f"Geometry profiles mismatch; missing={missing}, extra={extra}")
    for joint, fields in GEOMETRY_FIELDS.items():
        if not isinstance(geometry[joint], dict):
            raise P2ConfigError(f"{joint} geometry must be a dictionary")
        missing = sorted(set(fields) - set(geometry[joint]))
        if missing:
            raise P2ConfigError(f"{joint} geometry is missing: {', '.join(missing)}")
        for name in fields:
            try:
                value = float(geometry[joint][name])
            except (TypeError, ValueError) as exc:
                raise P2ConfigError(f"{joint}.{name} must be numeric") from exc
            if not math.isfinite(value):
                raise P2ConfigError(f"{joint}.{name} must be finite")
            zero_allowed = joint == "lap joint" and name == "we"
            if value < 0:
                raise P2ConfigError(f"{joint}.{name} cannot be negative")
            if value == 0 and not zero_allowed:
                raise P2ConfigError(f"{joint}.{name} must be greater than zero")
    return geometry


def validate_configuration(parameters: Mapping[str, Any], geometry: Mapping[str, Any]):
    checked_parameters = validate_parameters(parameters)
    checked_geometry = validate_geometry(geometry, checked_parameters)
    return checked_parameters, checked_geometry


def configuration_hints(parameters: Mapping[str, Any], geometry: Mapping[str, Any]) -> list[str]:
    """Return non-blocking modelling hints for a valid executable configuration."""
    hints = []
    joint = parameters["simulation_type"]
    spacing = float(parameters["dx"])
    active_geometry = geometry[joint]

    if float(parameters["safety_factor"]) > 1:
        hints.append("A safety factor above 1 may exceed the calculated explicit stability limit.")
    if float(parameters["s_per_frame_part1"]) <= 0:
        hints.append("A non-positive main save interval saves every solver iteration.")
    if parameters["use_sparse_saving_in_just_diffusion"] and float(parameters["s_per_frame_just_diffusion_sparse"]) <= 0:
        hints.append("A non-positive sparse save interval saves every room-temperature iteration.")

    if joint == "butt joint":
        if int(parameters["no_of_weld_beads"]) % 2:
            hints.append("Butt-joint bead placement is normally paired; an even bead count is recommended.")
        if active_geometry["su_w"] < active_geometry["we"]:
            hints.append("The backing support is narrower than the weld opening.")
    elif joint == "lap joint":
        if active_geometry["fr_le"] >= active_geometry["le"]:
            hints.append("The interface gap length leaves no contacting portion beneath the upper plate.")
        if active_geometry["we"] >= min(active_geometry["th"], active_geometry["su_h"]):
            hints.append("The interface gap is at least as large as one of the plate thicknesses.")
        if 0 < active_geometry["we"] < spacing:
            hints.append("The interface gap is smaller than one mesh cell and may disappear from the mesh.")

    misaligned = []
    for name, value in active_geometry.items():
        cells = float(value) / spacing
        if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-9):
            misaligned.append(GEOMETRY_FIELDS[joint][name])
    for name in ("bead_height", "bead_width"):
        cells = float(parameters[name]) / spacing
        if not math.isclose(cells, round(cells), rel_tol=0.0, abs_tol=1e-9):
            misaligned.append(PARAMETER_FIELDS[name][1].split(" [", 1)[0])
    if misaligned:
        hints.append("Not exact multiples of the mesh spacing (rounding may occur): " + ", ".join(misaligned) + ".")
    return hints


def recover_full_file(target: Path, default: Path):
    source = Path(default).read_text(encoding="utf-8")
    ast.parse(source, filename=str(default))
    _run_python_source(source, Path(target))
    _atomic_replace(Path(target), source)


def _sr_material_block(material: str, variant: str) -> str:
    # From PhD thesis SR serivation.
    if variant == "lower":
        ranges = [
            ("] -inf, 20 ]", "1.311965e-9 * 20**2.603497"),
            ("] 20, 100 ]", "1.311965e-9 * T_C**2.603497"),
            ("] 100, 740 ]", "0.335881 * exp(-22869 / (R * T_K))"),
            ("] 740, 1500 ]", "0.673600 * exp(-45086 / (R * T_K))"),
            ("] 1500, 2000 ]", "0.257000 * exp(-17154 / (R * T_K))"),
            ("] 2000, +inf ]", "0.257000 * exp(-17154 / (R * 2273.15))"),
        ]
    else:
        ranges = [
            ("] -inf, 20 ]", "9.957656e-3 * 20**0.337459 * exp(-9523 / (R * 293.15))"),
            ("] 20, 100 ]", "9.957656e-3 * T_C**0.337459 * exp(-9523 / (R * T_K))"),
            ("] 100, 740 ]", "0.085967 * exp(-11390 / (R * T_K))"),
            ("] 740, 1500 ]", "0.673600 * exp(-45086 / (R * T_K))"),
            ("] 1500, 2000 ]", "0.257000 * exp(-17154 / (R * T_K))"),
            ("] 2000, +inf ]", "0.257000 * exp(-17154 / (R * 2273.15))"),
        ]
    return "\n".join([f"material: {material}", *(f"{bounds}: D_H = {expr}" for bounds, expr in ranges)])


def sr_preset_spec(variant: str, default_spec: str | None = None) -> str:
    normalized = variant.strip().lower()
    if normalized == "default":
        if default_spec is None:
            default_spec = str(load_parameter_values(DEFAULT_PARAM_PATH)["microstructure_hydrogen_diff"])
        return default_spec
    if normalized not in {"lower sr", "upper sr", "lower", "upper"}:
        raise P2ConfigError(f"Unknown D_H preset: {variant}")
    mode = "lower" if normalized.startswith("lower") else "upper"
    none = "material: none\n] -inf, +inf ]: D_H = 0"
    metals = [_sr_material_block(name, mode) for name in ("base_metal", "weld_metal", "HAZ")]
    return "\n\n".join([none, *metals])


def material_curves(spec: str, variable: str, materials, temperatures):
    from b3_Functions import build_lookup_table

    grid = np.asarray(temperatures, dtype=float)
    table = build_lookup_table(spec, list(materials), grid, variable)
    if not np.isfinite(table).all() or np.any(table < 0.0):
        raise P2ConfigError(f"{variable} formulas produced invalid values")
    return table

