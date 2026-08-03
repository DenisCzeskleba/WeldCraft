"""
Lattice Visualizer — SC/BCC/FCC with dopants (PyVista/VTK)

This script builds crystalline lattices at scale, optionally places dopants
(substitutional/interstitial), and renders them efficiently using GPU-instanced
glyphs when available. It also supports export (meshes/screenshots), unit-cell
overlays, picking (find Hydrogen), and a documented Python configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Union
import argparse
import ast
import importlib.util
import json
import os
import pprint
import sys
import time
from pathlib import Path
import numpy as np
import pyvista as pv
import math

# VTK import for instanced glyphs
try:
    # Preferred: VTK 9.x via vtkmodules
    from vtkmodules.vtkRenderingCore import vtkGlyph3DMapper, vtkActor
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    HAVE_GLYPH3D_MAPPER = True
except Exception:
    try:
        # Fallback: some wheels expose classes through pyvista._vtk
        from pyvista import _vtk as vtk
        vtkGlyph3DMapper = vtk.vtkGlyph3DMapper          # type: ignore[attr-defined]
        vtkActor = vtk.vtkActor
        vtkSphereSource = vtk.vtkSphereSource
        HAVE_GLYPH3D_MAPPER = True
    except Exception:
        # No hardware instancing available — still keep vtk types for other uses
        from pyvista import _vtk as vtk  # noqa: F401
        HAVE_GLYPH3D_MAPPER = False
# --- Persistent documented config support ---
CONFIG_OVERRIDE: Optional[str] = None
DEFAULT_CONFIG_BASENAME = "config.py"
DEFAULT_CONFIG_TEMPLATE = "config_default.py"
WELDCRAFT_READY_FILE_ENV_VAR = "WELDCRAFT_STARTUP_READY_FILE"
MAX_TRUE_DOPANT_SPHERES = 500

# These settings change the scene extent, focal point, or explicitly requested
# camera. Retaining the previous camera for them can leave a one-cell lattice
# stranded at the focal point of a million-atom scene.
CAMERA_REFRAME_KEYS = {
    "target_atoms",
    "lattice",
    "lattice_size_behavior",
    "demo_cell_auto",
    "demo_cell_force",
    "r",
    "camera_preset",
    "camera_direction",
    "camera_view_up",
    "camera_distance_scale",
    "camera_normalize_demo_atom_size",
    "camera_parallel_projection",
    "camera_view_angle",
    "defaults",
}


def runtime_directory() -> Path:
    """Return the directory containing the script or packaged executable."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def mark_weldcraft_startup_ready() -> bool:
    """Complete the optional WeldCraft launcher handshake without GUI dependencies."""

    ready_file = os.getenv(WELDCRAFT_READY_FILE_ENV_VAR, "").strip()
    if not ready_file:
        return False

    try:
        Path(ready_file).touch()
    except OSError:
        return False
    return True


def guess_default_config() -> Optional[str]:
    """
Search likely locations/env var for the persistent Python config file.
    """

    if CONFIG_OVERRIDE:
        p = Path(CONFIG_OVERRIDE)
        if not p.is_absolute():
            for base in (Path.cwd(), runtime_directory()):
                cand = base / CONFIG_OVERRIDE
                if cand.exists():
                    return str(cand)
        if p.exists():
            return str(p)

    env = os.getenv("LATTICE_CONFIG")
    if env and Path(env).exists():
        return env

    for name in (DEFAULT_CONFIG_BASENAME,):
        # The persistent application config lives beside the script/executable.
        # Prefer it over an unrelated file named config.py in the caller's
        # working directory. Explicit --config and LATTICE_CONFIG overrides
        # remain available for code-driven runs.
        for base in (runtime_directory(), Path.cwd()):
            cand = base / name
            if cand.exists():
                return str(cand)
    return None


def default_config_template_path() -> Path:
    """Locate the tracked, fully commented default configuration template."""

    candidates = [
        runtime_directory() / "01_Resources" / DEFAULT_CONFIG_TEMPLATE,
        runtime_directory() / DEFAULT_CONFIG_TEMPLATE,
        Path(__file__).resolve().parent / "01_Resources" / DEFAULT_CONFIG_TEMPLATE,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the documented lattice configuration template: "
        f"{DEFAULT_CONFIG_TEMPLATE}"
    )


def persistent_config_path() -> Path:
    """Return the local configuration beside the source or frozen executable."""

    return runtime_directory() / DEFAULT_CONFIG_BASENAME


def ensure_config_file() -> Path:
    """Create the local persistent config from the documented defaults if needed."""

    path = persistent_config_path()
    if path.exists():
        return path

    template = default_config_template_path()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        template.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return path


# ------------------ Data models ------------------
@dataclass
class Species:
    """
Dataclass for a dopant/secondary species (visual color, size, placement mode, etc.).
    """

    name: str
    color: str
    # visual radius is computed later from size_scale; keep as internal cache
    radius: Optional[float] = None
    # Persisted config does not require explicit generated positions.
    positions: List[Tuple[float, float, float]] = field(default_factory=list)

    mode: str = "substitutional"         # "substitutional" | "interstitial"

    # Random placement controls:
    fraction: float = 0.0                 # for substitutional species (0..1)
    count: int = 0                        # for interstitial species (absolute)

    # Interstitial family, either one name or a per-lattice mapping.
    interstitial_site: Optional[
        Union[str, Dict[str, Optional[str]]]
    ] = None
    # Optional exact interstitial in fractional conventional-cell coordinates.
    # This may be one legacy [x, y, z] position or a per-lattice mapping.
    forced_interstitial_position: Optional[
        Union[
            Tuple[float, float, float],
            Dict[str, Optional[Tuple[float, float, float]]],
        ]
    ] = None

    # Single intuitive knob per dopant — relative to base Fe visual radius
    # Example: H size_scale=0.5 (half the grey), A size_scale=1.1 (10% larger than grey)
    size_scale: float = 1.0


@dataclass
class Config:

    """
Dataclass for all runtime settings (lattice, sizes, rendering, dopants, overlays).
    """

    # Lattice type:
    # "Simple Cubic" (use this for the cleanest visualization) | "BCC" | "FCC"
    lattice: str = "Simple Cubic"

    # Lattice size (unit cells)
    Nx: int = 10
    Ny: int = 10
    Nz: int = 10

    # Lattice parameter
    a: float = 1.0

    # Physical sizing
    target_atoms: int = 1_000_000  # user sets this (>=1)
    r: float = 0.124  # nm, atomic radius of Fe (default: ~0.124 nm @ RT)

    # Visual scale: sphere radii as a fraction of r (to keep atoms small on-screen)
    base_radius_scale: float = 0.25
    dopant_radius_scale: float = 0.25

    # Base Fe
    base_color: str = "grey"
    base_radius: float = 0.23
    base_displacements: Dict[Tuple[int, int, int], Tuple[float, float, float]] = field(default_factory=dict)

    # Additional species
    dopants: List[Species] = field(default_factory=lambda: [
        Species(name="A", color="red", mode="substitutional", fraction=0.0, size_scale=1.10),
        Species(name="B", color="blue", mode="substitutional", fraction=0.0, size_scale=1.05),
        Species(name="H", color="black", mode="interstitial", count=0, size_scale=0.50),
    ])

    # Rendering & interaction
    background: str = "white"
    show_axes: bool = True
    display_window: bool = True
    save_png: bool = False
    png_path: str = "02_Results/lattice_visualization.png"
    png_include_lattice_name: bool = True
    png_avoid_overwrite: bool = True
    png_scale: int = 2
    png_transparent_background: bool = False
    window_size: Tuple[int, int] = (1600, 1200)
    anti_aliasing: str = "msaa"  # "msaa" | "ssaa" | "fxaa" | "none"
    multi_samples: int = 8
    visual_preset: str = "screen"  # "screen" | "thesis" | "publication" | "outline"
    sphere_theta: int = 48
    sphere_phi: int = 48
    sphere_specular: float = 0.0
    sphere_ambient: float = 0.0
    sphere_diffuse: float = 1.0
    base_atom_opacity: float = 1.0
    base_atom_outline: bool = False
    base_atom_outline_color: str = "#202124"
    base_atom_outline_width: float = 2.5
    base_atom_outline_depth_offset: float = -2.0
    base_atom_outline_as_tubes: bool = True
    max_atoms_for_outlines: int = 30_000
    camera_preset: str = "custom"  # "custom" | "isometric" | "low_isometric"
    camera_direction: Tuple[float, float, float] = (-1.0, -1.0, 1.0)
    camera_view_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_distance_scale: float = 3.0
    camera_normalize_demo_atom_size: bool = True
    camera_parallel_projection: bool = False
    camera_view_angle: float = 30.0
    axis_location: str = "outer"
    axis_use_3d_text: bool = False
    axis_font_size: int = 32
    axis_line_width: float = 1.75
    deduplicate_axis_zero_labels: bool = True
    enable_picking: bool = True

    # NEW: zoom behavior
    # "focal" (default VTK dolly to focal point) | "cursor" (wheel zooms toward the mouse cursor)
    zoom_mode: str = "cursor"
    # Message shown on screen (bottom-left) when picking is enabled
    pick_instruction: str = "Right click to find the Hydrogen"

    # Rendering mode
    render_mode: str = "auto"  # "auto" | "spheres" | "impostor_points"

    # Export policy for huge scenes
    max_atoms_for_true_spheres: int = 30000

    # Impostor points
    points_impostor_size: float = 3.0

    # Data thinning / cropping
    stride: int = 1
    slab: Optional[Tuple[float, float]] = None  # z in [z0, z1) in lattice units

    # Chunking (instanced glyphs only)
    chunking_enabled: bool = True
    chunk_target_atoms: int = 125_000
    chunk_max_actors: int = 8
    chunk_axis: str = "z"

    # Adaptive resolution thresholds (for base sphere source)
    adaptive_resolution: bool = True
    res_thresh_1: int = 100_000
    res_thresh_2: int = 300_000
    res_thresh_3: int = 1_000_000
    res_cap_1: int = 16
    res_cap_2: int = 12
    res_cap_3: int = 8

    # Unit-cell overlay & legend
    show_unit_cell_overlay: bool = False
    overlay_color: str = "black"
    overlay_alpha: float = 0.65
    overlay_marker_scale: float = 0.6  # as a fraction of cfg.base_radius
    overlay_marker_opacity: float = 0.55
    overlay_marker_specular: float = 0.0
    tetrahedral_color: str = "green"
    octahedral_color: str = "orange"
    cubic_color: str = "purple"
    draw_bravais_overlay: bool = True
    interstitial_site_view: Optional[str] = None  # "all" | "canonical" | "picture"
    picture_site_faces: Union[
        Tuple[int, int, int],
        Dict[str, Tuple[int, int, int]],
    ] = (1, 1, 0)
    overlay_periodic: str = "both_faces"  # "both_faces" | "canonical"
    show_overlay_legend: bool = True  # show legend when unit-cell overlay is on
    overlay_legend_loc: str = "upper right"  # 'upper right' | 'upper left' | 'lower left' | 'lower right'
    overlay_legend_text_color: str = "#3A3A3A"
    overlay_legend_padding: int = 8
    overlay_legend_font_size: int = 18
    overlay_legend_x_offset: float = 0.025

    demo_cell_auto: bool = True  # auto-activate if target_atoms <= threshold for the chosen lattice
    demo_cell_force: Optional[bool] = None  # set to True/False to override auto (None = auto)
    random_seed: Optional[int] = None  # Optional repeatable seed for random dopant placement.


def validate_config(cfg: Config) -> None:
    """Reject invalid or physically impossible settings with clear messages."""

    errors = []
    lattice = str(cfg.lattice or "").strip().lower().replace("_", " ")
    if lattice not in {"simple cubic", "sc", "bcc", "fcc"}:
        errors.append("lattice must be Simple Cubic (SC), BCC, or FCC")
    if int(cfg.target_atoms) < 1:
        errors.append("target_atoms must be at least 1")
    if not math.isfinite(float(cfg.r)) or float(cfg.r) <= 0:
        errors.append("r must be a positive finite radius")
    if not math.isfinite(float(cfg.base_radius_scale)) or float(cfg.base_radius_scale) <= 0:
        errors.append("base_radius_scale must be positive")
    if int(cfg.stride) < 1:
        errors.append("stride must be at least 1")
    if int(cfg.png_scale) < 1:
        errors.append("png_scale must be at least 1")
    if len(cfg.window_size) != 2 or any(int(value) < 1 for value in cfg.window_size):
        errors.append("window_size must contain a positive width and height")
    if len(cfg.camera_direction) != 3 or not all(
        math.isfinite(float(value)) for value in cfg.camera_direction
    ) or math.sqrt(sum(float(value) ** 2 for value in cfg.camera_direction)) <= 1e-12:
        errors.append("camera_direction must contain three finite values and cannot be zero")
    if len(cfg.camera_view_up) != 3 or not all(
        math.isfinite(float(value)) for value in cfg.camera_view_up
    ) or math.sqrt(sum(float(value) ** 2 for value in cfg.camera_view_up)) <= 1e-12:
        errors.append("camera_view_up must contain three finite values and cannot be zero")
    if int(cfg.sphere_theta) < 3 or int(cfg.sphere_phi) < 3:
        errors.append("sphere_theta and sphere_phi must both be at least 3")

    substitutional_fraction = 0.0
    for index, species in enumerate(cfg.dopants, start=1):
        label = species.name.strip() or f"dopant {index}"
        if not species.name.strip():
            errors.append(f"dopant {index} must have a name")
        if species.mode not in {"substitutional", "interstitial"}:
            errors.append(
                f"{label}: mode must be 'substitutional' or 'interstitial'"
            )
        if not math.isfinite(float(species.size_scale)) or float(species.size_scale) <= 0:
            errors.append(f"{label}: size_scale must be positive")
        if not math.isfinite(float(species.fraction)) or not 0.0 <= float(species.fraction) <= 1.0:
            errors.append(f"{label}: fraction must be between 0 and 1")
        if int(species.count) < 0:
            errors.append(f"{label}: count cannot be negative")
        if species.mode == "substitutional":
            substitutional_fraction += float(species.fraction)
    if substitutional_fraction > 1.0 + 1e-12:
        errors.append(
            "the combined substitutional concentration cannot exceed 1.0"
        )

    if errors:
        raise ValueError("Invalid lattice configuration:\n- " + "\n- ".join(errors))


# ------------------ Config I/O ------------------
def _normalize_base_displacements(raw_disps) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
    """
Normalize displacement dict keys/values to integer tuples and float triples.
    """

    if not raw_disps:
        return {}
    out = {}
    for k, v in raw_disps.items():
        if isinstance(k, (list, tuple)):
            key = tuple(int(x) for x in k)
        elif isinstance(k, str):
            s = k.strip().strip("()[]")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) != 3:
                continue
            key = tuple(int(p) for p in parts)
        else:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 3:
            val = tuple(float(x) for x in v)
        else:
            continue
        out[key] = val
    return out


def _default_settings() -> Dict[str, object]:
    """Return a deep, serializable copy of the dataclass defaults."""

    return asdict(Config())


def _load_python_settings(path: Path) -> Dict[str, object]:
    """Load a SETTINGS mapping from a Python config module without importing it globally."""

    module_name = f"weldcraft_p5_config_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load configuration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "SETTINGS", None)
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must define a SETTINGS dictionary")
    return dict(raw)


def _load_raw_settings(path: Path) -> Dict[str, object]:
    ext = path.suffix.lower()
    if ext == ".py":
        return _load_python_settings(path)
    if ext == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} must contain a JSON object")
        return raw
    raise ValueError(
        f"Unsupported configuration format {path.suffix!r}; use a documented .py config"
    )


def load_config(path: Optional[str]) -> Config:
    """Load the documented Python configuration and normalize its values."""

    config_path = Path(path) if path else ensure_config_file()
    if not config_path.is_absolute():
        for base in (Path.cwd(), runtime_directory()):
            candidate = base / config_path
            if candidate.exists():
                config_path = candidate
                break
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = _default_settings()
    raw.update(_load_raw_settings(config_path))

    dopants = []
    for d in raw.get("dopants", []):
        if isinstance(d, Species):
            dopants.append(d)
            continue
        if isinstance(d, dict) and "interstitial_offset" in d:
            d = {k: v for k, v in d.items() if k != "interstitial_offset"}
        dopants.append(Species(**d))
    raw["dopants"] = dopants

    raw["base_displacements"] = _normalize_base_displacements(
        raw.get("base_displacements", {})
    )
    cfg = Config(**raw)
    validate_config(cfg)
    return cfg


def apply_visual_preset(cfg: Config) -> None:
    """Apply optional coordinated appearance settings after loading the config."""

    preset = str(cfg.visual_preset or "screen").strip().lower()
    if preset in ("", "custom"):
        return
    if preset not in ("screen", "default", "thesis", "publication", "outline"):
        raise ValueError(
            f"Unknown visual_preset {cfg.visual_preset!r}; "
            "use 'custom', 'screen', 'thesis', 'publication', or 'outline'."
        )

    if preset in ("screen", "default"):
        cfg.background = "#FFFFFF"
        cfg.base_color = "#555555"
        cfg.base_atom_opacity = 1.0
        cfg.base_atom_outline = False
        cfg.overlay_color = "#222222"
        cfg.overlay_alpha = 0.6
        cfg.overlay_marker_opacity = 0.55
        cfg.overlay_marker_specular = 0.0
        cfg.sphere_specular = 0.0
        cfg.sphere_ambient = 0.0
        cfg.sphere_diffuse = 1.0
        cfg.tetrahedral_color = "#008000"
        cfg.octahedral_color = "#FFA500"
        cfg.cubic_color = "#800080"
        for sp in cfg.dopants:
            if sp.name.strip().lower().startswith("h"):
                sp.color = "#0000FF"
        return

    # Thesis palette: neutral steel, muted hydrogen blue, soft green/teal site
    # families, and near-black construction lines. Flatter lighting reproduces
    # cleanly in print and avoids the heavy black Fe spheres of screen mode.
    cfg.background = "#FFFFFF"
    cfg.base_color = "#9A9FA5"
    cfg.overlay_color = "#202124"
    cfg.overlay_alpha = 0.78
    cfg.overlay_marker_opacity = 0.76
    cfg.overlay_marker_specular = 0.0
    cfg.sphere_specular = 0.0
    cfg.sphere_ambient = 0.32
    cfg.sphere_diffuse = 0.68
    cfg.tetrahedral_color = "#5BB97D"
    cfg.octahedral_color = "#C83E4D"
    cfg.cubic_color = "#202124"
    for sp in cfg.dopants:
        if sp.name.strip().lower().startswith("h"):
            sp.color = "#3F6FAE"

    if preset == "outline":
        # A translucent Fe shell plus a camera-aware silhouette makes atoms
        # hidden by an isometric projection remain legible. Site markers and
        # dopants deliberately retain their normal opacity.
        cfg.base_atom_opacity = 0.74
        cfg.base_atom_outline = True
        cfg.base_atom_outline_color = "#202124"
        cfg.base_atom_outline_width = 2.5


def apply_camera_preset(cfg: Config) -> None:
    """Apply an optional camera-only preset after loading the config."""

    preset = str(cfg.camera_preset or "custom").strip().lower().replace("-", "_")
    if preset in ("", "custom", "manual", "none"):
        return
    if preset in ("isometric", "full_isometric"):
        cfg.camera_direction = (-1.0, -1.0, 1.0)
    elif preset in ("low_isometric", "reference"):
        # Reconstructed from the supplied PyVista screenshot: symmetric X/Y
        # azimuth, a lower ~18-degree camera elevation, Z upright, and VTK's
        # normal perspective projection.
        cfg.camera_direction = (-1.0, -1.0, 0.45)
    else:
        raise ValueError(
            f"Unknown camera_preset {cfg.camera_preset!r}; "
            "use 'custom', 'isometric', or 'low_isometric'."
        )

    cfg.camera_view_up = (0.0, 0.0, 1.0)
    cfg.camera_distance_scale = 3.0
    cfg.camera_parallel_projection = False
    cfg.camera_view_angle = 30.0


def _settings_for_serialization(cfg: Config) -> Dict[str, object]:
    """Convert dataclasses into the Python-literal representation used by config.py."""

    return asdict(cfg)


def _source_offset(lines: List[str], line: int, column: int) -> int:
    return sum(len(item) for item in lines[: line - 1]) + column


def _format_dopants_for_config(dopants: List[Dict[str, object]]) -> str:
    """Format the dynamic dopant list with its explanatory field comments."""

    literal = pprint.pformat(dopants, indent=8, sort_dicts=False, width=112)
    return (
        "[\n"
        "        # Fields available for every species:\n"
        "        # - name: label used in the display, legend, and individual mesh export.\n"
        "        # - color: atom color as a hexadecimal value or a standard color name.\n"
        "        # - radius: derived display radius; normally leave this as None.\n"
        "        # - positions: optional explicit fractional lattice coordinates; leave []\n"
        "        #   to generate positions from the placement settings and random seed.\n"
        "        # - mode: 'substitutional' replaces host sites; 'interstitial' occupies\n"
        "        #   legal spaces between host atoms.\n"
        "        # - fraction: host-site fraction used only in substitutional mode (0..1).\n"
        "        # - count: absolute atom count used only in interstitial mode.\n"
        "        # - interstitial_site: 'tetra', 'octa', 'cubic', 'any', or a mapping\n"
        "        #   with separate BCC/FCC/SC choices. Hydrogen commonly uses tetra in\n"
        "        #   BCC, octa in FCC, and cubic in SC.\n"
        "        # - forced_interstitial_position: optional exact [x, y, z] site, or a\n"
        "        #   BCC/FCC/SC mapping. None selects a legal site automatically. Typical\n"
        "        #   examples are BCC [0.25, 0.0, 0.5], FCC [0.5, 0.0, 0.0], and\n"
        "        #   SC [0.5, 0.5, 0.5].\n"
        "        # - size_scale: displayed radius relative to a host atom (0.5 = half).\n"
        f"{literal[1:-1]}\n"
        "    ]"
    )


def _render_documented_config(cfg: Config) -> str:
    template_path = default_config_template_path()
    source = template_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(template_path))
    settings_node = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SETTINGS" for target in node.targets)
        and isinstance(node.value, ast.Dict)
    )
    values = _settings_for_serialization(cfg)
    lines = source.splitlines(keepends=True)
    replacements = []
    for key_node, value_node in zip(settings_node.keys, settings_node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        key = key_node.value
        if key not in values:
            continue
        start = _source_offset(lines, value_node.lineno, value_node.col_offset)
        end = _source_offset(lines, value_node.end_lineno, value_node.end_col_offset)
        if key == "dopants":
            replacement = _format_dopants_for_config(values[key])
        else:
            replacement = pprint.pformat(values[key], sort_dicts=False, width=112)
        replacements.append((start, end, replacement))

    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]

    return source


def dump_config(cfg: Config, path: str):
    """Atomically write the fully commented Python config or an explicit JSON override."""

    validate_config(cfg)
    destination = Path(path)
    if destination.suffix.lower() == ".py":
        payload = _render_documented_config(cfg)
    elif destination.suffix.lower() == ".json":
        payload = json.dumps(_settings_for_serialization(cfg), indent=2, default=str)
    else:
        raise ValueError("Configuration output must use .py or .json")

    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(destination)


def _normalized_lattice_key(lattice: str) -> str:
    """Return the short configuration key for a supported lattice."""

    lattice_name = str(lattice or "").strip().lower().replace("_", " ")
    if lattice_name in ("simple cubic", "sc"):
        return "sc"
    if lattice_name in ("bcc", "fcc"):
        return lattice_name
    return lattice_name


def _resolve_lattice_setting(value, lattice: str):
    """Resolve either one value or a BCC/FCC/SC keyed configuration value."""

    if not isinstance(value, dict):
        return value

    normalized_values = {}
    for key, item in value.items():
        normalized_key = str(key).strip().lower().replace("_", " ")
        if normalized_key == "simple cubic":
            normalized_key = "sc"
        normalized_values[normalized_key] = item

    return normalized_values.get(
        _normalized_lattice_key(lattice),
        normalized_values.get("default"),
    )


def _picture_site_faces_for_lattice(cfg: Config) -> Tuple[int, int, int]:
    """Return the selected periodic faces for the active lattice."""

    faces = _resolve_lattice_setting(cfg.picture_site_faces, cfg.lattice)
    if not isinstance(faces, (list, tuple)) or len(faces) != 3:
        raise ValueError(
            "picture_site_faces must be [x, y, z] or a lattice-aware "
            "mapping containing three-value lists."
        )
    resolved = tuple(int(value) for value in faces)
    if any(value not in (0, 1) for value in resolved):
        raise ValueError("picture_site_faces values must each be 0 or 1")
    return resolved


def output_path_with_lattice_name(path: str, lattice: str) -> str:
    """Append the active lattice to a configured PNG filename once."""

    requested = Path(path)
    lattice_label = _normalized_lattice_key(lattice).upper()
    if requested.stem.lower().endswith(f" {lattice_label.lower()}"):
        return str(requested)
    return str(
        requested.with_name(
            f"{requested.stem} {lattice_label}{requested.suffix}"
        )
    )


def resolve_runtime_output_path(path: str) -> str:
    """Anchor a relative configured output path beside the program."""

    requested = Path(path)
    if not requested.is_absolute():
        requested = runtime_directory() / requested
    return str(requested.resolve())


def next_available_output_path(path: str) -> str:
    """Return a download-style numbered filename without overwriting a file."""

    requested = Path(path)
    if not requested.exists():
        return str(requested)

    counter = 1
    while True:
        candidate = requested.with_name(
            f"{requested.stem} ({counter}){requested.suffix}"
        )
        if not candidate.exists():
            return str(candidate)
        counter += 1


def _compute_counts(cfg: Config):
    """
Compute counts of base/substitutional/interstitial atoms after filters.
    """

    # base sites w/ chosen lattice, after stride/slab/displacements and removing substitutionals
    base_idx = generate_lattice_sites(cfg.Nx, cfg.Ny, cfg.Nz, cfg.lattice)
    base_idx = apply_stride_and_slab_indices(base_idx, cfg.stride, cfg.slab)
    base_lat = apply_manual_displacements(base_idx, cfg.base_displacements)
    base_lat = remove_base_sites_for_substitutionals(base_lat, cfg.dopants)

    base_count = int(base_lat.shape[0])
    sub_counts = {sp.name: len(sp.positions) for sp in cfg.dopants if sp.mode == "substitutional"}
    int_counts = {sp.name: len(sp.positions) for sp in cfg.dopants if sp.mode == "interstitial"}

    total = base_count + sum(sub_counts.values()) + sum(int_counts.values())
    return base_count, sub_counts, int_counts, total


# ------------------ Export helpers ------------------
def ensure_dir(d: str):
    """
Create directory if it does not exist.
    """

    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def save_mesh(mesh: pv.PolyData, path: str):
    """
Save a PolyData mesh using the file extension to choose format.
    """

    if mesh is None or not mesh.n_points:
        return
    mesh.save(path)  # PyVista picks format from extension


def export_all(base_mesh: Optional[pv.PolyData], dop_meshes,
               export_dir: Optional[str], export_merged: Optional[str]):
    """
Export base and dopant meshes individually and/or as a merged file.
    """
    if not export_dir and not export_merged:
        return
    if export_dir:
        ensure_dir(export_dir)
        if base_mesh is not None and base_mesh.n_points:
            save_mesh(base_mesh, os.path.join(export_dir, "base.vtp"))
        for mesh, sp in dop_meshes:
            if mesh is not None and mesh.n_points:
                save_mesh(mesh, os.path.join(export_dir, f"{sp.name}.vtp"))
    if export_merged:
        parts = [m for m, sp in dop_meshes if m is not None and m.n_points]
        if base_mesh is not None and base_mesh.n_points:
            parts.insert(0, base_mesh)
        if parts:
            # ``append_polydata`` is not part of the pinned PyVista 0.46 API.
            # Merge without welding coincident points so each sphere keeps its
            # original topology and the result remains a PolyData export.
            merged = parts[0].merge(parts[1:], merge_points=False)
            save_mesh(merged, export_merged)


# ------------------ Lattice utilities ------------------
def _basis_and_interstitials(lattice: str):
    """
Return basis positions and catalogued interstitial sites for the lattice.
    """

    lat = (lattice or "Simple Cubic").strip().lower()
    if lat in ("simple cubic", "sc"):
        basis = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # Simple cubic has one conventional cubic (CN=8) hole at the body
        # centre. Quarter-coordinate points are not tetrahedral holes: each has
        # only one nearest SC atom, rather than four equidistant neighbours.

        # Cubic (CN=8): body center (½,½,½). (This is *not* octahedral.)
        cubic = [(0.5, 0.5, 0.5)]

        inter = {
            "octa": [],
            "tetra": [],
            "cubic": [np.array(p, dtype=np.float32) for p in cubic],
        }

    elif lat == "bcc":
        # Basis
        basis = np.array([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.5]], dtype=np.float32)
        # Octahedral in BCC: 6 face centers and 12 edge centers as drawn on a
        # closed conventional cell. Periodic deduplication yields 6 sites/cell:
        # 3 from shared faces and 3 from shared edges.
        octa_faces = [
            (0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
            (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
            (0.0, 0.5, 0.5), (1.0, 0.5, 0.5),
        ]
        octa_edges = []
        for axis in range(3):
            for u in (0.0, 1.0):
                for v in (0.0, 1.0):
                    p = [0.0, 0.0, 0.0]
                    p[axis] = 0.5
                    p[(axis + 1) % 3] = u
                    p[(axis + 2) % 3] = v
                    octa_edges.append(tuple(p))
        octa = octa_faces + octa_edges
        # Tetrahedral in BCC: 12 positions like (1/4,1/2,0) and permutations with 1/4↔3/4
        tetra = []
        vals_q = (0.25, 0.75)
        mids = (0.5,)
        zeros = (0.0, 1.0)
        # patterns: (q, 1/2, 0) and permutations; also flip q between 1/4 and 3/4
        patterns = [
            (vals_q, mids, zeros),
            (mids, vals_q, zeros),
            (vals_q, zeros, mids),
            (mids, zeros, vals_q),
            (zeros, vals_q, mids),
            (zeros, mids, vals_q),
        ]
        for X, Y, Z in patterns:
            for x in X:
                for y in Y:
                    for z in Z:
                        tetra.append((x, y, z))
        inter = {"octa": [np.array(p, dtype=np.float32) for p in octa],
                 "tetra": [np.array(p, dtype=np.float32) for p in tetra]}

    elif lat == "fcc":
        # Basis
        basis = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.5, 0.5],
                          [0.5, 0.0, 0.5],
                          [0.5, 0.5, 0.0]], dtype=np.float32)
        # Octahedral in FCC: 1 body center + 12 edge centers. Face centers are
        # Fe basis sites and must not be catalogued as interstitials.
        octa = [(0.5, 0.5, 0.5)]
        edges = []
        for axis in range(3):
            for u in (0.0, 1.0):
                for v in (0.0, 1.0):
                    p = [0.0, 0.0, 0.0]
                    p[axis] = 0.5
                    p[(axis + 1) % 3] = u
                    p[(axis + 2) % 3] = v
                    edges.append(tuple(p))
        octa.extend(edges)
        # Tetrahedral in FCC: all 8 with coords in {1/4, 3/4}
        tetra = []
        for x in (0.25, 0.75):
            for y in (0.25, 0.75):
                for z in (0.25, 0.75):
                    tetra.append((x, y, z))
        inter = {"octa": [np.array(p, dtype=np.float32) for p in octa],
                 "tetra": [np.array(p, dtype=np.float32) for p in tetra]}
    else:
        raise ValueError(f"Unknown lattice type: {lattice}")
    return basis, inter


def _periodic_site_representatives(frac_pts, mode: str, picture_faces=None):
    """
    Return periodic site representatives for an overlay or a single demo cell.

    ``canonical`` uses [0,1), while ``picture`` chooses the explicitly
    configured equivalent face (0 or 1) independently for X, Y, and Z.
    ``all`` preserves both faces of the closed conventional cell.
    """

    if frac_pts is None:
        return []
    pts = np.asarray(frac_pts, dtype=np.float32)
    if pts.size == 0:
        return []
    pts = pts.reshape((-1, 3))

    normalized_mode = str(mode or "all").strip().lower()
    if normalized_mode == "both_faces":
        normalized_mode = "all"
    if normalized_mode == "all":
        return [tuple(map(float, p)) for p in pts]
    if normalized_mode not in ("canonical", "picture"):
        raise ValueError(
            f"Unknown interstitial_site_view {mode!r}; use 'all', 'canonical', or 'picture'."
        )

    pts = np.mod(pts, 1.0)
    pts[np.isclose(pts, 0.0, atol=1e-7)] = 0.0
    if normalized_mode == "picture":
        selected_faces = np.asarray(
            picture_faces if picture_faces is not None else (1, 1, 0),
            dtype=int,
        )
        for axis in range(3):
            if selected_faces[axis] not in (0, 1):
                raise ValueError("picture_site_faces values must each be 0 or 1")
            if selected_faces[axis] == 1:
                pts[np.isclose(pts[:, axis], 0.0, atol=1e-7), axis] = 1.0
    pts = np.round(pts, 6)
    uniq = np.unique(pts, axis=0)
    return [tuple(map(float, p)) for p in uniq]


def generate_sc_indices(Nx, Ny, Nz) -> np.ndarray:
    """
Generate integer (i,j,k) indices for a simple cubic grid.
    """

    i, j, k = np.indices((Nx, Ny, Nz), dtype=np.int32)
    out = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.float32, copy=False)
    return out


def generate_cell_indices(Nx, Ny, Nz) -> np.ndarray:
    """
Generate integer (i,j,k) indices for unit-cell positions.
    """

    i, j, k = np.indices((Nx, Ny, Nz), dtype=np.int32)
    return np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1).astype(np.float32, copy=False)


def generate_lattice_sites(Nx, Ny, Nz, lattice: str) -> np.ndarray:
    """
Create fractional lattice coordinates by adding basis to each cell.
    """

    # Returns fractional lattice coords (float32), including basis
    cells = generate_cell_indices(Nx, Ny, Nz)
    basis, _ = _basis_and_interstitials(lattice)
    # broadcast add basis to every cell
    cells3 = cells[:, None, :]    # (M,1,3)
    basis3 = basis[None, :, :]    # (1,B,3)
    pts = (cells3 + basis3).reshape(-1, 3).astype(np.float32, copy=False)
    return pts


def _ceil_to_step(val: float, step: float) -> float:
    """
Utility to ceil a value to the next step multiple.
    """

    return float(math.ceil(val / step) * step)


def apply_stride_and_slab_indices(idx_lat: np.ndarray, stride: int,
                                  slab: Optional[Tuple[float, float]]) -> np.ndarray:
    """
Thin points by stride and crop a z-slab in lattice units.
    """
    pts = idx_lat
    if stride and stride > 1:
        ijk = np.rint(pts).astype(np.int32, copy=False)
        keep = (ijk[:, 0] % stride == 0) & (ijk[:, 1] % stride == 0) & (ijk[:, 2] % stride == 0)
        pts = pts[keep]
    if slab is not None:
        z0, z1 = float(slab[0]), float(slab[1])
        keep = (pts[:, 2] >= z0) & (pts[:, 2] < z1)
        pts = pts[keep]
    return np.ascontiguousarray(pts, dtype=np.float32)


def build_integer_index_map(pts_lat_rounded_int: np.ndarray):
    """
Map integer-rounded lattice coords to point indices for fast lookup.
    """

    idx_map: Dict[Tuple[int, int, int], List[int]] = {}
    for n, ijk in enumerate(pts_lat_rounded_int):
        key = (int(ijk[0]), int(ijk[1]), int(ijk[2]))
        idx_map.setdefault(key, []).append(n)
    return idx_map


def apply_manual_displacements(pts_lat: np.ndarray,
                               disps: Dict[Tuple[int, int, int], Tuple[float, float, float]]) -> np.ndarray:
    """
Apply per-index displacements to lattice points.
    """
    if not disps:
        return pts_lat
    out = pts_lat.copy()
    rounded = np.rint(out).astype(np.int32, copy=False)
    idx_map = build_integer_index_map(rounded)
    for key, val in disps.items():
        ids = idx_map.get((int(key[0]), int(key[1]), int(key[2])))
        if not ids:
            continue
        out[ids] += np.asarray(val, dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def world_from_lattice(pts_lat: np.ndarray, a: float) -> np.ndarray:
    """
Convert fractional lattice coordinates to world coordinates using 'a'.
    """

    return np.ascontiguousarray(pts_lat.astype(np.float32, copy=False) * np.float32(a), dtype=np.float32)


def lattice_constant_from_r(lattice: str, r_nm: float) -> float:
    """
Compute lattice constant 'a' from atomic radius r for SC/BCC/FCC.
    """

    lat = (lattice or "Simple Cubic").strip().lower()
    if lat in ("simple cubic", "sc"):
        return 2.0 * r_nm
    if lat == "bcc":
        return 4.0 * r_nm / math.sqrt(3.0)
    if lat == "fcc":
        return 2.0 * math.sqrt(2.0) * r_nm
    raise ValueError(f"Unknown lattice type: {lattice}")


def _basis_count(lattice: str) -> int:
    """
Return number of Fe basis atoms per conventional cell for the lattice.
    """

    basis, _ = _basis_and_interstitials(lattice)
    return int(basis.shape[0])


def choose_cubic_cell_counts_for_target(target_atoms: int, lattice: str) -> Tuple[int,int,int]:
    """
    Return Nx,Ny,Nz (cubic) so that Nx*Ny*Nz*basis_count >= target_atoms.
    Simple and robust: set N = ceil((target/b)^(1/3)).
    """
    b = max(1, _basis_count(lattice))
    n_cells_needed = max(1, int(math.ceil(float(target_atoms) / float(b))))
    N = int(math.ceil(n_cells_needed ** (1.0/3.0)))
    return N, N, N


def normalize_physical_config(cfg: Config) -> None:
    """
Derive dependent fields (a, Nx,Ny,Nz, radii) and demo-cell switch.
    """

    cfg.a = lattice_constant_from_r(cfg.lattice, float(cfg.r))

    # --- demo-cell detection ---
    thresh = _elemental_threshold(cfg.lattice)
    demo_on = (cfg.demo_cell_force if cfg.demo_cell_force is not None
               else (cfg.demo_cell_auto and int(cfg.target_atoms) <= thresh))

    if demo_on:
        cfg.Nx = cfg.Ny = cfg.Nz = 1
    else:
        cfg.Nx, cfg.Ny, cfg.Nz = choose_cubic_cell_counts_for_target(int(cfg.target_atoms), cfg.lattice)

    # Base Fe visual size from r
    cfg.base_radius = float(cfg.base_radius_scale) * float(cfg.r)

    # Dopants: single rule — radius = size_scale * base_radius
    for sp in cfg.dopants:
        sc = float(sp.size_scale) if sp.size_scale is not None else 1.0
        sp.radius = max(1e-6, sc) * float(cfg.base_radius)

    cfg._demo_cell_active = bool(demo_on)


def _elemental_cell_positions(lattice: str) -> np.ndarray:
    """Return Fe lattice positions for a single conventional cell (fractional coords including 0/1)."""
    L = (lattice or "Simple Cubic").strip().lower()
    corners = np.array([(x,y,z) for x in (0.0,1.0) for y in (0.0,1.0) for z in (0.0,1.0)],
                       dtype=np.float32)  # 8
    if L in ("simple cubic", "sc"):
        return corners
    if L == "bcc":
        body = np.array([[0.5,0.5,0.5]], dtype=np.float32)
        return np.vstack([corners, body])  # 9
    if L == "fcc":
        faces = np.array([
            [0.5,0.5,0.0], [0.5,0.5,1.0],
            [0.5,0.0,0.5], [0.5,1.0,0.5],
            [0.0,0.5,0.5], [1.0,0.5,0.5],
        ], dtype=np.float32)
        return np.vstack([corners, faces])  # 14
    raise ValueError(f"Unknown lattice type: {lattice}")


def _single_cell_periodic_copies(points: np.ndarray) -> np.ndarray:
    """Expand canonical boundary sites to all visible one-cell copies."""

    points = np.asarray(points, dtype=np.float32)
    if not points.size:
        return np.empty((0, 3), dtype=np.float32)
    copies = []
    seen = set()
    for point in points:
        canonical = np.mod(point, 1.0)
        choices = [
            (0.0, 1.0) if abs(float(value)) < 1e-6 else (float(value),)
            for value in canonical
        ]
        for x_value in choices[0]:
            for y_value in choices[1]:
                for z_value in choices[2]:
                    copy_position = (x_value, y_value, z_value)
                    key = tuple(int(round(value * 8.0)) for value in copy_position)
                    if key not in seen:
                        seen.add(key)
                        copies.append(copy_position)
    return np.asarray(copies, dtype=np.float32)


def _elemental_threshold(lattice: str) -> int:
    """
Smallest atom count per lattice used to auto-switch to demo cell.
    """

    L = (lattice or "Simple Cubic").strip().lower()
    if L in ("simple cubic", "sc"): return 8
    if L == "bcc": return 9
    if L == "fcc": return 14
    return 8


# ------------------ Random placement ------------------
def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """
Return a NumPy default random Generator.
    """

    return np.random.default_rng(seed)


def _choose_unique_sites(Nx: int, Ny: int, Nz: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """
Choose unique unit cells (i,j,k) without replacement.
    """

    total = Nx * Ny * Nz
    if count <= 0:
        return np.empty((0, 3), dtype=np.int32)
    if count > total:
        raise ValueError(f"Requested {count} sites, but only {total} lattice nodes exist.")
    flat = rng.choice(total, size=count, replace=False)
    k = flat // (Nx * Ny)
    r = flat % (Nx * Ny)
    j = r // Nx
    i = r % Nx
    return np.stack([i, j, k], axis=1).astype(np.int32)


def _choose_unique_lattice_sites(Nx: int, Ny: int, Nz: int, nbasis: int,
                                 count: int, rng: np.random.Generator) -> np.ndarray:
    """
Choose unique lattice sites including basis index without replacement.
    """

    total = Nx * Ny * Nz * nbasis
    if count <= 0:
        return np.empty((0, 4), dtype=np.int32)  # (i,j,k,basis_idx)
    if count > total:
        raise ValueError(f"Requested {count} sites, but only {total} lattice sites exist.")
    flat = rng.choice(total, size=count, replace=False)
    cell = flat // nbasis
    bidx = flat % nbasis
    k = cell // (Nx * Ny)
    r = cell % (Nx * Ny)
    j = r // Nx
    i = r % Nx
    return np.stack([i, j, k, bidx], axis=1).astype(np.int32)


def _append_substitutional_random(cfg: Config, sp: Species, taken: set):
    """
Append random substitutional positions avoiding collisions.
    """

    if sp.fraction <= 0.0:
        return
    basis, _ = _basis_and_interstitials(cfg.lattice)
    nbasis = int(basis.shape[0])
    total_sites = cfg.Nx * cfg.Ny * cfg.Nz * nbasis
    requested_total = int(round(float(sp.fraction) * float(total_sites)))

    # reserve already fixed positions (round to 1/8 for safe hashing)
    for p in sp.positions:
        t = tuple(int(x) for x in np.rint(np.asarray(p, dtype=np.float32)*8.0))
        taken.add(t)

    # Existing positions may be intentional fixed sites or positions written by
    # ``--dump-config``. Fill only the remainder so loading and dumping the same
    # configuration repeatedly cannot double the substitutional population.
    need = max(0, requested_total - len(sp.positions))
    if need <= 0:
        return

    rng = _rng(cfg.random_seed)
    picks = []
    tries = 0
    while len(picks) < need:
        tries += 1
        if tries > 10000:
            break
        batch = max(need * 2, 2048)
        cand = _choose_unique_lattice_sites(cfg.Nx, cfg.Ny, cfg.Nz, nbasis,
                                            min(batch, total_sites), rng)
        for (ii, jj, kk, bb) in cand:
            pos = np.asarray([ii, jj, kk], dtype=np.float32) + basis[bb]
            key = tuple(int(x) for x in np.rint(pos*8.0))
            if key in taken:
                continue
            taken.add(key)
            picks.append(tuple(float(x) for x in pos))
            if len(picks) >= need:
                break
    sp.positions = list(sp.positions) + picks[:need]


def _append_interstitial_random(cfg: Config, sp: Species):
    """
    Place exactly one interstitial per randomly chosen unit cell.
    Site choice is randomized per atom among legal interstitials for the lattice.
    If interstitial_site names a catalogue family, restrict to that family;
    otherwise "any" samples the union of all families. Avoid collisions with
    Fe basis and already occupied dopant positions.
    """
    remaining = max(0, int(sp.count) - len(sp.positions))
    if remaining <= 0:
        return

    rng = _rng(cfg.random_seed)
    basis, inter = _basis_and_interstitials(cfg.lattice)
    nbasis = int(basis.shape[0])

    # Build candidate offsets
    resolved_site = _resolve_lattice_setting(sp.interstitial_site, cfg.lattice)
    site_key = str(resolved_site or "any").strip().lower()
    if site_key in inter:
        families = inter.get(site_key, [])
    elif site_key == "any":
        families = sum((v for v in inter.values()), [])
    else:
        raise ValueError(
            f"{sp.name}: unknown interstitial family {site_key!r} for "
            f"{cfg.lattice}; use one of {sorted(inter)} or 'any'."
        )

    # If no catalogued sites, skip cleanly (no legacy numeric offset)
    if not families:
        print(f"[warn] no interstitial families defined for {cfg.lattice}; placed 0 for {sp.name}.")
        return

    # Use one representative per periodic site so shared faces/edges do not
    # bias random placement merely because the overlay draws them repeatedly.
    canonical_sites = _periodic_site_representatives(families, "canonical")
    sites = np.asarray(canonical_sites, dtype=np.float32)  # (S,3)

    # Hash of forbidden lattice positions (Fe basis + existing dopants) rounded to 1/8
    forbidden = set()
    # Existing dopant sites:
    for d in cfg.dopants:
        for p in d.positions:
            forbidden.add(tuple(int(x) for x in np.rint(np.asarray(p, dtype=np.float32)*8.0)))

    # Fill the requested total. An explicitly fixed site, when configured,
    # already occupies one of those slots rather than replacing the count.
    chosen = []
    tries = 0
    # Oversample cells in batches
    cells_batch = np.empty((0, 3), dtype=np.int32)

    while len(chosen) < remaining and tries < remaining * 50:
        tries += 1
        if cells_batch.size == 0:
            # get a new batch of unique cells
            need = max(128, remaining - len(chosen))
            cells_batch = _choose_unique_sites(cfg.Nx, cfg.Ny, cfg.Nz, min(need, cfg.Nx*cfg.Ny*cfg.Nz), rng)

        cell = cells_batch[-1].astype(np.float32); cells_batch = cells_batch[:-1]
        off = sites[rng.integers(0, sites.shape[0])]
        pos = cell + off
        key = tuple(int(x) for x in np.rint(pos * 8.0))

        # Reject if conflicts with existing dopants:
        if key in forbidden:
            continue
        # Reject if hits ANY Fe lattice site (this cell or a neighbor):
        # For some basis vector b, (pos - basis[b]) must be integral.
        clash = False
        pos64 = pos.astype(np.float64, copy=False)
        for b in range(nbasis):
            d = pos64 - basis[b].astype(np.float64, copy=False)
            r = d - np.rint(d)
            if np.all(np.abs(r) < 1e-6):
                clash = True
                break
        if clash:
            continue
        forbidden.add(key)
        chosen.append(tuple(float(x) for x in pos))

    if len(chosen) < remaining:
        # We placed as many as possible without collision
        placed_total = len(sp.positions) + len(chosen)
        print(
            f"[warn] placed {placed_total} / {sp.count} interstitials "
            f"for {sp.name} due to site conflicts."
        )

    sp.positions = list(sp.positions) + chosen


def _assign_forced_interstitial_position(cfg: Config, sp: Species) -> bool:
    """Validate and assign an exact catalogued interstitial position."""

    raw_position = _resolve_lattice_setting(
        sp.forced_interstitial_position,
        cfg.lattice,
    )
    if raw_position is None:
        if isinstance(sp.forced_interstitial_position, dict):
            resolved_site = _resolve_lattice_setting(
                sp.interstitial_site,
                cfg.lattice,
            )
            print(
                f"[info] {sp.name}: no forced interstitial position is set "
                f"for {cfg.lattice}; selecting a random catalogued "
                f"{resolved_site or 'interstitial'} site."
            )
        return False
    if sp.mode != "interstitial":
        raise ValueError(
            f"{sp.name}: forced_interstitial_position is only valid for "
            "mode: 'interstitial'."
        )

    position = np.asarray(raw_position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            f"{sp.name}: forced_interstitial_position must contain three "
            "finite fractional coordinates, either directly or below the "
            f"{cfg.lattice} lattice key."
        )

    cell_limits = np.asarray([cfg.Nx, cfg.Ny, cfg.Nz], dtype=float)
    if np.any(position < 0.0) or np.any(position > cell_limits):
        raise ValueError(
            f"{sp.name}: forced interstitial {position.tolist()} lies outside "
            f"the configured {cfg.Nx} x {cfg.Ny} x {cfg.Nz} cells."
        )

    basis, inter = _basis_and_interstitials(cfg.lattice)
    resolved_site = _resolve_lattice_setting(sp.interstitial_site, cfg.lattice)
    site_key = str(resolved_site or "any").strip().lower()
    if site_key in inter:
        allowed_sites = inter.get(site_key, [])
    elif site_key == "any":
        allowed_sites = sum((sites for sites in inter.values()), [])
    else:
        raise ValueError(
            f"{sp.name}: unknown interstitial family {site_key!r} for "
            f"{cfg.lattice}; use one of {sorted(inter)} or 'any'."
        )
    if not allowed_sites:
        raise ValueError(
            f"{sp.name}: no {site_key!r} interstitial sites are defined for "
            f"{cfg.lattice}."
        )

    # Compare modulo whole-cell translations so a periodic image at 1 is
    # equivalent to its canonical coordinate at 0.
    is_catalogued = False
    for site in allowed_sites:
        delta = position - np.asarray(site, dtype=float)
        delta -= np.rint(delta)
        if np.all(np.abs(delta) < 1e-6):
            is_catalogued = True
            break
    if not is_catalogued:
        raise ValueError(
            f"{sp.name}: forced interstitial {position.tolist()} is not a "
            f"catalogued {site_key} site in {cfg.lattice}."
        )

    # Reject a coordinate that is periodically equivalent to an Fe basis site.
    for base_site in basis:
        delta = position - np.asarray(base_site, dtype=float)
        delta -= np.rint(delta)
        if np.all(np.abs(delta) < 1e-6):
            raise ValueError(
                f"{sp.name}: forced interstitial {position.tolist()} overlaps "
                "an Fe lattice site."
            )

    sp.positions = [tuple(float(value) for value in position)]
    return True


def assign_random_positions(cfg: Config, dopants: List[Species]) -> None:
    """
Populate dopants with random positions per their mode and counts.
    """

    taken_sub = set()
    for sp in dopants:
        if sp.mode == "substitutional":
            _append_substitutional_random(cfg, sp, taken_sub)
        elif _assign_forced_interstitial_position(cfg, sp):
            _append_interstitial_random(cfg, sp)
        else:
            _append_interstitial_random(cfg, sp)


# ------------------ Geometry builders ------------------
def glyph_spheres(points_world: np.ndarray, radius: float, theta: int, phi: int) -> pv.PolyData:
    """
Build a sphere glyph for each point (true geometry for exports/small scenes).
    """

    if points_world.size == 0:
        return pv.PolyData()
    sphere = pv.Sphere(radius=radius, theta_resolution=theta, phi_resolution=phi)
    cloud = pv.PolyData(points_world)
    return cloud.glyph(geom=sphere, scale=False, orient=False)


def add_points_impostor(pl: pv.Plotter, points_world: np.ndarray, color: str, size_px: float):
    """
Render fast point impostors as spheres for huge scenes.
    """

    if points_world.size == 0:
        return None
    cloud = pv.PolyData(points_world)
    return pl.add_points(cloud, color=color, render_points_as_spheres=True, point_size=size_px)


# ------------------ Instanced rendering helpers ------------------
def adaptive_base_res(n_atoms: int, cfg: Config) -> Tuple[int, int]:
    """
Use the configured sphere resolution for smaller scenes, then cap it as atom
count increases to keep large instanced scenes responsive.
    """

    configured = (max(3, int(cfg.sphere_theta)), max(3, int(cfg.sphere_phi)))
    if not cfg.adaptive_resolution:
        return configured
    if n_atoms >= cfg.res_thresh_3:
        cap = max(3, int(cfg.res_cap_3))
        return min(configured[0], cap), min(configured[1], cap)
    if n_atoms >= cfg.res_thresh_2:
        cap = max(3, int(cfg.res_cap_2))
        return min(configured[0], cap), min(configured[1], cap)
    if n_atoms >= cfg.res_thresh_1:
        cap = max(3, int(cfg.res_cap_1))
        return min(configured[0], cap), min(configured[1], cap)
    return configured


def _color_to_rgb01(c) -> Tuple[float, float, float]:
    """
Convert color spec to (r,g,b) floats in [0,1].
    """

    try:
        from pyvista.plotting.colors import Color
        col = Color(c)
        r, g, b = col.float_rgb
        return float(r), float(g), float(b)
    except Exception:
        pass
    try:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(c)
        return float(r), float(g), float(b)
    except Exception:
        pass
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        try:
            r = int(c[1:3], 16) / 255.0
            g = int(c[3:5], 16) / 255.0
            b = int(c[5:7], 16) / 255.0
            return float(r), float(g), float(b)
        except Exception:
            pass
    return 0.5, 0.5, 0.5


def _register_png_scalable_text(plotter: pv.Plotter, actor) -> None:
    """Track a 2D text actor whose pixel size should follow png_scale."""

    actors = getattr(plotter, "_png_scalable_text_actors", None)
    if actors is None:
        actors = []
        plotter._png_scalable_text_actors = actors
    actors.append(actor)


def _scale_viewport_text_for_png(plotter: pv.Plotter, scale: int) -> None:
    """Preserve on-page text proportions in a scaled high-resolution PNG."""

    scale = max(1, int(scale))
    if scale == 1:
        return

    for actor in getattr(plotter, "_png_scalable_text_actors", []):
        text_property = actor.GetTextProperty()
        text_property.SetFontSize(
            max(1, int(round(text_property.GetFontSize() * scale)))
        )

    cube_axes = getattr(plotter, "_numbered_axes_actor", None)
    if cube_axes is not None:
        cube_axes.SetScreenSize(float(cube_axes.GetScreenSize()) * scale)
        for axis_index in range(3):
            for text_property in (
                cube_axes.GetTitleTextProperty(axis_index),
                cube_axes.GetLabelTextProperty(axis_index),
            ):
                text_property.SetFontSize(
                    max(1, int(round(text_property.GetFontSize() * scale)))
                )

    corner_axes = getattr(plotter, "_corner_axes_actor", None)
    if corner_axes is not None:
        for caption_actor in (
            corner_axes.GetXAxisCaptionActor2D(),
            corner_axes.GetYAxisCaptionActor2D(),
            corner_axes.GetZAxisCaptionActor2D(),
        ):
            text_property = caption_actor.GetCaptionTextProperty()
            text_property.SetFontSize(
                max(1, int(round(text_property.GetFontSize() * scale)))
            )


def make_instanced_actor(points_world: np.ndarray, radius: float, color: str,
                         theta: int, phi: int, specular: float = 0.0,
                         ambient: float = 0.0, diffuse: float = 1.0,
                         opacity: float = 1.0):
    """
Create a VTK instanced glyph actor (hardware instancing path).
    """

    if not HAVE_GLYPH3D_MAPPER or points_world.size == 0:
        return None

    sphere = vtkSphereSource()
    sphere.SetRadius(float(radius))
    sphere.SetThetaResolution(int(theta))
    sphere.SetPhiResolution(int(phi))
    sphere.Update()

    pts = np.ascontiguousarray(points_world, dtype=np.float32)
    poly = pv.PolyData(pts)

    mapper = vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceConnection(sphere.GetOutputPort())
    mapper.ScalingOff()
    mapper.OrientOff()

    actor = vtkActor()
    actor.SetMapper(mapper)
    r, g, b = _color_to_rgb01(color)
    actor.GetProperty().SetColor(r, g, b)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetSpecular(float(specular))
    actor.GetProperty().SetAmbient(float(ambient))
    actor.GetProperty().SetDiffuse(float(diffuse))
    actor.GetProperty().SetOpacity(float(opacity))
    # The source is retained by the live renderer so the visual radius can be
    # changed without rebuilding the window or replacing the actor.
    actor._weldcraft_glyph_source = sphere
    return actor


def chunk_points_z(points_world: np.ndarray,
                   target: int, max_actors: int) -> List[np.ndarray]:
    """
Split points by Z into chunks to reduce driver load.
    """

    n = points_world.shape[0]
    if n == 0:
        return []
    num = int(np.ceil(n / max(1, target)))
    num = min(max(1, num), max_actors)
    if num == 1:
        return [points_world]

    order = np.argsort(points_world[:, 2])
    pts_sorted = points_world[order]
    chunks = []
    step = int(np.ceil(n / num))
    for s in range(0, n, step):
        e = min(n, s + step)
        chunks.append(pts_sorted[s:e])
    return chunks


def draw_unit_cell_overlay(pl: pv.Plotter, cfg: Config):
    """Draw a single conventional cell wireframe at the origin + example site markers."""

    # wireframe cube (0..a on each axis)
    a = float(cfg.a)

    edges = [
        ((0,0,0), (a,0,0)), ((0,a,0), (a,a,0)), ((0,0,a), (a,0,a)), ((0,a,a), (a,a,a)),  # x edges
        ((0,0,0), (0,a,0)), ((a,0,0), (a,a,0)), ((0,0,a), (0,a,a)), ((a,0,a), (a,a,a)),  # y edges
        ((0,0,0), (0,0,a)), ((a,0,0), (a,0,a)), ((0,a,0), (0,a,a)), ((a,a,0), (a,a,a)),  # z edges
    ]

    if getattr(cfg, "draw_bravais_overlay", True):

        L = (cfg.lattice or "Simple Cubic").strip().lower()
        mid = a * 0.5

        if L == "fcc":
            # Face centers: connect each face center to its four face corners
            faces = [
                # z = 0 face
                ((mid, mid, 0.0), (0.0, 0.0, 0.0), (a, 0.0, 0.0), (0.0, a, 0.0), (a, a, 0.0)),
                # z = a face
                ((mid, mid, a),   (0.0, 0.0, a),   (a, 0.0, a),   (0.0, a, a),   (a, a, a)),
                # y = 0 face
                ((mid, 0.0, mid), (0.0, 0.0, 0.0), (a, 0.0, 0.0), (0.0, 0.0, a), (a, 0.0, a)),
                # y = a face
                ((mid, a,   mid), (0.0, a,   0.0), (a, a,   0.0), (0.0, a,   a), (a, a,   a)),
                # x = 0 face
                ((0.0, mid, mid), (0.0, 0.0, 0.0), (0.0, a,   0.0), (0.0, 0.0, a), (0.0, a,   a)),
                # x = a face
                ((a,   mid, mid), (a,   0.0, 0.0), (a,   a,   0.0), (a,   0.0, a), (a,   a,   a)),
            ]
            for fc, c1, c2, c3, c4 in faces:
                edges.append((fc, c1))
                edges.append((fc, c2))
                edges.append((fc, c3))
                edges.append((fc, c4))

        elif L == "bcc":
            # Body center: connect to all eight corners
            center = (mid, mid, mid)
            corners = [
                (0.0, 0.0, 0.0), (a, 0.0, 0.0), (0.0, a, 0.0), (a, a, 0.0),
                (0.0, 0.0, a), (a, 0.0, a), (0.0, a, a), (a, a, a),
            ]
            for c in corners:
                edges.append((center, c))

        else:
            # Simple Cubic (or unknown): no extra crosses, just the cube
            pass

    for p0, p1 in edges:
        pl.add_mesh(pv.Line(p0, p1), color=cfg.overlay_color, opacity=cfg.overlay_alpha,
                    render_lines_as_tubes=False, line_width=2)

    # markers: pick a few representative sites (not all of them)
    _, inter = _basis_and_interstitials(cfg.lattice)

    # families
    octa = inter.get("octa", [])
    tetra = inter.get("tetra", [])
    cubic = inter.get("cubic", [])

    # Select all closed-cell markers, canonical representatives, or equivalent
    # representatives chosen for visibility from the configured camera.
    site_view = (
        getattr(cfg, "interstitial_site_view", None)
        or getattr(cfg, "overlay_periodic", "both_faces")
    )
    site_view = str(site_view).strip().lower()
    picture_faces = _picture_site_faces_for_lattice(cfg)
    octa = _periodic_site_representatives(octa, site_view, picture_faces)
    tetra = _periodic_site_representatives(tetra, site_view, picture_faces)
    cubic = _periodic_site_representatives(cubic, site_view, picture_faces)

    # An occupied interstitial replaces its candidate marker. Keeping both
    # spheres at the same coordinate creates a misleading extra site and can
    # also cause transparency/depth artefacts around the occupied atom.
    occupied_sites = []
    for species in getattr(cfg, "dopants", []):
        if species.mode != "interstitial":
            continue
        for position in species.positions:
            occupied_sites.append(np.mod(np.asarray(position, dtype=float), 1.0))

    def _remove_occupied_markers(frac_pts):
        remaining = []
        for point in frac_pts:
            canonical_point = np.mod(np.asarray(point, dtype=float), 1.0)
            if any(
                np.allclose(canonical_point, occupied, atol=1e-6)
                for occupied in occupied_sites
            ):
                continue
            remaining.append(point)
        return remaining

    octa = _remove_occupied_markers(octa)
    tetra = _remove_occupied_markers(tetra)
    cubic = _remove_occupied_markers(cubic)

    # scale marker spheres small vs base spheres
    r_mark = float(cfg.base_radius) * float(cfg.overlay_marker_scale)

    def _add_markers(frac_pts, color, label):
        """
Def '_add_markers'.
        """

        if not frac_pts:
            return None
        lat = np.vstack(frac_pts).astype(np.float32)
        world = world_from_lattice(lat, cfg.a)
        mesh = glyph_spheres(world, r_mark, cfg.sphere_theta, cfg.sphere_phi)
        if mesh is not None and mesh.n_points:
            if hasattr(pl, "_weldcraft_radius_meshes"):
                pl._weldcraft_radius_meshes.append((mesh, world.copy()))
            pl.add_mesh(
                mesh,
                color=color,
                smooth_shading=True,
                opacity=float(cfg.overlay_marker_opacity),
                specular=float(cfg.overlay_marker_specular),
                ambient=float(cfg.sphere_ambient),
                diffuse=float(cfg.sphere_diffuse),
            )
            # add one label near the first marker for a clean legend
            pl.add_point_labels([tuple(world[0])], [label], show_points=False,
                                text_color="black", font_size=14, always_visible=True,
                                fill_shape=True, shape_opacity=0.7)
        return mesh

    # Do not add a separate Fe marker: an actual basis atom already occupies
    # that position. A duplicate is invisible for opaque atoms but appears as
    # a misleading smaller atom inside translucent Fe shells.
    _add_markers(octa, cfg.octahedral_color, "")
    _add_markers(tetra, cfg.tetrahedral_color, "")
    _add_markers(cubic, cfg.cubic_color, "")

    # ----- Structured legend (always shown when overlay is on) -----
    legend_rows = [
        ("Host Lattice (Fe)", cfg.base_color, False),
        ("Interstitials:", None, True),
    ]
    if tetra:
        legend_rows.append(("Tetrahedral", cfg.tetrahedral_color, False))
    if octa:
        legend_rows.append(("Octahedral", cfg.octahedral_color, False))
    if cubic:
        legend_rows.append(("Cubic", cfg.cubic_color, False))

    # Append any dopants that are actually present (placed positions > 0)
    present = []
    for d in getattr(cfg, "dopants", []):
        n = len(getattr(d, "positions", []) or [])
        if n > 0:
            color = (getattr(d, "color", None) or "black")
            present.append((f"Occupied ({d.name})", color))

    # Stable order for readability
    present.sort(key=lambda x: x[0].lower())
    legend_rows.extend((label, color, False) for label, color in present)

    if not getattr(cfg, "show_overlay_legend", True):
        return

    # vtkLegendBoxActor applies each entry color to both its marker and text.
    # Separate 2D bullet/text actors retain colored dots while keeping all
    # wording—including the light-grey Fe entry—a readable dark neutral.
    legend_loc = str(getattr(cfg, "overlay_legend_loc", "upper right")).lower()
    on_right = "right" in legend_loc
    on_upper = "upper" in legend_loc
    base_x = 0.735 if on_right else 0.035
    x_start = min(0.95, max(0.0, base_x + float(cfg.overlay_legend_x_offset)))
    y_start = 0.925 if on_upper else 0.285
    row_step = 0.052
    font_size = max(1, int(cfg.overlay_legend_font_size))
    viewport_width = max(1, int(cfg.window_size[0]))
    text_gap = 0.012 + max(0, int(cfg.overlay_legend_padding)) / viewport_width

    for index, (label, bullet_color, heading) in enumerate(legend_rows):
        y_pos = y_start - index * row_step
        row_x = x_start
        if bullet_color is not None:
            bullet_actor = pl.add_text(
                "\u2022",
                position=(row_x, y_pos),
                font_size=font_size + 4,
                color=bullet_color,
                name=f"_overlay_legend_bullet_{index}",
                viewport=True,
            )
            _register_png_scalable_text(pl, bullet_actor)
            label_x = row_x + text_gap
        else:
            label_x = row_x

        # Fe and the heading use the configured neutral; interstitial families
        # and occupied atoms use the same color as their corresponding dot.
        text_color = (
            cfg.overlay_legend_text_color
            if index < 2
            else bullet_color
        )
        text_actor = pl.add_text(
            label,
            position=(label_x, y_pos),
            font_size=font_size,
            color=text_color,
            name=f"_overlay_legend_text_{index}",
            viewport=True,
        )
        if heading:
            text_actor.GetTextProperty().SetBold(True)
        _register_png_scalable_text(pl, text_actor)


# ------------------ Scene construction ------------------
def _remove_lattice_positions(base_pts_lat: np.ndarray, to_remove) -> np.ndarray:
    """Remove exact eighth-lattice positions from a lattice-center array."""

    if not to_remove or not base_pts_lat.size:
        return base_pts_lat

    # Pack the eighth-lattice coordinates into one compact integer and let
    # NumPy perform the membership test in compiled code. The former tuple/set
    # loop was slow and memory-heavy for hundreds of thousands of dopants.
    base_keys = np.rint(base_pts_lat * 8.0).astype(np.int64, copy=False)
    remove_keys = np.rint(np.asarray(to_remove, dtype=np.float32) * 8.0).astype(
        np.int64, copy=False
    )
    minimum = np.minimum(base_keys.min(axis=0), remove_keys.min(axis=0))
    maximum = np.maximum(base_keys.max(axis=0), remove_keys.max(axis=0))
    spans = maximum - minimum + 1

    def _pack(keys):
        shifted = keys - minimum
        return (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2] + shifted[:, 2]

    packed_base = _pack(base_keys)
    packed_remove = np.unique(_pack(remove_keys))
    keep_mask = ~np.isin(
        packed_base,
        packed_remove,
        assume_unique=True,
        kind="sort",
    )
    return np.ascontiguousarray(base_pts_lat[keep_mask], dtype=np.float32)


def remove_base_sites_for_substitutionals(base_pts_lat: np.ndarray, dopants: List[Species]) -> np.ndarray:
    """Remove lattice centers occupied by substitutional species."""

    to_remove = []
    for sp in dopants:
        if sp.mode != "substitutional" or not sp.positions:
            continue
        to_remove.extend(sp.positions)
    return _remove_lattice_positions(base_pts_lat, to_remove)


def build_scene_points(cfg: Config):
    """
Def 'build_scene_points'.
    """

    demo_cell_active = getattr(cfg, "_demo_cell_active", False)
    if demo_cell_active:
        base_lat = _elemental_cell_positions(cfg.lattice)
        displayed_substitutionals = []
        for species in cfg.dopants:
            if species.mode == "substitutional" and species.positions:
                displayed_substitutionals.extend(
                    _single_cell_periodic_copies(species.positions).tolist()
                )
        base_lat = _remove_lattice_positions(base_lat, displayed_substitutionals)
    else:
        base_idx = generate_lattice_sites(cfg.Nx, cfg.Ny, cfg.Nz, cfg.lattice)
        base_idx = apply_stride_and_slab_indices(base_idx, cfg.stride, cfg.slab)
        base_lat = apply_manual_displacements(base_idx, cfg.base_displacements)
        base_lat = remove_base_sites_for_substitutionals(base_lat, cfg.dopants)

    base_world = world_from_lattice(base_lat, cfg.a)

    dop_meshes = []
    dopant_world_centers = []
    for sp in cfg.dopants:
        if not sp.positions:
            dop_meshes.append((None, sp)); continue
        pos_lat = np.asarray(sp.positions, dtype=np.float32)
        if demo_cell_active and sp.mode == "substitutional":
            # Corner and face sites have periodic copies on the opposite cell
            # boundaries. Display all copies so a substitution truly recolors
            # the one-cell lattice rather than sitting on top of a host atom.
            pos_lat = _single_cell_periodic_copies(pos_lat)
        elif (
            demo_cell_active
            and sp.mode == "interstitial"
            and str(cfg.interstitial_site_view or "").strip().lower() == "picture"
        ):
            # A boundary interstitial at 0 and its image at 1 are the same
            # periodic site. In picture mode, draw the camera-facing image.
            pos_lat = np.asarray(
                _periodic_site_representatives(
                    pos_lat,
                    "picture",
                    _picture_site_faces_for_lattice(cfg),
                ),
                dtype=np.float32,
            )
        dop_world = world_from_lattice(pos_lat, cfg.a)
        # True expanded sphere meshes preserve the historical small-scene
        # result, but scale catastrophically with a large concentration. Large
        # species are kept as centers and rendered through vtkGlyph3DMapper.
        dop_mesh = None
        if dop_world.shape[0] <= MAX_TRUE_DOPANT_SPHERES:
            dop_mesh = glyph_spheres(
                dop_world,
                sp.radius,
                cfg.sphere_theta,
                cfg.sphere_phi,
            )
        dop_meshes.append((dop_mesh, sp))
        dopant_world_centers.append((sp, dop_world, float(sp.radius)))

    return base_world, dop_meshes, dopant_world_centers


# ------------------ Interaction helpers ------------------
def _display_to_world(renderer, x, y):
    """Map display coords (x,y) at the Z-buffer to world coordinates."""
    renderer.SetDisplayPoint(x, y, 0)
    renderer.DisplayToWorld()
    wx, wy, wz, w = renderer.GetWorldPoint()
    if w == 0:
        return None
    return np.array([wx / w, wy / w, wz / w], dtype=float)


def enable_zoom_to_mouse(pl: pv.Plotter, sensitivity: float = 1.0):
    """
    Zoom toward the exact point under the cursor using a VTK picker.
    sensitivity > 1.0 : zoom-in step per wheel tick (backward uses 1/sensitivity).
    """
    ren = pl.renderer
    iren = pl.iren.interactor

    # -- robust picker creation that respects current imports --
    picker = None
    try:
        # if pyvista._vtk is available (second import branch), use that
        from pyvista import _vtk as _vtk  # type: ignore
        picker = _vtk.vtkCellPicker()
    except Exception:
        try:
            # fall back to vtkmodules (first import branch)
            from vtkmodules.vtkRenderingCore import vtkCellPicker
            picker = vtkCellPicker()
        except Exception:
            try:
                # last resort: a more generic picker
                from vtkmodules.vtkRenderingCore import vtkPropPicker
                picker = vtkPropPicker()
            except Exception:
                picker = None

    def _pick_world_point(x: int, y: int):
        """
Def '_pick_world_point'.
        """

        # precise pick first
        if picker is not None:
            try:
                if picker.Pick(x, y, 0.0, ren):
                    return np.array(picker.GetPickPosition(), dtype=float)
            except Exception:
                pass
        # fallback: project cursor ray into world (near plane)
        try:
            return _display_to_world(ren, x, y)
        except Exception:
            return None

    def _zoom_toward(target: np.ndarray, factor: float):

        cam = ren.GetActiveCamera()
        pos = np.array(cam.GetPosition(), dtype=float)
        foc = np.array(cam.GetFocalPoint(), dtype=float)

        # alpha = 1 - 1/factor -> smooth fraction toward target
        alpha = 1.0 - (1.0 / float(factor))
        cam.SetPosition(*(pos + (target - pos) * alpha))
        cam.SetFocalPoint(*(foc + (target - foc) * alpha))
        pl.render()

    pl.enable_trackball_style()

    def _on_wheel(obj, ev, forward: bool):
        # swallow VTK's default dolly (center-zoom)
        try:
            obj.AbortFlagOn()
        except Exception:
            pass

        x, y = iren.GetEventPosition()
        target = _pick_world_point(x, y)
        if target is None:
            return

        factor = float(sensitivity) if forward else (1.0 / float(sensitivity))
        # note: if sensitivity == 1.0, alpha==0 -> no motion
        _zoom_toward(target, factor)

    pl._weldcraft_zoom_observer_tags = (
        pl.iren.add_observer("MouseWheelForwardEvent", lambda o, e: _on_wheel(o, e, True)),
        pl.iren.add_observer("MouseWheelBackwardEvent", lambda o, e: _on_wheel(o, e, False)),
    )


# ------------------ Toast helpers (robust) ------------------
def restore_numbered_axes(plotter: pv.Plotter) -> None:
    """Restore the fixed zero-based label ranges after scene actor changes."""

    actor = getattr(plotter, "_numbered_axes_actor", None)
    ranges = getattr(plotter, "_numbered_axes_ranges", None)
    if actor is None or ranges is None:
        return
    try:
        actor.SetXAxisRange(float(ranges[0]), float(ranges[1]))
        actor.SetYAxisRange(float(ranges[2]), float(ranges[3]))
        actor.SetZAxisRange(float(ranges[4]), float(ranges[5]))
        if getattr(plotter, "_deduplicate_axis_zero_labels", False):
            # Retain X's origin label and suppress only the duplicate zero
            # strings on Y and Z. Nonzero labels are left untouched.
            for axis_index in (1, 2):
                labels = actor.GetAxisLabels(axis_index)
                if labels is None:
                    continue
                for index in range(labels.GetNumberOfValues()):
                    label = labels.GetValue(index).strip()
                    try:
                        is_zero = abs(float(label)) < 1e-12
                    except ValueError:
                        is_zero = False
                    if is_zero:
                        labels.SetValue(index, "")
                labels.Modified()
    except Exception:
        pass


def clear_toast(pl: pv.Plotter):
    """Remove any existing toast and its timer observer, if present."""
    # remove the actor
    actor = getattr(pl, "_toast_actor", None)
    if actor is not None:
        try:
            pl.remove_actor(actor)
        except Exception:
            pass
        finally:
            pl._toast_actor = None
    # remove the timer observer
    obs_id = getattr(pl, "_toast_timer_obs", None)
    if obs_id is not None:
        try:
            pl.iren.interactor.RemoveObserver(obs_id)
        except Exception:
            pass
        finally:
            pl._toast_timer_obs = None


def show_toast(pl: pv.Plotter, message: str, seconds: float = 5.0,
               position: str = "lower_right", font_size: int = 14,
               bg: Optional[str] = "yellow", bg_opacity: float = 0.9,
               frame: bool = False, frame_color: str = "black"):

    """
    Show a temporary HUD message for `seconds`, then auto-clear.
    Replaces any existing toast immediately.
    """
    # clear an existing toast before showing a new one
    clear_toast(pl)

    # add new text actor
    actor = pl.add_text(message, position=position, font_size=font_size,
                        color="black", name="_toast")
    pl._toast_actor = actor

    # optional background/frame
    try:
        tp = actor.GetTextProperty()
        if bg is not None:
            from pyvista.plotting.colors import Color
            r, g, b = Color(bg).float_rgb
            tp.SetBackgroundColor(r, g, b)
            tp.SetBackgroundOpacity(float(bg_opacity))
        if frame:
            from pyvista.plotting.colors import Color
            fr, fg, fb = Color(frame_color).float_rgb
            tp.SetFrame(1)
            tp.SetFrameColor(fr, fg, fb)
    except Exception:
        pass

    # start one-shot timer; store observer id so we can cancel/clear reliably
    iren = pl.iren.interactor
    ms = int(max(0.05, seconds) * 1000)
    timer_id = iren.CreateOneShotTimer(ms)

    def _on_timer(obj, ev):

        try:
            # Only react to our own timer if the API provides the id
            if hasattr(obj, "GetTimerEventId"):
                if obj.GetTimerEventId() != timer_id:
                    return
        except Exception:
            pass
        finally:
            # clear and detach in any case to avoid leaks
            clear_toast(pl)
            restore_numbered_axes(pl)
            try:
                obj.RemoveObserver(cid)
            except Exception:
                pass
            pl.render()

    cid = iren.AddObserver("TimerEvent", _on_timer)
    pl._toast_timer_obs = cid


def enable_picker(plotter: pv.Plotter, cfg: Config,
                  hydrogen_centers_world: np.ndarray, hydrogen_radius: float):

    """Right-click picking; detects H hits and shows a toast. Clears toast on misses."""
    # Instruction overlay (bottom-left)
    pick_help_actor = plotter.add_text(
        cfg.pick_instruction,
        position="upper_left",
        font_size=12,
        color="black",
        name="_pick_help",
    )
    _register_png_scalable_text(plotter, pick_help_actor)

    # Tolerance so users don't need a pixel-perfect hit
    tol = float(hydrogen_radius) * 1.2 if hydrogen_radius > 0 else 0.0

    def _on_pick(picked, *args):

        try:
            world = None
            # common PyVista pick payload paths
            try:
                arr = np.asarray(picked)
                if arr.shape == (3,):
                    world = arr.astype(float)
            except Exception:
                pass
            if world is None and hasattr(picked, "points"):
                pts = np.asarray(picked.points)
                if pts.size >= 3:
                    world = pts[0].astype(float)
            if world is None and hasattr(picked, "GetPickPosition"):
                world = np.array(picked.GetPickPosition(), dtype=float)
            if world is None:
                print(f"[pick] unknown payload type: {type(picked)}")
                # treat as miss: clear any existing toast
                clear_toast(plotter)
                restore_numbered_axes(plotter)
                plotter.render()
                return

            # lattice readout (still handy for debugging)
            lat = world / cfg.a
            ijk = tuple(np.rint(lat).astype(int))
            print(f"[pick] world={world}, lattice={lat}, nearest_index={ijk}")

            # H detection
            hit_h = False
            if tol > 0.0 and hydrogen_centers_world.size > 0:
                d = np.linalg.norm(hydrogen_centers_world - world[None, :], axis=1)
                hit_h = bool(np.any(d <= tol))

            if hit_h:
                show_toast(plotter, "You found a hydrogen atom, congrats!",
                           seconds=5.0, position="lower_right",
                           bg="yellow", bg_opacity=0.9, frame=True)
            else:
                # not hydrogen: clear any existing toast immediately
                clear_toast(plotter)
            restore_numbered_axes(plotter)
            plotter.render()

        except Exception as e:
            print(f"[pick] error: {e}")

    plotter.enable_point_picking(
        callback=_on_pick,
        use_picker=True,
        show_message=False,   # we show our own instruction text
        # PyVista's default pick marker becomes scene geometry and makes the
        # cube-axes actor recalculate its ranges from the marker bounds.
        show_point=False,
        left_clicking=False,  # right-click to pick
    )


# ------------------ Render ------------------
def _camera_snapshot(plotter: pv.Plotter):
    camera = plotter.camera
    return {
        "position": tuple(camera.GetPosition()),
        "focal_point": tuple(camera.GetFocalPoint()),
        "view_up": tuple(camera.GetViewUp()),
        "parallel": bool(camera.GetParallelProjection()),
        "view_angle": float(camera.GetViewAngle()),
    }


def _restore_camera_snapshot(plotter: pv.Plotter, snapshot) -> None:
    if not snapshot:
        return
    camera = plotter.camera
    camera.SetPosition(*snapshot["position"])
    camera.SetFocalPoint(*snapshot["focal_point"])
    camera.SetViewUp(*snapshot["view_up"])
    camera.SetParallelProjection(snapshot["parallel"])
    camera.SetViewAngle(snapshot["view_angle"])


def _update_base_radius_in_place(plotter: pv.Plotter, new_radius: float) -> bool:
    """Update the host-atom radius while retaining the existing scene/window."""

    old_radius = getattr(plotter, "_weldcraft_base_radius", None)
    if old_radius is None or old_radius <= 0 or new_radius <= 0:
        return False

    # Instanced spheres have a shared sphere source. Changing that source grows
    # every sphere around its own center and therefore preserves all lattice
    # coordinates.
    ratio = float(new_radius) / float(old_radius)
    changed = False
    for actor in getattr(plotter, "_weldcraft_base_glyph_actors", []):
        source = getattr(actor, "_weldcraft_glyph_source", None)
        if source is not None:
            source.SetRadius(float(source.GetRadius()) * ratio)
            source.Modified()
            changed = True

    # Outlined host atoms, dopants, and site markers use merged sphere meshes.
    # PyVista emits each source sphere's points as one contiguous block, so the
    # mesh can be resized around its recorded atom/site center without moving
    # that center or scaling the actor around the world origin.
    for mesh, centers in getattr(plotter, "_weldcraft_radius_meshes", []):
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
            continue
        points = np.asarray(mesh.points)
        if points.shape[0] % centers.shape[0] != 0:
            continue
        points_per_sphere = points.shape[0] // centers.shape[0]
        grouped = points.reshape(centers.shape[0], points_per_sphere, 3)
        resized = centers[:, None, :] + (grouped - centers[:, None, :]) * ratio
        mesh.points = resized.reshape(-1, 3)
        mesh.Modified()
        changed = True

    if changed:
        plotter._weldcraft_base_radius = float(new_radius)
        plotter.render()
    return changed


def _control_signature(path: Optional[str]):
    if not path:
        return None
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _install_live_control(plotter: pv.Plotter, config_path: Optional[str],
                          control_file: Optional[str]) -> None:
    """Poll the small command file used by the toolbox process."""

    if not config_path or not control_file or not hasattr(plotter, "iren"):
        return

    state = {"signature": _control_signature(control_file)}

    def _poll(_step=0):
        signature = _control_signature(control_file)
        if signature is None or signature == state["signature"]:
            return
        state["signature"] = signature
        try:
            with open(control_file, "r", encoding="utf-8") as handle:
                command = json.load(handle)
            if command.get("action") != "update":
                return

            new_cfg = load_config(config_path)
            apply_visual_preset(new_cfg)
            apply_camera_preset(new_cfg)
            normalize_physical_config(new_cfg)
            assign_random_positions(new_cfg, new_cfg.dopants)
            changed = set(command.get("changed", []))

            # Radius changes are frequent while dragging the basic slider, so
            # use the inexpensive actor/source update whenever possible.
            if changed == {"base_radius_scale"} and _update_base_radius_in_place(
                plotter, float(new_cfg.base_radius)
            ):
                plotter._weldcraft_live_config = new_cfg
                return

            snapshot = (
                None
                if changed.intersection(CAMERA_REFRAME_KEYS)
                else _camera_snapshot(plotter)
            )
            plot(
                new_cfg,
                export_dir=None,
                export_merged=None,
                screenshot=None,
                no_show=True,
                plotter=plotter,
                preserve_camera=snapshot,
            )
            plotter._weldcraft_live_config = new_cfg
        except Exception as exc:
            print(f"[warning] display update failed: {exc}")

    # Keep the poll callable on the plotter. The display loop below calls it
    # between PyVista event updates; this avoids version-dependent VTK timer
    # behavior on Windows.
    plotter._weldcraft_live_poll = _poll


def plot(cfg: Config, export_dir: Optional[str], export_merged: Optional[str],
         screenshot: Optional[str], no_show: bool, plotter: Optional[pv.Plotter] = None,
         config_path: Optional[str] = None, control_file: Optional[str] = None,
         preserve_camera=None):
    """
    Assemble scene, choose render path, handle overlay/picking, and show/export.
    """

    # On Windows, clicking a PyVista window's close button destroys the VTK
    # render window before a final screenshot callback can run. For "both"
    # mode, render the configured view to PNG first, then build a fresh normal-
    # resolution interactive plotter. This also keeps PNG-only font scaling out
    # of the displayed window.
    if screenshot and not no_show:
        plot(
            cfg,
            export_dir=export_dir,
            export_merged=export_merged,
            screenshot=screenshot,
            no_show=True,
        )
        print(f"[info] high-quality PNG saved: {screenshot}")
        plot(
            cfg,
            export_dir=None,
            export_merged=None,
            screenshot=None,
            no_show=False,
            config_path=config_path,
            control_file=control_file,
        )
        return

    # Build points and dopant meshes (+ centers for picking)
    base_world, dop_meshes, dopant_world_centers = build_scene_points(cfg)
    n_atoms = 0 if base_world is None else base_world.shape[0]

    # Decide rendering path
    use_impostor = (cfg.render_mode == "impostor_points")
    use_instanced = (cfg.render_mode in ("auto", "spheres")) and HAVE_GLYPH3D_MAPPER and (n_atoms > 0)

    # Build base mesh only for export and only if it's small enough
    base_mesh: Optional[pv.PolyData] = None
    if (export_dir or export_merged) and (n_atoms <= cfg.max_atoms_for_true_spheres):
        base_mesh = glyph_spheres(base_world, cfg.base_radius, cfg.sphere_theta, cfg.sphere_phi)

    # Reuse the existing plotter for live updates. Clearing actors is much
    # faster and, importantly, leaves the native render window and interactor
    # alive. The camera is restored by the caller for these updates.
    created_plotter = plotter is None
    pl = plotter or pv.Plotter(
        off_screen=no_show and (screenshot is not None),
        window_size=tuple(int(v) for v in cfg.window_size),
    )
    if not created_plotter:
        try:
            pl.disable_picking()
        except Exception:
            pass
        for tag in getattr(pl, "_weldcraft_zoom_observer_tags", ()):
            try:
                pl.iren.remove_observer(tag)
            except Exception:
                pass
        pl._weldcraft_zoom_observer_tags = ()
        pl.clear()
        pl._weldcraft_base_radius_actors = []
        pl._weldcraft_base_glyph_actors = []
        pl._weldcraft_radius_meshes = []
        pl._weldcraft_base_radius = float(cfg.base_radius)

    pl._weldcraft_base_radius = float(cfg.base_radius)
    pl._weldcraft_base_radius_actors = []
    pl._weldcraft_base_glyph_actors = []
    pl._weldcraft_radius_meshes = []
    pl.set_background(cfg.background)
    aa_type = str(cfg.anti_aliasing).strip().lower()
    if aa_type in ("", "none", "off", "false"):
        pl.disable_anti_aliasing()
    elif aa_type == "msaa":
        pl.enable_anti_aliasing(aa_type, multi_samples=int(cfg.multi_samples))
    else:
        pl.enable_anti_aliasing(aa_type)

    # Zoom mode switch
    if cfg.zoom_mode == "cursor":
        enable_zoom_to_mouse(pl, sensitivity=1.0)
    else:
        pl.enable_trackball_style()

    # Base atoms. The outline preset uses true geometry so PyVista can derive
    # camera-aware silhouettes; large scenes fall back to translucent instancing
    # rather than materializing an impractically large merged mesh.
    use_base_outlines = bool(
        cfg.base_atom_outline
        and not use_impostor
        and n_atoms <= cfg.max_atoms_for_outlines
    )
    if use_base_outlines:
        outlined_base = glyph_spheres(
            base_world,
            cfg.base_radius,
            cfg.sphere_theta,
            cfg.sphere_phi,
        )
        if outlined_base.n_points:
            pl._weldcraft_radius_meshes.append((outlined_base, base_world.copy()))
            base_actor = pl.add_mesh(
                outlined_base,
                color=cfg.base_color,
                opacity=float(cfg.base_atom_opacity),
                smooth_shading=True,
                specular=cfg.sphere_specular,
                ambient=cfg.sphere_ambient,
                diffuse=cfg.sphere_diffuse,
            )
            pl._weldcraft_base_radius_actors.append(base_actor)
            silhouette_actor = pl.add_silhouette(
                outlined_base,
                color=cfg.base_atom_outline_color,
                line_width=float(cfg.base_atom_outline_width),
                opacity=1.0,
            )
            silhouette_mapper = silhouette_actor.GetMapper()
            silhouette_mapper.SetResolveCoincidentTopologyToPolygonOffset()
            silhouette_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(
                0.0,
                float(cfg.base_atom_outline_depth_offset),
            )
            silhouette_actor.GetProperty().SetRenderLinesAsTubes(
                bool(cfg.base_atom_outline_as_tubes)
            )
            pl._weldcraft_base_radius_actors.append(silhouette_actor)
    elif use_impostor:
        base_actor = add_points_impostor(pl, base_world, cfg.base_color, cfg.points_impostor_size)
        if base_actor is not None:
            pl._weldcraft_base_radius_actors.append(base_actor)
    elif use_instanced:
        theta, phi = adaptive_base_res(n_atoms, cfg)
        chunks = (chunk_points_z(base_world, cfg.chunk_target_atoms, cfg.chunk_max_actors)
                  if cfg.chunking_enabled and n_atoms > 0 else [base_world])
        for ch in chunks:
            actor = make_instanced_actor(
                ch,
                cfg.base_radius,
                cfg.base_color,
                theta,
                phi,
                specular=cfg.sphere_specular,
                ambient=cfg.sphere_ambient,
                diffuse=cfg.sphere_diffuse,
                opacity=cfg.base_atom_opacity,
            )
            if actor is not None:
                pl.renderer.AddActor(actor)
                pl._weldcraft_base_glyph_actors.append(actor)
    else:
        if n_atoms <= cfg.max_atoms_for_true_spheres:
            baked = glyph_spheres(base_world, cfg.base_radius, cfg.sphere_theta, cfg.sphere_phi)
            if baked is not None and baked.n_points:
                pl._weldcraft_radius_meshes.append((baked, base_world.copy()))
                base_actor = pl.add_mesh(
                    baked,
                    color=cfg.base_color,
                    opacity=float(cfg.base_atom_opacity),
                    smooth_shading=True,
                    specular=cfg.sphere_specular,
                    ambient=cfg.sphere_ambient,
                    diffuse=cfg.sphere_diffuse,
                )
                pl._weldcraft_base_radius_actors.append(base_actor)
        else:
            print("[info] vtkGlyph3DMapper not available; falling back to impostor points for large scene.")
            base_actor = add_points_impostor(pl, base_world, cfg.base_color, cfg.points_impostor_size)
            if base_actor is not None:
                pl._weldcraft_base_radius_actors.append(base_actor)

    # Small dopant populations retain true sphere meshes for exact historical
    # appearance. Large concentrations use the same GPU-instanced mapper as the
    # host lattice, avoiding multi-gigabyte expanded meshes.
    hydrogen_centers_world = np.empty((0, 3), dtype=float)
    hydrogen_radius = 0.0
    total_scene_atoms = n_atoms + sum(
        int(centers.shape[0]) for _species, centers, _radius in dopant_world_centers
    )
    dopant_theta, dopant_phi = adaptive_base_res(total_scene_atoms, cfg)
    for mesh, sp in dop_meshes:
        centers = next(
            (centers for species, centers, _radius in dopant_world_centers if species is sp),
            None,
        )
        if centers is None or not len(centers):
            continue
        if mesh is not None and mesh.n_points:
            if len(centers):
                pl._weldcraft_radius_meshes.append((mesh, np.asarray(centers).copy()))
            pl.add_mesh(
                mesh,
                color=sp.color,
                smooth_shading=True,
                specular=cfg.sphere_specular,
                ambient=cfg.sphere_ambient,
                diffuse=cfg.sphere_diffuse,
            )
        elif use_impostor or not HAVE_GLYPH3D_MAPPER:
            add_points_impostor(
                pl,
                np.asarray(centers),
                sp.color,
                max(1.0, float(cfg.points_impostor_size) * float(sp.size_scale)),
            )
        else:
            chunks = (
                chunk_points_z(centers, cfg.chunk_target_atoms, cfg.chunk_max_actors)
                if cfg.chunking_enabled
                else [centers]
            )
            for chunk in chunks:
                actor = make_instanced_actor(
                    np.asarray(chunk),
                    float(sp.radius),
                    sp.color,
                    dopant_theta,
                    dopant_phi,
                    specular=cfg.sphere_specular,
                    ambient=cfg.sphere_ambient,
                    diffuse=cfg.sphere_diffuse,
                    opacity=1.0,
                )
                if actor is not None:
                    pl.renderer.AddActor(actor)
                    pl._weldcraft_base_glyph_actors.append(actor)
    # Collect H centers for picking feedback
    for sp, centers_w, rad in dopant_world_centers:
        if sp.mode == "interstitial" and sp.name.lower().startswith("h") and centers_w.size:
            hydrogen_centers_world = centers_w
            hydrogen_radius = rad
            break  # assume one H species

    # Camera
    extent = np.array([cfg.Nx, cfg.Ny, cfg.Nz], dtype=np.float32) * np.float32(cfg.a)
    center = 0.5 * extent
    camera_extent = extent
    if (
        bool(cfg.camera_normalize_demo_atom_size)
        and getattr(cfg, "_demo_cell_active", False)
    ):
        # The old distance was proportional to lattice constant, which made
        # equal-radius Fe spheres look smaller in FCC than BCC. Use the BCC
        # hard-sphere spacing as a common camera reference in all one-cell
        # comparison views. Coordinates and numbered-axis values still retain
        # the active lattice's actual lattice constant.
        reference_a = 4.0 * float(cfg.r) / math.sqrt(3.0)
        camera_extent = (
            np.array([cfg.Nx, cfg.Ny, cfg.Nz], dtype=np.float32)
            * np.float32(reference_a)
        )
    dist = float(np.linalg.norm(camera_extent)) * float(cfg.camera_distance_scale)
    camera_direction = np.asarray(cfg.camera_direction, dtype=float)
    direction_norm = float(np.linalg.norm(camera_direction))
    if direction_norm <= 1e-12:
        raise ValueError("camera_direction must contain at least one non-zero value")
    camera_direction /= direction_norm
    camera_position = center.astype(float) + dist * camera_direction
    pl.camera.SetPosition(*camera_position)
    pl.camera.SetFocalPoint(*center)
    pl.camera.SetViewUp(*(float(v) for v in cfg.camera_view_up))
    pl.camera.SetParallelProjection(bool(cfg.camera_parallel_projection))
    pl.camera.SetViewAngle(float(cfg.camera_view_angle))

    # Axes
    if cfg.show_axes:
        pl._corner_axes_actor = pl.add_axes()  # corner XYZ triad

    # Unit-cell overlay & site legend (optional)
    if cfg.show_unit_cell_overlay:
        draw_unit_cell_overlay(pl, cfg)

    # Picking (right click)
    if cfg.enable_picking and (not no_show or not created_plotter):
        enable_picker(pl, cfg, hydrogen_centers_world, hydrogen_radius)

    # Numbered axes with tick marks (math-style). Add this after all geometry so
    # later actors cannot make PyVista replace the zero-based display range with
    # the negative/positive sphere-surface bounds.
    if cfg.show_axes:
        # Substitutional atoms are still lattice sites. Include them when
        # determining the coordinate span so the axes remain stable as host
        # atoms are recolored, including the valid 100%-substitution case where
        # no host centers remain.
        lattice_center_parts = [base_world] if base_world.size else []
        lattice_center_parts.extend(
            centers
            for species, centers, _radius in dopant_world_centers
            if species.mode == "substitutional" and centers.size
        )
        if lattice_center_parts:
            lattice_centers = np.vstack(lattice_center_parts)
            lengths = np.ptp(lattice_centers, axis=0)
        else:
            lengths = extent.astype(float)
        Lx, Ly, Lz = (float(value) for value in lengths)

        # exact 0 on the left; round the max to avoid fp-noise
        def _rng(L): return (0.0, round(L, 6))

        bx = _rng(Lx)
        by = _rng(Ly)
        bz = _rng(Lz)
        bounds = (bx[0], bx[1], by[0], by[1], bz[0], bz[1])

        axes_actor = pl.show_bounds(
            axes_ranges=bounds,
            show_xaxis=True, show_yaxis=True, show_zaxis=True,
            xtitle="x [nm]", ytitle="y [nm]", ztitle="z [nm]",
            location=cfg.axis_location,
            ticks="outside",
            font_size=max(1, int(cfg.axis_font_size)),
            bold=True,
            fmt="%.2f",  # keep labels tidy; adjust precision if you like
            minor_ticks=False,
            use_3d_text=bool(cfg.axis_use_3d_text),
        )
        for line_property in (
            axes_actor.GetXAxesLinesProperty(),
            axes_actor.GetYAxesLinesProperty(),
            axes_actor.GetZAxesLinesProperty(),
        ):
            line_property.SetLineWidth(max(1.0, float(cfg.axis_line_width)))
        pl._numbered_axes_actor = axes_actor
        pl._numbered_axes_ranges = bounds
        pl._deduplicate_axis_zero_labels = bool(cfg.deduplicate_axis_zero_labels)
        restore_numbered_axes(pl)

    # Exports (meshes)
    export_all(base_mesh, dop_meshes, export_dir, export_merged)

    # Display and/or lossless PNG output. A scaled PNG increases saved
    # resolution without making the interactive window correspondingly huge.
    if screenshot:
        os.makedirs(os.path.dirname(screenshot) or ".", exist_ok=True)
        png_scale = max(1, int(cfg.png_scale))
        png_window_size = (
            max(1, int(cfg.window_size[0])) * png_scale,
            max(1, int(cfg.window_size[1])) * png_scale,
        )
        transparent = bool(cfg.png_transparent_background)
        _scale_viewport_text_for_png(pl, png_scale)
        pl.show(auto_close=False)
        pl.screenshot(
            screenshot,
            window_size=png_window_size,
            transparent_background=transparent,
        )
        pl.close()
    elif not no_show:
        if control_file and config_path:
            _install_live_control(pl, config_path, control_file)
        if control_file and config_path:
            # Non-blocking mode leaves the native window alive while this
            # process services both interaction events and toolbox updates.
            pl.show(auto_close=False, interactive_update=True)
            interactor = pl.iren.interactor
            # A plotter used for the configured PNG is closed immediately
            # before this interactive plotter is created. On Windows/VTK the
            # new interactor can occasionally inherit a completed state and
            # make the display loop exit as soon as the window appears.
            interactor.SetDone(0)
            mark_weldcraft_startup_ready()
            try:
                while interactor.GetDone() == 0:
                    pl._weldcraft_live_poll()
                    pl.update(25)
                    time.sleep(0.025)
            except Exception as exc:
                print(f"[info] display loop ended: {exc}")
            finally:
                try:
                    pl.close()
                except Exception:
                    pass
        else:
            mark_weldcraft_startup_ready()
            pl.show()
    elif not created_plotter:
        _restore_camera_snapshot(pl, preserve_camera)
        # Recalculate near/far clipping planes after every live rebuild. This
        # is essential when switching between a large lattice and a one-cell
        # example: retaining the former scene's clipping distances can hide
        # the much smaller replacement even though its camera and focal point
        # are otherwise correct.
        pl.reset_camera_clipping_range()
        pl.render()
    else:
        pl.close()


# ------------------ Startup summary (optional) ------------------
def print_startup_summary(config_path: Optional[str], cfg: Config,
                          export_dir: Optional[str], export_merged: Optional[str],
                          screenshot: Optional[str], no_show: bool):
    """
Print a human-readable summary of the current run.
    """

    est_atoms = (cfg.Nx * cfg.Ny * cfg.Nz) // max(1, (cfg.stride ** 3))

    # Compute actual counts after random assignment has happened
    base_count, sub_counts, int_counts, total = _compute_counts(cfg)

    if getattr(cfg, "_demo_cell_active", False):
        print("mode:          elemental crystal (single conventional cell)")
    print("----- Lattice Viewer -----")
    print(f"config:        {config_path or '(built-in defaults)'}")
    print(f"lattice:       {cfg.lattice} | a = {cfg.a} nm | r = {cfg.r} nm")
    print(f"size (cells):  {cfg.Nx} x {cfg.Ny} x {cfg.Nz}  (target_atoms >= {cfg.target_atoms})")
    print(f"base radius:   {cfg.base_radius}, color: {cfg.base_color}")
    print(f"dopants:       {[d.name for d in cfg.dopants if d.positions] or 'none'}")
    print(f"render_mode:   {cfg.render_mode}")
    print(f"visual preset: {cfg.visual_preset}")
    print(f"site view:     {cfg.interstitial_site_view or cfg.overlay_periodic}")
    print(f"atoms:         base={base_count}, substitutionals={sub_counts or {}}, interstitials={int_counts or {}}, total={total}")
    print(f"zoom_mode:     {cfg.zoom_mode}")
    if cfg.slab: print(f"slab z-range:  {cfg.slab}")
    if cfg.stride and cfg.stride > 1: print(f"stride:        {cfg.stride}")
    if cfg.chunking_enabled:
        print(f"chunking:      target={cfg.chunk_target_atoms}, max_actors={cfg.chunk_max_actors}, axis={cfg.chunk_axis}")
    print(f"export dir:    {export_dir or '-'}")
    print(f"export merged: {export_merged or '-'}")
    print(f"screenshot:    {screenshot or '-'}")
    print(f"no_show:       {no_show}")
    print("--------------------------")


# ------------------ CLI ------------------
def parse_args():
    """
Define/parse command-line arguments for the viewer.
    """

    p = argparse.ArgumentParser(description="Simple-cubic lattice visualizer (PyVista) with config + export")
    p.add_argument("--config", type=str, default=None, help="Path to a Python or JSON config override")
    p.add_argument("--dump-config", type=str, default=None, help="Write current config to a documented .py or .json file")
    p.add_argument("--export-dir", type=str, default=None, help="Directory to save base/species meshes as .vtp")
    p.add_argument("--export-merged", type=str, default=None, help="Path to save merged mesh (.vtp/.ply/.obj/.stl)")
    p.add_argument("--screenshot", type=str, default=None, help="Path to save a screenshot (PNG)")
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window (batch/export)")
    p.add_argument("--force-display", action="store_true", help="Open the interactive display even when config disables it")
    p.add_argument("--control-file", type=str, default=None,
                   help="Command file used by the toolbox for live display updates")
    return p.parse_args()


def main():
    """
Entry point wiring: config, normalization, placements, optional dump, run plot.
    """

    args = parse_args()

    config_path = args.config or guess_default_config()
    if config_path is None:
        config_path = str(ensure_config_file())
    cfg = load_config(config_path)
    apply_visual_preset(cfg)
    apply_camera_preset(cfg)

    screenshot = args.screenshot
    using_configured_png = screenshot is None and cfg.save_png
    if using_configured_png:
        screenshot = resolve_runtime_output_path(str(cfg.png_path))
        if cfg.png_include_lattice_name:
            screenshot = output_path_with_lattice_name(
                screenshot,
                cfg.lattice,
            )
    if screenshot is not None and cfg.png_avoid_overwrite:
        screenshot = next_available_output_path(screenshot)
    no_show = bool(args.no_show or (not args.force_display and not cfg.display_window))

    # Enforce physical sizing from (target_atoms, r, lattice)
    normalize_physical_config(cfg)

    # Assign random placements once (stable counts)
    assign_random_positions(cfg, cfg.dopants)

    if args.dump_config:
        dump_config(cfg, args.dump_config)
        print(f"config written to {args.dump_config}")

    print_startup_summary(config_path, cfg,
                          export_dir=args.export_dir,
                          export_merged=args.export_merged,
                          screenshot=screenshot,
                          no_show=no_show)

    plot(cfg,
         export_dir=args.export_dir,
         export_merged=args.export_merged,
         screenshot=screenshot,
         no_show=no_show,
         config_path=config_path,
         control_file=args.control_file)


if __name__ == "__main__":
    main()
