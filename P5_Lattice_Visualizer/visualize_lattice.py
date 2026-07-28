"""
Lattice Visualizer — SC/BCC/FCC with dopants (PyVista/VTK)

This script builds crystalline lattices at scale, optionally places dopants
(substitutional/interstitial), and renders them efficiently using GPU-instanced
glyphs when available. It also supports export (meshes/screenshots), unit-cell
overlays, picking (find Hydrogen), and a YAML/JSON-driven configuration.

Note: The only changes here are documentation strings and comments; no code logic
has been removed or added.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
import argparse
import json
import os
import sys
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
try:
    import yaml  # optional
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# --- Green-arrow / default config support ---
CONFIG_OVERRIDE: Optional[str] = None
DEFAULT_CONFIG_BASENAME = "User Input.yaml"
WELDCRAFT_READY_FILE_ENV_VAR = "WELDCRAFT_STARTUP_READY_FILE"


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
Search likely locations/env var for a default config file path.
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

    for name in ("User Input.yaml", "config.yml", "lattice.yaml", "lattice.yml"):
        for base in (Path.cwd(), runtime_directory()):
            cand = base / name
            if cand.exists():
                return str(cand)
    return None


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
    # YAML no longer provides positions; default to empty list
    positions: List[Tuple[float, float, float]] = field(default_factory=list)

    mode: str = "substitutional"         # "substitutional" | "interstitial"

    # Random placement controls:
    fraction: float = 0.0                 # for substitutional species (0..1)
    count: int = 0                        # for interstitial species (absolute)

    # Interstitial family (optional): "octa" | "tetra"; if None, we pick from all legal
    interstitial_site: Optional[str] = None

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
    sphere_theta: int = 32
    sphere_phi: int = 32
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
    res_thresh_1: int = 100_000
    res_thresh_2: int = 300_000
    res_thresh_3: int = 1_000_000

    # Unit-cell overlay & legend
    show_unit_cell_overlay: bool = False
    overlay_color: str = "black"
    overlay_alpha: float = 0.65
    overlay_marker_scale: float = 0.6  # as a fraction of cfg.base_radius
    draw_bravais_overlay: bool = True
    overlay_periodic: str = "both_faces"  # "both_faces" | "canonical"
    show_overlay_legend: bool = True  # show legend when unit-cell overlay is on
    overlay_legend_loc: str = "upper right"  # 'upper right' | 'upper left' | 'lower left' | 'lower right'

    demo_cell_auto: bool = True  # auto-activate if target_atoms <= threshold for the chosen lattice
    demo_cell_force: Optional[bool] = None  # set to True/False to override auto (None = auto)


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


def load_config(path: Optional[str]) -> Config:
    """
Load configuration from YAML/JSON and normalize legacy fields.
    """

    if path is None:
        return Config()
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            if not HAVE_YAML:
                print("pyyaml not installed; install it or use JSON.", file=sys.stderr)
                sys.exit(2)
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    dopants = []
    for d in raw.get("dopants", []):
        # Drop legacy key to avoid TypeError if present
        if isinstance(d, dict) and "interstitial_offset" in d:
            d = {k: v for k, v in d.items() if k != "interstitial_offset"}
        dopants.append(Species(**d))
    raw["dopants"] = dopants

    if "base_displacements" in raw:
        raw["base_displacements"] = _normalize_base_displacements(raw["base_displacements"])

    return Config(**raw)


def dump_config(cfg: Config, path: str):
    """
Write the current configuration to a YAML or JSON file.
    """

    d = asdict(cfg)
    if path.lower().endswith((".yaml", ".yml")):
        if not HAVE_YAML:
            raise RuntimeError("pyyaml not installed; cannot dump YAML.")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


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
            merged = pv.append_polydata(parts)
            save_mesh(merged, export_merged)


# ------------------ Lattice utilities ------------------
def _basis_and_interstitials(lattice: str):
    """
Return basis positions and catalogued interstitial sites for the lattice.
    """

    lat = (lattice or "Simple Cubic").strip().lower()
    if lat in ("simple cubic", "sc"):
        basis = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        # --- Interstitials for SC ---
        # Tetrahedral: the 8 (1/4,1/4,1/4)-type positions
        tetra = []
        for x in (0.25, 0.75):
            for y in (0.25, 0.75):
                for z in (0.25, 0.75):
                    tetra.append((x, y, z))

        # Cubic (CN=8): body center (½,½,½). (This is *not* octahedral.)
        cubic = [(0.5, 0.5, 0.5)]

        # No true octahedral (CN=6) sites in perfect SC at equal distances
        inter = {
            "octa": [],  # intentionally empty for SC
            "tetra": [np.array(p, dtype=np.float32) for p in tetra],
            "cubic": [np.array(p, dtype=np.float32) for p in cubic],}

    elif lat == "bcc":
        # Basis
        basis = np.array([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.5]], dtype=np.float32)
        # Octahedral in BCC: the 6 face centers
        octa = [(0.5, 0.5, 0.0), (0.5, 0.5, 1.0),
                (0.5, 0.0, 0.5), (0.5, 1.0, 0.5),
                (0.0, 0.5, 0.5), (1.0, 0.5, 0.5)]
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
        # Octahedral in FCC: 1 body center + 12 edge centers
        octa = [(0.5, 0.5, 0.5)]
        half = 0.5
        for axis in range(3):
            for u in (0.0, 1.0):
                p = [half, half, half]
                p[axis] = u
                octa.append(tuple(p))
        # that added 2*3 = 6 face centers; now add the 12 edge centers:
        # edges have exactly one coordinate at 0.5 and the other two at 0 or 1
        edges = []
        for axis in range(3):
            for u in (0.0, 1.0):
                for v in (0.0, 1.0):
                    p = [0.0, 0.0, 0.0]
                    p[axis] = 0.5
                    p[(axis + 1) % 3] = u
                    p[(axis + 2) % 3] = v
                    edges.append(tuple(p))
        # keep unique (a few duplicates appear when mixing definitions)
        octa = list({tuple(p) for p in (octa + edges)})
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
def _rng() -> np.random.Generator:
    """
Return a NumPy default random Generator.
    """

    return np.random.default_rng()


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
    need = int(round(float(sp.fraction) * float(total_sites)))
    if need <= 0:
        return

    # reserve already fixed positions (round to 1/8 for safe hashing)
    for p in sp.positions:
        t = tuple(int(x) for x in np.rint(np.asarray(p, dtype=np.float32)*8.0))
        taken.add(t)

    rng = _rng()
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
    If interstitial_site is "octa" or "tetra", restrict to that family;
    otherwise sample the union of all families. Avoid collisions with Fe basis
    and already occupied dopant positions.
    """
    if sp.count <= 0:
        return

    rng = _rng()
    basis, inter = _basis_and_interstitials(cfg.lattice)
    nbasis = int(basis.shape[0])

    # Build candidate offsets
    site_key = (sp.interstitial_site or "any").strip().lower()
    if site_key in ("octa", "tetra"):
        families = inter.get(site_key, [])
    else:
        families = sum((v for v in inter.values()), [])

    # If no catalogued sites, skip cleanly (no legacy numeric offset)
    if not families:
        print(f"[warn] no interstitial families defined for {cfg.lattice}; placed 0 for {sp.name}.")
        return

    sites = np.vstack(families).astype(np.float32, copy=False)  # (S,3)

    # Hash of forbidden lattice positions (Fe basis + existing dopants) rounded to 1/8
    forbidden = set()
    # Existing dopant sites:
    for d in cfg.dopants:
        for p in d.positions:
            forbidden.add(tuple(int(x) for x in np.rint(np.asarray(p, dtype=np.float32)*8.0)))

    # We'll greedily try cells until we place sp.count interstitials
    chosen = []
    tries = 0
    # Oversample cells in batches
    cells_batch = np.empty((0, 3), dtype=np.int32)

    while len(chosen) < sp.count and tries < sp.count * 50:
        tries += 1
        if cells_batch.size == 0:
            # get a new batch of unique cells
            need = max(128, sp.count - len(chosen))
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

    if len(chosen) < sp.count:
        # We placed as many as possible without collision
        print(f"[warn] placed {len(chosen)} / {sp.count} interstitials for {sp.name} due to site conflicts.")

    sp.positions = list(sp.positions) + chosen


def assign_random_positions(cfg: Config, dopants: List[Species]) -> None:
    """
Populate dopants with random positions per their mode and counts.
    """

    taken_sub = set()
    for sp in dopants:
        if sp.mode == "substitutional":
            _append_substitutional_random(cfg, sp, taken_sub)
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
        return
    cloud = pv.PolyData(points_world)
    pl.add_points(cloud, color=color, render_points_as_spheres=True, point_size=size_px)


# ------------------ Instanced rendering helpers ------------------
def adaptive_base_res(n_atoms: int, cfg: Config) -> Tuple[int, int]:
    """
Lower sphere resolution as atom count increases (performance heuristic).
    """

    if n_atoms >= cfg.res_thresh_3:
        return 8, 8
    if n_atoms >= cfg.res_thresh_2:
        return 12, 12
    if n_atoms >= cfg.res_thresh_1:
        return 16, 16
    return 24, 24


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


def make_instanced_actor(points_world: np.ndarray, radius: float, color: str,
                         theta: int, phi: int):
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
    actor.GetProperty().SetSpecular(0.2)
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
    basis, inter = _basis_and_interstitials(cfg.lattice)

    # families
    octa = inter.get("octa", [])
    tetra = inter.get("tetra", [])
    cubic = inter.get("cubic", [])

    # only collapse to canonical [0,1) if you ask for it
    if getattr(cfg, "overlay_periodic", "both_faces") == "canonical":
        def _unique_overlay_points(frac_pts):
            """
Def '_unique_overlay_points'.
            """

            if not frac_pts:
                return []
            pts = np.asarray(frac_pts, dtype=np.float32)
            pts = np.mod(pts, 1.0)  # map 1.0 → 0.0
            pts = np.round(pts, 6)
            uniq = np.unique(pts, axis=0)
            return [tuple(map(float, p)) for p in uniq]

        octa = _unique_overlay_points(octa)
        tetra = _unique_overlay_points(tetra)
        cubic = _unique_overlay_points(cubic)

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
        mesh = glyph_spheres(world, r_mark, 16, 16)
        if mesh is not None and mesh.n_points:
            pl.add_mesh(mesh, color=color, smooth_shading=True, opacity=0.95, specular=0.2)
            # add one label near the first marker for a clean legend
            pl.add_point_labels([tuple(world[0])], [label], show_points=False,
                                text_color="black", font_size=14, always_visible=True,
                                fill_shape=True, shape_opacity=0.7)
        return mesh

    # Fe lattice site marker (use the first basis site)
    _add_markers([basis[0]], cfg.base_color, "")
    _add_markers(octa, "orange", "")
    _add_markers(tetra, "green", "")
    _add_markers(cubic, "purple", "")

    # ----- Legend (always shown when overlay is on) -----
    L = (cfg.lattice or "Simple Cubic").strip().lower()
    if L in ("simple cubic", "sc") and cubic:
        labels = [
            ("Basis lattice site (Fe)", cfg.base_color),
            ("Tetrahedral interstitial site", "green"),
            ("Cubic interstitial site", "purple"),
        ]
    else:
        labels = [
            ("Basis lattice site (Fe)", cfg.base_color),
            ("Tetrahedral interstitial site", "green"),
            ("Octahedral interstitial site", "orange"),
        ]

    # Append any dopants that are actually present (placed positions > 0)
    present = []
    for d in getattr(cfg, "dopants", []):
        n = len(getattr(d, "positions", []) or [])
        if n > 0:
            mode = getattr(d, "mode", "substitutional")
            color = (getattr(d, "color", None) or "black")
            present.append((f"Occupied {mode} ({d.name})", color))

    # Stable order for readability
    present.sort(key=lambda x: x[0].lower())
    labels.extend(present)

    legend_loc = getattr(cfg, "overlay_legend_loc", "upper left")
    legend_size = (0.24, 0.20)

    try:
        # Prefer circular chips if your PyVista build supports it
        pl.add_legend(
            labels=labels,
            face="circle",  # chip shape (falls back below if unsupported)
            bcolor="w",
            size=legend_size,
            loc=legend_loc,
            background_opacity=0.7,
        )
    except TypeError:
        # Older builds: no 'face' kwarg — still works fine
        pl.add_legend(
            labels=labels,
            bcolor="white",
            border=True,
            size=legend_size,
            loc=legend_loc,
        )


# ------------------ Scene construction ------------------
def remove_base_sites_for_substitutionals(base_pts_lat: np.ndarray, dopants: List[Species]) -> np.ndarray:
    """
Def 'remove_base_sites_for_substitutionals'.
    """

    to_remove = []
    for sp in dopants:
        if sp.mode != "substitutional" or not sp.positions:
            continue
        for p in sp.positions:
            to_remove.append(tuple(int(x) for x in np.rint(np.asarray(p, dtype=np.float32)*8.0)))
    if not to_remove:
        return base_pts_lat
    remove_set = set(to_remove)
    keys = np.rint(base_pts_lat * 8.0).astype(np.int32, copy=False)
    keep_mask = np.array(
        [tuple(int(v) for v in xyz) not in remove_set for xyz in keys],
        dtype=bool
    )
    return np.ascontiguousarray(base_pts_lat[keep_mask], dtype=np.float32)


def build_scene_points(cfg: Config):
    """
Def 'build_scene_points'.
    """

    if getattr(cfg, "_demo_cell_active", False):
        base_lat = _elemental_cell_positions(cfg.lattice)
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
        dop_world = world_from_lattice(pos_lat, cfg.a)
        dop_mesh = glyph_spheres(dop_world, sp.radius, cfg.sphere_theta, cfg.sphere_phi)
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

    pl.iren.add_observer("MouseWheelForwardEvent", lambda o, e: _on_wheel(o, e, True))
    pl.iren.add_observer("MouseWheelBackwardEvent", lambda o, e: _on_wheel(o, e, False))


# ------------------ Toast helpers (robust) ------------------
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
    plotter.add_text(cfg.pick_instruction, position="upper_left",
                     font_size=12, color="black", name="_pick_help")

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
                plotter.render()

        except Exception as e:
            print(f"[pick] error: {e}")

    plotter.enable_point_picking(
        callback=_on_pick,
        use_picker=True,
        show_message=False,   # we show our own instruction text
        show_point=True,
        left_clicking=False,  # right-click to pick
    )


# ------------------ Render ------------------
def plot(cfg: Config, export_dir: Optional[str], export_merged: Optional[str],
         screenshot: Optional[str], no_show: bool):
    """
Assemble scene, choose render path, handle overlay/picking, and show/export.
    """

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

    # Plotter
    pl = pv.Plotter(off_screen=no_show and (screenshot is not None))
    pl.set_background(cfg.background)

    # Zoom mode switch
    if cfg.zoom_mode == "cursor":
        enable_zoom_to_mouse(pl, sensitivity=1.0)
    else:
        pl.enable_trackball_style()

    # Base atoms
    if use_impostor:
        add_points_impostor(pl, base_world, cfg.base_color, cfg.points_impostor_size)
    elif use_instanced:
        theta, phi = adaptive_base_res(n_atoms, cfg)
        chunks = (chunk_points_z(base_world, cfg.chunk_target_atoms, cfg.chunk_max_actors)
                  if cfg.chunking_enabled and n_atoms > 0 else [base_world])
        for ch in chunks:
            actor = make_instanced_actor(ch, cfg.base_radius, cfg.base_color, theta, phi)
            if actor is not None:
                pl.renderer.AddActor(actor)
    else:
        if n_atoms <= cfg.max_atoms_for_true_spheres:
            baked = glyph_spheres(base_world, cfg.base_radius, cfg.sphere_theta, cfg.sphere_phi)
            if baked is not None and baked.n_points:
                pl.add_mesh(baked, color=cfg.base_color, smooth_shading=True, specular=0.2)
        else:
            print("[info] vtkGlyph3DMapper not available; falling back to impostor points for large scene.")
            add_points_impostor(pl, base_world, cfg.base_color, cfg.points_impostor_size)

    # Dopants as true spheres
    hydrogen_centers_world = np.empty((0, 3), dtype=float)
    hydrogen_radius = 0.0
    for mesh, sp in dop_meshes:
        if mesh is not None and mesh.n_points:
            pl.add_mesh(mesh, color=sp.color, smooth_shading=True, specular=0.25)
    # Collect H centers for picking feedback
    for sp, centers_w, rad in dopant_world_centers:
        if sp.mode == "interstitial" and sp.name.lower().startswith("h") and centers_w.size:
            hydrogen_centers_world = centers_w
            hydrogen_radius = rad
            break  # assume one H species

    # Camera
    extent = np.array([cfg.Nx, cfg.Ny, cfg.Nz], dtype=np.float32) * np.float32(cfg.a)
    center = 0.5 * extent
    dist = float(np.linalg.norm(extent)) * 2.2
    pl.camera.SetPosition(center[0], 1.2 * center[1], center[2] + dist)
    pl.camera.SetFocalPoint(*center)
    pl.camera.SetViewUp(0, 1, 0)
    pl.camera.Azimuth(25)
    pl.camera.Elevation(20)

    # Axes
    if cfg.show_axes:
        pl.add_axes()  # corner XYZ triad

    # Numbered axes with tick marks (math-style)
        Lx = float(np.max(base_world[:, 0]) - np.min(base_world[:, 0]))
        Ly = float(np.max(base_world[:, 1]) - np.min(base_world[:, 1]))
        Lz = float(np.max(base_world[:, 2]) - np.min(base_world[:, 2]))

        # exact 0 on the left; round the max to avoid fp-noise
        def _rng(L): return (0.0, round(L, 6))

        bx = _rng(Lx)
        by = _rng(Ly)
        bz = _rng(Lz)
        bounds = (bx[0], bx[1], by[0], by[1], bz[0], bz[1])

        # build clean tick arrays so the first tick is exactly 0.0 (no "-0.0")
        def _ticks(L, n=5):

            vals = np.linspace(0.0, L, n)
            vals[np.isclose(vals, 0.0, atol=1e-12)] = 0.0
            return vals

        tx = _ticks(bx[1])
        ty = _ticks(by[1])
        tz = _ticks(bz[1])

        pl.show_bounds(
            bounds=bounds,
            show_xaxis=True, show_yaxis=True, show_zaxis=True,
            xtitle="x (nm)", ytitle="y (nm)", ztitle="z (nm)",
            location="outer",
            ticks="outside",  # <- explicit ticks; first is 0.0 exactly
            fmt="%.2f",  # keep labels tidy; adjust precision if you like
            minor_ticks=False,
        )

    # Unit-cell overlay & site legend (optional)
    if cfg.show_unit_cell_overlay:
        draw_unit_cell_overlay(pl, cfg)

    # Picking (right click)
    if cfg.enable_picking and not no_show:
        enable_picker(pl, cfg, hydrogen_centers_world, hydrogen_radius)

    # Exports (meshes)
    export_all(base_mesh, dop_meshes, export_dir, export_merged)

    # Show
    if screenshot:
        os.makedirs(os.path.dirname(screenshot) or ".", exist_ok=True)
        pl.show(screenshot=screenshot, auto_close=True)
    elif not no_show:
        mark_weldcraft_startup_ready()
        pl.show()


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
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config (optional)")
    p.add_argument("--dump-config", type=str, default=None, help="Write current config to file (YAML/JSON)")
    p.add_argument("--export-dir", type=str, default=None, help="Directory to save base/species meshes as .vtp")
    p.add_argument("--export-merged", type=str, default=None, help="Path to save merged mesh (.vtp/.ply/.obj/.stl)")
    p.add_argument("--screenshot", type=str, default=None, help="Path to save a screenshot (PNG)")
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window (batch/export)")
    return p.parse_args()


def main():
    """
Entry point wiring: config, normalization, placements, optional dump, run plot.
    """

    args = parse_args()

    config_path = args.config or guess_default_config()
    cfg = load_config(config_path) if config_path else Config()

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
                          screenshot=args.screenshot,
                          no_show=args.no_show)

    plot(cfg,
         export_dir=args.export_dir,
         export_merged=args.export_merged,
         screenshot=args.screenshot,
         no_show=args.no_show)


if __name__ == "__main__":
    main()
