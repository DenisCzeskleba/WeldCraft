# P4 Hydrogen Permeation Flux

P4 creates publication-oriented black-and-white diagrams for recognizing how
one-dimensional hydrogen permeation curves respond to ideal transport
parameters, changing entry conditions, reversible kinetic traps, and residual
hydrogen left in a prepared specimen.

The module is deliberately independent from P1. P1 remains the interactive
general diffusion program; P4 is a focused numerical response-atlas and thesis
diagram generator.

## Layout and publication boundary

- `01_Resources/` contains the shipped configuration and diagram presets.
- `02_Results/` contains generated HDF5 data and diagram exports. All
  generated contents are local-only; only `.gitkeep` is published.
- `03_CodeBase/` contains the explicit solver, case builder, HDF5 support,
  diagram renderer, command-line entry point, and tests.

An optional local `03_CodeBase/config.py` can define a partial `CONFIG`
dictionary overriding `01_Resources/config_default.py`. The local configuration
is ignored by Git and is intended to become the persistent GUI configuration.

## Numerical model

The model uses normalized reference quantities

```text
tau_ref = L_ref^2 / D_ref
J_ref   = D_ref C_ref / L_ref
```

and advances the one-dimensional concentration field explicitly. The internal
time step is restricted by both the standard diffusion stability limit and the
fastest enabled trapping rate. Only requested output times are retained.

The shipped physical reference is a 0.5 mm membrane with
`D_ref = 6e-5 mm^2/s`, giving `tau_ref = 69.44 min`. Review diagrams use real
time in minutes and state these reference values on the plate. Their common
comparison window ends at `1.25 tau_ref = 86.81 min`; slower curves are not
forced to reach steady state inside the panel. Diagnostic shape comparisons
use `J/J_ss`, while amplitude comparisons use `J/J_ref`. Plots can instead show
the Fourier number `Fo = Dt/L^2`, per-curve `J/J_ss`, physical seconds, or
physical molar flux when a reference concentration is configured. The term
`J_max` is intentionally avoided because prefilled specimens can overshoot the
eventual steady flux.

### Kinetic trapping

P4 implements the McNabb-Foster kinetic capture/detrapping picture. Trap
capacity is expressed relative to `C_ref`; retention is configured through a
release half-time relative to `tau_ref`. The diagnostic plate varies these two
influences separately so storage and release kinetics are not conflated.

McNabb-Foster kinetics and Oriani local equilibrium are related but distinct.
The source documentation identifies both original references; Oriani
equilibrium is not implemented as a second solver mode in this version.

### Aged prefilling

The default prefill case begins uniformly at 20% of `C_ref`, applies zero
concentration at both surfaces, and ages the profile until its symmetric centre
peak reaches 10%. Experimental time is then reset and the normal entry/sink
permeation boundaries are applied. This creates a diffusion-generated central
bulge rather than imposing an arbitrary Gaussian.

## Running P4

Use the mandatory WeldCraft interpreter from the repository root:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --preset overview
```

Useful commands:

```powershell
# List shipped presets
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --list-presets

# Generate one plate and display it after saving
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --preset trapping --show

# Rerender a saved result without repeating the simulation
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --rerender '.\P4_Hydrogen Permeation Flux\02_Results\hydrogen_permeation_atlas.h5'
```

During diagram development, the `overview` preset creates the separate ideal,
entry-condition, and trapping diagnostic plates as PNG only. The prefill and
response-map presets remain available but are intentionally excluded from that
default review set. Final vector exports remain supported with
`--formats pdf,svg,png`. HDF5 preserves the underlying fields, flux histories,
configurations, and derived metrics for later GUI use.

The P0 launcher already names and describes P4, but intentionally presents an
informational message until the graphical interface is added.

## GUI feasibility and implementation plan

Adding a GUI is highly feasible without rewriting the numerical solver. P4
already exposes the important application boundaries: settings and preset
loading, case construction, HDF5 persistence, and figure rendering. The main
missing pieces are a settings writer/validator suitable for form input,
background execution and progress reporting, and a result viewer.

The planned implementation is:

1. Add a PyQt5 entry path to `permeation_atlas.py` with `--gui`; keep the
   current no-flag/`--cli` behavior and all existing CLI options unchanged.
2. Add a P4-local GUI support module beside the existing code. It will manage
   validated form settings, ignored runtime `config.py` persistence, safe
   result filenames, overwrite confirmation, and worker-to-GUI status events.
   Generic launcher signaling remains in `Resources/Common`.
3. Build the window around three workflows: **Run** (preset, numerical and
   diagram options), **Results** (load an existing P4 HDF5 file and select a
   figure/case), and **Export** (PNG/PDF/SVG, normalization, time axis, and
   response metric). The first version can preview the generated PNG exports;
   direct embedded Matplotlib plots can follow if interactive inspection is
   needed.
4. Run `build_atlas_cases`, `save_atlas_hdf5`, and `render_figures` in a
   `QThread` worker. The worker must never update Qt widgets directly. Add
   staged progress signals first; add cooperative cancellation to the solver
   loop as a follow-up once the basic run path is stable.
5. Bind the P0 launcher’s slot 4 to the new script with `--gui`, replacing the
   informational placeholder. The GUI will import
   `Common.launch_ready.StartupReadySignal`, create the main window, and emit
   the existing `WELDCRAFT_STARTUP_READY_FILE` handshake after the window is
   visible. P0 already injects that environment variable, monitors the child,
   minimizes/restores itself, and re-enables the slot after exit.
6. Test the integration at three levels: unchanged CLI and numerical tests;
   settings/HDF5/export tests for GUI-managed runs; and a launcher/headless
   Qt smoke test that verifies the ready-file handshake and clean child exit.

The public-facing name is now **P4 Hydrogen Permeation Flux**. The internal
script name `permeation_atlas.py`, result stem `hydrogen_permeation_atlas`, and
HDF5 format identifier remain stable for CLI and saved-file compatibility; the
word “atlas” continues to describe the collection of response plates rather
than the module’s display name.

## Tests

From `03_CodeBase` run:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  -m unittest -v test_permeation_atlas.py
```

The tests cover the ideal Fourier-series reference, convergence, explicit trap
conservation and bounds, surface histories, aged prefilling, normalization,
HDF5 round trips, presets, figure exports, CLI discovery, and launcher text.
