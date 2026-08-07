# P4 Hydrogen Permeation Flux

P4 creates publication-oriented black-and-white diagrams for recognizing how
one-dimensional hydrogen permeation curves respond to ideal transport
parameters, changing entry conditions, reversible kinetic traps, and residual
hydrogen left in a prepared specimen. It provides both a persistent PyQt5 GUI
and a compatible command-line workflow.

P4 is deliberately independent from P1. P1 remains the interactive general
diffusion program; P4 is a focused numerical response-atlas and thesis diagram
generator.

## Layout and local state

- `01_Resources/` contains shipped defaults and protected figure presets.
- `02_Results/` contains local HDF5 data and PNG/PDF/SVG exports. Only
  `.gitkeep` is published.
- `03_CodeBase/` contains the solver, case builder, persistence, renderer, GUI,
  CLI entrypoint, and tests.

The ignored `03_CodeBase/config.py` is created automatically and stores:

- `CONFIG`: validated scientific, numerical, and diagram settings shared by
  the GUI and default CLI.
- `GUI_STATE`: output choices, last-result path, and window state.
- `USER_PROFILES`: named scientific and presentation profiles.

Only valid edits replace persistent state. A broken manually edited file can
be restored to shipped defaults from the GUI recovery dialog.

## Numerical model

The model uses normalized reference quantities

```text
tau_ref = L_ref^2 / D_ref
J_ref   = D_ref C_ref / L_ref
```

and advances the one-dimensional concentration field explicitly. The internal
time step is restricted by the diffusion stability limit and the fastest
enabled trapping rate. Only requested output times are retained.

The shipped reference is a 0.5 mm membrane with
`D_ref = 6e-5 mm^2/s`, giving `tau_ref = 69.44 min`. Review diagrams use real
time in minutes and end their common comparison window at
`1.25 tau_ref = 86.81 min`. Diagnostic shape comparisons use `J/J_ss`, while
amplitude comparisons use `J/J_ref`. Figures can instead use Fourier number,
seconds, or physical molar flux when a reference concentration is configured.

### Kinetic trapping

P4 implements the McNabb-Foster kinetic capture/detrapping picture. Trap
capacity is relative to `C_ref`; retention is configured through a release
half-time relative to `tau_ref`. In the local normalized equations, the trap
occupancy `theta` follows

```text
dtheta/dt = k_cap C (1 - theta) - k_det theta
dC_mobile/dt = diffusion - (N_T/C_ref) dtheta/dt
```

`N_T/C_ref` is storage capacity, not a literal count of atoms. `k_cap` is a
capture-rate coefficient; the default `k_cap * tau_ref = 20` means capture is
fast compared with reference diffusion at `C/C_ref = 1`, not that 20 atoms or
20 traps are inserted. The overview capacity-panel comparison now fixes the
equivalent binding energy at `E_B = 30 kJ/mol` (about `3.24 min` detrapping
half-time under the illustrative calibration).

The left overview panel presents an equivalent binding-energy sweep. It uses
the declared calibration

```text
p = p0 exp[-(E_D + E_B)/(R T)]
t_half,det = ln(2) / p
```

Here, `p` is the first-order detrapping coefficient [1/s], `p0` is its
pre-exponential factor or attempt-frequency scale [1/s], `E_B` is the trap
binding energy [J/mol or kJ/mol], `E_D` is the lattice-diffusion activation
energy [J/mol or kJ/mol], `R` is the universal gas constant, and `T` is the
absolute temperature [K]. The half-time `t_half,det` is therefore the time
required for half of an initially occupied trap population to detrap when
release follows first-order kinetics. The prefactor and activation energies
are calibration parameters; they should not be treated as properties that can
be inferred from a single normalized permeation curve.

with `T = 293.15 K`, `p0 = 5e3 1/s`, and `E_D = 4.5 kJ/mol` in the shipped
SCM435-derived illustrative calibration. Those calibration values are explicit
assumptions, not universal material constants; replace them when fitting a
particular steel or experiment. The `infinity` curve is the finite-window limit `p -> 0` (no
detrapping). With finite trap capacity it is not a permanently lower steady
flux: traps eventually fill, so the long-time flux approaches the trap-free
steady state.

McNabb-Foster kinetics and Oriani local equilibrium are related but distinct;
Oriani equilibrium is not implemented as a second solver mode.

### Aged prefilling

The default prefill case begins uniformly at 20% of `C_ref`, applies zero
concentration at both surfaces, and ages the profile until its symmetric centre
peak reaches 10%. Experimental time is then reset and normal entry/sink
permeation boundaries are applied.

## Graphical interface

Open P4 from slot 4 of the WeldCraft launcher or run:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' --gui
```

The left panel exposes every current physical and sweep setting, with solver
safety and publication controls behind **Show advanced settings**. Shipped
figure presets choose response families; named user profiles preserve
scientific and presentation settings without capturing result filenames or
last-opened files.

The **Setup / Run** tab shows the selected preset, exact solver-call count,
result name, output choices, progress, and run log. Every run writes reusable
HDF5 data. PNG is selected initially; PDF and SVG can be checked as needed.
All selected outputs are prepared temporarily and only replace existing files
after the complete operation succeeds.

The **Results / Export** tab opens current or existing P4 HDF5 files, previews
each stored response plate in an embedded Matplotlib canvas, lists case metrics,
and rerenders PNG/PDF/SVG without simulation. Presentation changes update the
preview directly. Scientific changes keep the old result visible with a stale
data banner until **Run simulation** is used again.

The last HDF5 path is remembered across sessions but is not loaded
automatically. Cooperative cancellation stops between cases, inside long
numerical loops, and between figure exports; incomplete temporary outputs are
removed and previous completed files remain intact.

## Command line

No mode flag and `--cli` both use the CLI. Without `--config`, it uses the same
persistent `CONFIG` as the GUI.

```powershell
# Generate the overview preset
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --preset overview

# List shipped presets
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --list-presets

# Explicit CLI and vector/raster exports
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --cli --preset trapping --formats pdf,svg,png

# Rerender saved data without simulation
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Flux\03_CodeBase\permeation_atlas.py' `
  --rerender '.\P4_Hydrogen Permeation Flux\02_Results\hydrogen_permeation_flux.h5'
```

The public application name and default result stem are **Hydrogen Permeation
Flux**. Some internal Python function names and the HDF5 format identifier
retain `atlas` for backward compatibility with existing saved result files.

## Tests

From `03_CodeBase` run:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  -m unittest discover -v -p 'test_permeation*.py'
```

Tests cover the numerical reference, convergence, trapping and prefill bounds,
HDF5 compatibility, publication rendering, transactional outputs, persistent
settings and profiles, GUI startup, cancellation, result reopening, CLI
dispatch, launcher wiring, and the startup-ready handshake.
