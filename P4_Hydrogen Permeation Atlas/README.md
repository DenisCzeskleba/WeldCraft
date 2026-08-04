# P4 Hydrogen Permeation Atlas

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
  '.\P4_Hydrogen Permeation Atlas\03_CodeBase\permeation_atlas.py' `
  --preset overview
```

Useful commands:

```powershell
# List shipped presets
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Atlas\03_CodeBase\permeation_atlas.py' `
  --list-presets

# Generate one plate and display it after saving
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Atlas\03_CodeBase\permeation_atlas.py' `
  --preset trapping --show

# Rerender a saved result without repeating the simulation
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  '.\P4_Hydrogen Permeation Atlas\03_CodeBase\permeation_atlas.py' `
  --rerender '.\P4_Hydrogen Permeation Atlas\02_Results\hydrogen_permeation_atlas.h5'
```

During diagram development, the `overview` preset creates the separate ideal,
entry-condition, and trapping diagnostic plates as PNG only. The prefill and
response-map presets remain available but are intentionally excluded from that
default review set. Final vector exports remain supported with
`--formats pdf,svg,png`. HDF5 preserves the underlying fields, flux histories,
configurations, and derived metrics for later GUI use.

The P0 launcher already names and describes P4, but intentionally presents an
informational message until the graphical interface is added.

## Tests

From `03_CodeBase` run:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' `
  -m unittest -v test_permeation_atlas.py
```

The tests cover the ideal Fourier-series reference, convergence, explicit trap
conservation and bounds, surface histories, aged prefilling, normalization,
HDF5 round trips, presets, figure exports, CLI discovery, and launcher text.
