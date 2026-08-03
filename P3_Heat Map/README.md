# P3 Heat Map

P3 is a focused standalone thermal-welding simulator and visualization tool. It
is useful for manual heating calibration, experimenting with a moving heat
source, inspecting temperature traces at four points, fitting cooling behavior
from measurement data, and creating annotated heat-map animations.

Its compact all-in-one implementation also makes P3 a practical smaller-scale
example of the thermal concepts used in the full P2 welding and hydrogen-
diffusion simulation. P3 and P2 are independent programs and do not share
runtime code.

## Layout and publication boundary

- `01_Resources/` contains raw measurements, converted data, and workbooks. Its
  contents are local-only; only `.gitkeep` is published.
- `02_Results/` contains generated HDF5 data, plots, and animations. Its contents
  are local-only; only `.gitkeep` is published.
- `03_CodeBase/` contains the versioned runnable scripts.
- `00_Development_Archive/` preserves earlier development iterations and
  scratch notes locally and is ignored in full.

The project-local `.gitignore` enforces these boundaries. Existing local files
were reorganized without deleting them.

## Scripts

`03_CodeBase/heat_map.py` runs the 2D thermal simulation, writes
`02_Results/simple_heat_map.h5`, and normally renders
`02_Results/heat_map_animation.mp4`. Rendering requires an FFmpeg installation
that Matplotlib can use.

`03_CodeBase/figure_out_cooling.py` reads
`01_Resources/Curve fit Sub 150 cooling/011_prepaired.CSV` and compares two
empirical cooling fits with a simple convection calculation. It writes
`02_Results/cooling_fit_comparison.png`.

From the repository root, use the workspace Python environment:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' '.\P3_Heat Map\03_CodeBase\heat_map.py'
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' '.\P3_Heat Map\03_CodeBase\figure_out_cooling.py'
```

The main WeldCraft launcher also exposes P3 through its **Heat Map** button.

## Standard result names

- Simulation snapshots: `02_Results/simple_heat_map.h5`
- Rendered animation: `02_Results/heat_map_animation.mp4`
- Cooling-fit figure: `02_Results/cooling_fit_comparison.png`

Each new heat-map run replaces the standard HDF5 file and animation. Move or
rename results you want to retain before starting another run.

## Relationship to P2

The ideas that carried forward into P2 are the explicit Fourier temperature
step, a temperature-only calibration path, cooling-boundary calibration, HDF5
snapshots, and extraction of temperature histories at four locations. P2 now
owns the more complete versions: multiple joint geometries and beads,
temperature-dependent diffusivity, face-based Robin cooling, provenance
metadata, and optional hydrogen diffusion.

P3's `conv_variable = 3 W/m^2/K` is not directly transferable to P2's
`t_conv_air` or `t_conv_cu`. P3 applies its cooling term throughout the interior
metal mask as a per-cell sink, whereas P2 applies Robin conditions at exposed
faces with different units and geometry.

## Model scope and interpretation

- The model is two-dimensional and uses an explicit finite-difference step.
- The heat source is a moving, fixed-temperature rectangle, not a calibrated
  energy input or Goldak-style volumetric source.
- Thermal diffusivity is effectively fixed by material region; it is not updated
  with temperature.
- Cooling is a simplified volumetric sink, not a surface-only convection model;
  radiation, latent heat, and evolving material properties are absent.
- The reflective boundary helper chooses one nearby interior cell and is intended
  only for non-interacting metal/air boundaries.
- Geometry, monitoring points, run duration, save rate, and plot limits are
  hardcoded in the script.
- The HDF5 output has snapshots and times but no configuration/provenance
  metadata.
- The cooling-fit script trims one dataset by hardcoded row numbers and performs
  an extrapolative empirical fit; it does not establish a unique physical heat
  transfer coefficient.

Use P3 for focused thermal calibration, experimentation, teaching, and animation
work. Use P2 when the task requires its full welding geometry, material model,
or coupled hydrogen-diffusion behavior.
