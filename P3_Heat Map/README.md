# P3 Heat Map

P3 is a focused standalone thermal-welding simulator and visualization tool. It
is useful for manual heating calibration, experimenting with a moving heat
source, inspecting temperature traces at four points, and creating annotated
heat-map animations.

Its compact three-part implementation also makes P3 a practical smaller-scale
example of the thermal concepts used in the full P2 welding and hydrogen-
diffusion simulation. P3 and P2 are independent programs and do not share
runtime code.

## Layout and publication boundary

- `01_Resources/` contains the shipped `config_default.py` reset template.
  Cooling measurements and related workbooks are kept with the P2 cooling-curve
  analysis.
- `02_Results/` contains generated HDF5 data, plots, and animations. Its contents
  are local-only; only `.gitkeep` is published.
- `03_CodeBase/` contains the versioned runnable scripts.
- `00_Development_Archive/` preserves earlier development iterations and
  scratch notes locally and is ignored in full.

The project-local `.gitignore` enforces these boundaries. Generated results
remain workspace-local and are not published.

## Scripts

`03_CodeBase/heat_map.py --gui` opens the P1-style graphical interface. The GUI
loads persistent settings from `03_CodeBase/config.py`, creates it from
`01_Resources/config_default.py` when needed, runs the 2D thermal simulation in a background
worker, and displays the stored result frames and diagrams. Animation export is
optional and requires an FFmpeg installation that Matplotlib can use.

The P3 code is split into three roles:

- `config.py` stores the user-editable settings and is recreated from the shipped
  defaults when missing.
- `functions.py` contains configuration validation, mesh creation, simulation,
  HDF5 loading, and plotting helpers.
- `heat_map.py` contains the GUI, launcher entry point, and CLI dispatch.

For direct scripted execution, run `heat_map.py` or `heat_map.py --cli`.
Append `--render` to render the configured MP4 after the CLI simulation.

From the repository root, use the workspace Python environment:

```powershell
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' '.\P3_Heat Map\03_CodeBase\heat_map.py' --cli
& 'F:\99_Virtual-Environments\02_WeldCraft\Scripts\python.exe' '.\P3_Heat Map\03_CodeBase\heat_map.py' --gui
```

The main WeldCraft launcher also exposes P3 through its **Heat Map** button.

## Standard result names

- Simulation snapshots: `02_Results/simple_heat_map.h5`
- Rendered animation: `02_Results/heat_map_animation.mp4`

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
- User-facing geometry, monitoring points, run duration, save rate, and plot
  limits are stored in `config.py` and exposed through the GUI; solver internals
  remain code-only or advanced settings.
- The HDF5 output stores snapshots and times together with the validated
  configuration and basic format metadata.

Use P3 for focused thermal calibration, experimentation, teaching, and animation
work. Use P2 when the task requires its full welding geometry, material model,
or coupled hydrogen-diffusion behavior.
