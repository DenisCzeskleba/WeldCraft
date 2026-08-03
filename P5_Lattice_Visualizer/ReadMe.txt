ReadMe - WeldCraft P5 Lattice Visualizer
=======================================

What this program does
----------------------
P5 visualizes Simple Cubic (SC), Body-Centered Cubic (BCC), and Face-Centered
Cubic (FCC) lattices. It supports substitutional and interstitial dopants,
single-cell teaching views, very large atom counts, unit-cell overlays,
Hydrogen picking, screenshots, and mesh export.

How to use the application
--------------------------
1. Install Python 3.10 and the required libraries:

   python -m pip install -r requirements.txt

2. Open the WeldCraft launcher and select Lattice Visualizer.
3. Configure the floating toolbox.
4. Settings are saved immediately to the fully commented `config.py`.
5. Use Open Display to open the separate PyVista display window. The button is
   called Update Display while that window is open. Slider and field changes are
   also applied automatically without closing the display or resetting its view.

The toolbox is intentionally separate from the renderer. This keeps VTK's
specialized interaction stable while allowing the settings window to remain a
convenient floating control panel.

Persistent configuration
------------------------
`config.py` is the local, persistent, human-readable configuration. It exposes
the `SETTINGS` mapping and `get_value(name)` function. The GUI writes it
atomically after valid edits and applies changes to the open display window.

`01_Resources/config_default.py` is the tracked reset template. It contains the
full explanations that used to be stored as comments in `User Input.yaml`.
Those explanations are retained in generated `config.py` files and are also
represented by GUI tooltips and this document.

Basic controls
--------------
The normal view contains the settings needed for routine lattice illustrations:

- crystal structure and approximate host-atom count;
- lattice-size behavior and host-atom size;
- substitutional concentrations and sizes for species A and B;
- hydrogen count and size;
- visible interstitial-site copies, marker size, and marker visibility;
- an optional random seed for repeatable random placement.

Continuous visual values use a slider and an editable number field. Counts use
number fields because their useful range is too large for an accurate slider.
Lattice size behavior is deliberately visible in both Basic and Advanced. It
can use one cell automatically for small examples, always respect the requested
atom count, or deliberately show one conventional cell.

Advanced controls
-----------------
Advanced options replaces the basic page with six focused tabs:

- Structure: physical host radius, host colour, and lattice-size behavior.
- Dopants: named species, colour, substitutional/interstitial placement,
  concentration or count, relative size, site family, and an optional fixed
  fractional position.
- Appearance: coordinated presets, background, host visibility, outlines, and
  surface lighting.
- Guides & Camera: unit-cell and lattice guides, site colours, legend,
  interaction, camera, projection, and numbered axes.
- Quality: lattice sampling, edge smoothing, sphere smoothness, and automatic
  simplification of very large lattices.
- Output: PNG file name, resolution, transparency, overwrite behavior, and
  display dimensions.

The renderer's safe internal limits remain documented in `config.py`, but are
not presented as normal visual controls that users would have to coordinate.

Direct renderer usage
---------------------
The renderer remains fully usable without the toolbox. Edit the documented
`config.py` and run:

   python visualize_lattice.py

This follows `display_window`, `save_png`, and all other settings in the same
configuration used by the GUI. It opens the normal interactive PyVista window
when display output is enabled.

For batch work without a display window, run:

   python visualize_lattice.py --no-show

Use `--config path_to_config.py` for an explicit Python settings module,
`--dump-config path_to_config.py` to write a documented configuration, and the
existing export/screenshot options for scripted output.

Tips
----
- Choose Show one-cell example under Lattice size behavior for teaching or
  clean SC/BCC/FCC comparisons.
- For ppm illustrations, choose a high host-atom count and a small dopant concentration.
- If a large scene is slow, increase Lattice sampling step, lower Sphere
  smoothness, or enable Simplify very large lattices.
