ReadMe - Lattice Visualization Tool
===================================

What this program does
----------------------
This program visualizes basic crystalline lattices such as Simple Cubic (SC),
Body-Centered Cubic (BCC), and Face-Centered Cubic (FCC). You can view small
unit cells to understand the geometry, or you can scale up to very large
systems to illustrate how rare events (like parts-per-million dopant atoms)
look in a crystal structure.

It allows you to:
- Explore how atoms are arranged in different lattice types.
- Insert dopants (different atom types) at substitutional or interstitial
  positions to simulate impurities.
- Adjust visualization style, colors, overlays, and rendering options to
  highlight features of interest.
- Create large-scale structures to illustrate very small dopant fractions
  (ppm scale) in an intuitive visual way.

How to use the program
----------------------
1. Install Python 3.10 and the required libraries with:

   python -m pip install -r requirements.txt

2. Open the file "User Input.yaml". This is where you control everything the
   program does.
3. Edit parameters in User Input.yaml to set the lattice, dopants, overlays,
   and visualization style.
4. Run the program with:

   python visualize_lattice.py

   It will read User Input.yaml and display the structure.

Editing the User Input.yaml
-----------------------
All important settings are in User Input.yaml. You don’t need to touch the Python
file itself. Here are the key options in simple terms:

- lattice: Choose "SC", "BCC", or "FCC" to pick the crystal type.
- target_atoms: Roughly how many atoms to show. The program adjusts the cell
  counts to get close to this number.
- Nx, Ny, Nz: Manual number of unit cells in x, y, and z. Normally overridden
  by target_atoms unless you force demo_cell.
- r: The atomic radius (in nm). Defines spacing between atoms.
- base_color: The color of the base atoms (iron by default). Example: "#555555".
- dopants: A list of additional atoms you want to insert. For each dopant:
  * name: Label for the dopant (e.g., "H").
  * color: Visual color (e.g., "blue").
  * mode: "substitutional" (replace base atoms) or "interstitial" (fit into
    holes in the lattice).
  * fraction: For substitutional dopants, fraction of sites to replace (0.01 = 1%).
  * count: For interstitial dopants, number of atoms to add.
  * interstitial_site: Restrict placement to a catalogue family. This may be
    one value or a lattice-aware mapping, for example:
        interstitial_site:
          BCC: tetra
          FCC: octa
          SC: cubic
  * forced_interstitial_position: Optional exact position in fractional
    conventional-cell coordinates. It may be one legacy [x, y, z] coordinate,
    or a lattice-aware mapping such as:
        forced_interstitial_position:
          BCC: [0.25, 0.0, 0.5]
          FCC: [0.5, 0.5, 0.5]
          SC: [0.5, 0.5, 0.5]
    The selected coordinate must belong to the chosen interstitial family.
    A missing/null entry falls back to random placement instead of crashing
    after the lattice type is changed.
  * size_scale: How big dopant atoms look relative to the base atoms.
  * positions: Optionally give explicit coordinates (advanced use).

- sphere_theta / sphere_phi: Control how smooth the spheres look. Higher values
  = smoother but slower. Lower values = blocky but faster.
- window_size: Interactive window and screenshot size in pixels.
- display_window / save_png: Select interactive display only, PNG output only,
  both, or neither. If both are true, the configured camera view is saved first
  and then a separate normal-resolution interactive window opens. Interactive
  rotations made afterward do not alter that PNG.
- png_path / png_scale: Set the lossless PNG destination and saved-resolution
  multiplier. With window_size [1600, 1200] and png_scale 2, the PNG is
  3200 x 2400 while the displayed window remains 1600 x 1200.
- png_include_lattice_name: Adds the active lattice before the extension, for
  example lattice_visualization FCC.png.
- png_avoid_overwrite: When true, an existing name.png is preserved and the new
  image is saved as name (1).png, then name (2).png, and so forth. With the
  lattice suffix this becomes lattice_visualization FCC (1).png, etc.
- png_transparent_background: Save the PNG with transparency instead of the
  configured solid background.
- deduplicate_axis_zero_labels: Shows a single shared-origin zero label while
  retaining all nonzero X/Y/Z ticks.
- axis_font_size / axis_line_width: Control the publication-scale numbered-axis
  text and line thickness. The defaults are enlarged for figures that will be
  reduced on a page.
- base_atom_outline_depth_offset / base_atom_outline_as_tubes: Keep outline
  contours solid and slightly in front of translucent Fe surfaces, avoiding
  depth-fighting stipple and white fringe pixels at overlapping edges.
- camera_preset: "custom" uses camera_direction and the other explicit camera
  settings; "isometric" uses the original equal-axis view; "low_isometric"
  reproduces the lower-elevation perspective reconstructed from the reference
  screenshot. Camera presets are independent of visual_preset.
- anti_aliasing: Edge smoothing. "fxaa" is recommended for the translucent
  outline preset because MSAA can produce white/colored fringe pixels where
  transparent surfaces and silhouettes overlap. "msaa" remains crisp for
  opaque presets; "ssaa" is softer because it downsamples the whole frame.
- sphere_specular: Controls reflective highlights; 0.0 gives a matte surface.
- visual_preset: "screen" preserves the configured colors; "thesis" and
  "publication" apply the thesis palette (silver Fe, blue H, green/teal site
  families, near-black lines) and flatter lighting. "outline" retains that
  palette but draws Fe as translucent shells with solid, camera-aware outlines,
  which helps reveal atoms hidden by an isometric projection.
- adaptive_resolution and res_cap_1/2/3: Keep very large instanced scenes
  responsive by capping base-sphere smoothness at the configured thresholds.
- render_mode: "auto" (smart choice), "spheres" (full spheres), or
  "impostor_points" (fast, simplified spheres).
- stride: Keep every nth atom. For example stride=2 shows half the atoms.

Overlays:
- show_unit_cell_overlay: Draws the outline of the conventional unit cell.
- draw_bravais_overlay: Draws extra lines for BCC/FCC to highlight the
  structure.
- overlay_color: Color of the overlay lines.
- overlay_alpha: Transparency of the overlay lines.
- overlay_marker_scale: Adjusts size of overlay markers.
- overlay_marker_opacity / overlay_marker_specular: Control marker transparency
  and shininess.
- interstitial_site_view: "all" shows both periodic faces; "canonical" folds
  duplicate boundary sites into [0,1); "picture" uses the equivalent faces
  selected by picture_site_faces. This controls the single-cell
  overlay wherever it is enabled; in demo mode, "picture" also moves a boundary
  dopant to its selected equivalent periodic image.
- picture_site_faces: May be one [x, y, z] face selection or separate BCC/FCC/SC
  selections. An occupied interstitial replaces the candidate-site marker at
  the same periodic coordinate instead of drawing two overlapping spheres.
- overlay_periodic: Choose whether overlays repeat on both faces or just the
  canonical unit cell. This is the legacy name for interstitial_site_view.
- show_overlay_legend: Adds a legend for overlays.
- overlay_legend_loc: Where the legend appears.
- overlay_legend_text_color / overlay_legend_padding: Control the legend's
  neutral text color and internal spacing independently of its colored dots.
- overlay_legend_font_size: Controls the structured legend heading and row
  text size.
- overlay_legend_x_offset: Moves the structured legend horizontally as a
  fraction of the viewport width; positive values move right.

Demo mode:
- demo_cell_force: If true, only show one conventional unit cell.
- demo_cell_auto: Automatically turn on demo mode if atom count is small.

Tips:
-----
- Use demo_cell_force for teaching to clearly show one unit cell.
- For ppm illustrations, set target_atoms very high (1e6) and add a small
  dopant fraction. This creates a big lattice with a few impurities.
- If the program is slow, lower sphere_theta and sphere_phi, or switch
  render_mode to "impostor_points".

That’s it! Edit User Input.yaml, run visualize_lattice.py, and enjoy exploring
crystal lattices interactively.
