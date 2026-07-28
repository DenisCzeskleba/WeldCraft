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
  * interstitial_site: Restrict to "octa" or "tetra" holes if you want.
  * size_scale: How big dopant atoms look relative to the base atoms.
  * positions: Optionally give explicit coordinates (advanced use).

- sphere_theta / sphere_phi: Control how smooth the spheres look. Higher values
  = smoother but slower. Lower values = blocky but faster.
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
- overlay_periodic: Choose whether overlays repeat on both faces or just the
  canonical unit cell.
- show_overlay_legend: Adds a legend for overlays.
- overlay_legend_loc: Where the legend appears.

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
