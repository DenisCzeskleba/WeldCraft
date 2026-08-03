"""Documented default configuration for the WeldCraft P5 Lattice Visualizer.

The GUI copies this documented shape into the local ``config.py`` and updates
values in-place through the renderer's atomic configuration writer.  Keep the
explanations in this file: ``config.py`` is intentionally a human-readable
working configuration, not a comment-free cache.
"""

# Physical sizing (the controls most users normally need)
SETTINGS = {
    "lattice": "FCC",  # SC (Simple Cubic), BCC, or FCC.
    "Nx": 10,  # Manual unit-cell count in x; normally derived from target_atoms.
    "Ny": 10,  # Manual unit-cell count in y; normally derived from target_atoms.
    "Nz": 10,  # Manual unit-cell count in z; normally derived from target_atoms.
    "a": 1.0,  # Derived lattice spacing in nm; recalculated from the physical radius.
    # Approximate host-atom count. A value near 1,000,000 is useful for ppm
    # illustrations. Automatic one-cell mode applies at SC <= 8, BCC <= 9,
    # and FCC <= 14 unless the single-cell behavior is explicitly overridden.
    "target_atoms": 1,
    "r": 0.124,  # Fe atomic radius in nm at room temperature.
    "base_radius_scale": 0.5,  # Visual radius relative to the physical Fe radius.
    "dopant_radius_scale": 0.25,  # Compatibility value for configurations made before per-species sizes.
    "base_color": "#555555",  # Base lattice color; #555555 is a neutral grey.
    "base_radius": 0.23,  # Derived displayed host radius; recalculated before rendering.
    "base_displacements": {},  # Optional advanced map from integer lattice sites to [dx, dy, dz] offsets.

    # Additional species.  Substitutional species replace base atoms by
    # fraction; interstitial species occupy legal catalogue sites by count.
    "dopants": [
        {
            "name": "H",
            "color": "#0000FF",  # Atom color, stored as a standard hexadecimal color.
            "radius": None,  # Derived from size_scale before rendering.
            "positions": [],  # Optional explicit positions; normally filled by placement rules.
            "mode": "interstitial",  # Occupy spaces between host atoms.
            "fraction": 0.0,  # Used only for substitutional placement.
            "count": 1,  # Absolute number of interstitial atoms to place.
            # H is tetrahedral in BCC, octahedral in FCC, and cubic in SC.
            "interstitial_site": {"BCC": "tetra", "FCC": "octa", "SC": "cubic"},
            # Optional exact site in fractional conventional-cell coordinates.
            "forced_interstitial_position": {
                "BCC": [0.25, 0.0, 0.5],
                "FCC": [0.5, 0.0, 0.0],
                "SC": [0.5, 0.5, 0.5],
            },
            "size_scale": 0.5,  # Displayed radius relative to a host atom.
        },
        {
            "name": "A",
            "color": "#FF0000",
            "radius": None,  # Derived from size_scale before rendering.
            "positions": [],  # Optional explicit positions; normally selected randomly.
            "mode": "substitutional",  # Replace host atoms.
            "fraction": 0.0,  # Fraction of host sites replaced by this species.
            "count": 0,  # Used only for interstitial placement.
            "interstitial_site": None,  # No site-family restriction for a substitutional species.
            "forced_interstitial_position": None,  # No fixed interstitial position.
            "size_scale": 1.2,  # Displayed radius relative to a host atom.
        },
        {
            "name": "B",
            "color": "#000000",
            "radius": None,  # Derived from size_scale before rendering.
            "positions": [],  # Optional explicit positions; normally selected randomly.
            "mode": "substitutional",  # Replace host atoms.
            "fraction": 0.0,  # Fraction of host sites replaced by this species.
            "count": 0,  # Used only for interstitial placement.
            "interstitial_site": None,  # No site-family restriction for a substitutional species.
            "forced_interstitial_position": None,  # No fixed interstitial position.
            "size_scale": 1.5,  # Displayed radius relative to a host atom.
        },
    ],

    # Basic display and interaction
    "background": "#FFFFFF",  # Background color.
    "show_axes": True,  # Show the corner orientation triad and numbered axes.
    "enable_picking": True,  # Enable right-click Hydrogen picking.
    "zoom_mode": "cursor",  # cursor zooms toward the mouse; focal uses VTK's default.
    "pick_instruction": "Right click to find the Hydrogen",

    # Conventional-cell and interstitial overlays
    "show_unit_cell_overlay": True,
    "draw_bravais_overlay": True,
    "overlay_marker_scale": 0.45,
    "overlay_color": "#222222",
    "overlay_alpha": 0.6,
    "overlay_marker_opacity": 0.55,
    "overlay_marker_specular": 0.0,
    "tetrahedral_color": "#008000",  # Tetrahedral-site marker color.
    "octahedral_color": "#FFA500",  # Octahedral-site marker color.
    "cubic_color": "#800080",  # Cubic-site marker color.
    "interstitial_site_view": "picture",  # all, canonical, or picture.
    # Periodic faces selected by the camera-facing site view for each lattice.
    "picture_site_faces": {"BCC": [1, 1, 0], "FCC": [0, 0, 0], "SC": [1, 1, 0]},
    "overlay_periodic": "both_faces",  # Compatibility value for older periodic-site configurations.
    "show_overlay_legend": True,
    "overlay_legend_loc": "upper right",
    "overlay_legend_text_color": "#3A3A3A",
    "overlay_legend_padding": 8,
    "overlay_legend_font_size": 18,
    "overlay_legend_x_offset": 0.025,

    # Demo mode is useful for teaching and comparing one SC/BCC/FCC cell.
    "demo_cell_auto": True,
    "demo_cell_force": None,  # Set True or False to override automatic demo mode.
    "random_seed": None,  # Optional repeatable seed for random dopant placement.

    # Rendering and output
    "visual_preset": "outline",  # custom, screen, thesis/publication, or outline.
    "render_mode": "auto",  # Internal selection: automatic spheres, geometric spheres, or point spheres.
    "display_window": True,  # Open the interactive lattice display.
    "save_png": True,  # Save a PNG when opening the display.
    "png_path": "02_Results/lattice_visualization.png",  # Output file relative to the application folder.
    "png_include_lattice_name": True,  # Add the lattice name to the output file name.
    "png_avoid_overwrite": True,  # Add a number instead of replacing an existing file.
    "png_scale": 2,  # Resolution multiplier; 2 turns 1600 x 1200 into 3200 x 2400.
    "png_transparent_background": False,  # Save an alpha channel instead of an opaque background.
    "window_size": [1600, 1200],  # Display width and height and the base PNG dimensions.
    "anti_aliasing": "fxaa",  # FXAA is recommended for translucent outline atoms.
    "multi_samples": 8,  # Number of samples used only by multisample edge smoothing.
    "sphere_theta": 48,
    "sphere_phi": 48,
    "sphere_specular": 0.0,
    "sphere_ambient": 0.0,
    "sphere_diffuse": 1.0,
    "base_atom_opacity": 1.0,
    "base_atom_outline": False,  # Draw silhouettes around host atoms.
    "base_atom_outline_color": "#202124",  # Silhouette color.
    "base_atom_outline_width": 2.5,  # Silhouette line thickness.
    "base_atom_outline_depth_offset": -2.0,  # Prevent white gaps where silhouettes meet atom surfaces.
    "base_atom_outline_as_tubes": True,  # Use continuous rounded silhouette lines.
    "max_atoms_for_outlines": 30000,  # Disable expensive silhouettes above this host-atom count.
    "points_impostor_size": 3.0,  # Compatibility size for the internal point-sphere fallback.

    # Camera and numbered-axis presentation
    "camera_preset": "custom",  # custom, isometric, or low_isometric.
    "camera_direction": [-1.0, -0.7, 0.35],  # Direction from scene center toward the camera.
    "camera_view_up": [0.0, 0.0, 1.0],  # Direction that remains upright on screen.
    "camera_distance_scale": 3.0,  # Camera distance relative to lattice extent.
    "camera_normalize_demo_atom_size": True,  # Match apparent atom size across one-cell lattice types.
    "camera_parallel_projection": False,  # False uses perspective; True uses orthographic projection.
    "camera_view_angle": 30.0,  # Perspective field of view in degrees.
    "axis_location": "outer",  # Placement of numbered coordinate axes.
    "axis_use_3d_text": False,  # Use crisp screen text rather than geometry-based text.
    "axis_font_size": 32,  # Coordinate title and number size.
    "axis_line_width": 1.75,  # Coordinate axis and tick thickness.
    "deduplicate_axis_zero_labels": True,  # Show one shared zero label at the origin.

    # Data thinning and cropping
    "stride": 1,  # Keep every nth node; higher values thin very large scenes.
    "slab": None,  # Optional advanced fractional z-range [z0, z1) for cropping.

    # Large-scene performance controls
    # The remaining values are safe renderer limits. The toolbox exposes one
    # plain-language "Simplify very large lattices" switch instead of asking
    # users to coordinate these implementation details themselves.
    "max_atoms_for_true_spheres": 1020030,  # Largest exported scene that may be expanded into full sphere meshes.
    "chunking_enabled": True,  # Split large instanced scenes into manageable rendering groups.
    "chunk_target_atoms": 125000,  # Preferred number of host atoms in one rendering group.
    "chunk_max_actors": 8,  # Maximum number of host-atom rendering groups.
    "chunk_axis": "z",  # Axis along which rendering groups are divided.
    "adaptive_resolution": True,  # Reduce sphere detail automatically as scenes become very large.
    "res_thresh_1": 100000,  # First atom-count boundary for automatic detail reduction.
    "res_thresh_2": 300000,  # Second atom-count boundary for automatic detail reduction.
    "res_thresh_3": 600000,  # Third atom-count boundary for automatic detail reduction.
    "res_cap_1": 16,  # Maximum sphere detail above the first boundary.
    "res_cap_2": 12,  # Maximum sphere detail above the second boundary.
    "res_cap_3": 8,  # Maximum sphere detail above the third boundary.
}


def get_value(name):
    """Return one documented default setting by name."""

    try:
        return SETTINGS[name]
    except KeyError as exc:
        raise ValueError(f"Parameter '{name}' not found in config") from exc
