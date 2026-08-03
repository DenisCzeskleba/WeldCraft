"""Shipped default settings for the P3 Heat Map application.

This file is copied to ``03_CodeBase/config.py`` on first launch. Users may
change the generated config through the GUI; deleting it restores these
defaults.
"""

SETTINGS = {
    # Geometry [mm]
    "le": 125.0,
    "ri": 125.0,
    "we": 20.0,
    "th": 300.0,
    "su_h": 50.0,
    "su_w": 120.0,
    "fr_ab": 5.0,
    "fr_be": 5.0,
    "weld_bead_thickness": 10.0,
    # Numerical mesh
    "dx": 1.0,
    "dy": 1.0,
    # Temperatures [degC]
    "t_cool": 160.0,
    "t_hot": 1500.0,
    "t_room": 25.0,
    # Thermal diffusivity [mm^2/s]
    "diff_coeff_bm": 5.36768,
    "diff_coeff_wm": 5.0,
    "diff_coeff_haz": 4.5,
    "diff_coeff_air": 0.0,
    # Moving weld source
    "weld_length": 350.0,
    "weld_speed": 600.0,
    "weld_temp": 1500.0,
    "weld_spot_size": 20.0,
    "time_before_weld_start": 5.0,
    # Cooling/material approximation
    "conv_variable": 3.0,
    "c": 486.0,
    "rho": 7850.0,
    # Runtime and snapshot output
    "sim_time": 900.0,
    "save_so_often_per_sec": 0.5,
    "slow_down_beginning": True,
    "h5_filename": "simple_heat_map.h5",
    "animation_filename": "heat_map_animation.mp4",
    "figure_filename": "heat_map_figure.png",
    "disable_overwrite_warning": False,
    # Viewer and export options
    "heatmap_style": "hot",
    "heatmap_vmin": 50.0,
    "heatmap_vmax": 800.0,
    "show_contours": True,
    "contour_levels": [200, 250, 300, 350, 400, 450, 500],
    "show_monitoring_points": True,
    "show_mesh_lines": False,
    "monitoring_distances": [5, 10, 25, 50],
    "monitoring_y_offset": 30.0,
    "weld_zoom_margin": 20.0,
    "animation_fps": 30,
    "animation_dpi": 160,
    "animation_frame_stride": 1,
    # Advanced boundary/debug controls
    "use_boundary_adjustment": False,
    "make_animation_here": False,
}


def get_value(name):
    try:
        return SETTINGS[name]
    except KeyError as exc:
        raise ValueError(f"Parameter '{name}' not found in config") from exc
