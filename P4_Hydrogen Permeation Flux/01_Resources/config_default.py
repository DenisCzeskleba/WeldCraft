"""Shipped defaults for the P4 Hydrogen Permeation Flux tool.

The GUI creates an ignored ``03_CodeBase/config.py`` from these values and
keeps it shared with direct CLI runs. This shipped file remains unchanged.
"""

DEFAULT_CONFIG = {
    "simulation": {
        "end_time_ref": 2.5,
        "n_nodes": 201,
        "n_output": 401,
        "diffusion_safety": 0.45,
        "reaction_safety": 0.08,
        "max_internal_steps": 50_000_000,
        "reference_length_mm": 0.5,
        "reference_diffusivity_mm2_s": 6.0e-5,
        "reference_concentration_mol_mm3": None,
    },
    "ideal": {
        "diffusivity_ratios": [0.05, 0.10, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0],
        "length_ratios": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "solubility_ratios": [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "surface": {
        # Begin a progressive effective entry-condition change while the
        # reference transient is still rising. Both onset and time constant
        # are expressed relative to the ideal reference 50% crossing time.
        "onset_fraction_of_ideal_t50": 0.5,
        "time_constant_fraction_of_ideal_t50": 1.0,
        "entry_concentration_ratios": [0.30, 0.50, 0.60, 0.80, 0.90, 0.95, 1.0, 1.05, 1.10],
    },
    "trapping": {
        "end_time_ref": 3.0,
        "capture_rate_ref": 20.0,
        "capture_rate_refs": [2.0, 5.0, 10.0, 20.0, 40.0],
        "capture_sweep_capacity_ratio": 1.0,
        "capture_sweep_release_half_time_ref": 0.5,
        "capacity_ratios": [0.0, 0.5, 1.0, 1.5, 2.0],
        "capacity_sweep_release_half_time_ref": 0.5,
        "release_half_times_ref": [0.1, 0.25, 0.5, 1.0, 2.0],
        "release_sweep_capacity_ratio": 1.0,
        "combined_cases": [
            {"label": "trap-free", "capacity_ratio": 0.0, "release_half_time_ref": 0.5},
            {"label": "few / fast release", "capacity_ratio": 0.5, "release_half_time_ref": 0.1},
            {"label": "few / slow release", "capacity_ratio": 0.5, "release_half_time_ref": 2.0},
            {"label": "many / fast release", "capacity_ratio": 2.0, "release_half_time_ref": 0.1},
            {"label": "many / slow release", "capacity_ratio": 2.0, "release_half_time_ref": 2.0},
        ],
        "map_capacity_ratios": [0.25, 0.5, 1.0, 1.5, 2.0],
        "map_release_half_times_ref": [0.1, 0.25, 0.5, 1.0, 2.0],
    },
    "prefill": {
        "initial_fraction": 0.20,
        "target_center_fraction": 0.10,
        "target_center_fractions": [0.025, 0.05, 0.10, 0.15],
        "maximum_age_time_ref": 20.0,
    },
    "diagram": {
        "normalization": "common_reference",
        "time_axis": "minutes",
        "comparison_window_ref": 1.25,
        "response_metric": "t50",
        # PNG only while the diagnostic plates are being developed.  The CLI
        # still accepts --formats pdf,svg,png for final thesis export.
        "formats": ["png"],
        "dpi": 300,
        "figure_scale": 1.0,
        "font_scale": 1.0,
        "line_width_scale": 1.0,
        "marker_scale": 1.0,
        "grid_visible": True,
        "grid_style": ":",
        "legend_mode": "original",
        "show_title": True,
        "title_override": "",
    },
}
