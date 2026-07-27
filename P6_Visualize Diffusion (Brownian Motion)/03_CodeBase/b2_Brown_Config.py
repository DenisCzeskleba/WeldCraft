"""
Editable settings for the Brownian motion diffusion visualizer.

This file is intentionally plain Python, like the P2 parameter config.
Change values here to control the simulation and the generated presentation.
"""

from fractions import Fraction


# ---------------------- Main Simulation Settings ---------------------- #
# See ../01_Resources/README.md for execution and step definitions.
# Recommended: "molecular_wiggle", "random_sequential_wiggle", "event_driven_wiggle"
# "molecular_wiggle": direct synchronous reference model; closest to explicit molecular wiggles, but slowest.
# "random_sequential_wiggle": compact asynchronous wiggles with exact no-move probabilities and attempt accounting.
# "event_driven_wiggle": uniformized rate selection; null waiting is preserved but skipped geometrically.
# DEPRECATED "forced_jump": retained for old comparisons only; ignores the area-characteristic physics below.

simulation_mode = "event_driven_wiggle"
y = 650  # Height (y)
x = 1300  # Width (x)

steps = 5_000_000_000  # use 10_000_000 notation, or 10000000. DO NOT use 100.000.000
max_radius_to_jump = 10

# ---------------------- Matrix Source ---------------------- #
MATRIX_SOURCE = "random"  # Options: "random", "image", "lattice"


# Used when MATRIX_SOURCE = "random" or "lattice"
max_sol_a = Fraction(10, 100)
max_sol_b = Fraction(5, 100)

# Used when MATRIX_SOURCE = "lattice"
LATTICE_STYLE = "prime"
LATTICE_START_SPACING = 5
LATTICE_MIN_SPACING = 1

# Used when USE_IMAGE_MATRIX = True
image_name = "Gef\u00fcge_array.tiff"
max_sol_white = Fraction(40, 100)
max_sol_black = Fraction(2, 100)
show_image_matrix_plot = True

# ---------------------- Initial Concentration ---------------------- #
concentration_a = 50
concentration_b = 50
concentration_spot = 50  # only used if USE_SPOT = True
concentration_trap_layer = 50  # only used if USE_TRAP_LAYER = True

# ---------------------- Layer / Boundary Options ---------------------- #
USE_SPOT = True
SPOT_DIAMETER = 80
SPOT_CENTER_X = 3 * x // 2  # Column position of the spot center
SPOT_CENTER_Y = y // 2  # Row position of the spot center

USE_TRAP_LAYER = True
TRAP_LAYER_CENTER_X = x // 2
TRAP_LAYER_WIDTH = 20
max_sol_trap_layer = Fraction(20, 100)

USE_SINK_SOURCE = True
SINK_SOURCE_THICKNESS = 10
SOURCE_SIDE = "left"  # Options: "left" or "right"
num_subregions = 1


# ---------------------- Movement Probability ---------------------- #
# Global probability scale for wiggle-derived movement. A uniform change affects kinetics, not equilibrium preference.
base_movement_probability = 1.0

# ---------------------- Area Characteristics ---------------------- #
# Four user-facing area types are supported. A/B are the base halves, the optional trap layer overrides
# A/B, and the optional spot overrides every area it overlaps.
#
# affinity:
#   Relative equilibrium preference. Only ratios matter. A move toward a higher-affinity area keeps its
#   ordinary rate; the reverse move is reduced by the affinity ratio.
# mobility:
#   Symmetric kinetic scale from 0 to 1. It changes movement speed without changing equilibrium preference.
#   The mobility of a boundary transition is the geometric mean of the two areas' mobilities.
AREA_CHARACTERISTICS = {
    "a": {"affinity": 1.0, "mobility": 1.0},
    "b": {"affinity": 1.0, "mobility": 1.0},
    "spot": {"affinity": 1.0, "mobility": 1.0},
    "trap_layer": {"affinity": 3.0, "mobility": 1.0},
}

# --------------------- Numerics stuff --------------------- #

random_seed = None  # None selects a fresh seed per run; set an integer to reproduce a run exactly.
random_size = 10 ** 7  # Deprecated forced_jump only: number of precomputed random values.
max_ram_mb = 2000  # Adjustable memory target for HDF5 frame buffering
save_every_steps = 500000
delete_old_h5 = True


# ---------------------- Animation Panels ---------------------- #
SHOW_MAIN_SIMULATION_PANEL = True
SHOW_CONCENTRATION_PROFILE_PANEL = True
SHOW_DIFFUSION_SPEED_PANEL = False

# ---------------------- Main Panel Render Mode ---------------------- #
MAIN_RENDER_MODE = "pixels"  # Options: "pixels", "dots"
DOT_SIZE_AVAILABLE = 12
DOT_SIZE_HYDROGEN = 12
DOT_ALPHA_AVAILABLE = 0.85
DOT_ALPHA_HYDROGEN = 0.95


# ---------------------- Animation Colors ---------------------- #
COLOR_EMPTY = "#440154"  # "#440154" 
COLOR_AVAILABLE_SPOT = "#0000FF"
COLOR_HYDROGEN = "#FF0000"
COLOR_CONCENTRATION_LINE = "#0000FF"
DIFFUSION_SPEED_COLORS = [
    "#FF0000",
    "#FFA500",
    "#008000",
    "#800080",
    "#A52A2A",
    "#00FFFF",
    "#000000",
]

# ---------------------- Output Files ---------------------- #
h5_filename = "random_motion.h5"
animation_output_folder = "Saved Animations"
animation_filename = "brownian_motion_animation.mp4"

# ---------------------- Video Output ---------------------- #
render_every_nth_frame = 5  # Render every Nth saved HDF5 frame; use larger values for huge runs to keep videos and memory smaller.
animation_fps = 12
animation_main_pixel_scale = 2  # Integer video pixels per matrix cell; 2 preserves sparse one-cell sites clearly.
animation_side_panel_width_px = 480
animation_title_font_size = 28
animation_axis_label_font_size = 20
animation_tick_font_size = 18
animation_legend_font_size = 16
animation_artist = "Denis Czeskleba"
ffmpeg_path = (
    r"C:\Users\DCzes\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-6.0-full_build\bin\ffmpeg.exe"
)
