"""
Editable settings for the Brownian motion diffusion visualizer.

This file is intentionally plain Python, like the P2 parameter config.
Change values here to control the simulation and the generated presentation.
"""

from fractions import Fraction


# ---------------------- Main Simulation Settings ---------------------- #
y = 200  # Height (y)
x = 400  # Width (x)

steps = 40000
max_radius_to_jump = 10

# ---------------------- Matrix Source ---------------------- #
USE_IMAGE_MATRIX = False
image_name = "Gef\u00fcge_array.tiff"

# Used when USE_IMAGE_MATRIX = True
max_sol_white = Fraction(40, 100)
max_sol_black = Fraction(2, 100)
show_image_matrix_plot = True

# Used when USE_IMAGE_MATRIX = False
max_sol_a = Fraction(5, 100)
max_sol_b = Fraction(10, 100)


# ---------------------- Initial Concentration ---------------------- #
concentration_a = 50
concentration_b = 50


# ---------------------- Layer / Boundary Options ---------------------- #
USE_SPOT = True
SPOT_DIAMETER = 50
SPOT_CENTER_X = 3 * x // 4  # Column position of the spot center
SPOT_CENTER_Y = y // 2  # Row position of the spot center

USE_TRAP_LAYER = False
TRAP_LAYER_WIDTH = 19

USE_SINK_SOURCE = True
SINK_SOURCE_THICKNESS = 10
SOURCE_SIDE = "left"  # Options: "left" or "right"
num_subregions = 1


# ---------------------- Movement Probability ---------------------- #
base_movement_probability = 1.0

# --------------------- Numerics stuff --------------------- #

random_size = 10 ** 7  # Number of precomputed random numbers
max_ram_mb = 2000  # Adjustable memory target for HDF5 frame buffering
save_every_steps = 1000
delete_old_h5 = True


# ---------------------- Animation Panels ---------------------- #
SHOW_MAIN_SIMULATION_PANEL = True
SHOW_CONCENTRATION_PROFILE_PANEL = False
SHOW_DIFFUSION_SPEED_PANEL = False


# ---------------------- Animation Colors ---------------------- #
COLOR_EMPTY = "#440154"
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
animation_fps = 12
animation_dpi = 300
animation_artist = "Denis Czeskleba"
animation_bitrate = 3000
ffmpeg_path = (
    r"C:\Users\DCzes\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-6.0-full_build\bin\ffmpeg.exe"
)
