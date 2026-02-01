"""
Parameter configuration file for the welding and hydrogen diffusion simulation.

This file contains all relevant constants, boundary conditions, and material properties
used in the simulation. Adjust values here to configure a specific simulation setup.

Sections include:
    - Dimensional step sizes (dx, dy)
    - Initial and boundary conditions for temperature and hydrogen
    - Temperature- and hydrogen-dependent diffusion coefficients
    - Time step calculation and overrides
    - Welding parameters (geometry, bead timing, hydrogen input)
    - Cooling and post-weld diffusion parameters
    - Output options and debugging controls

IMPORTANT:
    - Ensure dx == dy unless solver logic has been updated (BY YOU) for anisotropic meshing.
    - Double-check solver stability when modifying diffusion coefficients or dt.

"""

model_version = "0.15 - alpha"  # For provenance. Don't change, unless you customize logic. Then its yours, yay!
simulation_type = "butt joint"  # Options: "lap joint", "butt joint" and "iso3690"

# ---------------------------------------- Dimensional step size, gradient --------------------------------------------
dx = 0.5  # step size in x direction - if not equal to dy, tripple check solver logic!
dy = 0.5  # step size in y direction - if not equal to dx, tripple check solver logic!
dx2 = dx*dx  # partial differential x
dy2 = dy*dy  # partial differential y
inv_dx2 = 1.0 / (dx*dx)  # Inverse partial differential x
inv_dy2 = 1.0 / (dy*dy)  # Inverse partial differential y

# -------------------- Relevant Starting Conditions - Temperature und Hydrogen concentration --------------------------
t_cool = 160  # Interpass temperature basically. Used for BCs and initialization
t_hot = 1550  # Weld bead temperature. Adjust as needed
t_room = 25  # Surrounding temperature. Use 0.001 instead of 0 for ISO3690 for example
h_cont_initial = 0  # No hydrogen present in bm at the beginning of the simulation

lowest_temp_in_sim = min(t_cool, t_hot, t_room)  # find min temp
highest_temp_in_sim = max(t_cool, t_hot, t_room)  # find max temp

# ---------------------------------------- TEMPERATURE diffusion coefficients -----------------------------------------
bm_rho = 8100e-9  # [kg/mm³]
bm_cp = 460  # [J/kgK]
bm_k = 0.02  # [W/mmK]
diff_coeff_bm = bm_k / (bm_rho * bm_cp)  # D = 5.36768 e-6  m²/s = 5.36768 mm²/s  **1
diff_coeff_wm = 5.5  # Similar but programs logic  NEEDS this to be different (even if just by e-8 or something)
diff_coeff_haz = 5  # # Similar but programs logic  NEEDS this to be different (even if just by e-8 or something)
diff_coeff_air = 0  # No diffusion in air. If != 0, check ... everything!!!
highest_diff_coeff = max(diff_coeff_bm, diff_coeff_wm, diff_coeff_haz)

# ------------------------------------------ Convection / Cooling / Robin BC ------------------------------------------
t_conv_air = 0.1e-5  # [W/mm²/K], ≈ 5 W/m²K, top plate, still air. Guess / calibrate from temperature measurements
t_conv_h2 = 5e-4   # [W/mm²/K], ≈ 500 W/m²K, underside, forced hydrogen. Guess or calibrate UNUSED - Change dt!!!!
# Doesnt make sense to use "convection" but easy to tweak. consider using sink temperature T_cu(t) later!
t_conv_cu = 3e-4   # [W/mm²/K], ≈ 300 W/m²K, Copper contact.


# Precompute handy factors (for constant h, k). Robin uses 2*h/(k*Δ)
coef_robin_x_air = 2.0 * t_conv_air / (bm_k * dx)  # [1/mm]
coef_robin_y_air = 2.0 * t_conv_air / (bm_k * dy)  # [1/mm]

coef_robin_x_h2 = 2.0 * t_conv_h2 / (bm_k * dx)  # [1/mm]
coef_robin_y_h2 = 2.0 * t_conv_h2 / (bm_k * dy)  # [1/mm]

coef_robin_x_cu = 2.0 * t_conv_cu / (bm_k * dx)  # [1/mm]
coef_robin_y_cu = 2.0 * t_conv_cu / (bm_k * dy)  # [1/mm]

# ------------------------------------------ HYDROGEN diffusion coefficients ------------------------------------------
# diff_coeff_h = 8.8 * 10 ** - 9  # Not quite in [mm² / s]! Needs * (T ** 2.2285) - applied during simulation **2
# diff_coeff_h = 6.8016 * 10 ** - 8  # Low?? value from our project (QL BM) - Some Copy paste error here?????
# diff_coeff_h = 7.7908 * 10 ** - 8  # high?? value from our project (TM BM) - Some Copy paste error here?????
diff_coeff_h = 4.8309355118 * 10 ** - 8  # Low value from our final project report
# diff_coeff_h = 6.7479734134 * 10 ** - 8  # High value from our final project report
# diff_coeff_h = 8.5 * 10 ** - 8  # standard value you often used
diff_coeff_h_air = 0  # Not looking at hydrogen in the air
minimum_h_diff_in_sample = diff_coeff_h * (lowest_temp_in_sim ** 2.2285)  # Determines time step dt

# ----------------------------------------------- Time Differential  --------------------------------------------------
safety_factor = 1  # should usually be ok to just use 1 (1 = stable, lower maybe better temporal convergence)
dt = safety_factor * (dx2 * dy2 / (2 * highest_diff_coeff * (dx2 + dy2)))  # highest time step to still be stable
dt_big_calc = safety_factor * (dx2 * dy2 / (2 * minimum_h_diff_in_sample * (dx2 + dy2)))

use_big_dt_override = True  # use the 600s save time override for the last stage (diffusion at RT)
big_dt_override = 300  # calculate every so many seconds, even if you could go faster, to safe the states regardless
if use_big_dt_override and dt_big_calc > big_dt_override:  # change this here!
    dt_big = big_dt_override
    print("Big_dt Overwrite ON, new dt_big: " + str(dt_big))
else:
    dt_big = dt_big_calc
    print("Big_dt Overwrite OFF, using dt_big: " + str(dt_big))

# ---------------------------------------- Offsets - Temperature and Diffusion ----------------------------------------
temperature_offset = -5  # The offset is used to make the graphs display the suroundings in grey
hydrogen_offset = -5  # The offset is used to make the graphs display the suroundings in grey

# --------------------------- Welding - Temps and Timey Whimey Whibbly Whobbly stuff -----------------------------------
weld_simulation_style = "ellipse"  # More realistic weld beads ("ellipse"). "rectangle" not supported currently
no_of_weld_beads = 16  # no of "blocks" during welding: Butt joint: must be %2, fit bead_height! Lap Joint: max 4
bead_height = 2.5  # Half of weld beads * height should probably be weld thickness (th) (2.8 for iso?
bead_width = 12  # Using half of weld width (we) for blocks, for ellipses maybe 3/4-ish of weld width? 60%
bead_scales = [(1.0, 1.0), (1.0, 1.0), (1.6, 1.6), (3.0, 3.0)]  # Used for lap joint and iso3690

haz_creation_temperature = 1350  # Checks if some bm area got hotter than this, creates HAZ there
haz_creation_check_time_window = 10  # no need to check for ever after welding, but maybe for 10s?

add_bead_mode = "regular_intervals"  # Options: "regular_intervals" and "interpass_temperature_controlled"
# add_bead_mode = "regular_intervals"  # Simply add a weld bead every so many seconds
# add_bead_mode = "interpass_temperature_controlled"  # Add a weld bead when a specific point reaches some temperature

temp_weld_metal = t_hot  # Temperature of new weld block
hydro_weld_metal = 100  # Hydrogen in the new weld block, for now set as "100%"
reference_from_iso3690 = 2.5  # this is used for lap joints with hydrogen from the inside. "2.5 is 100%"
h_on_the_inside = 0  # Used for the constant h lap joint boundary condition on the inside of the weld / pipeline
pipe_line_inner_hydrogen = "variable"  # Options: "constant", "variable" | variable means sieverts law (temp depend)

time_before_first_weld = 5  # Run simulation before starting with first bead. Not necessary, makes for nicer videos [s]
time_for_weld_bead = 480  # Time between welds. Weld block gets added at 0. Temp. held for time_heat_hold. [s]
time_after_last_weld = 480  # Time after last weld. BC in sample edge is held at t_cool this long. [s]
time_heat_hold = 3  # Force the new weld block to have this temp for so long [s]
# check this logic again, for example: what hapens if time_for_weld_bead < time_heat_hold?

time_welding = no_of_weld_beads * time_heat_hold + (no_of_weld_beads - 1) * time_for_weld_bead + time_after_last_weld
time_cooling_to_rt = 2 * 60 * 60  # For now set to 1.5h. During, forced linear cooling as BC in sample metal
time_diffusion_at_rt = 7 * 24 * 60 * 60  # 2d * 24h * 60min * 60s

total_time_to_first_weld = time_before_first_weld
total_time_to_cooling = time_before_first_weld + time_welding
total_time_to_rt = total_time_to_cooling + time_cooling_to_rt
total_max_time = total_time_to_rt + time_diffusion_at_rt

# ------------------------------------- Welding - Temps and Timey Whimey stuff -----------------------------------------
animation_frame_stride = 5

# ------------------------------------------------ Debug Helpers -------------------------------------------------------
debug_bead_plots = False  # Set to False to disable

# ------------------------------------------- Diagram and save options -------------------------------------------------
file_name = r"02_Results\00_diffusion_array.h5"  # diffusion_array.h5"
animation_name = r"02_Results\00_diffusion_animation.mp4"  # diffusion_animation.mp4
s_per_frame_part1 = 3.0
# ----------------------------------------------------------------------------------------------------------------------
#                                               Space to print things:

# print(dt)
# print(dt_big)
# print(steps_h_diffuse)
# print(bead_height)


# ---------------------------------------------------------------------------------------------------------------------
"""
**1 Temperaturleitfähigkeit, hier D = lambda / (rho*c), lambda wärmeleitfähigkeit, rho dichte, c spez. Wärmekapazität
    BÖHLER S690 MICROCLEAN: https://www.bohler.de/app/uploads/sites/92/2023/09/productdb/api/s690-microclean_de.pdf

**2 Adapted from Boellinghaus 1995: 8.8 * e-11 * T ** 2.2285 [cm² / s]! Temperature T is in °C here!
    D_H = (8.8 * 10 ** -9 * T ** 2.2285) # [mm² / s]
    https://nagra.ch/wp-content/uploads/2022/08/e_ntb09-004.pdf p.19

"""
