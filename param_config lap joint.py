# Change the relevant values here, so you don't have to scroll through the code so much
# this is just a save of the settings for kjells stuff. you needed to work on IIW
# it ONLY READS THE param_config.py, so this is just a save? you think, you kinda forgot and found this

# param_config.py
# dimensions initialization.py
import math

# ---------------------------------------- Dimensional step size, gradient --------------------------------------------
dx = 0.25  # step size in x direction - if not equal to dy, tripple check solver logic!
dy = 0.25  # step size in y direction - if not equal to dx, tripple check solver logic!
dx2 = dx*dx  # partial differential x
dy2 = dy*dy  # partial differential y

# -------------------- Relevant Starting Conditions - Temperature und Hydrogen concentration --------------------------
t_cool = 160  # Interpass temperature basically. Used for BCs and initialization
t_hot = 1600  # Weld bead temperature. Adjust as needed
t_room = 25  # Surrounding temperature. Remember that the graphs display a different color for t_room! -offset!!
h_cont_initial = 0  # No hydrogen present in bm at the beginning of the simulation
lowest_temp_in_sim = min(t_cool, t_hot, t_room)  # find min temp
highest_temp_in_sim = max(t_cool, t_hot, t_room)  # find max temp

# ---------------------------------------- TEMPERATURE diffusion coefficients -----------------------------------------
# diff_coeff_bm = 15  # Fake but just needed something to make the animation better quickly!
diff_coeff_bm = 5.36768  # D = 20 [W/mK] / (8100 [kg/m³] * 460 [J/kgK] = 5.36768 e-6  m²/s = 5.36768 mm²/s  **1
diff_coeff_wm = 4  # For now just made up numbers
diff_coeff_haz = 4.8  # For now just made up numbers - picked something inbetween bm and wm
diff_coeff_air = 0  # no convection for now! If != 0, check BCs! right now you have Neumann Boundary Condition!
highest_diff_coeff = max(diff_coeff_bm, diff_coeff_wm, diff_coeff_haz)

# ------------------------------------------ HYDROGEN diffusion coefficients ------------------------------------------
# diff_coeff_h = 8.8 * 10 ** - 8  # Not quite in [mm² / s]! Needs * (T ** 2.2285) - applied during simulation **3
# diff_coeff_h = 9.5 * 10 ** - 9  # make sure this does NOT go above T-Diffusion values after T^2.2285! convergence!
# diff_coeff_h = 7.53973 * 10 ** - 8  # Low value from our project
# diff_coeff_h = 1.12213 * 10 ** - 7  # high value from our project
diff_coeff_h = 8.5 * 10 ** - 8  # high one for comparison for PA --- faking it till you got time to fix the boundary!
diff_coeff_h_air = 0  # Not looking at hydrogen in the air (no convection - actualy no convection in T for now either)
minimum_h_diff_in_sample = diff_coeff_h * (lowest_temp_in_sim ** 2.2285)  # Determines time step

# ----------------------------------------------- Time Differential  --------------------------------------------------
dt = dx2 * dy2 / (2 * highest_diff_coeff * (dx2 + dy2))  # highest time step to still be stable
dt_big_calc = dx2 * dy2 / (2 * minimum_h_diff_in_sample * (dx2 + dy2))  # highest time step to still be stable

use_big_dt_override = False  # use the 900s save time override for the last stage (diffusion at RT)
big_dt_override = 900  # calculate every so many seconds, even if you could go faster, to safe the states regardless
if use_big_dt_override and dt_big_calc > big_dt_override:  # change this here!
    dt_big = big_dt_override
    print("Big_dt Overwrite active, new dt_big: " + str(dt_big))
else:
    dt_big = dt_big_calc
    print("Big_dt Overwrite in-active, using dt_big: " + str(dt_big))
# ---------------------------------------- Offsets - Temperature and Diffusion ----------------------------------------
temperature_offset = -5  # The offset is used to make the graphs display the suroundings in grey
hydrogen_offset = -5  # The offset is used to make the graphs display the suroundings in grey

# ------------------------------------ Welding - Temps and Timey Whimey stuff -----------------------------------------
no_of_weld_beads = 3  # no of "blocks" to add during welding - make sure its %2 and fits bead_height!
bead_height = 2  # Half of weld beads * height should probably be weld thickness (th)
bead_width = 2  # Using half of weld width (we) for blocks, for ellipses maybe 3/4-ish of weld width?

haz_creation_temperature = 1350  # Checks if some bm area got hotter than this, creates HAZ there
haz_creation_time_limit_multiplier = 5  # No need to check this every step, but maybe 5 times longer than holding time?

add_bead_mode = "regular_intervals"  # Simply add a weld bead every so many seconds
# add_bead_mode = "interpass_temperature_controlled"  # Add a weld bead when a specific point reaches some temperature

temp_weld_metal = t_hot  # Temperature of new weld block
hydro_weld_metal = 8  # Hydrogen in the new weld block

time_before_first_weld = 6  # Run simulation before starting with first bead. Not necessary, makes for nicer videos [s]
time_for_weld_bead = 250  # Time between welds. Weld block gets added at 0. Temp. held for time_heat_hold. [s]
time_after_last_weld = 100  # Time after last weld. BC in sample edge is held at t_cool this long. [s]
time_heat_hold = 5  # Force the new weld block to have this temp for so long [s]

time_welding = no_of_weld_beads * time_for_weld_bead + time_after_last_weld
time_cooling_to_rt = 1200  # For now set to 1.5h. During, forced linear cooling as BC in sample metal
time_diffusion_at_rt = 2 * 24 * 60 * 60  # 3d * 24h * 60min * 60s
# time_diffusion_at_rt = 1  # debug, delete later
total_time_to_first_weld = time_before_first_weld
total_time_to_cooling = time_before_first_weld + no_of_weld_beads * time_for_weld_bead + time_after_last_weld
total_time_to_rt = time_before_first_weld + no_of_weld_beads * time_for_weld_bead + time_after_last_weld + \
                   time_cooling_to_rt
total_max_time = time_before_first_weld + no_of_weld_beads * time_for_weld_bead + time_after_last_weld + \
                 time_cooling_to_rt + time_diffusion_at_rt

# ---------------------------------------------- Calculation steps ----------------------------------------------------
# # Eventually this will be depreciated!
# steps_per = math.ceil(1/dt)  # "rounded up" time steps per second
# steps_before = steps_per * time_before_first_weld  # steps before the first weld bead
# steps_welding = steps_per * total_time_welding  # steps per sec * total welding time
# steps_cooling_to_rt = steps_per * total_time_cooling_to_rt  # steps per sec * 2 hours for example
# steps_h_diffuse = math.ceil(total_time_diffusion_at_rt/dt_big)  # e.g. 7 days | 604800s / 544 =  1112 steps
#
# steps_to_rt = steps_before + steps_welding + steps_cooling_to_rt
# nsteps = steps_before + steps_welding + steps_cooling_to_rt + steps_h_diffuse  # Total steps

# ------------------------------------------- Diagram and save options ------------------------------------------------
file_name = "diffusion_array.h5"
s_per_frame_part1 = 2
# ---------------------------------------------------------------------------------------------------------------------
#                                               Space to print things:

# print(dt)
# print(dt_big)
# print(steps_h_diffuse)
# print(bead_height)


# ---------------------------------------------------------------------------------------------------------------------
"""
**1 Temperaturleitfähigkeit, hier D = lambda / (rho*c), lambda wärmeleitfähigkeit, rho dichte, c spez. Wärmekapazität
    BÖHLER S690 MICROCLEAN: https://www.bohler.de/app/uploads/sites/92/2023/09/productdb/api/s690-microclean_de.pdf

**2 From Tobi Diss p.65. for alpha iron (until ~723)

**3 Adapted from Boellinghaus 1995: 8.8 * e-11 * T ** 2.2285 [cm² / s]! Temperature T is in °C here!
    D_H = (8.8 * 10 ** -9 * T ** 2.2285) # [mm² / s]
    https://nagra.ch/wp-content/uploads/2022/08/e_ntb09-004.pdf p.19

"""
