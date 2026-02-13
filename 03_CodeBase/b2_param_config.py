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

from b4_functions import in_results, get_spec_value_at_temp, find_min_max_value

""" ------------------------------------------------ User Settings  ------------------------------------------------ """
""" ---------------------- Main Simulation Settings ----------------------------- """
model_version = "0.4.0"  # For provenance. Don't change, unless you customize logic. Then its yours, Yay!
simulation_type = "butt joint"  # Options: "lap joint", "butt joint" and "iso3690"
diffusion_scheme = 0  # 0 = centered D * Laplacian) | 1 = flux-conservative | 2 = flux + mu-driven (solubility incl.)

""" ---------------------- Spacial Discretization (Step Size) ------------------- """
dx = 0.5  # step size in x direction - if not equal to dy, tripple check solver logic!
dy = 0.5  # step size in y direction - if not equal to dx, tripple check solver logic!

""" ---------------------- Weld Bead Settings ----------------------------------- """
add_bead_mode = "regular_intervals"  # Options: "regular_intervals" and "interpass_temperature_controlled"

no_of_weld_beads = 14  # no of "blocks" during welding: Butt joint: must be %2, fit bead_height! Lap Joint: max 4
bead_height = 3.0  # Half of weld beads * height should probably be weld thickness (th) (2.8 for iso?
bead_width = 12  # Using half of weld width (we) for blocks, for ellipses maybe 3/4-ish of weld width? 60%?
bead_scales = [(1.0, 1.0), (1.0, 1.0), (1.6, 1.6), (3.0, 3.0)]  # Used for lap joint and iso3690

""" ---------------------- Temporal Discretization and Settings ------------------ """
time_before_first_weld = 5  # Run simulation before starting with first bead. Not necessary, makes for nicer videos [s]
time_for_weld_bead = 480  # Time between welds. Weld block gets added at 0. Temp. held for time_heat_hold. [s]
time_after_last_weld = 480  # Time after last weld. BC in sample edge is held at t_cool this long. [s]
time_heat_hold = 3  # Force the new weld block to have this temp for so long [s]

time_cooling_to_rt = 1 * 60 * 60  # For now set to 1.5h. During, forced linear cooling as BC in sample metal
time_diffusion_at_rt = 1 * 24 * 60 * 60  # 2d * 24h * 60min * 60s

safety_factor = 1  # 1 = stable (lower maybe better temporal convergence) | Default and recommended = 1
use_big_dt_override = True  # Diffusion at RT slow -> large automatic dt possible. Manual override to use smaller dt?
big_dt_override = 900  # If override = True, calculate every so many seconds, even if you could go faster.

guess_adaptive_stable_dt = False  # Try to use a bigger dt where possible - highly experimental right now

""" ---------------------- Relevant (Starting) Conditions - Thermal ------------- """
t_cool = 160  # Interpass temperature basically. Used for BCs, initialization and as starting temperature
t_hot = 1550  # Weld bead temperature. Adjust as needed
t_room = 25  # Surrounding temperature. Use 0.001 instead of 0 for ISO3690!

haz_creation_temperature = 1350  # Checks if some bm area got hotter than this, creates HAZ there
haz_creation_check_time_window = 10  # no need to check for ever after welding, but maybe for 10s?

t_conv_air = 1e-7  # [W/mm²/K] | ≈ 0.1 W/m²K | Top plate, still air. Guess / calibrate from temperature measurements
t_conv_cu = 3e-4   # [W/mm²/K], ≈ 300 W/m²K, Copper contact. "Convection" doesn't really make sense. Consider temperature T_cu(t) later!

""" ---------------------- Relevant (Starting) Conditions - Hydrogen ----------- """
hydro_weld_metal = 100  # Hydrogen in the new weld block, for now set as "100%"
h_cont_initial = 0  # Almost always = 0, cuz no hydrogen present in BM in the beginning

reference_from_iso3690 = 2.5  # Used for lap joints with hydrogen from the inside. Read as "2.5 is 100%"
pipe_line_inner_hydrogen = "variable"  # Options: "constant", "variable" | variable means sieverts law (temp depend)
h_on_the_inside = 0  # Used for the constant h lap joint boundary condition on the inside of the weld / pipeline

t_conv_h2 = 5e-4   # - UNUSED - [W/mm²/K], ≈ 500 W/m²K, underside, forced hydrogen. Guess or calibrate - Change dt?

""" ---------------------- Save and Animation Options --------------------------- """
file_name = str(in_results("00_diffusion_array.h5", mkdir=True))  # diffusion_array.h5"
animation_name = str(in_results("00_diffusion_animation.mp4", mkdir=True))  # diffusion_animation.mp4

s_per_frame_part1 = 5  # Save every so many seconds (dt is usually < 0.001s)
animation_frame_stride = 5  # Only render every n-th frame (used in animation/video scripts)

use_sparse_saving_in_just_diffusion = True  # If True, save less often after welding (long RT diffusion)
s_per_frame_just_diffusion_sparse = 300.0  # Seconds per save during just-diffusion phase when sparse saving is ON

""" ---------------------- Microstructure Parameters (D | D_H | S) -------------- """
microstructures = ["none", "base_metal", "weld_metal", "HAZ"]  # Check solver logic if adding more, but should work

# Thermal conductivity as per DIN EN 1993-1-2:2010-12 **1
microstructure_thermal_diff = """
material: none
] -inf, +inf ]: D = 0

material: base_metal
] -inf, 20 ]:   D = (54 - 0.0333*20) / (7850 * (425 + 0.773*20 - 1.69e-3*20**2 + 2.22e-6*20**3)) * 1e6
] 20, 600 ]:    D = (54 - 0.0333*T_C) / (7850 * (425 + 0.773*T_C - 1.69e-3*T_C**2 + 2.22e-6*T_C**3)) * 1e6
] 600, 800 ]:   D = 5.7029134143 + (((T_C - 600)/200)**2 * (3 - 2*((T_C - 600)/200))) * (-1.3734270193)
] 800, 900 ]:   D = 27.3 / (7850 * (545 + 17820/(T_C - 731))) * 1e6
] 900, 1200 ]:  D = 27.3 / (7850 * 650) * 1e6
] 1200, +inf ]: D = 27.3 / (7850 * 650) * 1e6

material: weld_metal
] -inf, 20 ]:   D = 1.005 * (54 - 0.0333*20) / (7850 * (425 + 0.773*20 - 1.69e-3*20**2 + 2.22e-6*20**3)) * 1e6
] 20, 600 ]:    D = 1.005 * (54 - 0.0333*T_C) / (7850 * (425 + 0.773*T_C - 1.69e-3*T_C**2 + 2.22e-6*T_C**3)) * 1e6
] 600, 800 ]:   D = 5.7029134143 + (((T_C - 600)/200)**2 * (3 - 2*((T_C - 600)/200))) * (-1.3734270193)
] 800, 900 ]:   D = 1.005 * 27.3 / (7850 * (545 + 17820/(T_C - 731))) * 1e6
] 900, 1200 ]:  D = 1.005 * 27.3 / (7850 * 650) * 1e6
] 1200, +inf ]: D = 1.005 * 27.3 / (7850 * 650) * 1e6

material: HAZ
] -inf, 20 ]:   D = 0.995 * (54 - 0.0333*20) / (7850 * (425 + 0.773*20 - 1.69e-3*20**2 + 2.22e-6*20**3)) * 1e6
] 20, 600 ]:    D = 0.995 * (54 - 0.0333*T_C) / (7850 * (425 + 0.773*T_C - 1.69e-3*T_C**2 + 2.22e-6*T_C**3)) * 1e6
] 600, 800 ]:   D = 5.7029134143 + (((T_C - 600)/200)**2 * (3 - 2*((T_C - 600)/200))) * (-1.3734270193)
] 800, 900 ]:   D = 0.995 * 27.3 / (7850 * (545 + 17820/(T_C - 731))) * 1e6
] 900, 1200 ]:  D = 0.995 * 27.3 / (7850 * 650) * 1e6
] 1200, +inf ]: D = 0.995 * 27.3 / (7850 * 650) * 1e6
""".strip()

#  Microstructure/temperature dependend hydrogen diffusion coefficient **2
microstructure_hydrogen_diff = """
material: none
] -inf, +inf ]: D_H = 0

material: base_metal
] -inf, 20 ]:    D_H_mean = 0.07465 * exp(-11072 / (R * 293.15))
] 20, 200 ]:     D_H_mean = 0.07465 * exp(-11072 / (R * T_K))
] 200, 740 ]:    D_H_mean = 0.1104  * exp(-12437 / (R * T_K))
] 740, 1450 ]:   D_H_mean = 0.8753  * exp(-46396 / (R * T_K))
] 1450, 1540 ]:  D_H_mean = 1.2104  * exp(-37785 / (R * T_K))
] 1540, 2000 ]:  D_H_mean = 1.1578  * exp(-37007 / (R * T_K))

material: weld_metal
] -inf, 20 ]:    D_H = 0.07465 * exp(-11072 / (R * 293.15))
] 20, 200 ]:     D_H = 0.07465 * exp(-11072 / (R * T_K))
] 200, 740 ]:    D_H = 0.1104  * exp(-12437 / (R * T_K))
] 740, 1450 ]:   D_H = 0.8753  * exp(-46396 / (R * T_K))
] 1450, 1540 ]:  D_H = 1.2104  * exp(-37785 / (R * T_K))
] 1540, 2000 ]:  D_H = 1.1578  * exp(-37007 / (R * T_K))

material: HAZ
] -inf, 20 ]:    D_H = 0.07465 * exp(-11072 / (R * 293.15))
] 20, 200 ]:     D_H = 0.07465 * exp(-11072 / (R * T_K))
] 200, 740 ]:    D_H = 0.1104  * exp(-12437 / (R * T_K))
] 740, 1450 ]:   D_H = 0.8753  * exp(-46396 / (R * T_K))
] 1450, 1540 ]:  D_H = 1.2104  * exp(-37785 / (R * T_K))
] 1540, 2000 ]:  D_H = 1.1578  * exp(-37007 / (R * T_K))
""".strip()

# Solubility used for chemical potential driven diffusion. Due to lack of data, relative values only for now!
# Idea: baseline S=1 below transformation, linear ramp to S=10 across 740–800°C, then constant.
microstructure_solubility = """
material: none
] -inf, +inf ]: S = 0

material: base_metal
] -inf, 740 ]:  S = 0.5
] 740, 800 ]:   S = 0.5 + 4.5 * ((T_C - 740) / (800 - 740))
] 800, +inf ]:  S = 5

material: weld_metal
] -inf, 740 ]:  S = 1
] 740, 800 ]:   S = 1 + 9 * ((T_C - 740) / (800 - 740))
] 800, +inf ]:  S = 10

material: HAZ
] -inf, 740 ]:  S = 1
] 740, 800 ]:   S = 1 + 9 * ((T_C - 740) / (800 - 740))
] 800, +inf ]:  S = 10
""".strip()

""" ------------------------------------------ Advanced and Resultant Setting  ------------------------------------- """
""" ---------------------- Spacial Discretization (Gradients) ------------------- """
dx2 = dx*dx  # partial differential x
dy2 = dy*dy  # partial differential y
inv_dx2 = 1.0 / (dx*dx)  # Inverse partial differential x
inv_dy2 = 1.0 / (dy*dy)  # Inverse partial differential y

""" ---------------------- Resulting Conditions - Thermal ----------------------- """
lowest_temp_in_sim = min(t_cool, t_hot, t_room)  # find min temp
highest_temp_in_sim = max(t_cool, t_hot, t_room)  # find max temp

precalc_grid_step = 0.5  # Default 1 | speed impact minor but 0.1 is likely overkill
precalc_min_temp = lowest_temp_in_sim - precalc_grid_step  # Padded for safety and rounding errors
precalc_max_temp = highest_temp_in_sim + precalc_grid_step  # Padded for safety and rounding errors

max_microstructure_thermal_diff = find_min_max_value(microstructure_thermal_diff, microstructures, "D",
                                                     t_min=precalc_min_temp, t_max=precalc_max_temp,
                                                     step_c=precalc_grid_step, agg="max")
max_hydrogen_diff_at_room_temp = (
    get_spec_value_at_temp(microstructure_hydrogen_diff, microstructures, t_room, "D_H", agg="max"))

""" ---------------------- Timey Whimey Whibbly Whobbly ----------------------- """
dt = safety_factor * (dx2 * dy2 / (2 * max_microstructure_thermal_diff * (dx2 + dy2)))  # highest time step to still be stable
dt_big_calc = safety_factor * (dx2 * dy2 / (2 * max_hydrogen_diff_at_room_temp * (dx2 + dy2)))

time_welding = no_of_weld_beads * time_heat_hold + (no_of_weld_beads - 1) * time_for_weld_bead + time_after_last_weld
total_time_to_first_weld = time_before_first_weld
total_time_to_cooling = time_before_first_weld + time_welding
total_time_to_rt = total_time_to_cooling + time_cooling_to_rt
total_max_time = total_time_to_rt + time_diffusion_at_rt

if use_big_dt_override and dt_big_calc > big_dt_override:  # change this here!
    dt_big = big_dt_override
    print("Big_dt Overwrite ON, new dt_big: " + str(dt_big))
else:
    dt_big = dt_big_calc
    print("Big_dt Overwrite OFF, using dt_big: " + str(dt_big))

""" ---------------------- Convection / Cooling / Robin BC -------------------- """
bm_k = 0.02  # [W/mmK] thermal conductivity

# Precompute handy factors (for constant h, k). Robin uses 2*h/(k*Δ)
coef_robin_x_air = 2.0 * t_conv_air / (bm_k * dx)  # [1/mm]
coef_robin_y_air = 2.0 * t_conv_air / (bm_k * dy)  # [1/mm]

coef_robin_x_h2 = 2.0 * t_conv_h2 / (bm_k * dx)  # [1/mm]
coef_robin_y_h2 = 2.0 * t_conv_h2 / (bm_k * dy)  # [1/mm]

coef_robin_x_cu = 2.0 * t_conv_cu / (bm_k * dx)  # [1/mm]
coef_robin_y_cu = 2.0 * t_conv_cu / (bm_k * dy)  # [1/mm]

""" ---------------------- Save and Animation Options --------------------------- """
temperature_offset = -5  # The offset is used to make the graphs display the suroundings in grey
hydrogen_offset = -5  # The offset is used to make the graphs display the suroundings in grey

""" ---------------------- Informational Metadata (stored in .h5 (JSON)) ------ """
convention = "rows: top->bottom, cols: left->right"
units_temperature = "degC"
units_hydrogen = "%"
units_diffusivity = "mm^2/s"

""" ------------------------------------------------- Debug Helpers  ------------------------------------------------"""
debug_bead_plots = False  # Set to False to disable

""" ---------------------- Space to print things: ----------------------------- """
# print(dt)
# print(dt_big)
# print(steps_h_diffuse)
# print(bead_height)

""" ------------------------------------------ Literature, Sources for Values  --------------------------------------"""
"""
**1 Temperaturleitfähigkeit, hier D = lambda / (rho*c), lambda wärmeleitfähigkeit, rho dichte, c spez. Wärmekapazität
    DIN EN 1993-1-2:2010-12 for the values between 25 and 1200°C, they are constant then (we dont worry about liquid)
    
EXAMPLE    material: base_metal
] 600, 735 ]:   D = (54 - 0.0333*T_C) / (7850 * (666 + 13002/(738 - T_C))) * 1e6
] 735, 800 ]:   D = (54 - 0.0333*T_C) / (7850 * (545 + 17820/(T_C - 731))) * 1e6

# Here we have the characteristic thermodynamically modelled dip due to phase transformation
# For simplicity and ease of temperature field calibration, we interpolate between 600 and 800.
# but the above is directly from EN1993

# Alternativly, direct interpolation: 
] 600, 800 ]:   D = 5.7029134143 + (((T_C - 600)/200)**2 * (3 - 2*((T_C - 600)/200))) * (-1.3734270193)

# Respectivly for the hydrogen diffusion:

# Hydrogen Diff Min
] -inf, 20 ]:  D_H = 8.7615 * (10 ** - 9) * (20 ** 2.2285)
] 20, 200 ]:   D_H = 8.7615 * (10 ** - 9) * (T_C ** 2.2285)
] 200, 740 ]:  D_H = 8.9963 * (10 ** - 9) * (T_C ** 2.2480)
] 740, 1450 ]: D_H = 0.6736  * exp(-45086 / (R * T_K))
] 1450, 1540 ]:D_H = 28.7905  * exp(-93534 / (R * T_K))
] 1540, 2000 ]:D_H = 0.246  * exp(-15450 / (R * T_K))

# Hydrogen Diff Max
] -inf, 20 ]:    D_H_max = 0.076 * exp(-9562 / (R * 293.15))
] 20, 200 ]:     D_H_max = 0.076 * exp(-9562 / (R * T_K))
] 200, 740 ]:    D_H_max = 8.1056 * (10 ** -6) * (T_C ** 1.2528)
] 740, 1450 ]:   D_H_max = 1.0691 * exp(-41624 / (R * T_K))
] 1450, 1540 ]:  D_H_max = 0.437  * exp(-17273 / (R * T_K))
] 1540, 2000 ]:  D_H_max = 0.437  * exp(-17273 / (R * T_K))

# Hydrogen Diff Mean
] -inf, 20 ]:    D_H_mean = 0.07465 * exp(-11072 / (R * 293.15))
] 20, 200 ]:     D_H_mean = 0.07465 * exp(-11072 / (R * T_K))
] 200, 740 ]:    D_H_mean = 0.1104  * exp(-12437 / (R * T_K))
] 740, 1450 ]:   D_H_mean = 0.8753  * exp(-46396 / (R * T_K))
] 1450, 1540 ]:  D_H_mean = 1.2104  * exp(-37785 / (R * T_K))
] 1540, 2000 ]:  D_H_mean = 1.1578  * exp(-37007 / (R * T_K))

# Or for Solubility:
material: none
] -inf, +inf ]: S = 0
material: base_metal
] -inf, +inf ]: S = 1
material: weld_metal
] -inf, +inf ]: S = 1
material: HAZ
] -inf, +inf ]: S = 1

**2 Adapted from Boellinghaus 1995: 8.8 * e-11 * T ** 2.2285 [cm² / s]! Temperature T is in °C here!
    D_H = (8.8 * 10 ** -9 * T ** 2.2285) # [mm² / s]
"""
