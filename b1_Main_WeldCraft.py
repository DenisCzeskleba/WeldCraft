"""
Main simulation driver for WeldCraft.

This script initializes and executes the thermal–hydrogen diffusion simulation
for various welding configurations (e.g., lap joint, butt joint, ISO 3690).
It handles setup, time stepping, phase transitions (welding, cooling, diffusion),
and periodic saving of simulation data to an HDF5 file.

Modules required:
- b3_initialization.py: defines geometry, materials, and initial conditions
- b2_param_config.py: contains all the "knobs" the user can tune
- b4_functions.py: provides numerical operations and boundary handling
"""

import sys
from b3_initialization import *
from tqdm import tqdm
import h5py
import os
import math

# -------------------------------------------------- SIMULATION TYPE --------------------------------------------------
simulation_type = get_value("simulation_type")  # Simlulation type - Options: "lap joint", "butt joint" and "iso3690"
add_bead_mode = get_value("add_bead_mode")  # Options: "regular_intervals" and "interpass_temperature_controlled"

# ------------------------------------- Simulation size and other dimensions [mm] -------------------------------------
try:  # this should end the sim if the wrong type is chosen, no need to check again later?
    joint_edge, dim_rows, dim_columns, le, ri, we, th, su_h, su_w, fr_le, fr_ri, fr_ab, fr_be = weld_sample(simulation_type)

except ValueError as e:
    print(f"Error: {e}")
    raise SystemExit

# --------------------------------------------- Step size, steps, gradient --------------------------------------------
dx, dy = get_value("dx"), get_value("dy")  # Spacial discretization
nx, ny = int(dim_rows / dx), int(dim_columns / dy)  # for readabilty - x should be horizontal, y vertical
dx2, dy2 = get_value("dx2"), get_value("dy2")  # Precalculate dx2, dy2

# --------------------------------------- initialization and starting conditions --------------------------------------
u0, u, h0, h, D, D_H, t_cool, t_hot, t_room = initialize(simulation_type, nx, ny, dx, dy, le, we, th, su_h, su_w, fr_le, fr_ab)

# -------------------------------------------------- Simulation Setup -------------------------------------------------

# ---------- Welding parameters ----------
bead_height = get_value("bead_height")  # Set the height for a weld bead
bead_width = get_value("bead_width")  # Set the width of a weld bead // Unused??
t_weld_metal = get_value("temp_weld_metal")  # Temperature of the new welding block
h_weld_metal = get_value("hydro_weld_metal")  # Hydrogen content. Usually just in %
D_bm = get_value("diff_coeff_bm")  # Diffusion coefficient of base metal (Temperature)
D_weld_metal = get_value("diff_coeff_wm")  # Diffusion coefficient of weld metal (Temperature)
D_haz = get_value("diff_coeff_haz")  # Diffusion coefficient of HAZ (Temperature)
D_H_weld_metal = get_value("diff_coeff_h")  # Diffusion coefficient of weld metal (Hydrogen)
keep_temp_duration = get_value("time_heat_hold")  # Force the new weld block to have this temp for so long
haz_creation_temperature = get_value("haz_creation_temperature")  # Temperature for HAZ creation
room_temp = get_value("t_room")  # Room temperature
pipe_line_inner_hydrogen = get_value("pipe_line_inner_hydrogen")
hydro_inside = get_value("h_on_the_inside")
boundary_temperature = get_value("t_cool")
inv_dx2 = get_value("inv_dx2")
inv_dy2 = get_value("inv_dy2")
coef_robin_x_air = get_value("coef_robin_x_air")
coef_robin_y_air = get_value("coef_robin_y_air")
coef_robin_x_h2 = get_value("coef_robin_x_h2")
coef_robin_y_h2 = get_value("coef_robin_y_h2")
coef_robin_x_cu = get_value("coef_robin_x_cu")
coef_robin_y_cu = get_value("coef_robin_y_cu")

# ---------- Time differential ----------
# H-Diffusion is always much much lower than heat, so just use heat
dt = get_value("dt")  # highest time step to still be stable: dx2 * dy2 / (2 * highest_diff_coeff * (dx2 + dy2))
dt_big = get_value("dt_big")  # After cooling to RT: only simuluate H diffusion. Bigger time steps possible!

print("Time step for this simulation: " + str(dt) + "s")  # For debug
dt_changed = False

# -------- Simulation Time Setup ---------
if add_bead_mode == "regular_intervals":
    change_times = generate_change_steps()  # Generate the list of times where a new weld bead starts

    # Combined safety check for maximum supported beads per joint type
    max_beads = {"lap joint": 4, "iso3690": 3}
    if simulation_type in max_beads and len(change_times) > max_beads[simulation_type]:
        print(f"\n !! [ERROR] {simulation_type} doesn't support {len(change_times)} beads. "
              f"Please reduce the number of beads or switch to a different joint type. !!")
        sys.exit(1)  # <-- CLEAN exit, no traceback, safe for batch runs

    print("A new weld bead will be added at the following times (s): " + ", ".join(map(str, change_times)))


else:
    # This will get re-added for interpass_temperature_controlled option
    change_times = generate_change_steps(dt)  # CHANGE THIS WHEN YOU RE-IMPLEMENT IT

time_before_first_weld = get_value("time_before_first_weld")  # in seconds
time_welding = get_value("time_welding")
time_cooling_to_rt = get_value("time_cooling_to_rt")
time_diffusion_at_rt = get_value("time_diffusion_at_rt")

total_time_to_first_weld = get_value("total_time_to_first_weld")
total_time_to_cooling = get_value("total_time_to_cooling")
total_time_to_rt = get_value("total_time_to_rt")
total_max_time = get_value("total_max_time")

current_time = 0.0  # Initialize simulation time

# ---------- Numerics Mumbo Jumbo ----------
# Figure out calculated area (D > 0) and the edges of that area (faces)
mask, faces = compute_mask_and_faces(D)  # D is initialized with base metal temperature diffusion coefficient

# Initialize variables for bead addition
cwi = 0  # current weld indicator
cci = 0  # current change indicator

current_phase = None  # Initial phase
current_forced_part_temperature = t_cool  # Initial border temperature

start_cooling_temp = -999  # Drop into ice water - cooling setup for ISO3690
new_area = np.zeros_like(D, dtype=bool)  # For adding beads. Dont reheat/hydrogenate prev. weld metal

# ---------- Save matrices ----------
file_name = get_value("file_name")  # File name to save the matrices as
save_every_x_s = get_value("s_per_frame_part1")
print("Saving a frame every " + str(save_every_x_s) + "s during main simulation loop")
save_counter = 0  # used in the saving loop
last_save_time = 0  # used to track when the last save occurred

if True:  # Change to False if you want to keep the file or mess with the filename
    if os.path.exists(file_name):  # Delete old files
        os.remove(file_name)

# Create file, write coordinates once, and stamp file-level convention
with h5py.File(file_name, 'a') as hf:
    # Coordinate vectors (cell centers, in mm, origin at top-left: rows increase downward)
    x_coords = (np.arange(nx) + 0.5) * dx  # mm, left→right (+0.5 to get cell center)
    y_coords = (np.arange(ny) + 0.5) * dy  # mm, top→bottom (+0.5 to get cell center)

    hf.create_dataset('/x', data=x_coords)
    hf['/x'].attrs['units'] = 'mm'
    hf['/x'].attrs['definition'] = 'cell centers, x = columns (left→right)'

    hf.create_dataset('/y', data=y_coords)
    hf['/y'].attrs['units'] = 'mm'
    hf['/y'].attrs['definition'] = 'cell centers, y = rows (top→bottom)'

    # File-level metadata for clarity
    hf.attrs['convention'] = 'rows: top→bottom, cols: left→right'
    hf.attrs['units_temperature'] = '°C'
    hf.attrs['units_hydrogen'] = '%'
    hf.attrs['units_diffusivity'] = 'mm^2/s'

    # --- Weld phase timing information ---
    hf.attrs['time_before_first_weld'] = time_before_first_weld
    hf.attrs['time_welding'] = time_welding
    hf.attrs['time_cooling_to_rt'] = time_cooling_to_rt
    hf.attrs['time_diffusion_at_rt'] = time_diffusion_at_rt

    # --- Cumulative times ---
    hf.attrs['total_time_to_first_weld'] = total_time_to_first_weld
    hf.attrs['total_time_to_cooling'] = total_time_to_cooling
    hf.attrs['total_time_to_rt'] = total_time_to_rt
    hf.attrs['total_max_time'] = total_max_time

# ---------- Optimization attempts ----------
dudx2 = np.zeros((ny, nx))  # preallocate the derivative matrices
dudy2 = np.zeros((ny, nx))  # this should save some time recreating empty ones each loop
dhdx2 = np.zeros((ny, nx))
dhdy2 = np.zeros((ny, nx))

# Dont check for HAZ all the time!
check_for_HAZ_creation = 0
HAZ_creation_time_start = 0

max_u0 = get_value("highest_temp_in_sim")  # precalculate diffusion values for different temperatures
precomputed_powers = np.arange(max_u0 + 1) ** 2.2285  # Create an array to store precomputed values of u0**2.2285

# Preset some functions/values for the hydrogen inside pipeline / ISO3690 boundary conditions
# cache: solubility lookup (100 bar table) and the inside row index
sol_fn = get_h_solubility_fn()                      # build once
row_inside_const = int((th + su_h + fr_ab) / dx)    # compute once
ign_ab = int(fr_ab / dx)                  # used for iso3690

# handy constant to avoid recomputing every step
two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)

# Main loop and progress bar
bar_format = '{desc} {percentage:3.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]'
pbar = None

# sys.exit()  # debug, obviously remove \(e^{\frac{-11072}{R\cdot T}}\)

# Main Loop
while current_time <= total_max_time:
    # Determine the current phase based on simulation time

    # ---- PRE-WELDING PHASE ----
    if current_time < total_time_to_first_weld:
        # Pre-welding phase - possibly add over heating and cooling before weld experiment
        insert_stuff_here_later = 1

        # Simulation book keeping and progress bar stuff
        if current_phase != "pre-welding":
            safe_close_pbar(pbar)
            phase_total_time = total_time_to_first_weld - current_time
            total_iterations = int(np.ceil(phase_total_time / dt))
            pbar = tqdm(total=total_iterations, bar_format=bar_format, desc='Simulating pre-welding phase')
            current_phase = "pre-welding"

    # ---- WELDING PHASE ----
    elif current_time < total_time_to_cooling:
        # Add weld beads, manipulate diffusion coefficients etc.
        if change_times[cwi] <= current_time <= change_times[cwi] + keep_temp_duration:

            if cci == 0:

                check_for_HAZ_creation = 1  # Auto-HAZ creation
                HAZ_creation_time_start = current_time

            # Change stuff, like adding a weld bead and such (in b4_functions.py)
            # Info: cci is used to determine if we should also manipulate the diffusion values
            u0, h0, D, D_H, mask, faces, new_area = manipulate_simulation(simulation_type, u0, h0, D, D_H, cwi, cci, t_weld_metal,
                                                                h_weld_metal, D_bm, D_weld_metal, D_haz, D_H_weld_metal,
                                                                dx, dy, le, we, fr_ab, fr_be, su_h, fr_le, th, mask,
                                                                faces, new_area)

            # Debug to show temperature diffusion coefficients
            if get_value("debug_bead_plots") and cci == 0:
                debug_show_DH(D, dy, dy, title=f"Temperature diffusion coefficient", block=True)

            # You manipulated the simulation! Indicate this!
            cci += 1

            # If this is the last time this gets triggered this bead (if current time plus next dt would be bigger)
            if current_time + dt >= change_times[cwi] + keep_temp_duration:
                cci = 0  # Make it 0 again so you can redo the bead dimensions
                if cwi == len(change_times) - 1:
                    pass  # reached last change time - pass
                else:
                    cwi += 1

        # Add HAZ
        if check_for_HAZ_creation == 1:
            if np.max(u) >= haz_creation_temperature:
                possible_locations = u > haz_creation_temperature
                haz_location = possible_locations & (D != D_weld_metal)
                D[haz_location] = D_haz
            if current_time - HAZ_creation_time_start > get_value("haz_creation_check_time_window"):
                check_for_HAZ_creation = 0

        # Simlation book keeping and progressbar stuff
        if current_phase != "welding":
            safe_close_pbar(pbar)
            phase_total_time = total_time_to_cooling - current_time
            total_iterations = int(np.ceil(phase_total_time / dt))
            pbar = tqdm(total=total_iterations, bar_format=bar_format, desc='Simulating welding phase')
            current_phase = "welding"

    # ---- COOLING TO ROOM TEMPERATURE PHASE ----
    elif current_time < total_time_to_rt:
        cooling_time_elapsed = current_time - total_time_to_cooling

        # Currently we force the whole part to be evenly tempered.
        k = 4.605170186 / time_cooling_to_rt  # 99% cooled by the end

        if start_cooling_temp == -999:
            hot_cells = u[u > t_cool]
            start_cooling_temp = np.mean(hot_cells) if hot_cells.size else np.mean(u)

        # if simulation_type == "lap joint" or simulation_type == "butt joint":  # More realistic cooling for Butt/Lap
        #     current_forced_part_temperature = t_room + (t_cool - t_room) * math.exp(-k * cooling_time_elapsed)
        # elif simulation_type == "iso3690":
        #     current_forced_part_temperature = t_room + (start_cooling_temp - t_room) * math.exp(-k * cooling_time_elapsed)

        current_forced_part_temperature = t_room + (start_cooling_temp - t_room) * math.exp(-k * cooling_time_elapsed)

        # Simlation book keeping and progressbar stuff
        if current_phase != "cooling":
            safe_close_pbar(pbar)
            phase_total_time = total_time_to_rt - current_time
            total_iterations = int(np.ceil(phase_total_time / dt))
            pbar = tqdm(total=total_iterations, bar_format=bar_format,
                        desc='Simulating cooling phase')
            current_phase = "cooling"

    # ---- DIFFUSION AT ROOM TEMPERATURE PHASE ----
    else:
        # Change to much bigger dt and to static temps and simplify diffusion (not temp depend. anymore)
        if not dt_changed:
            dt = dt_big  # Switch to larger time step for diffusion phase
            dt_changed = True

            # Update diffusion coefficients for this phase
            D_H_in_air = get_value("diff_coeff_h_air")
            static_D_H = get_value("minimum_h_diff_in_sample")
            D_H = np.where(D_H != D_H_in_air, static_D_H, D_H)

            # Set temperature to room temperature
            t_static_room = get_value("t_room")
            u = np.where(u >= t_static_room, t_static_room, u)
            u0 = np.where(u0 >= t_static_room, t_static_room, u0)

        # Simulation book keeping and progressbar stuff
        if current_phase != "just diffusion":
            safe_close_pbar(pbar)
            phase_total_time = total_max_time - current_time
            total_iterations = int(np.ceil(phase_total_time / dt)) + 1
            pbar = tqdm(total=total_iterations, bar_format=bar_format,
                        desc='Simulating diffusion at RT')
            current_phase = "just diffusion"

    # Actual calculation here
    u0, u, h0, h, dudx2, dudy2, dhdx2, dhdy2, = do_timestep(simulation_type, current_phase, u0, u, D, h0, h, D_H, mask,
                                                            faces, dt, dx2, dy2, ign_ab, fr_le, th, su_h, we, dx, dudx2,
                                                            dudy2, dhdx2, dhdy2, room_temp, precomputed_powers, sol_fn,
                                                            row_inside_const, two_over_sqrt_pi,
                                                            current_forced_part_temperature, pipe_line_inner_hydrogen,
                                                            hydro_inside, boundary_temperature, inv_dx2, inv_dy2,
                                                            coef_robin_x_air, coef_robin_y_air, coef_robin_x_h2,
                                                            coef_robin_y_h2, coef_robin_x_cu, coef_robin_y_cu, t_room,
                                                            joint_edge)

    # Saving the simulation data
    if current_phase != "just diffusion":  # normal state, save every so often (~2s)

        if current_time >= last_save_time + save_every_x_s or current_time == 0:
            last_save_time = current_time

            u_save_name = f'u_snapshot_{save_counter:05d}'  # Unique name for 'u'
            h_save_name = f'h_snapshot_{save_counter:05d}'  # Unique name for 'h'
            t_save_name = f't_snapshot_{save_counter:05d}'  # Unique name for 'time'
            d_save_name = f'd_snapshot_{save_counter:05d}'  # Unique name for 'd' (heat diffusion coeff)
            save_counter += 1

            with h5py.File(file_name, 'a') as hf:
                # Create a dataset with the unique name and store the data
                hf.create_dataset(u_save_name, data=u)
                hf.create_dataset(h_save_name, data=h)
                hf.create_dataset(t_save_name, data=current_time)
                hf.create_dataset(d_save_name, data=D)

    else:  # Just diffusion phase, save every pic!

        # Use this if you wanna simulate for years after welding to save more rarely
        # if int(current_time) // 7200 > int(
        #         last_save_time) // 7200:
        #     last_save_time = current_time

        u_save_name = f'u_snapshot_{save_counter:05d}'  # Unique name for 'u'
        h_save_name = f'h_snapshot_{save_counter:05d}'  # Unique name for 'h'
        t_save_name = f't_snapshot_{save_counter:05d}'  # Unique name for 'time'
        d_save_name = f'd_snapshot_{save_counter:05d}'  # Unique name for 'd' (heat diffusion coeff)
        save_counter += 1

        with h5py.File(file_name, 'a') as hf:
            # Create a dataset with the unique name and store the data
            hf.create_dataset(u_save_name, data=u)
            hf.create_dataset(h_save_name, data=h)
            hf.create_dataset(t_save_name, data=current_time)
            hf.create_dataset(d_save_name, data=D)

    # Update time, make sure to include the very end of the given time window
    if current_time + dt <= total_max_time:
        current_time += dt  # Keep track of current time of simulation
    elif current_time == total_max_time:
        current_time += 1
    elif current_time + dt > total_max_time:
        current_time = total_max_time

    pbar.update(1)

pbar.close()
time_to_print = format_simulation_time(current_time)
print(f"\nSimulation finished. Total simulation time: {time_to_print}")
