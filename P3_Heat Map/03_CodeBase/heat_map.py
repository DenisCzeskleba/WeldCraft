"""Focused thermal-only simulator for a moving weld heat map."""

from pathlib import Path
import sys

import numpy as np
from numba import jit
from scipy.ndimage import binary_erosion
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "02_Results"
RESULTS_DIR.mkdir(exist_ok=True)
RESOURCES_DIR = PROJECT_DIR.parent / "Resources"
if str(RESOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(RESOURCES_DIR))

from Common.launch_ready import mark_startup_ready


mark_startup_ready()

"""
The following functions are copies of the current version for the bigger simulation, check there for details!

Essentially I split the space derivate calculation off, for clearity and ease of reading the math.
The update_u function (the heat diffusion) does not depend on values in another matrix (like how hydrogen diffussion
depends on the temperature field)

Do not mess with these functions unless you are sure numba can handle the logic. The @jit decorator is pretty fussy.
"""


@jit(nopython=True, cache=True)
def compute_field_derivatives(field, func_dx2, func_dy2, field_dx2, field_dy2, func_ny, func_nx):
    """
    Compute the second spatial derivatives of a given scalar field.

    This function calculates the second derivatives with respect to x and y for
    each interior point of the field, using central difference. The boundaries
    are left unchanged.

    Parameters:
    - field (numpy.ndarray): The 2D array of the scalar field (e.g., temperature, concentration).
    - dx2 (float): The denominator in the difference formula for the x-direction.
    - dy2 (float): The denominator in the difference formula for the y-direction.
    - field_dx2 (numpy.ndarray): The 2D array to store the second derivative with respect to x.
    - field_dy2 (numpy.ndarray): The 2D array to store the second derivative with respect to y.

    Returns:
    - tuple of numpy.ndarray: The arrays (field_dx2, field_dy2) containing the second derivatives.
    - essentially dudx2, dudy2 and dhdx2, dhdy2 respectivly.
    """

    # Compute derivatives for the interior points, leaving boundaries as is
    for j in range(1, func_ny - 1):
        for i in range(1, func_nx - 1):
            field_dx2[j, i] = (field[j, i + 1] - 2 * field[j, i] + field[j, i - 1]) / func_dx2
            field_dy2[j, i] = (field[j + 1, i] - 2 * field[j, i] + field[j - 1, i]) / func_dy2

    return field_dx2, field_dy2


@jit(nopython=True, cache=True)
def update_u_with_jit(u, u0, diffusion_matrix, func_dt, func_dudx2, func_dudy2, func_ny, func_nx):
    """
    Update the temperature field (u) based on heat diffusion effects over a time step. (Fourier)

    This function computes the new values of the temperature field (u) by applying a
    diffusion formula according to Fourier's law, which considers the second spatial derivatives
    in both the x and y directions (dudx2 and dudy2). It integrates the effects of a diffusion
    coefficient matrix (D) and a time step (dt) to simulate the diffusion process.

    Parameters:
    - u (numpy.ndarray): The 2D array to be updated, representing the current state of the temperature field.
    - u0 (numpy.ndarray): The 2D array representing the previous state of the temperature field.
    - D (numpy.ndarray): The 2D array of diffusion coefficients applicable to each point in the temperature field.
    - dt (float): The time step for the update.
    - dudx2 (numpy.ndarray): The 2D array containing the second derivative of the temperature field with respect to x.
    - dudy2 (numpy.ndarray): The 2D array containing the second derivative of the temperature field with respect to y.

    Returns:
    - numpy.ndarray: The updated 2D array of the temperature field (u) after applying the heat diffusion update.
    """

    for j in range(func_ny):
        for i in range(func_nx):
            u[j, i] = u0[j, i] + diffusion_matrix[j, i] * func_dt * (func_dudx2[j, i] + func_dudy2[j, i])

    return u


def apply_mask(func_diff_matrix, show_boundary=0):
    """
    This is a copy of the simplified version of "apply_diffusion_masks".

    It potentially allows multipple valid neighbors inside the area for a point on the boundary. (1)
    As well as multiple points on the boundary for points in the calculated area. (2)
    This can obviously lead to undesired interactions, such as sources or sinks depending on (1) or (2).

    ONLY use this for NON-INTERACTIVE boundaries, i.e. between weld sample and air for Neumann BC.("reflective")
    This might get depreciated, if "apply_diffusion_masks" is efficient enough for complicated geometries.

    This is a complicated bla for what is basically, "find where air is". Might not get used if we do real convection.
    """

    # Find all the spots where D is greater than 0, mark those spots as True others as False
    mask_func = func_diff_matrix > 0
    # Find the edges of that, basically the "simulation area boundary"
    boundary = mask_func ^ binary_erosion(mask_func, structure=np.ones((3, 3)))  # Find the boundary of the sample

    # Get the location of the boundary pixels
    boundary_indices_func = np.argwhere(boundary)

    # Compute the inner part by subtracting the boundary from the mask
    inner_mask = np.logical_and(mask_func, np.logical_not(boundary))

    # Get the location of the inner part - remember that the "edges" are NOT excluded (so first/last row/column!)
    inner_indices_func = np.argwhere(inner_mask)

    # Create a new list with the same size/dimensions to store the valid neighbors
    valid_neighbors_indices_func = np.zeros_like(boundary_indices_func)

    # Loop through
    for index, (i, j) in enumerate(boundary_indices_func):

        # Find the neighboring pixels that are not on the boundary and have a value > 0
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

        # Are any of them viable? They are good if they have D>0 (mask) and are not on the boundary themselves
        valid_neighbors = [(x, y) for x, y in neighbors if
                           0 <= x < func_diff_matrix.shape[0] and 0 <= y < func_diff_matrix.shape[1] and mask_func[
                               x, y] and not boundary[x, y]]

        # This could still result in a few pixels which are not included. Adaptable range?
        if not valid_neighbors:

            for radius in range(1, 3):  # max_search_radius = 3
                valid_neighbors = [(x, y) for x in range(i - radius, i + radius + 1) for y in
                                   range(j - radius, j + radius + 1) if (
                                           0 <= x < func_diff_matrix.shape[0] and 0 <= y < func_diff_matrix.shape[1] and
                                           mask_func[x, y] and not
                                           boundary[x, y])]

                # Check if any valid neighbors are found
                if valid_neighbors:
                    break  # Exit the loop as soon as valid neighbors are found

        # All points on the boundary now have a valid neighbor (corners maybe 2) Save pixel coords and their partner
        if valid_neighbors:
            valid_neighbors_indices_func[index] = valid_neighbors[0]

    # For debugging
    if show_boundary != 0:
        # For debugging! - Plot D, mask, and calc_area
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(func_diff_matrix, cmap='viridis')
        plt.title('D Matrix')

        plt.subplot(132)
        plt.imshow(mask_func, cmap='binary')
        plt.title('Mask (D > 0)')

        plt.subplot(133)
        plt.imshow(boundary, cmap='binary')
        plt.title('Boundary')

        plt.show()

    # For some reason they get saved as floats above? tf? - solve later
    valid_neighbors_indices_func = valid_neighbors_indices_func.astype(int)

    return inner_mask, inner_indices_func, boundary_indices_func, valid_neighbors_indices_func


@jit(nopython=True, cache=True)
def apply_convection(temp_map_func, t_room_func, weld_convection_func, inner_area_func, func_ny, func_nx):
    """
    Description here...
    """

    for j in range(func_ny):
        for i in range(func_nx):
            # inner_area_func is a boolean mask so if its not in it, below gets multipied by 0 and nothing changes.
            temp_map_func[j, i] -= (temp_map_func[j, i] - t_room_func) * weld_convection_func * inner_area_func[j, i]

    return temp_map_func


"""
---------------------------------------------------- Dimensions -------------------------------------------------------
#                       <----- le -------> <- we -> <----- ri ----->
#                                                                           | fr_ab
#                                      
#                                    |----|------|----|  ⅄
#                                    |    |      |    |  | su_h
#                                    |    |      |    |  Y
#                       __________________        _________________                 dim_x,dim_y = total dimensions
#                      |                  |      |                 |      ⅄         le, ri = width of base metall
#                      |                  |      |                 |      |         we = weld width
#                      |                  |      |                 |      | th      th = weld thickness
#                      |                  |      |                 |      |         su_h = weld support height
#                      |__________________|      |_________________|      Y         su_w = weld support width
#                                    |    |      |    |  ⅄                          fr_ab,fr_be = free above/below
#                                    |    |      |    |  | su_h
#                                    |____|______|____|  Y                             * [mm]
#                                                                         | fr_be
#                       <-- fr_le --><----- su_w -----><-- fr_ri -->
"""

le = ri = 125
we = 20
th = 300
su_h = 50
su_w = 120  # 50 width, times 2 plus 20 for weld width
fr_le = int(le - ((su_w - we) / 2))
fr_ri = int(ri - ((su_w - we) / 2))
fr_ab = fr_be = 5
last_weld_bead = le + int(we / 2)  # this is where our simulated weld bead starts (from the left)
weld_bead_thickness = int(we / 2)  # the thickness of that weld (make sure that makes sense)

# For visual purposes, space above and below but no free space left and right
dim_rows = le + ri + we  # The rows have so many many entries in them (1 per mm * the amount of points within 1 mm)
dim_columns = th + (2 * su_h) + fr_ab + fr_be  # same with columns

# ---------------------------------------- Dimensional step size, gradient --------------------------------------------
dx = dy = 1  # step size - if not equal, TRIPPLE check everything! This was always meant to be equal!
dx2 = dx * dx  # partial differential x
dy2 = dy * dy  # partial differential y

# -------------------- Relevant Starting Conditions - Temperature und Hydrogen concentration --------------------------
t_cool = 160  # Interpass temperature basically. Used for BCs and/or initialization
t_hot = 1500  # Weld bead temperature. Adjust as needed
t_room = 25  # Surrounding temperature. Remember that the graphs display a different color for t_room! -offset!!

# ---------------------------------------- TEMPERATURE diffusion coefficients -----------------------------------------
diff_coeff_bm = 5.36768  # D = 20 [W/mK] / (8100 [kg/m³] * 460 [J/kgK] = 5.36768 e-6  m²/s = 5.36768 mm²/s  **1
diff_coeff_wm = 5  # For now just made up numbers
diff_coeff_haz = 4.5  # For now just made up numbers - picked something inbetween bm and wm
diff_coeff_air = 0  # no diffusion into air! If != 0, check BCs! right now you have Neumann Boundary Condition!
highest_diff_coeff = max(diff_coeff_bm, diff_coeff_wm, diff_coeff_haz)

dt = (dx2 * dy2) / (2 * highest_diff_coeff * (dx2 + dy2))  # highest time step to still be stable
print("dt: " + str(dt))

# -------------------------------- Make initial matrices - temp_map and temp_diff -------------------------------------
# Make initial "Heat-Matrix":
nx, ny = int(dim_rows / dx), int(dim_columns / dy)  # for readabilty - x should be horizontal, y vertical
temp_map = t_cool * np.ones((ny, nx))  # x should be horizontal! But in Python the first entry is no. of rows!
temp_map[:int(fr_ab / dy), :] = t_room  # Free space ABOVE sample
temp_map[-int(fr_be / dy):, :] = t_room  # Free space BELOW sample
temp_map[:int((fr_ab + su_h) / dy), :int(fr_le / dx)] = t_room  # Free space TOP LEFT
temp_map[:int((fr_ab + su_h) / dy), -int(fr_ri / dx):] = t_room  # Free space TOP RIGHT
temp_map[-int((fr_be + su_h) / dy):, :int(fr_le / dx)] = t_room  # Free space BOTTOM LEFT
temp_map[-int((fr_be + su_h) / dy):, -int(fr_ri / dx):] = t_room  # Free space BOTTOM RIGHT

# This is the actual "weld line" where we later simulate heat. Apply some temp offset for visibility?
temp_map[int(fr_ab / dy):-int(fr_be / dy), int(last_weld_bead / dx):
                                           int((last_weld_bead + weld_bead_thickness) / dx)] = t_cool - 5 # Weld area

# Make initial "Diffusion-Matrix" based on the Heat-Matrix:
temp_diffusion = temp_map.copy()  # just copy that matrix
temp_diffusion[temp_map == t_cool] = diff_coeff_bm  # set the diffusion coefficient to bm value where bm has temperature
temp_diffusion[temp_map == t_room] = diff_coeff_air  # set the diffusion coefficient 0 every where else

# This is the actual "weld line" where we later simulate heat. Set different diffusion coeff here?
temp_diffusion[int(fr_ab / dy):-int(fr_be / dy), int(last_weld_bead / dx):
                                                 int((last_weld_bead + weld_bead_thickness) / dx)] = diff_coeff_wm

# ----------------------------------------------- Weld Setup ----------------------------------------------------------
weld_length = 350  # [mm] This will be centered around the midle (y-direction)!
weld_speed = 600  # 600 [mm/min] basically the setting for our SAW machine, will be converted further down!
weld_temp = t_hot  # Incase you want to set it here
weld_spot_size = 20  # [mm] the "length" of the weld spot, the "width" results from setup above (last bead for now)
time_before_weld_start = 5  # [s] not really necessary but makes for nicer animations (can be set to 0)
weld_start_time = time_before_weld_start  # Rename for clearity, or adjust as needed here
weld_end_time = weld_start_time + (weld_length / weld_speed) * 60  # Calculate end time

# ----------------------------------------------- Convection Setup ----------------------------------------------------

# Convection: Air is a poor conductor, but has low density thus high diffusivity
# Air has roughly the same thermal diffusivity as steel.
# But if caloric:, maybe 10-30 W/m²K should be rightish, BUT this assumes we keep track of the air temp as well!
# Which is obvously almost always the case, because convection is used in cfd mostly to then simulate the fluid.
# If we discard air temperature change, which we do, the convection variable will obviously be !much! lower because
# the air is constantly at room temperature (of ~25°C).

# Parameters
t_room_conv = t_room  # get room temp, adjust here if you want
dt = dt  # here just for illustrative purposes, maybe delete for clarity, is set above
conv_variable = 3.00  # [W/m²K] This is determined / in "figure_out_cooling.py" for example. adjust as needed!

# Specific heat capacity (J/kg·K) for low carbon steel
c = 486  # 1010 Air, for steel use 486
# Density (kg/m³) for low carbon steel (if needed)
rho = 7850  # 1.3 Air, for steel use 7850

# Calculate volume and area, lets work with 1 mm for now?
thickness = dx * 1e-3  # [m] Thickness of the material layer affected by convection (m)
area = dx * dy * 1e-6  # [m²] Area of a mesh cell
V = area * thickness  # [m³] Volume of the material affected (m³)

# q = conv_variable * area * (current_temperature_convection - t_room_conv)  # Rate of heat transfer (W)
# delta_T = q * dt / (c * rho * V)  # Temperature change
# delta_T = (conv_variable * area * dt) / (c * rho * V) * (current_temperature_convection - t_room_conv)
# So most of this is simply a constant, we will call it: weld_convection
weld_convection = (conv_variable * area * dt) / (c * rho * V)
print("Weld_convection variable: " + str(weld_convection))

# # Polarer Gigantismus!! Remember that the above is before anything, a geometric description of energy flowing
# # through an area, leaving a volume. That means if you half dx, you quarter the area, but the volume is 1/8th.
# # We are not simulating the whole physical geometry, where the area/volume is fixed, regardless of mesh.
# # Hence, we calibrate this for 1 mm by 1 mm and scale the factor. Which makes sense, since we already put a lot into it.
# weld_convection = weld_convection / dx
#
# print("weld conv * dt = " + str(weld_convection*dt))

# -------------------------------------------------- Simulation Setup -------------------------------------------------
use_boundary_adjustment = False
# Total simulation time
sim_time = 900 * 1 * 1  # 3600 * 1 * 1  # seconds * hours * days
nsteps = int(sim_time / dt)

# Print some stuff to console for easier check on simulation setup
print("Array shape:", temp_map.shape)
print("Mesh points:", temp_map.size)

# ---------- Numerics Mumbo Jumbo ----------
# Initialize some dhdx2 matrices
dudx2 = np.zeros_like(temp_map)
dudy2 = np.zeros_like(temp_map)

# Initialize "old" temp_map. temp_map is the next step and temp_map_0 is the one that is one step "back"
temp_map_0 = temp_map.copy()
actual_time = 0  # keep track of simulation time

# find the border stuff
prev_end_row = 0

# ---------- Precalculate some stuff ----------
# Find the vertical mid point to center the weld around
mid_index = int(temp_map.shape[0] // 2)  # Integer division
start_col = int(last_weld_bead / dx)  # weld bead left border
end_col = int((last_weld_bead + weld_bead_thickness) / dx)  # weld bead right border
start_row = mid_index + int((weld_length // 2) / dy) - int((weld_spot_size // 2) / dy)  # weld bead top
end_row = int(start_row + (weld_spot_size / dy))  # weld bead bottom

# the air/metal boundary above and below, plus some mask and the inner area indices
inner_mask, inner_area_indices, boundary_indices, valid_neighbors_indices = apply_mask(temp_diffusion, 0)

# Finding indices for the parts that are metal (and are at interpass temperature)
first_col_indices = np.where(temp_map[:, 0] == t_cool)[0]
last_col_indices = np.where(temp_map[:, -1] == t_cool)[0]

# Creating boundary indices array for easy manipulation
boundary_indices_first = np.column_stack((first_col_indices, np.zeros_like(first_col_indices)))
boundary_indices_last = np.column_stack((last_col_indices, (nx - 1) * np.ones_like(last_col_indices)))

# Combining all indices
all_boundary_indices = np.vstack([boundary_indices_first, boundary_indices_last])

# ---------- Save matrices ----------
file_name = RESULTS_DIR / "simple_heat_map.h5"  # File name to save the matrix as
# Set how ofter you want to save, this, BY FAR, has the biggest impact on performance!!
save_so_often_per_sec = 0.5  # How often to save per second

if True:  # Change to False if you want to keep the file or mess with the filename
    if file_name.exists():  # Delete old files
        file_name.unlink()

save_interval = 1 / save_so_often_per_sec  # Calculate the save interval in seconds
save_counter = 0  # used in the saving loop
time_since_last_save = 0  # Time since the last save
slow_down_beginning = 1  # Used to change the animation speed by adjusting save interval... quick and dirty way, change?

# -------------------------------------------- Actual Simulation Loop -------------------------------------------------
"""
Since we already partially optimized the actual calculation to use machine code, I will use this here as well.

Check the bigger simulation for details but basically you do the spacial derivative, then apply the numeric Fourier 
heat diffusion. I will include the full function here in this all in one script but for clearity, 
this is the explicit NumPy version:

# Get the second (central) spacial derivative, w/o boundary, which will need to be handled seprately further down:
    dudx2[1:-1, 1:-1] = (u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2
    dudy2[1:-1, 1:-1] = (u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2

# Apply Head PDE (Fourier)
    u = u0 + diff_array * dt * (dudx2 + dudy2)
"""

for m in tqdm(range(nsteps), desc='Simulating Heat Map'):

    # keep track of time, also for saving frequency
    actual_time += dt
    time_since_last_save += dt

    # Compute the second spatial derivatives
    dudx2, dudy2 = compute_field_derivatives(temp_map, dx2, dy2, dudx2, dudy2, ny, nx)

    # Apply Fourier
    temp_map = update_u_with_jit(temp_map, temp_map_0, temp_diffusion, dt, dudx2, dudy2, ny, nx)

    # Apply convection
    temp_map = apply_convection(temp_map, t_room, weld_convection, inner_mask, ny, nx)

    # Apply the weld pool, basically just a spot that moves and has a specific temperature. Adjust as needed.
    if time_before_weld_start <= actual_time <= weld_end_time:
        # Calculate the current position of the weld spot
        distance_moved = int((weld_speed / 60) * (actual_time - time_before_weld_start) / dy)  # Distance the spot moved

        start_row = mid_index + int((weld_length // 2) / dy) - int((weld_spot_size // 2) / dy) - distance_moved
        end_row = int(start_row + (weld_spot_size / dy))
        start_col = int(start_col)  # Unnecessary, you precalc this but here for clearity. Little performance impact
        end_col = int(end_col)  # Unnecessary, you precalc this but here for clearity. Little performance impact

        # If you need to: Figure out the new boundary, apply weld temp and diff coeff
        if prev_end_row != end_row:

            # Apply the weld temperature to that spot
            temp_map[start_row:end_row, start_col:end_col] = weld_temp

            # Apply diffusion coefficient to that as well
            temp_diffusion[start_row:end_row, start_col:end_col] = diff_coeff_bm  # just one diffusion coefficient now

            # Change to True above if metal/air boundary changes during sim. For example if you set diff_coeff = 0
            if use_boundary_adjustment:
                inner_mask, inner_area_indices, boundary_indices, valid_neighbors_indices = apply_mask(temp_diffusion)
                prev_end_row = end_row

    # ----------------------------------------- Take care of the boundaries -------------------------------------------
    # No diffusion i.e. Neumanns "reflective boundary" for the air/metal boundary
    temp_map[boundary_indices[:, 0], boundary_indices[:, 1]] =  temp_map_0[
        valid_neighbors_indices[:, 0], valid_neighbors_indices[:, 1]]

    # Border of the metal on the left and right. Fixed Gradient - Neumann Boundary Condition (here: du/dy | du/dx = 0)
    # --------- Left boundary ---------
    temp_map[boundary_indices_first[:, 0], boundary_indices_first[:, 1]] = temp_map[
        boundary_indices_first[:, 0], boundary_indices_first[:, 1] + 1]
    # --------- Right boundary ---------
    temp_map[boundary_indices_last[:, 0], boundary_indices_last[:, 1]] = temp_map[
        boundary_indices_last[:, 0], boundary_indices_last[:, 1] - 1]

    # quick way to "slow" down the animation during welding (by saving more often)
    if slow_down_beginning == 1:
        if actual_time <= 60:
            save_so_often_per_sec = 4  # How often to save per second
            save_interval = 1 / save_so_often_per_sec  # Calculate the save interval in seconds
        else:
            save_so_often_per_sec = 0.5  # How often to save per second
            save_interval = 1 / save_so_often_per_sec  # Calculate the save interval in seconds

    # This takes most of the computational power! Consider how often you need to save!
    # Check if it's time to save again, then save.
    if time_since_last_save >= save_interval:
        # name for your temp_map

        temp_map_save_name = f'temp_map_{save_counter:05d}'  # Change at your own risk, this is fiddly af
        t_save_name = f't_snapshot_{save_counter:05d}'  # Unique name for 'time'
        save_counter += 1
        time_since_last_save = 0

        with h5py.File(file_name, 'a') as hf:
            # Create a dataset with the unique name and store the data
            hf.create_dataset(temp_map_save_name, data=temp_map)
            hf.create_dataset(t_save_name, data=actual_time)

    temp_map_0 = temp_map.copy()

# ------------------------------------------------ Visualization ------------------------------------------------------
"""
Here because we want an all in one script but this part should probably be seperate. 

For now you can set it on/off here.
"""

make_animation_here = True

if make_animation_here:
    loaded_u_arrays = []
    loaded_t_values = []

    with h5py.File(file_name, 'r') as hf:
        keys = sorted([key for key in hf.keys() if key.startswith('temp_map_') or key.startswith('t_snapshot_')])
        for key in tqdm(keys, desc="Loading data"):
            if key.startswith('temp_map_'):
                loaded_u_arrays.append(hf[key][:])
            elif key.startswith('t_snapshot_'):
                loaded_t_values.append(hf[key][()])

    # Size of the matrices:
    num_rows, num_cols = loaded_u_arrays[0].shape
    extent_heat_map = [0, num_cols * dx, num_rows * dy, 0]  # left, right, bottom, top

    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.90, bottom=0.1)
    gs = gridspec.GridSpec(12, 3, width_ratios=[1, 3, 1], wspace=0.04, hspace=1)
    ax_heatmap = fig.add_subplot(gs[:, 1])  # Main animation in the center column

    # Central heatmap plot - chose a style: "hot", "interval", "jet"
    style = "hot"
    if style == "hot":
        norm = mcolors.Normalize(vmin=50, vmax=800)
        cmap = plt.get_cmap('hot')
        im = ax_heatmap.imshow(loaded_u_arrays[0], cmap=cmap, norm=norm, interpolation='nearest',
                               aspect='equal', extent=extent_heat_map)
        plt.colorbar(im, ax=ax_heatmap)  # 'pad' is the space between
    elif style == "interval":
        boundaries = np.arange(100, 501, 20)  # 25, 75, 125, ..., 825
        colors = plt.cm.viridis(np.linspace(0, 1, len(boundaries) - 1))  # Generate color indices
        cmap = mcolors.LinearSegmentedColormap.from_list("CustomMap", colors)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        im = ax_heatmap.imshow(loaded_u_arrays[0], cmap=cmap, norm=norm, interpolation='nearest', extent=extent_heat_map)
        plt.colorbar(im, ax=ax_heatmap, boundaries=boundaries[:-1] + 25, ticks=boundaries)
    elif style == "jet":
        boundaries = np.arange(100, 301, 10)  # 25, 75, 125, ..., 825
        cmap = plt.get_cmap('jet')
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        im = ax_heatmap.imshow(loaded_u_arrays[0], cmap=cmap, norm=norm, interpolation='nearest', extent=extent_heat_map)
        plt.colorbar(im, ax=ax_heatmap, boundaries=boundaries[:-1] + 25, ticks=boundaries)

    ax_heatmap.set_title('Heat Map Animation')
    ax_heatmap.set_xlabel('X Axis')
    ax_heatmap.set_ylabel('Y Axis')
    time_text = ax_heatmap.text(0.35, 1.08, f'Time: {0} s', transform=plt.gca().transAxes, color='black', fontsize=20)

    # Zoomed in on weld spot
    ax_weld_spot = fig.add_subplot(gs[0:9, 0])  # Main animation in the center column
    norm = mcolors.Normalize(vmin=25, vmax=t_hot)
    cmap = plt.get_cmap('hot')
    #  start/end for rows
    c_start, c_end = start_col - int(20/dx), end_col + int(20/dx)  # left/right - change numbers as needed
    #  start/end for columns
    r_start, r_end = start_row - int(20/dy), end_row + int(20/dy)  # Up/down - change numbers as needed
    extent = (c_start, c_end, r_end, r_start)
    im_weld_spot = ax_weld_spot.imshow(loaded_u_arrays[0][r_start:r_end, c_start:c_end],
                                       cmap=cmap, norm=norm, interpolation='nearest', aspect='equal', extent=extent)
    plt.colorbar(im_weld_spot, ax=ax_weld_spot)  # colorbar
    ax_weld_spot.set_title('Weld Spot')
    ax_weld_spot.set_xlabel('')
    ax_weld_spot.set_ylabel('')

    # Define the positions for line plots (x, y)
    x = int((last_weld_bead + weld_bead_thickness) / dx)  # Right side of the last weld bead (left-right coord)
    y = int((fr_ab + su_h + th) / dy - (30 / dy))  # Change the number as needed the - starts at lower edge of sample
    distances = [5, 10, 25, 50]
    point_names = ["A", "B", "C", "D"]
    monitoring_positions = [(y, int(x + distances[0]/dx)), (y, int(x + distances[1]/dx)), (y, int(x + distances[2]/dx)),
                            (y, int(x + distances[3]/dx))]  # (y, x)
    # Contour levels
    levels = [200, 250, 300, 350, 400, 450, 500]

    # Right side line plots
    ax_lines_right = [fig.add_subplot(gs[3*i:3*(i+1), 2]) for i in range(len(monitoring_positions))]
    line_plots = []
    for i, (ax, pos) in enumerate(zip(ax_lines_right, monitoring_positions)):
        y_data = [u[pos[0], pos[1]] for u in loaded_u_arrays]
        line, = ax.plot(loaded_t_values, y_data, label="")
        line_plots.append(line)
        ax.set_title(f'Temperature at Point {point_names[i]} ({distances[i]} mm)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_xlim(0, loaded_t_values[-1])
        ax.set_ylim(0, 750)  # Adjust Y scaling on axis as needed

        # Add major grid lines to the line plots
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')

        # Only show x-axis labels and ticks on the last plot
        if i < len(monitoring_positions) - 1:
            ax.set_xticklabels([])  # Hide x-axis labels
            ax.set_xlabel('')  # Clear the x-axis label
        else:
            ax.set_xlabel('Time (s)')  # Only set x-axis label for the last subplot

        # Also mark these spots in the heat map animation itself
        # New method using physical coordinates
        for name, (row_idx, col_idx) in zip(point_names, monitoring_positions):
            x_coord = col_idx * dx
            y_coord = row_idx * dy
            ax_heatmap.plot(x_coord, y_coord, marker='+', color='black', markersize=10)
            ax_heatmap.text(x_coord - 8, y_coord - 15, name, color='black', fontsize=12)

    # Left side line plot - "draws" the interpass temperature
    # Extracting data for point D over all time steps for initial y-axis limits
    y_data_D = [u[monitoring_positions[3][0], monitoring_positions[3][1]] for u in loaded_u_arrays]
    y_min, y_max = min(y_data_D), max(y_data_D)

    ax_plot_left = fig.add_subplot(gs[9:12, 0])
    line, = ax_plot_left.plot(loaded_t_values, y_data_D)
    ax_plot_left.set_xlim(min(loaded_t_values), max(loaded_t_values))
    ax_plot_left.set_ylim(y_min * 0.95, y_max * 1.05)  # Give a bit of margin around the min and max
    dot, = ax_plot_left.plot([], [], 'ko', markersize=5)  # Initial empty plot for the black dot

    ax_plot_left.set_title(f'Interpass Temperature (Point D)')
    ax_plot_left.set_xlabel('Time (s)')
    ax_plot_left.set_ylabel('Temperature (°C)')
    ax_plot_left.grid(True)

    def update(frame):
        # Update main simulation
        im.set_data(loaded_u_arrays[frame])

        # Update zoom on weld spot
        # Calculate the current position of the weld spot
        real_time = loaded_t_values[frame]  # Directly use the real timestamp

        if real_time <= weld_end_time:
            distance_moved_update = int((weld_speed / 60) * (real_time - time_before_weld_start) / dy)
            start_row = mid_index + int((weld_length // 2) / dy) - int((weld_spot_size // 2) / dy) - distance_moved_update
            r_start_update = start_row - int(10 / dy)
            r_end_update = r_start_update + int(40 / dy)
            # Update the extent of the zoomed region
            extent_update = (c_start, c_end, r_end_update, r_start_update)
            im_weld_spot.set_extent(extent_update)  # Update the extent for correct axis labels
            # update the actual data for the zoom on weld spot
            im_weld_spot.set_data(loaded_u_arrays[frame][r_start_update:r_end_update, c_start:c_end])

        else:  # necessary to keep updating the weld spot data without moving the zoom
            # Looks complicated, but only the weld_end_time should be different here
            distance_moved_update = int((weld_speed / 60) * (weld_end_time - time_before_weld_start) / dy)
            start_row = mid_index + int((weld_length // 2) / dy) - int((weld_spot_size // 2) / dy) - distance_moved_update
            r_start_update = start_row - int(10 / dy)
            r_end_update = r_start_update + int(40 / dy)
            # Update the extent of the zoomed region
            extent_update = (c_start, c_end, r_end_update, r_start_update)
            im_weld_spot.set_extent(extent_update)  # Update the extent for correct axis labels
            # update the actual data for the zoom on weld spot
            im_weld_spot.set_data(loaded_u_arrays[frame][r_start_update:r_end_update, c_start:c_end])

        for coll in ax_heatmap.collections:  # Remove old contours
            coll.remove()
        X, Y = np.meshgrid(np.arange(0, num_cols * dx, dx), np.arange(0, num_rows * dy, dy))
        cont = ax_heatmap.contour(X, Y, loaded_u_arrays[frame], levels, colors='black', alpha=0.5)
        plt.clabel(cont, inline=True, fontsize=8, fmt='%2.0f°C')

        real_time = loaded_t_values[frame]  # Directly use the real timestamp

        # Update line data, black dot position
        line.set_data(loaded_t_values[:frame + 1], y_data_D[:frame + 1])
        dot.set_data([loaded_t_values[frame]], [y_data_D[frame]])

        if real_time > 18000:
            time_text.set_text(f'Time: {int(real_time / 3600)} h')
        elif real_time > 900:
            time_text.set_text(f'Time: {int(real_time / 60)} min')
        else:
            time_text.set_text(f'Time: {int(real_time)} s')

        # Return all artists that need to be updated
        artists = [im, im_weld_spot, time_text, line, dot]
        return artists

    print("Creating Animation")
    ani = FuncAnimation(fig, update, frames=len(loaded_u_arrays), repeat=False, blit=True)
    # ani = FuncAnimation(fig, update, frames=30, repeat=False, blit=True)  # for testing, just create 1s animations

    # Set up tqdm progress bar
    pbar = tqdm(total=len(loaded_u_arrays), desc="Saving Animation")

    def tqdm_callback(current_frame: int, total_frames: int):
        # Update the progress bar
        pbar.update(1)

    before_save = time.time()
    animation_file = RESULTS_DIR / "heat_map_animation.mp4"
    ani.save(str(animation_file), writer='ffmpeg', dpi=200, fps=30, progress_callback=tqdm_callback)
    after_save = time.time()

    # Close the tqdm progress bar and figure
    pbar.close()
    print(f"Saving completed in {after_save - before_save:.2f} seconds.")

    # Chose one, either show the animation after (good for inspecting points) or not
    plt.close(fig)
    # plt.show()
# ---------------------------------------------------------------------------------------------------------------------
