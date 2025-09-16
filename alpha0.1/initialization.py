# --------------------------------------------- Build the weld sample here ---------------------------------------------

import numpy as np
from functions import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import matplotlib.colors as mcolors
import math
from functions import *


def weld_sample(sim_type):

    if sim_type == "butt joint":
        #
        #                   <----- le -------> <- we -> <----- ri ----->
        #                                                                       | fr_ab
        #                   __________________          _________________                 dim_x,dim_y = total dimensions
        #                  |                  |        |                 |      ⅄         le, ri = width of base metall
        #                  |                  |        |                 |      |         we = weld width
        #                  |                  |        |                 |      | th      th = weld thickness
        #                  |                  |        |                 |      |         su_h = weld support height
        #                  |__________________|________|_________________|      Y         su_w = weld support width
        #                                   |            |  ⅄                             fr_ab,fr_be = free above/below
        #                                   |            |  | su_h
        #                                   |____________|  Y                             * [mm]
        #                                                                       | fr_be
        #                   <--- fr_le ----><--- su_w ---><--- fr_ri ---->

        le = ri = 60
        we = 20
        th = 30
        su_h = 10
        su_w = 30
        fr_le = le - (su_w - we)/2
        fr_ri = ri - (su_w - we)/2
        fr_ab = fr_be = 5

        # Calculate dimension stuff (naming odd, cuz of python convention for x/y, i/j)
        dim_rows = le + ri + we  # The rows have so many entries in them (1 per mm * the amount of points within 1 mm)
        dim_columns = th + su_h + fr_ab + fr_be  # same with columns

        # Set the edges here so we can easily do the boundary conditions later
        dx = get_value("dx")
        rows = slice(int(fr_ab/dx), int((fr_ab + th) / dx))  # vertical extent of plate
        butt_joint_edges = {
            "left": {"rows": rows, "col": 0, "nbr_col": 1},
            "right": {"rows": rows, "col": -1, "nbr_col": -2}}

        return butt_joint_edges, dim_rows, dim_columns, le, ri, we, th, su_h, su_w, fr_le, fr_ri, fr_ab, fr_be

    elif sim_type == "lap joint":
        #
        #                 <-------------- le ------------>
        #          fr_ab  |
        #                  ______________________________                                dim_x,dim_y = total dimensions
        #                 |                              | ⅄                             th = thickness upper plate
        #                 |                              | |                             su_h = thickness lower plate
        #                 |                              | | th                          fr_le = free width left (gap)
        #                 |_________________________|    | |                             fr_ri = free width right
        #             we  |_________________________|    |_Y_________________            we = gap width
        #                 |                                                  |  ⅄        fr_ab,fr_be = free above/below
        #                 |                                                  |  |        * [mm]
        #                 |                                                  |  | su_h
        #                 |                                                  |  |
        #                 |__________________________________________________|  Y
        #          fr_be  |
        #                  <---------- fr_le ------->    <------ fr_ri ----->

        le = 20
        fr_le = 18  # Optionally, this is the gap length between plates
        fr_ri = 20
        th = 4.2
        su_h = 4
        fr_ab = fr_be = 2
        we = 0.2  # Gap width, check spacial discretization (dx, dy)!! Gets subtracted from upper plate width!

        # Calculate dimension stuff (naming odd, cuz of python convention for x/y, i/j)
        dim_rows = le + fr_ri
        dim_columns = th + su_h + fr_ab + fr_be

        # Set the edges here so we can easily do the boundary conditions later
        dx = get_value("dx")

        left_above = slice(int(fr_ab / dx), int((fr_ab + th - we) / dx))  # left edge, above gap
        left_below = slice(int((fr_ab + th) / dx), int((fr_ab + th + su_h) / dx))  # left edge, below gap
        right_rows = slice(int((fr_ab + th) / dx), int((fr_ab + th + su_h) / dx))  # right edge (lower plate)

        lap_joint_edges = {
            "left_above": {"rows": left_above, "col": 0, "nbr_col": 1},
            "left_below": {"rows": left_below, "col": 0, "nbr_col": 1},
            "right": {"rows": right_rows, "col": -1, "nbr_col": -2}}

        # Unused
        ri = su_w = 0

        return lap_joint_edges, dim_rows, dim_columns, le, ri, we, th, su_h, su_w, fr_le, fr_ri, fr_ab, fr_be

    elif sim_type == "iso3690":
        #
        #                   <------------------- le ------------------->
        #
        #                   ___________________________________________       | fr_ab
        #                  |                                           |      ⅄
        #                  |                                           |      |         dim_x,dim_y = total dimensions
        #                  |                                           |      |         le = width of base metall
        #                  |                                           |      | th      th = weld thickness
        #                  |                                           |      |         fr_ab,fr_be = free above/below
        #                  |                                           |      |         * [mm]
        #                  |___________________________________________|      Y
        #                                                                     | fr_be
        #

        le = 30  # Width of ISO 3690 sample
        th = 10  # Height of ISO 3690 sample
        fr_le = 2  # Leave some space for nicer animations
        fr_ri = 2
        fr_ab = 5  # Above needs enough space for weld beads
        fr_be = 2

        # Calculate dimension stuff (naming odd, cuz of python convention for x/y, i/j)
        dim_rows = le + fr_le + fr_ri  # rows have so many entries in them (1 per mm * the amount of points within 1 mm)
        dim_columns = th + fr_ab + fr_be  # same with columns

        # Set the edges here so we can easily do the boundary conditions later
        dx = get_value("dx")
        dy = get_value("dy")

        rows = slice(int(fr_ab / dy), int((fr_ab + th) / dy))  # vertical extent of plate
        col_left = int(fr_le / dx)  # Horizontal extent of the plate, first column inside the plate
        col_right = int((fr_le + le) / dx) - 1  # last column inside the plate (inclusive)

        iso3690_weld_edges = {
            "left": {"rows": rows, "col": col_left, "nbr_col": col_left + 1},
            "right": {"rows": rows, "col": col_right, "nbr_col": col_right - 1}}

        # Unused
        ri = we = su_w = su_h = 0

        return iso3690_weld_edges, dim_rows, dim_columns, le, ri, we, th, su_h, su_w, fr_le, fr_ri, fr_ab, fr_be

    else:
        raise ValueError("Unknown joint type selected! Please use a valid joint type.")


def initialize(simulation_type, nx, ny, dx, dy, le, we, th, su_h, su_w, fr_le, fr_ab):

    # Define some Temperatures t_xxx
    t_cool, t_hot, t_room = get_value("t_cool"), get_value("t_hot"), get_value("t_room")

    # Get initial hydrogen concentration
    init_hydrogen_conc = get_value("h_cont_initial")

    # Get initial heat diffusion coefficients
    diff_coeff_bm = get_value("diff_coeff_bm")  # For base metal

    # Get initial hydrogen diffusion coefficients
    diff_coeff = get_value("diff_coeff_h")  # Not in [mm² / s]! Needs * (T ** 2.2285) - applied during simulation

    # Initialize matrix at cool temperature
    u0 = t_cool * np.ones((ny, nx))  # x should be horizontal! But in python the first entry is no. of rows!
    h0 = init_hydrogen_conc * np.ones((ny, nx))  # for now we just assume no atomic H in the weld sample when we start
    D = diff_coeff_bm * np.ones((ny, nx))  # initiate with D = D_bm everywhere, set differences later
    D_H = diff_coeff * np.ones((ny, nx))  # same but for hydrogen

    # Carve out the gaps in sample geometry, set to room temperature, set diffusion coefficients
    if simulation_type == "butt joint":

        for x in range(nx):  # no vectorization/optimization here for easier readability / modification
            for y in range(ny):

                # Above and below the whole thing for nicer pictures and boundary conditions
                # -5 for now for plotting purposes, change later maybe
                if y < (fr_ab / dx) or y > (fr_ab + th + su_h) / dx:
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

                # below weld samples | left and right of weld pool support
                if (x < fr_le / dx or x > ((fr_le+su_w) / dx)) and y > (th + fr_ab) / dx:
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

                # where the actual weld beads go later, for now its "empty" at room temperature
                if le/dx < x < ((le+we)/dx) and y < (th+fr_ab)/dy:
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

        u = u0.copy()
        h = h0.copy()

        return u0, u, h0, h, D, D_H, t_cool, t_hot, t_room

    elif simulation_type == "lap joint":

        for x in range(nx):  # no vectorization/optimization here for easier readability / modification
            for y in range(ny):

                # Above and below the whole thing for nicer pictures and boundary conditions
                # -5 for now for plotting purposes, change later
                if y < (fr_ab / dx) or y > (fr_ab + th + su_h) / dx:
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

                # Right side above
                # (depreciated for left below: x < le / dx and y > (th + fr_ab) / dx)
                if x > le / dx and y < (fr_ab + th) / dx:
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

                # Gap between the plates (For now, 1 dy thick only)
                if x <= fr_le / dx and ((fr_ab + th - we) / dy <= y < (fr_ab + th) / dy):
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

                # For simplicity apply base metal coefficients here to the overlap
                if (fr_le / dx < x <= le / dx) and ((fr_ab + th - we) / dy <= y < (fr_ab + th) / dy):
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    D[y, x] = get_value("diff_coeff_bm")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h")  # We want x horizontal! But is second in [,] notation

                # Initialize the lower plate w. hydrogen. Linear equillibrium with hydro_inside inside and 0 outside.
                if (fr_ab + th) / dy < y < (fr_ab + th + su_h) / dy:
                    fraction = (y - (fr_ab + th) / dy) / ((fr_ab + th + su_h) / dy - (fr_ab + th) / dy)
                    fraction = max(0.0, min(1.0, fraction))
                    h0[y, x] = get_value("h_on_the_inside") * fraction

        u = u0.copy()
        h = h0.copy()

        return u0, u, h0, h, D, D_H, t_cool, t_hot, t_room

    elif simulation_type == "iso3690":

        for x in range(nx):  # no vectorization/optimization here for easier readability / modification
            for y in range(ny):

                # Above and below the whole thing for nicer pictures and boundary conditions
                # -5 for now for plotting purposes, change later maybe
                if y < (fr_ab / dx) or y > (fr_ab + th) / dx:
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

                # Left and right edges for nicer videos
                if x < fr_le / dx or x >= ((fr_le + le) / dx):
                    # remember! we want x horizontal! BUT it needs to be second in [,] notation
                    u0[y, x] = t_room + get_value("temperature_offset")
                    h0[y, x] = get_value("hydrogen_offset")  # -5 for nicer display
                    D[y, x] = get_value("diff_coeff_air")  # We want x horizontal! But is second in [,] notation
                    D_H[y, x] = get_value("diff_coeff_h_air")  # We want x horizontal! But is second in [,] notation

        u = u0.copy()
        h = h0.copy()

        return u0, u, h0, h, D, D_H, t_cool, t_hot, t_room


def generate_change_steps():

    time_before_first_weld = get_value("time_before_first_weld")  # in seconds
    time_between_welds = get_value("time_for_weld_bead")  # in seconds
    number_of_changes = get_value("no_of_weld_beads")  # Total number of weld beads (Needs to be even)

    change_times = []  # Create empty list

    for i in range(number_of_changes):
        change_time = time_before_first_weld + i * time_between_welds
        change_times.append(change_time)

    return change_times


def reset_temp(u0):
    # Current version takes temperatures from u0 and sets everything "warm" to room temperature
    # Older version reset the geometry but that was just for rectangle, check Heat_PDE_LDDE_old for that version

    # Define some temperatures t_xxx
    t_room, t_offset = get_value("t_room"), get_value("temperature_offset")

    # Check where the temperature is not t_room - offset and just set it to room temp - allows for easier calcs later
    u0 = np.where(u0 >= t_room + t_offset, t_room, u0)

    return u0
