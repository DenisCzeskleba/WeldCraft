# -------------------------------------- function calls used in main file ----------------------------------------------
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import b2_param_config


def get_value(param_name):
    # Check if the parameter exists in the config file
    if hasattr(b2_param_config, param_name):
        return getattr(b2_param_config, param_name)
    else:
        raise ValueError(f"Parameter '{param_name}' not found in param_config")


def safe_close_pbar(pbar):
    """Safely close a tqdm progress bar without throwing errors."""
    if pbar is not None and hasattr(pbar, "close"):
        try:
            pbar.close()
        except Exception:
            pass


def format_simulation_time(seconds):
    intervals = (
        ('years', 31536000),  # 60 * 60 * 24 * 365
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )
    result = []

    for name, count in intervals:
        value = int(seconds // count)
        if value:
            seconds -= value * count
            result.append(f"{value} {name}")
    if not result:
        return "0 seconds"
    else:
        return ' '.join(result)


def generate_change_steps():

    time_before_first_weld = get_value("time_before_first_weld")  # in seconds
    time_between_welds = get_value("time_for_weld_bead")  # in seconds
    number_of_changes = get_value("no_of_weld_beads")  # Total number of weld beads (Needs to be even)

    change_times = []  # Create empty list

    for i in range(number_of_changes):
        change_time = time_before_first_weld + i * time_between_welds
        change_times.append(change_time)

    return change_times


# Manipulate the simulation area - add welds, change diffusion, heat input etc.
def manipulate_simulation(sim_type, u0, h0, D, D_H, cwi, cci, t_weld_metal, h_weld_metal, D_bm, D_weld_metal, D_haz,
                          D_H_weld_metal, dx, dy, le, we, fr_ab, fr_be, su_h, fr_le, th, mask, faces, new_area):

    if sim_type == "butt joint":

        # --- Bead axes in index units (rows=y, cols=x in array indexing) ---
        bead_height = get_value("bead_height") / dx
        bead_width = get_value("bead_width") / dy

        # Double check this logic if you change sample dimensions - should work but better check!
        bead_x2 = (fr_ab + th - (get_value("bead_height") * (cwi // 2))) / dx
        bead_y1 = (le / dy) + 1.0
        bead_y3 = ((le + we) / dy)

        # --- Ellipse center depends on left/right bead ---
        if cwi % 2 == 0:
            center_x, center_y = bead_x2, bead_y1  # left bead
        else:
            center_x, center_y = bead_x2, bead_y3  # right bead

        # --- Grids ---
        num_rows, num_cols = u0.shape
        row_grid, col_grid = np.ogrid[0:num_rows, 0:num_cols]

        # --- Ellipse equation (normalized distance) ---
        distance = np.sqrt(((row_grid - center_x) / bead_height) ** 2 + ((col_grid - center_y) / bead_width) ** 2)

        # Only compute the "newly created material cells" once per bead activation (cci==0),
        # then reuse that same new_area for the remainder of the heat-hold window.
        if cci == 0 and not isinstance(D, (int, float)):

            # --- Geometric bead shape mask (half ellipse) ---
            inside_ellipse = (distance <= 1.0)

            if cwi % 2 == 0:  # Even - left bead
                inside_ellipse &= (col_grid >= center_y)  # include the center line
            else:  # Odd - right bead
                inside_ellipse &= (col_grid <= center_y)  # include the center line

            fillable = (D != D_bm) & (D != D_haz) & (D != D_weld_metal)

            # base new area from geometry
            new_area = inside_ellipse & fillable

            # ---- "gravity fill": for each column, fill everything below the lowest selected pixel ----
            cols_with_weld = np.any(new_area, axis=0)
            if np.any(cols_with_weld):
                cols_idx = np.where(cols_with_weld)[0]

                # For each column, find the maximum row index that is True in new_area
                bottom_row_per_col = np.argmax(new_area[:, cols_idx][::-1, :], axis=0)
                bottom_row_per_col = (num_rows - 1) - bottom_row_per_col

                # Figure out how far down to go (free below + thickness of weld pool backing)
                r_limit = num_rows - 1 - int((fr_be + su_h) / dy)  # or something similar
                fill_mask = np.zeros_like(new_area, dtype=bool)

                for col, r_bottom in zip(cols_idx, bottom_row_per_col):
                    fill_mask[r_bottom:r_limit+1, col] = True

                # Only fill where it's allowed
                new_area |= (fill_mask & fillable)

            # Update material maps only once (geometry changes)
            D[new_area] = D_weld_metal
            D_H[new_area] = D_H_weld_metal
            mask, faces = compute_mask_and_faces(D)

        # Apply heat & hydrogen only to the (cached) new_area
        u0[new_area] = t_weld_metal
        h0[new_area] = h_weld_metal

    elif sim_type == "lap joint":  # Check if you fixed the elipse logic (dont rehydrate old WM!)

        # --- Bead semi-axes in index units (cols=x, rows=y) ---
        a_base = get_value("bead_width") / dx  # semi-axis along x (columns)
        b_base = get_value("bead_height") / dy  # semi-axis along y (rows)

        # --- Lap corner (anchor): right edge of overlap, bottom of top plate ---
        x0 = le / dx + 1
        y0 = (fr_ab + th) / dy - 1

        # Grids
        num_rows, num_cols = u0.shape
        row_grid, col_grid = np.ogrid[0:num_rows, 0:num_cols]

        # Scale schedule per pass
        scales = get_value("bead_scales")
        sx, sy = scales[min(int(cwi), len(scales) - 1)]

        # Current ellipse semi-axes
        a = sx * a_base
        b = sy * b_base

        # --- Place center so quarter-ellipse touches the corner (x0, y0) ---
        # For array coords: cols → right, rows → downward
        center_col = x0
        center_row = y0

        # Ellipse equation
        distance = ((row_grid - center_row) / b) ** 2 + ((col_grid - center_col) / a) ** 2

        # Quarter-ellipse region: right & above the lap corner
        inside_ellipse = (distance <= 1.0) & (col_grid >= x0) & (row_grid <= y0)

        # Select only NEWLY added weld material - weld goes into previous "air" only
        new_area = inside_ellipse & (D != D_bm) & (D != D_haz)

        # Update heat & hydrogen for new weld bead area
        u0[new_area] = t_weld_metal
        h0[new_area] = h_weld_metal

        # On first activation in this pass, update material maps
        if cci == 0 and not isinstance(D, (int, float)):
            D[new_area] = D_weld_metal
            D_H[new_area] = D_H_weld_metal
            mask, faces = compute_mask_and_faces(D)

    elif sim_type == "iso3690":

        # --- Bead semi-axes in index units (cols=x, rows=y) ---
        a_base = get_value("bead_width") / dx  # semi-axis along x (columns)
        b_base = get_value("bead_height") / dy  # semi-axis along y (rows)

        # Grids
        num_rows, num_cols = u0.shape
        row_grid, col_grid = np.ogrid[0:num_rows, 0:num_cols]

        # Scale schedule per pass
        scales = get_value("bead_scales")
        sx, sy = scales[min(int(cwi), len(scales) - 1)]

        # Current ellipse semi-axes
        a = sx * a_base
        b = sy * b_base

        # --- Bead positions along top edge (fractions of 'le') ---
        n_beads = get_value("no_of_weld_beads")  # guaranteed <=3 elsewhere

        if n_beads <= 1:
            pos = [0.5]
        elif n_beads == 2:
            pos = [1 / 3, 2 / 3]
        else:  # n_beads == 3
            pos = [1 / 3, 2 / 3, 0.5]

        # Use current weld index 'cwi' to pick the active bead position
        frac = pos[min(int(cwi), len(pos) - 1)]

        # --- Anchor: point on top surface at the chosen fraction across the plate ---
        x0 = (fr_le + frac * le) / dx + 1  # columns (rightwards)
        y0 = fr_ab / dy - 1  # rows (top surface)

        # For array coords: cols → right, rows → downward
        center_col = x0
        center_row = y0

        # Ellipse equation
        distance = ((row_grid - center_row) / b) ** 2 + ((col_grid - center_col) / a) ** 2

        # Half-ellipse above the top surface (add only "upper" half)
        inside_ellipse = (distance <= 1.0) & (row_grid <= y0)

        # On first activation in this pass, update material maps
        if cci == 0 and not isinstance(D, (int, float)):
            # Select only NEWLY added weld material - weld goes into previous "air" only
            new_area = inside_ellipse & (D != D_bm) & (D != D_haz) & (D != D_weld_metal)

            D[new_area] = D_weld_metal
            D_H[new_area] = D_H_weld_metal
            mask, faces = compute_mask_and_faces(D)

        # Update heat & hydrogen for new weld bead area
        u0[new_area] = t_weld_metal
        h0[new_area] = h_weld_metal


    return u0, h0, D, D_H, mask, faces, new_area


def compute_mask_and_faces(D: np.ndarray):
    """
    From the current temperature diffusion map D, build:
      - mask:      material cells (D>0)
      - faces:     (left_face, right_face, up_face, down_face, boundary_cells)
                   where each face mask selects material cells whose outward neighbor is air.
    Recompute this ONLY when geometry/material map changes (e.g., after adding a weld bead).
    """

    if not isinstance(D, np.ndarray):  # For debug: Fail early and loundly!
        raise TypeError("D must be a numpy array")

    mask = D > 0

    # build shifted masks without wrap
    in_left = np.zeros_like(mask, dtype=bool); in_left[:, 1:] = mask[:, :-1]
    in_right = np.zeros_like(mask, dtype=bool); in_right[:, :-1] = mask[:, 1:]
    in_up = np.zeros_like(mask, dtype=bool); in_up[1:, :] = mask[:-1, :]
    in_down = np.zeros_like(mask, dtype=bool); in_down[:-1, :] = mask[1:, :]

    # “face” means: this cell is material AND the outward neighbor is air
    left_face = mask & (~in_left)   # outward is left → interior neighbor is to the right
    right_face = mask & (~in_right)  # outward is right → interior neighbor is to the left
    up_face = mask & (~in_up)     # outward is up → interior neighbor is below
    down_face = mask & (~in_down)   # outward is down → interior neighbor is above

    boundary_cells = left_face | right_face | up_face | down_face
    faces = (left_face, right_face, up_face, down_face, boundary_cells)
    return mask, faces


def apply_dirichlet_faces_const(field, field0, D_h, faces, dt, inv_dx2, inv_dy2, outer_value):
    left_face, right_face, up_face, down_face, _ = faces
    apply_dirichlet_faces_const_jit(field, field0, D_h, left_face, right_face,
                                    up_face, down_face, dt, inv_dx2, inv_dy2, outer_value)


@jit(nopython=True)
def apply_dirichlet_faces_const_jit(field, field0, D_h, left_face, right_face,
                                    up_face, down_face, dt, inv_dx2, inv_dy2, outer_value):
    ny, nx = field0.shape
    for j in range(ny):
        for i in range(nx):
            if D_h[j, i] > 0:
                is_left  = left_face[j, i]
                is_right = right_face[j, i]
                is_up    = up_face[j, i]
                is_down  = down_face[j, i]
                if not (is_left or is_right or is_up or is_down):
                    continue

                u0 = field0[j, i]
                left_n  = (2.0*outer_value - u0) if is_left  else (field0[j, i-1] if i-1 >= 0 else u0)
                right_n = (2.0*outer_value - u0) if is_right else (field0[j, i+1] if i+1 < nx else u0)
                up_n    = (2.0*outer_value - u0) if is_up    else (field0[j-1, i] if j-1 >= 0 else u0)
                down_n  = (2.0*outer_value - u0) if is_down  else (field0[j+1, i] if j+1 < ny else u0)

                lap = (right_n - 2.0*u0 + left_n)*inv_dx2 + (up_n - 2.0*u0 + down_n)*inv_dy2
                field[j, i] = u0 + D_h[j, i] * dt * lap


def reflect_neumann_faces(u, u0, faces):
    left_face, right_face, up_face, down_face, _ = faces
    reflect_neumann_faces_kernel(u, u0, left_face, right_face, up_face, down_face)


@jit(nopython=True)
def reflect_neumann_faces_kernel(u, u0, left_face, right_face, up_face, down_face):
    ny, nx = u0.shape
    for j in range(ny):
        for i in range(nx):
            s = 0.0
            c = 0
            if left_face[j, i] and i+1 < nx:   # interior neighbor is (j, i+1)
                s += u0[j, i+1]; c += 1
            if right_face[j, i] and i-1 >= 0:  # interior neighbor is (j, i-1)
                s += u0[j, i-1]; c += 1
            if up_face[j, i] and j+1 < ny:     # interior neighbor is (j+1, i)
                s += u0[j+1, i]; c += 1
            if down_face[j, i] and j-1 >= 0:   # interior neighbor is (j-1, i)
                s += u0[j-1, i]; c += 1
            if c > 0:
                u[j, i] = s / c


def apply_dirichlet_faces(field, field0, D_h, precomputed_powers, u0_idx, faces, dt, inv_dx2, inv_dy2, outer_value):

    left_face, right_face, up_face, down_face, _ = faces

    apply_dirichlet_faces_hydro_jit(field, field0, D_h, precomputed_powers, u0_idx, left_face, right_face,
                                    up_face, down_face, dt, inv_dx2, inv_dy2, outer_value)


@jit(nopython=True, cache=True)
def apply_dirichlet_faces_hydro_jit(field, field0, D_h, precomputed_powers, u0_idx, left_face, right_face,
                                    up_face, down_face, dt, inv_dx2, inv_dy2, outer_value):
    """
    Apply Dirichlet BC at material–air faces using a ghost-cell derivation (value=g).
    Updates ONLY boundary nodes (interior untouched here).
    """
    ny, nx = field0.shape

    for j in range(ny):
        for i in range(nx):
            if D_h[j, i] > 0:
                is_left  = left_face[j, i]
                is_right = right_face[j, i]
                is_up    = up_face[j, i]
                is_down  = down_face[j, i]

                if not (is_left or is_right or is_up or is_down):
                    continue

                u0 = field0[j, i]

                left_n = (2.0 * outer_value - u0) if is_left else (field0[j, i - 1] if i - 1 >= 0 else u0)
                right_n = (2.0 * outer_value - u0) if is_right else (field0[j, i + 1] if i + 1 < nx else u0)
                up_n = (2.0 * outer_value - u0) if is_up else (field0[j - 1, i] if j - 1 >= 0 else u0)
                down_n = (2.0 * outer_value - u0) if is_down else (field0[j + 1, i] if j + 1 < ny else u0)

                lap = (right_n - 2.0 * u0 + left_n) * inv_dx2 + (up_n - 2.0 * u0 + down_n) * inv_dy2

                D_eff = D_h[j, i] * precomputed_powers[u0_idx[j, i]]
                field[j, i] = u0 + D_eff * dt * lap


def compute_u0_idx(u0: np.ndarray, table_len: int, buf: np.ndarray | None = None) -> np.ndarray:
    """
    Build/clamp int indices for precomputed_powers once per step.
    Reuses 'buf' if shape matches to avoid reallocations.
    """
    if buf is None or buf.shape != u0.shape or buf.dtype != np.int32:
        buf = np.empty(u0.shape, dtype=np.int32)
    np.clip(u0.astype(np.int32, copy=False), 0, table_len - 1, out=buf)
    return buf


@jit(nopython=True, fastmath=True, cache=True)
def compute_field_derivatives(field, dx2, dy2, field_dx2, field_dy2):
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

    ny, nx = field.shape

    # Compute derivatives for the interior points, leaving boundaries as is
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            field_dx2[j, i] = (field[j, i + 1] - 2 * field[j, i] + field[j, i - 1]) / dx2
            field_dy2[j, i] = (field[j + 1, i] - 2 * field[j, i] + field[j - 1, i]) / dy2

    return field_dx2, field_dy2


@jit(nopython=True, cache=True)
def update_u_with_jit(u, u0, D, dt, dudx2, dudy2):
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

    ny, nx = u0.shape

    for j in range(ny):
        for i in range(nx):
            if D[j, i] >= 0:
                u[j, i] = u0[j, i] + D[j, i] * dt * (dudx2[j, i] + dudy2[j, i])

    return u


@jit(nopython=True)
def update_h_with_jit(h, h0, dhdx2, dhdy2, D_H, dt, u0, u0_idx, precomputed_powers):
    """
    Update the hydrogen concentration field (h) based on diffusion over a time step, influenced by the heat field u0.

    This function computes the new values of the hydrogen concentration field h by applying a
    diffusion formula that takes into account the influence of another scalar field u0. It considers the second
    spatial derivatives of hydrogen concentration (dhdx2, dhdy2), a diffusion coefficient matrix D_H,
    and a time step dt. The diffusion rate is modulated by precomputed powers derived from the field u0,
    emphasizing the dynamic interaction between u0 and hydrogen diffusion.

    Parameters:
    - h (numpy.ndarray): The 2D array to be updated, representing the current state of hydrogen concentration.
    - h0 (numpy.ndarray): The 2D array representing the previous state of hydrogen concentration.
    - dhdx2 (numpy.ndarray): The 2D array containing the second derivative of the hydrogen concentration w respect to x.
    - dhdy2 (numpy.ndarray): The 2D array containing the second derivative of the hydrogen concentration w respect to y.
    - D_H (numpy.ndarray): The 2D array of diffusion coefficients specific to hydrogen, applicable to each point.
    - dt (float): The time step for the update.
    - u0 (numpy.ndarray): The 2D array representing another scalar field that influences hydrogen diffusion.
    - precomputed_powers (numpy.ndarray): Array of precomputed power values based on the field u0,
        used to modulate the diffusion rate of hydrogen.

    Returns:
    - numpy.ndarray: The updated 2D array of hydrogen concentration (h) after applying the diffusion update.
    """

    ny, nx = h0.shape

    # Directly calculate precalc_h0 equivalent using NumPy operations within Numba
    for j in range(ny):
        for i in range(nx):
            if D_H[j, i] >= 0:
                h[j, i] = h0[j, i] + (D_H[j, i] * dt * precomputed_powers[u0_idx[j, i]]) * (dhdx2[j, i] + dhdy2[j, i])
    return h


def do_timestep(sim_type, current_phase, u0, u, D, h0, h, D_H, mask, faces, dt, dx2,
                dy2, ign_ab, fr_le, th, su_h, we, dx, dudx2, dudy2, dhdx2, dhdy2, room_temp, precomputed_powers, sol_fn,
                row_inside_const, two_over_sqrt_pi, current_forced_part_temperature, pipe_line_inner_hydrogen,
                hydro_inside, boundary_temperature, inv_dx2, inv_dy2, coef_robin_x_air, coef_robin_y_air,
                coef_robin_x_h2, coef_robin_y_h2, coef_robin_x_cu, coef_robin_y_cu, t_room, joint_edge):

    # ------------------------------------------ Debug / Safe Guards  -------------------------------------------------
    assert u0 is not u, "u0 and u must be different arrays"  # Guards against accidental aliasing from the caller
    assert h0 is not h, "h0 and h must be different arrays"
    assert not np.shares_memory(u0, u), "u0 and u share memory (view overlap)"  # Catch overlapping views too
    assert not np.shares_memory(h0, h), "h0 and h share memory (view overlap)"

    u0_idx = compute_u0_idx(u0, precomputed_powers.shape[0])

    #  -------------------- Propagate with forward-difference in time, central-difference in space --------------------
    if current_phase != "just diffusion":  # normal calculation including heat, 99% of the time this happens.

        # --------------------- Heat ---------------------
        if current_phase == "cooling":  # we dont really need to simulate the heat here...
            # Set entries in `u` that are larger than `room_temp` to `current_forced_part_temperature`
            u[u > room_temp] = current_forced_part_temperature
        else:
            dudx2, dudy2 = compute_field_derivatives(u0, dx2, dy2, dudx2, dudy2)
            u = update_u_with_jit(u, u0, D, dt, dudx2, dudy2)

        # ------------------- Hydrogen -------------------
        dhdx2, dhdy2 = compute_field_derivatives(h0, dx2, dy2, dhdx2, dhdy2)
        h = update_h_with_jit(h, h0, dhdx2, dhdy2, D_H, dt, u0, u0_idx, precomputed_powers)

    else:  # Constant temperature! Hydrogen diffusion with dt_big and a changed D_H (happens in main loop)
        dhdx2, dhdy2 = compute_field_derivatives(h0, dx2, dy2, dhdx2, dhdy2)
        h = update_u_with_jit(h, h0, D_H, dt, dhdx2, dhdy2)  # Use the heat jit function for temperature independent D_H

    # ------------------------------------------- Boundary conditions -------------------------------------------------

    u0, u, h0, h = custom_boundary(sim_type, current_phase, current_forced_part_temperature, dt, D_H, D, u0, u, h0, h,
                                   ign_ab, th, su_h, we, dx, mask, faces, sol_fn, row_inside_const, two_over_sqrt_pi,
                                   precomputed_powers, pipe_line_inner_hydrogen, hydro_inside, boundary_temperature,
                                   inv_dx2, inv_dy2, coef_robin_x_air, coef_robin_y_air, coef_robin_x_h2,
                                   coef_robin_y_h2, coef_robin_x_cu, coef_robin_y_cu, t_room, joint_edge, u0_idx)

    if current_phase != "just diffusion":
        u0, u = u, u0  # swap references (no copy)
        h0, h = h, h0
    else:  # Temperature field doesnt evolve in "just diffusion"
        h0, h = h, h0

    return u0, u, h0, h, dudx2, dudy2, dhdx2, dhdy2


def pipeline_complicated_boundary(row_inside_const, dx, u0, D_h, sol_fn, h, two_over_sqrt_pi, dt,
                                  precomputed_powers):
    """
    Diffusion bound! We basically say, the very first atomic scale available latice spots are maximally
    filled with hydrogen, according to Sieverts Law ( Se excel or word file for details). So we are
    limited, by how fast hydrogen can diffuse away to make space for new hydrogen there.

    This enforces the semi-infinite diffusion uptake per step:
    Δh ≈ (2/√π) * (√(D·dt)/dx) * (H* - H_old), capped at Dirichlet when Fo≥O(1).
    """

    T_row_old = u0[row_inside_const, :]  # °C
    T_idx_row = T_row_old.astype(np.int32)
    np.clip(T_idx_row, 0, precomputed_powers.shape[0] - 1, out=T_idx_row)

    H_star_row = sol_fn(T_row_old)  # %

    D_base_row = np.maximum(D_h[row_inside_const, :], 0.0)  # mm^2/s
    col0, col1 = 0, h.shape[1]

    pipeline_complicated_boundary_jit(h[row_inside_const, :], H_star_row, D_base_row, T_idx_row,
                                      precomputed_powers, dt, dx, two_over_sqrt_pi,
                                      col0, col1)


@jit(nopython=True)
def pipeline_complicated_boundary_jit(h_row,  # 1D view: h[row, :]
                                      H_star_row,  # 1D: precomputed target % from table
                                      D_base_row,  # 1D: D_H[row, :] (mm^2/s), base (>=0)
                                      T_idx_row,  # 1D: int32 indices from u0[row, :] (clipped)
                                      precomputed_powers,  # 1D: same as used in update_h_with_jit
                                      dt, dx, two_over_sqrt_pi,
                                      col0, col1):
    """
    In-place update of one surface row with the diffusion-limited limiter.
    """

    for i in range(col0, col1):
        # T scaling for D (same scheme as bulk diffusion step)
        Deff = D_base_row[i] * precomputed_powers[T_idx_row[i]]  # mm^2/s
        if Deff < 0.0:
            Deff = 0.0

        # f = min(1, (2/sqrt(pi)) * sqrt(Deff*dt)/dx)
        f = two_over_sqrt_pi * np.sqrt(Deff * dt) / dx
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0

        # relaxed update: h += f * (H - h)
        h_row[i] = h_row[i] + f * (H_star_row[i] - h_row[i])


@jit(nopython=True)
def apply_robin_cooling_boundary_jit_faces(u, u0, D, left_face, right_face, up_face, down_face, dt, inv_dx2, inv_dy2,
                                           coef_x, coef_y, t_room):
    """
    Apply a uniform Robin (Newton cooling) boundary to material–air faces, after the interior update.

    Physics
    -------
    For a boundary node T, Newton cooling gives q = h (T - T∞).
    Using a ghost-cell derivation for the normal derivative, each exposed face contributes:
      X-faces:  + 2*(T_nbr - T)/dx^2  - (2*h/(k*dx))*(T - T∞)
      Y-faces:  + 2*(T_nbr - T)/dy^2  - (2*h/(k*dy))*(T - T∞)
    The update is explicit:
      u_boundary = u0_boundary + D * dt * (sum of face contributions)

    Inputs & Units
    --------------
    u, u0 : temperature fields [°C]
    D     : thermal diffusivity [mm^2/s]
    left/right/up/down_face : boolean masks (True where outward neighbor is air)
    dt    : time step [s]
    inv_dx2, inv_dy2 : 1/dx^2, 1/dy^2 [1/mm^2]
    coef_x, coef_y   : 2*h/(k*dx), 2*h/(k*dy) [1/mm]  (with h in W/mm^2/K, k in W/mm/K, dx,dy in mm)
    t_room : ambient temperature [°C]

    Notes
    -----
    - Only boundary cells are overwritten; interior nodes remain as computed by the main diffusion step.
    - Corner nodes naturally accumulate contributions from both exposed faces..
    """

    ny, nx = u0.shape

    for j in range(ny):
        for i in range(nx):
            is_left = left_face[j, i]
            is_right = right_face[j, i]
            is_up = up_face[j, i]
            is_down = down_face[j, i]

            if not (is_left or is_right or is_up or is_down):
                continue  # interior nodes untouched here

            T0 = u0[j, i]
            contrib = 0.0

            if is_left and i+1 < nx:
                contrib += 2.0 * (u0[j, i+1] - T0) * inv_dx2
                contrib +=  coef_x * (t_room - T0)

            if is_right and i-1 >= 0:
                contrib += 2.0 * (u0[j, i-1] - T0) * inv_dx2
                contrib +=  coef_x * (t_room - T0)

            if is_up and j+1 < ny:
                contrib += 2.0 * (u0[j+1, i] - T0) * inv_dy2
                contrib +=  coef_y * (t_room - T0)

            if is_down and j-1 >= 0:
                contrib += 2.0 * (u0[j-1, i] - T0) * inv_dy2
                contrib +=  coef_y * (t_room - T0)

            u[j, i] = T0 + D[j, i] * dt * contrib


@jit(nopython=True)
def apply_robin_cooling_iso3690_jit_faces(u, u0, D, left_face, right_face, up_face, down_face, dt, inv_dx2, inv_dy2,
                                           coef_x, coef_y, ign_ab, t_room):
    """
    Apply a uniform Robin (Newton cooling) boundary to steel–copper faces, after the interior update.

    See the other robin cooling for more info
    """

    ny, nx = u0.shape

    for j in range(ny):
        if j < ign_ab:
            continue
        for i in range(nx):
            is_left = left_face[j, i]
            is_right = right_face[j, i]
            is_down = down_face[j, i]

            if not (is_left or is_right or is_down):
                continue  # interior nodes untouched here

            T0 = u0[j, i]
            contrib = 0.0

            if is_left and i+1 < nx:
                contrib += 2.0 * (u0[j, i+1] - T0) * inv_dx2
                contrib +=  coef_x * (t_room - T0)

            if is_right and i-1 >= 0:
                contrib += 2.0 * (u0[j, i-1] - T0) * inv_dx2
                contrib +=  coef_x * (t_room - T0)

            if is_down and j-1 >= 0:
                contrib += 2.0 * (u0[j-1, i] - T0) * inv_dy2
                contrib +=  coef_y * (t_room - T0)

            # Only write if something changed (avoids touching unrelated cells)
            if contrib != 0.0:
                u[j, i] = T0 + D[j, i] * dt * contrib


def mask_faces_except_row(faces: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                          row_inside_const: int,
                          field_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given the faces tuple (lf, rf, uf, df, bc) and a constant row index (row_inside_const),
    return a new faces tuple where any faces ON that specific row are masked out.
    All other rows remain unchanged.

    Parameters
    ----------
    faces : tuple of bool arrays
        Tuple (lf, rf, uf, df, bc) from precomputed boundary masks.
    row_inside_const : int
        Index of the row inside the steel that we want to temporarily exclude.
    field_shape : tuple[int, int]
        Shape of the hydrogen or temperature field, e.g. h.shape.

    Returns
    -------
    faces_masked : tuple of bool arrays
        Same shape as faces but with inside-row faces masked out.
    """
    lf, rf, uf, df, bc = faces

    # Build a mask selecting all rows except the "inside row"
    row_idx = np.arange(field_shape[0])[:, None]
    not_inside = (row_idx != row_inside_const)

    # Apply mask to each face component
    lf_masked = lf & not_inside
    rf_masked = rf & not_inside
    uf_masked = uf & not_inside
    df_masked = df & not_inside
    bc_masked = bc & not_inside

    return lf_masked, rf_masked, uf_masked, df_masked, bc_masked


def custom_boundary(sim_type, current_phase, current_forced_part_temperature, dt, D_h, D, u0, u, h0, h, ign_ab, th,
                    su_h, we, dx, mask, faces, sol_fn, row_inside_const, two_over_sqrt_pi, precomputed_powers,
                    pipe_line_inner_hydrogen, hydro_inside, boundary_temperature, inv_dx2, inv_dy2, coef_robin_x_air,
                    coef_robin_y_air, coef_robin_x_h2, coef_robin_y_h2, coef_robin_x_cu, coef_robin_y_cu, t_room,
                    joint_edge, u0_idx):

    left_face, right_face, up_face, down_face, boundary_cells = faces

    # Options for material edges:
    # apply_butt_edges(u, u0, joint_edge, left=("reflective", None), right=("reflective", None))
    # apply_butt_edges(u, u0, joint_edge, left=("constant", 160), right=("constant", boundary_temperature))
    # apply_lap_edges(u, u0, joint_edge, left_above=("reflective", None), left_below=("reflective", None), right=("reflective", None))
    # apply_lap_edges(h, h0, joint_edge, left_above=("constant", 0), left_below=("constant", 0), right=("constant", 0))

    if current_phase == "pre-welding":
        """
        Pre-welding phase is here for possible future expansion. Right now its only used to make pretty animations.
        Heat:       We reflect on ALL internal boundaries effectivly leaving the initial conditions intact.
                    The outer most parts dont get identified as faces, so set first/last row/column manually!
                
        Hydrogen:   Since the new lap joint may be initialized with some initial hydrogen concentration,
                    we also reflect the hydrogen inside. Consider shortening this phase or changing the 
                    boundary condition if redistribution within the bulk is unwanted.
                    The outer most parts dont get identified as faces, so set first/last row/column manually!
        """

        # ------------- Boundary of Steel / Air -------------
        # Technically right now we calculate hydro twice on the inside (reflect, then robin)
        reflect_neumann_faces(u, u0, faces)  # TEMPERATURE
        reflect_neumann_faces(h, h0, faces)  # HYDROGEN

        if sim_type == "butt joint":
            # ------ TEMPERATURE: Neumann Boundary (reflective) on left and right edge of the sample
            apply_butt_edges(u, u0, joint_edge, left=("reflective", None), right=("reflective", None))

            # ------ HYDROGEN: Neumann Boundary (reflective) on left and right edge of the sample
            apply_butt_edges(h, h0, joint_edge, left=("reflective", None), right=("reflective", None))

        elif sim_type == "lap joint":

            # ------ TEMPERATURE: Neumann Boundary (reflective) on left and right edge of the sample
            apply_lap_edges(u, u0, joint_edge, left_above=("reflective", None), left_below=("reflective", None),
                            right=("reflective", None))

            # ------ HYDROGEN: Neumann Boundary (reflective) on left and right edge of the sample
            apply_lap_edges(h, h0, joint_edge, left_above=("reflective", None), left_below=("reflective", None),
                            right=("reflective", None))

            # ---- HYDROGEN INSIDE THE PIPELINE ----
            if pipe_line_inner_hydrogen == "constant":
                h[row_inside_const, :] = hydro_inside  # constant hydrogen concentration on the inside?

            elif pipe_line_inner_hydrogen == "variable":
                pipeline_complicated_boundary(row_inside_const, dx, u0, D_h, sol_fn, h, two_over_sqrt_pi, dt,
                                              precomputed_powers)

        elif sim_type == "iso3690":

            # ------ TEMPERATURE: Neumann Boundary (reflective) on left and right edge of the sample
            apply_butt_edges(u, u0, joint_edge, left=("reflective", None), right=("reflective", None))

            # ------ HYDROGEN: Neumann Boundary (reflective) on left and right edge of the sample
            apply_butt_edges(h, h0, joint_edge, left=("reflective", None), right=("reflective", None))

    elif current_phase == "welding":

        # ------------------------------------ Boundary of Steel / Air ------------------------------------------------
        # TEMPERATURE: Apply AIR cooling uniformly on all steel to air faces (Set via h_conv_air in param_config)
        apply_robin_cooling_boundary_jit_faces(u, u0, D, left_face, right_face, up_face, down_face, dt, inv_dx2,
                                               inv_dy2, coef_robin_x_air, coef_robin_y_air, t_room)

        if sim_type == "butt joint":

            # ------ TEMPERATURE: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_butt_edges(u, u0, joint_edge, left=("constant", boundary_temperature),
                             right=("constant", boundary_temperature))

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_butt_edges(h, h0, joint_edge, left=("constant", 0), right=("constant", 0))
            apply_dirichlet_faces(h, h0, D_h, precomputed_powers, u0_idx, faces, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Face Dirichlet (Set to 0)

        elif sim_type == "lap joint":

            # ------ TEMPERATURE: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_lap_edges(u, u0, joint_edge, left_above=("constant", boundary_temperature),
                            left_below=("constant", boundary_temperature), right=("constant", boundary_temperature))

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_lap_edges(h, h0, joint_edge, left_above=("constant", 0),
                            left_below=("constant", 0), right=("constant", 0))

            # ---- Stuff for inside the pipe line ----
            if pipe_line_inner_hydrogen == "constant":
                h[row_inside_const, :] = hydro_inside  # constant hydrogen concentration on the inside?

            elif pipe_line_inner_hydrogen == "variable":
                pipeline_complicated_boundary(row_inside_const, dx, u0, D_h, sol_fn, h, two_over_sqrt_pi, dt,
                                              precomputed_powers)

            # Zero H at all steel–air faces EXCEPT the inside row
            faces_not_inside = mask_faces_except_row(faces, row_inside_const, h.shape)
            apply_dirichlet_faces(h, h0, D_h, precomputed_powers, u0_idx, faces_not_inside, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Face Dirichlet (Set to 0)

        elif sim_type == "iso3690":
            # We cooled all around once as tho it was free floating in air.
            # So now, overwrite values left/right/below to cool correctly there
            apply_robin_cooling_iso3690_jit_faces(u, u0, D, left_face, right_face, up_face, down_face, dt, inv_dx2,
                                                  inv_dy2, coef_robin_x_cu, coef_robin_y_cu, ign_ab, t_room)

            apply_dirichlet_faces(h, h0, D_h, precomputed_powers, u0_idx, faces, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Face Dirichlet (Set to 0)

    elif current_phase == "cooling":

        # ------------------------------------ Boundary of Steel / Air ------------------------------------------------
        # TEMPERATURE: No boundary condition needed, as we set the temperature of the whole part in do_time_step

        if sim_type == "butt joint":
            # ------ TEMPERATURE: Nothing needed here

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_butt_edges(h, h0, joint_edge, left=("constant", 0), right=("constant", 0))
            apply_dirichlet_faces(h, h0, D_h, precomputed_powers, u0_idx, faces, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Face Dirichlet (Set to 0)

        elif sim_type == "lap joint":
            # ------ TEMPERATURE: Nothing needed here

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_lap_edges(h, h0, joint_edge, left_above=("constant", 0),
                            left_below=("constant", 0), right=("constant", 0))

            # ---- Stuff for inside the pipe line ----
            if pipe_line_inner_hydrogen == "constant":
                h[row_inside_const, :] = hydro_inside  # constant hydrogen concentration on the inside?

            elif pipe_line_inner_hydrogen == "variable":
                pipeline_complicated_boundary(row_inside_const, dx, u0, D_h, sol_fn, h, two_over_sqrt_pi, dt,
                                              precomputed_powers)

            # Zero H at all steel–air faces EXCEPT the inside row
            faces_not_inside = mask_faces_except_row(faces, row_inside_const, h.shape)
            apply_dirichlet_faces(h, h0, D_h, precomputed_powers, u0_idx, faces_not_inside, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Face Dirichlet (Set to 0)

        elif sim_type == "iso3690":
            # ------ TEMPERATURE: Nothing needed here

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_butt_edges(h, h0, joint_edge, left=("constant", 0), right=("constant", 0))
            apply_dirichlet_faces(h, h0, D_h, precomputed_powers, u0_idx, faces, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Face Dirichlet (Set to 0)

    elif current_phase == "just diffusion":  # no heat anymore, hydrogen stays the same

        # ------------------------------------ Boundary of Steel / Air ------------------------------------------------
        # TEMPERATURE: No boundary condition needed, as we set the temperature of the whole part in do_time_step.

        if sim_type == "butt joint":
            # ------ TEMPERATURE: Nothing needed here

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_butt_edges(h, h0, joint_edge, left=("constant", 0), right=("constant", 0))
            apply_dirichlet_faces_const(h, h0, D_h, faces, dt, inv_dx2, inv_dy2, outer_value=0.0)  # Constant Version!!

        elif sim_type == "lap joint":
            # ------ TEMPERATURE: Nothing needed here

            # ------ HYDROGEN: Current: Dirichlet Boundary ("constant) | Other: Neumann Boundary ("reflective")
            apply_lap_edges(h, h0, joint_edge, left_above=("constant", 0),
                            left_below=("constant", 0), right=("constant", 0))

            # ---- Stuff for inside the pipe line ----
            if pipe_line_inner_hydrogen == "constant":
                h[row_inside_const, :] = hydro_inside  # constant hydrogen concentration on the inside?

            elif pipe_line_inner_hydrogen == "variable":
                pipeline_complicated_boundary(row_inside_const, dx, u0, D_h, sol_fn, h, two_over_sqrt_pi, dt,
                                              precomputed_powers)

            # Zero H at all steel–air faces EXCEPT the inside row
            faces_not_inside = mask_faces_except_row(faces, row_inside_const, h.shape)
            apply_dirichlet_faces_const(h, h0, D_h, faces_not_inside, dt, inv_dx2, inv_dy2, outer_value=0.0)

    return u0, u, h0, h


# lazy singleton so we don't have to pass it around
_H_SOL_FN = None


def _apply_edge(field, field0, rows, col, nbr_col, mode, value):  # application helper: explicit, readable
    if mode == "constant":          # Dirichlet
        field[rows, col] = value
    elif mode == "reflective":      # Neumann (zero normal flux)
        field[rows, col] = field0[rows, nbr_col]
    else:
        raise ValueError(f"Unknown BC mode: {mode}")


def apply_butt_edges(field, field0, bidx, *, left=("reflective", None), right=("reflective", None)):
    # left
    mode, val = left
    r = bidx["left"]["rows"];  c = bidx["left"]["col"];  n = bidx["left"]["nbr_col"]
    _apply_edge(field, field0, r, c, n, mode, val)
    # right
    mode, val = right
    r = bidx["right"]["rows"]; c = bidx["right"]["col"]; n = bidx["right"]["nbr_col"]
    _apply_edge(field, field0, r, c, n, mode, val)


def apply_lap_edges(field, field0, lidx, *,
                    left_above=("reflective", None),
                    left_below=("reflective", None),
                    right=("reflective", None)):
    # left above gap
    mode, val = left_above
    r = lidx["left_above"]["rows"]; c = lidx["left_above"]["col"]; n = lidx["left_above"]["nbr_col"]
    _apply_edge(field, field0, r, c, n, mode, val)
    # left below gap
    mode, val = left_below
    r = lidx["left_below"]["rows"]; c = lidx["left_below"]["col"]; n = lidx["left_below"]["nbr_col"]
    _apply_edge(field, field0, r, c, n, mode, val)
    # right side
    mode, val = right
    r = lidx["right"]["rows"]; c = lidx["right"]["col"]; n = lidx["right"]["nbr_col"]
    _apply_edge(field, field0, r, c, n, mode, val)


def _build_h_solubility_fn():
    """
    Builds a vectorized callable f(T)->solubility_percent that maps temperature [°C]
    to hydrogen solubility *in percent* of your reference_from_iso3690 (100% reference).
    Uses np.interp (fast), clamps outside the tabulated range.
    """
    # Your table: 25°C .. 1000°C (5°C steps originally).
    # Keep as ml/100g here; we will normalize below.
    # equilibrium lattice solubility [ml/100g Fe] Sieverts Law
    # Includes y-fraction, Pressure (100bar), fugacity. See Excel for details
    solubs_ml_per_100g = np.array([
        0.009142952, 0.010159032, 0.012190964, 0.014222820, 0.016254561,
        0.019302052, 0.022349326, 0.025396342, 0.028443062, 0.032505241,
        0.037582735, 0.041643965, 0.047736008, 0.052811707, 0.059917853,
        0.067023082, 0.074127357, 0.082246031, 0.091378889, 0.100510458,
        0.110655900, 0.121814993, 0.132972455, 0.146158241, 0.159342080,
        0.173538802, 0.188748180, 0.203955284, 0.221189368, 0.239435433,
        0.257678772, 0.277948206, 0.299228960, 0.320506557, 0.343809382,
        0.368122890, 0.393446875, 0.420795097, 0.448139356, 0.477507308,
        0.507884798, 0.539271638, 0.571667642, 0.606086152, 0.641513322,
        0.677948980, 0.716406261, 0.755871552, 0.796344692, 0.838838613,
        0.882339930, 0.926848496, 0.973377043, 1.021925222, 1.071479947,
        1.122041085, 1.174621112, 1.228207161, 1.282799112, 1.340421650,
        1.398037252, 1.457670548, 1.519321233, 1.581976868, 1.646649428,
        1.712326611, 1.779008324, 1.847706361, 1.918420447, 1.990138548,
        2.063872284, 2.138609751, 2.215362455, 2.293118620, 2.372889639,
        2.453663861, 2.536452567, 2.620244232, 2.705038796, 2.792858543,
        2.880669729, 2.970494590, 3.062332905, 3.155173509, 3.249016355,
        3.344872237, 3.442740948, 3.541611553, 3.641484011, 3.743368911,
        3.846255484, 3.950143693, 4.056043979, 4.162945731, 4.271859292,
        4.381774158, 4.492690301, 4.605617918, 4.719546659, 4.834476499,
        4.951417496, 5.069359449, 5.188302337, 5.309256078, 5.431210621,
        5.554165944, 5.678122031, 5.804088621, 5.931055850, 6.059023701,
        6.187992160, 6.317961212, 6.449940380, 6.582920027, 6.716900139,
        6.851880705, 6.987861713, 7.124843151, 7.262825009, 7.402816520,
        7.542799143, 7.684791318, 7.827783792, 7.970767470, 8.115760553,
        8.261753909, 8.408747529, 8.555732476, 8.704726640, 8.854721046,
        9.005715687, 9.156701775, 9.309696901, 9.462683535, 9.616670449,
        9.772666271, 9.928653690, 10.085641370, 10.243629300, 10.401608990,
        10.561597450, 11.429492270, 12.319515800, 13.231666440, 14.165942640,
        15.123351130, 16.102882040, 17.105542110, 18.131329760, 19.179235300,
        20.251273550, 21.346434830, 22.464717660, 23.606120590, 24.771650200,
        25.960297000, 27.174075500, 28.410968330, 29.670974150, 30.956107320,
        32.266366380, 33.599734280, 34.957217520, 36.339822500, 37.747547820,
        39.179384430, 40.636338700, 42.117401690, 43.219536420, 43.723126810,
        44.228712890, 44.735287210, 45.243857380, 45.753415970, 46.263963120,
        46.775498940, 47.289030920, 47.803551770, 48.319061580, 48.835560480,
        49.354055850, 49.872533250, 50.393007280, 50.914470790, 51.436923890,
        51.960366690, 52.483792180, 53.009214710, 53.535627250, 54.063029890,
        54.591422750, 55.119798920, 55.649165540, 56.180529650, 56.711877440,
        57.244215960
    ], dtype=float)

    # Temperatures aligned to table (25°C -> first entry, 1000°C -> last).
    temps_C = np.arange(25.0, 1000.0 + 5.0, 5.0)
    assert temps_C.size == solubs_ml_per_100g.size, (
        f"Mismatch between table size ({solubs_ml_per_100g.size}) and expected 5°C steps ({temps_C.size})"
    )

    # Normalize to percent of your reference (100% := reference_from_iso3690)
    ref = float(get_value("reference_from_iso3690"))  # e.g., 2.5
    if ref <= 0:
        raise ValueError("reference_from_iso3690 must be > 0")
    solubs_percent = (solubs_ml_per_100g / ref) * 100.0  # now [% of weld reference]

    def f(T_C):
        T = np.asarray(T_C)
        T = np.clip(T, temps_C[0], temps_C[-1])
        # vectorized interpolation → returns percentages
        return np.interp(T, temps_C, solubs_percent, left=solubs_percent[0], right=solubs_percent[-1])

    return f


def get_h_solubility_fn():
    """Lazy getter so you can call it anywhere without refactoring signatures."""
    global _H_SOL_FN
    if _H_SOL_FN is None:
        _H_SOL_FN = _build_h_solubility_fn()
    return _H_SOL_FN


def reset_h_solubility_cache():
    global _H_SOL_FN
    _H_SOL_FN = None


def debug_show_DH(D, dx, dy,
                  title="Debug: D map",
                  xlabel="x [mm]",
                  ylabel="y [mm]",
                  cmap="viridis",
                  block=True,
                  with_colorbar=True):
    """
    Blocking debug visualization for the diffusion map (D).
    Shows the array scaled in mm, preserving correct aspect ratio.

    Parameters
    ----------
    D : np.ndarray
        2D field of temperature diffusion coefficients [mm²/s].
    dx, dy : float
        Physical step size in x and y [mm].
    title : str
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    cmap : str
        Matplotlib colormap name.
    block : bool
        If True, plt.show(block=True) halts execution until the window is closed.
    with_colorbar : bool
        Whether to draw a colorbar labeled in mm²/s.
    """
    # Get matrix dimensions in mm
    ny, nx = D.shape
    extent = [0, nx * dx, ny * dy, 0]  # [x_min, x_max, y_min, y_max]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(D, origin="upper", cmap=cmap, extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add colorbar if requested
    if with_colorbar:
        cbar = plt.colorbar(im, ax=ax, label="Temperature Diffusion Coefficient D [mm²/s]")
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show(block=block)
