import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from tqdm import tqdm

import math


@jit(nopython=True)
def compute_numerical_solution(u, alpha, dt, dx, nt):
    nx = len(u)
    for _ in range(nt):
        u_new = u.copy()
        for i in range(1, nx - 1):
            u_new[i] = u[i] + alpha * dt / dx ** 2 * (u[i + 1] - 2 * u[i] + u[i - 1])
        u = u_new
    return u


@jit(nopython=True)
def compute_field_derivatives(field, dx2_func, dy2_func, field_dx2, field_dy2):

    ny_func, nx_func = field.shape

    # Compute derivatives for the interior points, leaving boundaries as is
    for j in range(1, ny_func - 1):
        for i in range(1, nx_func - 1):
            field_dx2[j, i] = (field[j, i + 1] - 2 * field[j, i] + field[j, i - 1]) / dx2_func
            field_dy2[j, i] = (field[j + 1, i] - 2 * field[j, i] + field[j - 1, i]) / dy2_func

    return field_dx2, field_dy2


@jit(nopython=True)
def update_h_with_jit(h_func, h0_func, d_h_func, dt_func, dudx2, dudy2):

    ny_func, nx_func = h0_func.shape

    for j in range(ny_func):
        for i in range(nx_func):
            h_func[j, i] = h0_func[j, i] + d_h_func[j, i] * dt_func * (dudx2[j, i] + dudy2[j, i])

    return h_func


def numerical_solution(length, dx, alpha, u0, uL, dt, t_max, init_conc=0, worker=None):
    print(f"Currently calculating numerical solution (dx = {dx}):")  # keep track of progress in GUI

    x = np.arange(0, length + dx, dx)
    nx = len(x)
    nt = int(t_max / dt)
    u = np.ones(nx) * init_conc

    u[0] = u0
    u[-1] = uL

    bar_format = "{desc} {percentage:03.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    tqdm_bar = tqdm(range(nt), desc=f'Numerical solution (dx = {dx})', bar_format=bar_format)
    for _ in tqdm_bar:
        if worker and worker.stop_requested:
            tqdm_bar.close()
            return x, u, dt  # Return current state
        u = compute_numerical_solution(u, alpha, dt, dx, 1)
    tqdm_bar.refresh()  # Ensure the progress bar is updated at the end

    print(f"STEP_COMPLETED: Numerical solution (dx = {dx})")  # keep track of progress in GUI
    return x, u, dt


def analytical_solution(length, alpha, u0, uL, t, init_conc=0, points=None, show_flux=0, density=1, heat_capa=1, num_terms=100):

    if points is not None:
        x = np.array(points)
    else:
        x = np.linspace(0, length, 200)

    if u0 == uL:  # for some reason i need to figure out later, when they are the same, the calc is weird
        u0 += 1e-8
    u_s = u0 + (uL - u0) * x / length
    initial_condition = np.ones_like(x) * init_conc
    initial_condition -= u_s

    def calculate_B_n(n, length, initial_condition, x):
        lambda_n = n * np.pi / length
        B_n = (2 / length) * np.trapz(initial_condition * np.sin(lambda_n * x), x)
        return B_n

    v = np.zeros_like(x)
    error_estimate = float('inf')

    if num_terms == "auto":
        n = 1
        while error_estimate > 1e-12:
            lambda_n = n * np.pi / length
            B_n = calculate_B_n(n, length, initial_condition, x)
            v += B_n * np.sin(lambda_n * x) * np.exp(-alpha * lambda_n ** 2 * t)
            n += 1
            lambda_n = n * np.pi / length
            B_n = calculate_B_n(n, length, initial_condition, x)
            error_estimate = np.max(np.abs(B_n * np.sin(lambda_n * x) * np.exp(-alpha * lambda_n ** 2 * t)))
    else:
        for n in range(1, num_terms + 1):
            lambda_n = n * np.pi / length
            B_n = calculate_B_n(n, length, initial_condition, x)
            v += B_n * np.sin(lambda_n * x) * np.exp(-alpha * lambda_n ** 2 * t)
        n = num_terms + 1
        lambda_n = n * np.pi / length
        B_n = calculate_B_n(n, length, initial_condition, x)
        error_estimate = np.max(np.abs(B_n * np.sin(lambda_n * x) * np.exp(-alpha * lambda_n ** 2 * t)))

    u = v + u_s

    heat_flux = None
    if show_flux:
        # calculate thermal conductivity
        alpha_m = alpha * 1e-6  # transform alpha to m²/s
        k = alpha_m * density * heat_capa
        # Calculate the heat flux (q) using Fourier's law
        delta_u = u[-1] - u[-2]  # Temperature difference between the last two points
        delta_x = x[-1] - x[-2]  # Position difference between the last two points
        heat_flux = -k * (delta_u / delta_x)  # Negative sign due to the direction of heat flow

    return x, u, u_s, error_estimate, heat_flux


def calculate_error(u_analytical, u_numerical, x_numerical, x_analytical):
    # Interpolate the analytical solution to the numerical solution points
    u_analytical_interp = np.interp(x_numerical, x_analytical, u_analytical)
    error = np.mean(np.abs(u_analytical_interp - u_numerical))
    return error


def calculate_1d(length, alpha, u0, uL, dt, t, init_conc, dx_values, worker=None):
    x_numerical_list = []
    u_numerical_list = []
    dt_values = []
    errors = []

    all_x_points = set()

    # Do the numerical evaluation
    for dx in dx_values:
        x_numerical, u_numerical, dt = numerical_solution(length, dx, alpha, u0, uL, dt, t, init_conc, worker)
        x_numerical_list.append(x_numerical)
        u_numerical_list.append(u_numerical)
        dt_values.append(dt)
        all_x_points.update(x_numerical)

    # Convert set to sorted list
    all_x_points = sorted(all_x_points)

    # Do the analytical solution
    x_analytical, u_analytical, u_stable, error_estimate, flux_1d = analytical_solution(length, alpha, u0, uL, t, init_conc, points=all_x_points)

    # Calculate errors
    for x_numerical, u_numerical in zip(x_numerical_list, u_numerical_list):
        u_analytical_at_numerical_points = np.interp(x_numerical, x_analytical, u_analytical)
        error = calculate_error(u_analytical_at_numerical_points, u_numerical, x_numerical, x_numerical)
        errors.append(error)

    return x_analytical, u_analytical, error_estimate, x_numerical_list, u_numerical_list, dx_values, dt_values, errors


def adaptive_time_step(alpha, dx2, dy2):
    return dx2 * dy2 / (2 * alpha * (dx2 + dy2))


def sigmoid(t, alpha_0, alpha_1, t_0, k):
    return alpha_0 + (alpha_1 - alpha_0) / (1 + np.exp(-k * (t - t_0)))


def simulate_2d(dx, dim_x, dim_y, sim_time, init_hydrogen_conc, border_hydrogen, border_bc, diffusion_coefficient, warm_up_time, k_value, save_every_x_s, file_name, worker=None):
    print("Currently Simulating 2D diffusion:")  # keep track of progress in GUI

    # ----------------------------------------------- Numerics Mumbo Jumbo ---------------------------------------------
    interrupted = 0
    use_ramp = 0  # marker to signal which diffusion option we wanna use
    warmup_period = warm_up_time  # Initial warm-up period in seconds
    k = k_value
    dx = max(dx)
    dy = dx
    nx, ny = int(dim_x / dx) + 1, int(dim_y / dy) + 1  # points in x and y direction
    dx2 = dx * dx  # partial differential x
    dy2 = dy * dy  # partial differential y

    # Initialize matrices
    h0 = init_hydrogen_conc * np.ones((ny, nx))  # Initial concentration

    # Initialize the diffusion coefficient matrix
    if isinstance(diffusion_coefficient, list):  # If it got send more than a single number for diffusion_coefficient
        alpha_0, alpha_1, alpha_end_time = diffusion_coefficient
        d_h = alpha_0 * np.ones((ny, nx))  # Initial Diffusion Matrix
        dt = adaptive_time_step(alpha_0, dx2, dy2)  # highest stable INITIAL time step

        sim_time += warmup_period  # Extend the sim time to add the warm-up period

        # Ensure at least 100 steps during the ramp
        min_ramp_dt = alpha_end_time / 200
        if dt > min_ramp_dt:
            dt = min_ramp_dt
        use_ramp = 1  # signal to later use a ramp

    else:  # This got send a single diffusion coefficient, use that
        d_h = diffusion_coefficient * np.ones((ny, nx))  # Diffusion matrix
        dt = adaptive_time_step(diffusion_coefficient, dx2, dy2)  # highest stable time step
        alpha_0 = alpha_1 = diffusion_coefficient
        alpha_end_time = 0

    # Apply boundary conditions
    h0[0, :] = h0[1, :] if border_bc[0] else border_hydrogen[0]  # Top boundary
    h0[-1, :] = h0[-2, :] if border_bc[1] else border_hydrogen[1]  # Bottom boundary
    h0[:, 0] = h0[:, 1] if border_bc[2] else border_hydrogen[2]  # Left boundary
    h0[:, -1] = h0[:, -2] if border_bc[3] else border_hydrogen[3]  # Right boundary

    h = h0.copy()

    dhdx2 = np.zeros((ny, nx))  # preallocate the derivative matrices
    dhdy2 = np.zeros((ny, nx))  # this should save some time recreating empty ones each loop

    # --------------------------------------------------- Saving Options -----------------------------------------------
    real_time = 0  # Keep track of how much time passed
    last_save_time = 0  # Used to track when the last save occurred
    save_counter = 0  # used in the saving loop

    if True:  # Change to False if you want to keep the file or mess with the filename
        if os.path.exists(file_name):  # Delete old files
            os.remove(file_name)

    # Initialize the progress bar
    bar_format = "{desc} {percentage:03.0f}% | {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}]"
    tqdm_bar = tqdm(total=sim_time, desc='Simulating 2D diffusion', bar_format=bar_format)

    # Save the initial state
    with h5py.File(file_name, 'a') as hf:
        h_save_name = f'h_snapshot_{save_counter:05d}'  # Unique name for 'h'
        t_save_name = f't_snapshot_{save_counter:05d}'  # Unique name for 'time'
        hf.create_dataset(h_save_name, data=h)
        hf.create_dataset(t_save_name, data=real_time)
    save_counter += 1

    # The actual loop
    while real_time < sim_time:
        # Stop loop if simulation interrupted by user
        if worker and worker.stop_requested:
            tqdm_bar.close()
            interrupted = 1
            break

        # Figure out the diffusion matrix
        if use_ramp == 1:
            if warmup_period <= real_time <= alpha_end_time + warmup_period and real_time != 0:
                new_d_h = sigmoid(real_time - warmup_period, alpha_0, alpha_1, alpha_end_time / 2, k)
                d_h[:, :] = new_d_h  # set new diffusion coefficient matrix
                dt = adaptive_time_step(new_d_h, dx2, dy2)
                if dt > min_ramp_dt:
                    dt = min_ramp_dt
            if real_time > alpha_end_time + warmup_period:
                d_h[:, :] = alpha_1  # set it to the final alpha
                dt = adaptive_time_step(alpha_1, dx2, dy2)  # new highest stable time step
                use_ramp = 0  # don't use this loop anymore now

        # Calculate derivatives
        dhdx2, dhdy2 = compute_field_derivatives(h, dx2, dy2, dhdx2, dhdy2)

        # Calculate the matrix stuff
        h = update_h_with_jit(h, h0, d_h, dt, dhdx2, dhdy2)

        # Handle Boundary, either fixed or Neumann (dx, dy = 0)
        h[0, :] = h[1, :] if border_bc[0] else border_hydrogen[0]  # Top boundary
        h[-1, :] = h[-2, :] if border_bc[1] else border_hydrogen[1]  # Bottom boundary
        h[:, 0] = h[:, 1] if border_bc[2] else border_hydrogen[2]  # Left boundary
        h[:, -1] = h[:, -2] if border_bc[3] else border_hydrogen[3]  # Right boundary

        h0 = h.copy()  # The new becomes old ... how poetic of me
        real_time += dt  # Keep track of current time of simulation for later use in animation / diagrams

        # Save the matrix
        if math.floor(real_time / save_every_x_s) > math.floor(last_save_time / save_every_x_s):  # save every save_every_x_s [s]
            # It's a new interval, update last_save_time
            last_save_time = real_time

            h_save_name = f'h_snapshot_{save_counter:05d}'  # Unique name for 'h'
            t_save_name = f't_snapshot_{save_counter:05d}'  # Unique name for 'time'
            save_counter += 1

            with h5py.File(file_name, 'a') as hf:
                # Create a dataset with the unique name and store the data
                hf.create_dataset(h_save_name, data=h)
                hf.create_dataset(t_save_name, data=real_time)

        # Ensure real_time and dt are not None before updating the progress bar
        if real_time is not None and dt is not None:
            # Update the progress bar manually
            tqdm_bar.update(min(dt, sim_time - real_time))  # we might go over sim_time and tqdm no likey
            tqdm_bar.set_postfix({'real_time': f"{real_time:.2f}", 'dt': f"{dt:.2f}"})

    tqdm_bar.close()

    if interrupted == 0:
        print("STEP_COMPLETED: 2D simulation")  # keep track of progress in GUI
    elif interrupted == 1:
        print("Simulation stopped by user request.")


def load_sample_plot_point(file_name="charging_sample.h5"):
    print("Currently Loading data:")  # keep track of progress in GUI
    """
    Load the HDF5 file and plot the hydrogen concentration at the middle point over time.

    Parameters:
    file_name (str): Name of the HDF5 file to load. Default is "charging_sample.h5".
    """
    # Initialize lists to store time and hydrogen concentration
    time_points = []
    h_values = []

    try:
        # Open the HDF5 file
        with h5py.File(file_name, 'r') as hf:
            # Find the middle point of the sample
            sample_shape = hf['h_snapshot_00000'].shape
            x, y = sample_shape[1] // 2, sample_shape[0] // 2

            # Get the number of keys to set up the progress bar
            total_keys = sum(1 for key in hf.keys() if key.startswith('h_snapshot_'))

            # Iterate over all datasets in the file with progress bar
            bar_format = "{desc} {percentage:03.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            tqdm_bar = tqdm(hf.keys(), total=total_keys, desc="Loading data", bar_format=bar_format)
            for key in tqdm_bar:
                # Check if the dataset is a hydrogen concentration snapshot
                if key.startswith('h_snapshot_'):
                    # Get the time snapshot corresponding to this concentration snapshot
                    t_key = key.replace('h_snapshot_', 't_snapshot_')
                    if t_key in hf:
                        # Append the time value to the list (convert to hours)
                        time_points.append(hf[t_key][()] / 3600)  # Convert seconds to hours
                        # Append the hydrogen concentration value at the specified point to the list
                        h_values.append(hf[key][y, x])

        # Convert lists to numpy arrays for easier plotting
        time_points = np.array(time_points)
        h_values = np.array(h_values)

        # Sort the time points and corresponding hydrogen values (just in case they are not sorted)
        sorted_indices = np.argsort(time_points)
        time_points = time_points[sorted_indices]
        h_values = h_values[sorted_indices]

        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, h_values, marker='o')
        plt.xlabel('Time (hours)')
        plt.ylabel('Hydrogen Concentration')
        plt.title(f'Hydrogen Concentration at Middle Point ({x}, {y}) Over Time')
        plt.grid(True)

        # Set axis limits
        plt.xlim(left=0)  # Start x-axis at 0, automatically adjust the upper limit
        plt.ylim(bottom=0)  # Start y-axis at 0, automatically adjust the upper limit

        plt.show()


    except IndexError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("STEP_COMPLETED: Loading Data")  # keep track of progress in GUI


def load_sample_plot_line(time_step, file_name="charging_sample.h5"):
    """
    Load the HDF5 file and plot the hydrogen concentration along the middle line at a specific time step.

    Parameters:
    time_step (float): Desired time step in seconds.
    file_name (str): Name of the HDF5 file to load. Default is "charging_sample.h5".
    """
    try:
        # Open the HDF5 file
        with h5py.File(file_name, 'r') as hf:
            # Get the shape of the matrix to check bounds
            sample_shape = hf['h_snapshot_00000'].shape
            y_pos = sample_shape[0] // 2  # Middle line

            # Initialize lists to store times and dataset names
            time_points = []
            dataset_names = []

            # Iterate over all datasets in the file with progress bar
            for key in tqdm(hf.keys(), desc="Loading time steps"):
                if key.startswith('t_snapshot_'):
                    time_points.append(hf[key][()])
                    dataset_names.append(key.replace('t_snapshot_', 'h_snapshot_'))

            # Convert time_points to numpy array for easier manipulation
            time_points = np.array(time_points)

            # Find the index of the closest time step
            closest_index = np.argmin(np.abs(time_points - time_step))
            closest_time = time_points[closest_index]
            h_dataset_name = dataset_names[closest_index]

            print(f"Closest time to {time_step} is {closest_time}. Using dataset {h_dataset_name}.")

            # Extract the hydrogen concentration at the closest time step
            h_data = hf[h_dataset_name][y_pos, :]

            # Plotting the data
            plt.figure(figsize=(10, 6))
            plt.plot(h_data, marker='o')
            plt.xlabel('X Position')
            plt.ylabel('Hydrogen Concentration')
            plt.title(f'Hydrogen Concentration along Y={y_pos} at Time {closest_time} seconds')
            plt.grid(True)

            # Set axis limits
            plt.xlim(left=0)  # Start x-axis at 0, automatically adjust the upper limit
            plt.ylim(bottom=0)  # Start y-axis at 0, automatically adjust the upper limit

            plt.show()

    except IndexError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def create_hydrogen_animation(loaded_u_arrays, loaded_t_values, frame_interval=10, max_length=90, dpi=100, fps=30):
    print("Currently creating hydrogen animation:")

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from tqdm import tqdm

    """
    Create an animation of the hydrogen concentration heat map and plot additional diagrams.

    Parameters:
    loaded_u_arrays (list of np.array): List of hydrogen concentration arrays.
    loaded_t_values (list of float): List of time values corresponding to the concentration arrays.
    frame_interval (int): Interval at which to use frames for the animation.
    max_length (int): Maximum length of the animation in seconds.
    dpi (int): Dots per inch for the animation.
    fps (int): Frames per second for the animation.
    """

    # Ensure that the number of frames doesn't exceed the available frames
    total_frames = len(loaded_u_arrays)
    max_frames = min(total_frames, max_length * fps)
    frame_interval = max(1, total_frames // max_frames)  # Adjust frame_interval to fit within max_frames

    # Figure out which time to display:
    time_unit = "[s]"
    if max(loaded_t_values) >= 31557600:
        loaded_t_values = np.array(loaded_t_values) / 31557600
        time_unit = "[y]"
    elif max(loaded_t_values) >= 864000:
        loaded_t_values = np.array(loaded_t_values) / 86400
        time_unit = "[d]"
    elif max(loaded_t_values) >= 7200:
        loaded_t_values = np.array(loaded_t_values) / 3600
        time_unit = "[h]"
    elif max(loaded_t_values) >= 600:
        loaded_t_values = np.array(loaded_t_values) / 60
        time_unit = "[min]"
    else:
        pass  # no need to change anything, its in seconds and the unit is [s] already too

    # Size of the matrices
    num_rows, num_cols = loaded_u_arrays[0].shape
    extent_heat_map = [0, num_cols, num_rows, 0]  # left, right, bottom, top

    # Middle point and middle line
    middle_point = (num_cols // 2, num_rows // 2)
    middle_line = num_rows // 2

    # Create figure and gridspec layout
    fig = plt.figure(figsize=(16, 8), dpi=dpi)
    gs = fig.add_gridspec(12, 10, width_ratios=[6, 1, 1, 1, 1, 1, 1, 1, 1, 1], wspace=0.4, hspace=1.2)

    # Left: Heat map animation
    ax_heatmap = fig.add_subplot(gs[:, :6])
    norm = mcolors.Normalize(vmin=0, vmax=np.max(loaded_u_arrays))
    cmap = plt.get_cmap('viridis')
    im = ax_heatmap.imshow(loaded_u_arrays[0], cmap=cmap, norm=norm, interpolation='nearest', aspect='equal',
                           extent=extent_heat_map)
    plt.colorbar(im, ax=ax_heatmap)
    ax_heatmap.set_title('Hydrogen Concentration in Sample')
    ax_heatmap.set_xlabel('Width [0.1mm]')
    ax_heatmap.set_ylabel('Height [0.1mm]')
    time_text = ax_heatmap.text(0.35, 1.08, f'Time: {0:.2f} hours', transform=ax_heatmap.transAxes, color='black',
                                fontsize=20)

    # Top Right: Time series plot at a specific point
    ax_time_series = fig.add_subplot(gs[:6, 6:])
    y_data_point = [u[middle_point[1], middle_point[0]] for u in loaded_u_arrays]
    line_point, = ax_time_series.plot(loaded_t_values, y_data_point)
    dot_point, = ax_time_series.plot([], [], 'ko', markersize=5)
    ax_time_series.set_title(f'Hydrogen Concentration at Middle Point {middle_point}')
    ax_time_series.xaxis.set_label_position('top')
    ax_time_series.xaxis.tick_top()
    ax_time_series.set_xlabel('Time (hours)')
    ax_time_series.set_ylabel('Hydrogen Concentration')
    ax_time_series.grid(True)

    # Bottom Right: Line plot across x-axis at a specific y position
    ax_line_plot = fig.add_subplot(gs[6:, 6:])
    y_data_line = loaded_u_arrays[0][middle_line, :]
    line_line, = ax_line_plot.plot(y_data_line)
    ax_line_plot.set_title(f'Hydrogen Concentration along Y={middle_line}')
    ax_line_plot.set_xlabel('X Position [0.1mm]')
    ax_line_plot.set_ylabel('Hydrogen Concentration')
    ax_line_plot.grid(True)

    # Set up tqdm progress bar
    tqdm_bar = tqdm(total=len(range(0, total_frames, frame_interval)), desc="Saving Animation", bar_format="{desc} {percentage:03.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def update(frame):
        # Update heat map
        im.set_data(loaded_u_arrays[frame])

        # Update time text
        time_to_display = loaded_t_values[frame]
        time_text.set_text(f'Time: {time_to_display:.2f} {time_unit}')

        # Update time series plot at the point
        line_point.set_data(loaded_t_values[:frame + 1], y_data_point[:frame + 1])
        dot_point.set_data([loaded_t_values[frame]], [y_data_point[frame]])

        # Update line plot across x-axis at the y position
        line_line.set_ydata(loaded_u_arrays[frame][middle_line, :])

        # Update the progress bar
        tqdm_bar.update(1)

        return im, time_text, line_point, dot_point, line_line

    ani = FuncAnimation(fig, update, frames=range(0, total_frames, frame_interval), repeat=False, blit=True)

    ani.save('hydrogen_concentration_animation.mp4', writer='ffmpeg', dpi=dpi, fps=fps)

    # Close the tqdm progress bar and figure
    tqdm_bar.close()
    plt.close(fig)

    print("STEP_COMPLETED: Hydrogen animation")
