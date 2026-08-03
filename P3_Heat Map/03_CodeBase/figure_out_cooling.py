import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_DIR / "01_Resources" / "Curve fit Sub 150 cooling" / "011_prepaired.CSV"
RESULTS_DIR = PROJECT_DIR / "02_Results"
OUTPUT_FIGURE = RESULTS_DIR / "cooling_fit_comparison.png"
RESULTS_DIR.mkdir(exist_ok=True)

"""
This part works nicely for the lower temperatures (sub 150?), maybe we can use it for the "cool to room temperature". 
This is because there is no real movement of heat "within" the metal plate as it has the same temperature everywhere,
safe for the gradient within the metal resulting form cooling but no extra flow due to the heat input from welding.
"""

# ------------------------------------------------- Import and Convert ------------------------------------------------
# Load the two semicolon-delimited columns and normalize their decimal commas.
# NumPy is already a solver dependency, so this avoids requiring pandas only for
# a small import step.
raw_data = np.genfromtxt(INPUT_CSV, delimiter=";", dtype=str, encoding="utf-8")
if raw_data.ndim != 2 or raw_data.shape[1] != 2:
    raise ValueError(f"Expected two columns in {INPUT_CSV}, got shape {raw_data.shape}.")

numpy_array = np.char.replace(raw_data, ",", ".").astype(float)
print(numpy_array[:5])
print(f"Initial rows: {numpy_array.shape[0]}")

# ------------------------------------------------ Manipulate as needed -----------------------------------------------

trimmed_numpy_array = numpy_array[186100:414000]
print(f"Rows after trimming: {trimmed_numpy_array.shape[0]}")

# Apply a filter to keep only entries where the 'Time' value is an integer
integer_time_indices = (trimmed_numpy_array[:, 0] % 1 == 0)

# Apply the filter to the array
filtered_numpy_array = trimmed_numpy_array[integer_time_indices]
print(f"Rows with integer 'Time' values: {filtered_numpy_array.shape[0]}")

# Reduce the number of points further by taking every 10th point
reduced_numpy_array = filtered_numpy_array[::10]
print(f"Rows after taking every 10th point: {reduced_numpy_array.shape[0]}")

# ------------------------------------------------- Apply Curve fitting -----------------------------------------------
# Extract the time and temperature data from the trimmed array
time_data = reduced_numpy_array[:, 0]  # All rows, first column of the trimmed array
temperature_data = reduced_numpy_array[:, 1]  # All rows, second column of the trimmed array


# Define the model function
def model(t, a, x, b):
    return a * np.exp(-x * t) - b


def modified_newton_cooling(t, T_env, T_initial, k, c):
    """
    Modified Newton's law of cooling to include temperature-dependent cooling rate.
    T(t) = T_env + (T_initial - T_env) * exp(-k * (t^c))

    Where:
    T_env: Environmental temperature
    T_initial: Initial temperature
    k: Overall cooling rate constant
    c: Modifier for the cooling rate
    """
    return T_env + (T_initial - T_env) * np.exp(-k * (t ** c))


# Perform the curve fitting using these models
# The initial guess is very important, let's try some sensible values:
# T_env could be the last temperature in your data (assuming it has cooled down to environmental temp)
# T_initial could be the first temperature in your data
# k is a rate constant, which we don't have much info on, but we'll start with a small value
T_env_guess = temperature_data[-1]
T_initial_guess = temperature_data[0]
c_guess = 1  # Start with a simple model where cooling rate is constant
k_guess = 0.001  # this is a shot in the dark, you might need to adjust this

# Subtract the first time value from all time values to shift the start time to 0
time_data = time_data - time_data[0]

# Include the guess for the new parameter 'c' in the initial guesses
initial_guess_mod = [T_env_guess, T_initial_guess, k_guess, c_guess]

# Bounds to ensure physical relevance
# T_env should be less than the initial temperature and greater than zero, k should be positive
bounds_mod = ([0, 0, 0, 0], [np.inf, T_initial_guess, np.inf, np.inf])

# Perform the simple curve fitting model. Keep the cooling-rate constant x
# nonnegative so the optimizer cannot explore exponentially growing curves.
popt, pcov = curve_fit(
    model,
    time_data,
    temperature_data,
    bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
)

# Extracting the estimated parameters
a_est, x_est, b_est = popt
print(f"Siimple: Estimated parameters: a = {a_est}, x = {x_est}, b = {b_est}")

# Perform the modified curve fitting model
popt_mod, pcov_mod = curve_fit(modified_newton_cooling, time_data, temperature_data, p0=initial_guess_mod, bounds=bounds_mod)

# Extracting the estimated parameters
T_env_est, T_initial_est, k_est, c_est = popt_mod
print(f"Mod: Estimated parameters: T_env = {T_env_est}, T_initial = {T_initial_est}, k = {k_est}, c = {c_est}")

# Generate fitted data using the estimated model
fitted = model(time_data, *popt)
fitted_mod = modified_newton_cooling(time_data, *popt_mod)

# -------------------------------------------- Extrapolate to higher temps --------------------------------------------
# Set the new initial temperature for the prediction.
T_initial_new = 1500

# Use the fitted values for T_env, k, and c from the previous fitting
T_env_pred, k_pred, c_pred = T_env_est, k_est, c_est

# Define a range of time over which you want to predict the cooling
# This should be similar to the range used during fitting or longer if you want to extrapolate
time_pred = np.arange(0, max(time_data), 1)  # Here, I'm assuming time is in the same units as your data

# Apply the model with the new initial temperature
predicted_temperatures = modified_newton_cooling(time_pred, T_env_pred, T_initial_new, k_pred, c_pred)

# --------------------------------------------Shift the data for a better fit -----------------------------------------
# Find the first value of the mod fit temperature data
first_actual_temp = fitted[0]

# Find the corresponding time value in the predicted curve
# Get the absolute differences between the first actual temperature and all predicted temperatures
temp_differences = np.abs(predicted_temperatures - first_actual_temp)

# Find the index of the smallest difference
closest_index = np.argmin(temp_differences)

# Find the time stamp in the predicted curve that corresponds to this index
closest_time_stamp = time_pred[closest_index]

# Calculate the time shift needed
time_shift = closest_time_stamp

# Apply this time shift to the time data of the actual and fitted curves. # out if you dont want this
time_data = time_data + time_shift

# ---------------------------------------------------- Compare to Convection ------------------------------------------

# Convection: Air is a poor conductor, but has low density thus high diffusivity
# Air has roughly the same thermal diffusivity as steel.
# But if caloric:, maybe 10-30 W/m²K should be rightish, BUT this assumes we keep track of the air temp as well!
# Which is obvously almost always the case, because convection is used in cfd mostly to then simulate the fluid.
# If we discard air temperature change, which we do, this will obviously be much lower because the air is constantly
# at room temperature (of ~25°C).

# Parameters
t_room = temperature_data[-1]  # get room temp from last temperature measured
dt = time_data[1] - time_data[0]  # get "dt", just the difference in time_data entries (should be 10s or something?)
print("dt is: " + str(dt))
conv_variable = 3  # 0.09 W/m²K)

# Specific heat capacity (J/kg·K) for low carbon steel
c = 486  # 1010 Air, for steel use 486
# Density (kg/m³) for low carbon steel (if needed)
rho = 7850  # 1.3 Air, for steel use 7850

# Calculate volume and area, lets work with 1 mm for now?
thickness = 1e-3  # Thickness of the material layer affected by convection (m)
area = 1e-3 * 1e-3  # Area of a mesh cell (m²)

V = area * thickness  # Volume of the material affected (m³)

# Initialize the list to store the temperature at each step
convection_temperatures = []
current_temperature_convection = T_initial_new  # Initialize current temperature
convection_temperatures.append(current_temperature_convection)  # add first temperature

for step in range(len(time_pred)-1):

    # Adjusting the temperature change calculation
    q = conv_variable * area * (current_temperature_convection - t_room)  # Rate of heat transfer (W)
    # technically this happens for both, air and steel, which change temperature differently, we dont care about
    # the air tho
    # For real Simu: Think about this more, is boundary in air or metal? Adjust conv_variable as needed.
    delta_T = (q * dt) / (c * rho * V)  # Temperature change

    # Update the current temperature by subtracting the change
    current_temperature_convection -= delta_T

    # Append the new temperature to the list
    convection_temperatures.append(current_temperature_convection)

# ---------------------------------------------------- Show Preview ---------------------------------------------------

# Create the plot
plt.figure(figsize=(10, 5))  # You can adjust the figure size as needed

# Plot the actual temperature data
plt.plot(time_data, temperature_data, label='Actual Temperature', linestyle='-', marker='')

# Plot the fitted curve
plt.plot(time_data, fitted, label='Fitted Curve', linestyle='--', color='red')

# Plot the fitted curve
plt.plot(time_data, fitted_mod, label='Modified Fit Curve', linestyle='--', color='blue')

# Plot the prediction from the configured initial temperature.
plt.plot(time_pred, predicted_temperatures, label=f'Predicted Cooling from {T_initial_new}°C', linestyle='--', color='green')

# Plot the convection cooling curve
plt.plot(time_pred, convection_temperatures, label='Convection Cooling', linestyle='-.', color='orange')

# Add title and labels
plt.title('Temperature Over Time with Curve Fit')
plt.xlabel('Time')
plt.ylabel('Temperature')

# Show legend
plt.legend()

# Save under the standard results boundary, then display the plot.
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=200)
print(f"Saved figure to: {OUTPUT_FIGURE}")
plt.show()
