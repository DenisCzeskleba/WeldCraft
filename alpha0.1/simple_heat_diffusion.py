"""
For a simple heat simulation - static diffusion for now

"""

from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py
import os
import numexpr as ne

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
su_h = 100
su_w = 120  # 50 wifth, times 2 plus 20 for weld width
fr_le = le - (su_w - we) / 2
fr_ri = ri - (su_w - we) / 2
fr_ab = fr_be = 5

dim_rows = le + ri + we  # The rows have so many many entries in them (1 per mm * the amount of points within 1 mm)
dim_columns = th + su_h + fr_ab + fr_be  # same with columns

# ---------------------------------------- Dimensional step size, gradient --------------------------------------------
dx = dy = 1  # step size - if not equal, tripple check solver logic!
dx2 = dx*dx  # partial differential x
dy2 = dy*dy  # partial differential y

# -------------------- Relevant Starting Conditions - Temperature und Hydrogen concentration --------------------------
t_cool = 160  # Interpass temperature basically. Used for BCs and initialization
t_hot = 1500  # Weld bead temperature. Adjust as needed
t_room = 25  # Surrounding temperature. Remember that the graphs display a different color for t_room! -offset!!

# ---------------------------------------- TEMPERATURE diffusion coefficients -----------------------------------------
diff_coeff_bm = 5.36768  # D = 20 [W/mK] / (8100 [kg/m³] * 460 [J/kgK] = 5.36768 e-6  m²/s = 5.36768 mm²/s  **1
diff_coeff_wm = 4  # For now just made up numbers
diff_coeff_haz = 4.8  # For now just made up numbers - picked something inbetween bm and wm
diff_coeff_air = 0  # no diffusion into air! If != 0, check BCs! right now you have Neumann Boundary Condition!!
highest_diff_coeff = max(diff_coeff_bm, diff_coeff_wm, diff_coeff_haz)

# diff_1 = 8.8 * (10 ** -5) * 25 ** 2.2285  # [µm²/s] Diffusion coefficient 1 - for now just 10 ** 5 times the other one
# diff_2 = 8.8 * (10 ** -10) * 25 ** 2.2285  # Diffusion coefficient 2

diff_1 = 0
diff_2 = 5
# diff_2 = 8.8 * (10 ** -5) * 25 ** 2.2285  # [µm²/s] Diffusion coefficient 1 - for now just 10 ** 5 times the other one
# diff_1 = 8.8 * (10 ** -10) * 25 ** 2.2285  # Diffusion coefficient 2

print("Diffusion coeff. 1 is: " + str(diff_1))
print("Diffusion coeff. 2 is: " + str(diff_2))

highest_diff_coeff = max(diff_1, diff_2)
dt = (dx2 * dy2) / (2 * highest_diff_coeff * (dx2 + dy2))  # highest time step to still be stable
print("dt: " + str(dt))

# Start by opening a .tiff
image_path = "Bilder/duplexstahl1_array.tiff"
image = Image.open(image_path)

# Convert to NumPy array
array = np.array(image, dtype=int)

# Make a copy of the array, just to make sure that if diff_1 is 0, not everything is diff_2 later
diff_array = array.copy()
print("Unique values in array:", np.unique(array))
# Replace 1s with diff_1 and 0s with diff_2
diff_array = np.where(array == 1, diff_1, diff_array)
diff_array = np.where(array == 0, diff_2, diff_array)

# Initialize some dhdx2 matrices
dhdx2 = np.zeros_like(diff_array)
dhdy2 = np.zeros_like(diff_array)

# Initialize some initial h0
h0 = 0 * array

# Simulation time
sim_time = 480 * 1 * 1  # 3600 * 1 * 1  # seconds * hours * days
nsteps = int(sim_time / dt)
print("Total steps: " + str(nsteps))
print("Array shape:", array.shape)
print("Mesh points:", array.size)

# ---------- Save matrices ----------
file_name = "h_diff_tests.h5"  # File name to save the matrix as
if True:  # Change to False if you want to keep the file or mess with the filename
    if os.path.exists(file_name):  # Delete old files
        os.remove(file_name)
save_counter = 0  # used in the saving loop

for m in tqdm(range(nsteps), desc='Simulating Heat Map'):

    # actual calculation
    dhdx2[1:-1, 1:-1] = (h0[2:, 1:-1] - 2 * h0[1:-1, 1:-1] + h0[:-2, 1:-1]) / dx2
    dhdy2[1:-1, 1:-1] = (h0[1:-1, 2:] - 2 * h0[1:-1, 1:-1] + h0[1:-1, :-2]) / dy2

    h = ne.evaluate("h0 + (diff_array * dt * (dhdx2 + dhdy2))")  # calculation with numexpr = perf. boost??
    # h = h0 + diff_array * dt * (dhdx2 + dhdy2)  # "normal" calculation

    # Dirichlet Boundary Condition - Fixed Value
    boundary_value = 0
    h[0, :] = boundary_value  # Top boundary
    h[-1, :] = boundary_value  # Bottom boundary
    h[:, 0] = 5  # Left boundary
    h[:, -1] = boundary_value  # Right boundary

    # Fixed Gradient - Neumann Boundary Condition (here: du/dy | du/dx = 0)
    # h[0, :] = h[1, :]  # Top boundary (du/dy = 0)
    # h[-1, :] = h[-2, :]  # Bottom boundary (du/dy = 0)
    # h[:, 0] = h[:, 1]  # Left boundary (du/dx = 0)
    # h[:, -1] = h[:, -2]  # Right boundary (du/dx = 0)

    # saving stuff
    if m % 5 == 0:  # save the matrix every 150th calculation
        # print(np.sum(h) - np.sum(h0))
        h_save_name = f'h_snapshot_{save_counter:05d}'  # Unique name for 'h'
        save_counter += 1

        with h5py.File(file_name, 'a') as hf:
            # Create a dataset with the unique name and store the data
            hf.create_dataset(h_save_name, data=h)

    h0 = h.copy()

# ---------------------------------------------------------------------------------------------------------------------
"""
**1 Temperaturleitfähigkeit, hier D = lambda / (rho*c), lambda wärmeleitfähigkeit, rho dichte, c spez. Wärmekapazität
    BÖHLER S690 MICROCLEAN: https://www.bohler.de/app/uploads/sites/92/2023/09/productdb/api/s690-microclean_de.pdf

**2 From Tobi Diss p.65. for alpha iron (until ~723)

**3 Adapted from Boellinghaus 1995: 8.8 * e-13 * T ** 2.2285 [cm² / s]! Temperature T is in °C here!
    D_H = (8.8 * 10 ** -11 * T ** 2.2285) # [mm² / s]
    https://nagra.ch/wp-content/uploads/2022/08/e_ntb09-004.pdf p.19
"""
