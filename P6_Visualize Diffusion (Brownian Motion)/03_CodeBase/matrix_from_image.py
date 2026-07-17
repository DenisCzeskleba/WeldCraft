import numpy as np
from PIL import Image
from fractions import Fraction
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox


def create_matrix_from_image(image_path, max_sol_white, max_sol_black, show_plot=True):
    """
    Loads an image and creates a matrix where white and black areas get a certain concentration of '1's.

    Parameters:
        image_path (str): Path to the image.
        max_sol_white (Fraction): Fraction of '1's in the white regions.
        max_sol_black (Fraction): Fraction of '1's in the black regions.
        show_plot (bool): Whether to display the generated matrix for verification.

    Returns:
        np.ndarray: The generated matrix with '1's placed according to the defined concentration.
    """
    # Load image as grayscale
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Threshold: Anything above 128 is considered white, below is black
    white_mask = img_array > 128  # True for white areas
    black_mask = img_array <= 128  # True for black areas

    # Get total available spots in each region
    num_white_pixels = np.sum(white_mask)
    num_black_pixels = np.sum(black_mask)

    # Determine how many '1's to place in each region
    num_white_ones = int(num_white_pixels * max_sol_white)
    num_black_ones = int(num_black_pixels * max_sol_black)

    # Create empty matrix
    matrix = np.zeros_like(img_array, dtype=int)

    # Randomly select indices in white regions
    if num_white_ones > 0:
        white_indices = np.argwhere(white_mask)
        selected_white = white_indices[np.random.choice(len(white_indices), num_white_ones, replace=False)]
        matrix[selected_white[:, 0], selected_white[:, 1]] = 1

    # Randomly select indices in black regions
    if num_black_ones > 0:
        black_indices = np.argwhere(black_mask)
        selected_black = black_indices[np.random.choice(len(black_indices), num_black_ones, replace=False)]
        matrix[selected_black[:, 0], selected_black[:, 1]] = 1

    # Display the generated matrix if required
    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(matrix, cmap="gray", interpolation="nearest")
        plt.title("Generated Matrix from Image")
        plt.colorbar(label="Occupancy (1 = Filled)")
        plt.show()

        # Use Tkinter for confirmation
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        user_response = messagebox.askyesno("Matrix Verification", "Does the matrix look correct?")

        if not user_response:
            messagebox.showwarning("Aborting", "Check your image and parameters. Simulation will not continue.")
            exit()  # Stop execution

    return matrix


# # Example usage
# Example image path after migration: P6 root / "01_Resources" / "Bilder" / "Gef\u00fcge_array.tiff"
# max_sol_white = Fraction(20, 100)  # x% concentration in white areas
# max_sol_black = Fraction(2, 100)  # y% concentration in black areas
#
# generated_matrix = create_matrix_from_image(image_path, max_sol_white, max_sol_black)
#
# # Display the generated matrix
# import matplotlib.pyplot as plt
#
# plt.imshow(generated_matrix, cmap="gray")
# plt.title("Generated Matrix from Image")
# plt.show()
