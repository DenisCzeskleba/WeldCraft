import numpy as np
import matplotlib.pyplot as plt

def create_crystal_lattice_matrix(rows, cols, spacing, lattice_style='even'):
    """
    Create a matrix representing a uniform crystal lattice with different styles.

    Args:
    - rows (int): Number of rows in the matrix.
    - cols (int): Number of columns in the matrix.
    - spacing (int): Distance between points in the lattice.
    - lattice_style (str): Style of the lattice pattern ('even', 'prime', 'diagonal', 'border').

    Returns:
    - np.array: Matrix with the specified crystal lattice pattern.
    """
    matrix = np.zeros((rows, cols), dtype=int)
    if lattice_style == 'even':
        # Regular grid pattern
        matrix[spacing//2::spacing, spacing//2::spacing] = 1

    elif lattice_style == 'prime':
        # Simpler approach: Place points in alternating rows and columns
        for row in range(0, rows, spacing):
            for col in range(0, cols, spacing):
                matrix[row, col] = 1  # Place a point every "spacing" in the current row

        # Move down by spacing//2 rows, offset column by spacing//2
        for row in range(spacing // 2, rows, spacing):
            for col in range(spacing // 2, cols, spacing):
                matrix[row, col] = 1  # Place a point with offset every "spacing" in this row

    elif lattice_style == 'diagonal':
        # Diagonal lines across the matrix
        for start in range(0, rows, spacing):
            np.fill_diagonal(matrix[start:], 1)
            np.fill_diagonal(matrix[:, start:], 1)

    elif lattice_style == 'border':
        # Points only on the border of the matrix
        matrix[0::spacing, 0] = 1  # Left border
        matrix[0::spacing, -1] = 1  # Right border
        matrix[0, 0::spacing] = 1  # Top border
        matrix[-1, 0::spacing] = 1  # Bottom border

    elif lattice_style == 'checkerboard':
        # Create a checkerboard pattern where the black and white areas are "spacing x spacing" blocks
        for row in range(0, rows, spacing):
            for col in range(0, cols, spacing):
                if (row // spacing) % 2 == (col // spacing) % 2:
                    matrix[row:row + spacing, col:col + spacing] = 1  # Fill a spacing x spacing block

    elif lattice_style == 'random':
        # Calculate the number of points to place based on the spacing, using abs(spacing) + 1
        num_total_points = rows * cols
        num_points_to_place = int(
            num_total_points * (1 / (abs(spacing) + 1)))  # 1/(abs(spacing) + 1) fraction of points

        # Randomly choose the points
        indices = np.random.choice(num_total_points, num_points_to_place, replace=False)

        # Unravel the 1D indices to 2D indices for the matrix
        matrix[np.unravel_index(indices, (rows, cols))] = 1

    elif lattice_style == 'radial':
        center = (rows // 2, cols // 2)
        for r in range(spacing, min(center), spacing):
            angle_spacing = int(2*np.pi*r / spacing)
            for angle in range(0, angle_spacing):
                x = int(center[0] + r * np.cos(angle * (2*np.pi / angle_spacing)))
                y = int(center[1] + r * np.sin(angle * (2*np.pi / angle_spacing)))
                matrix[y % rows, x % cols] = 1

    return matrix

# Example usage
rows = 20
cols = 20
spacing = 5  # Adjust spacing as needed

styles = ['even', 'prime', 'diagonal', 'border', 'checkerboard', 'random', 'radial']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create 2x4 grid

# Flatten axes array to handle it easily in a loop
axes = axes.flatten()

# Loop over styles and axes
for i, (ax, style) in enumerate(zip(axes, styles)):
    lattice_matrix = create_crystal_lattice_matrix(rows, cols, spacing, lattice_style=style)
    ax.imshow(lattice_matrix, cmap='viridis', interpolation='nearest')
    ax.set_title(f'{style.capitalize()} Lattice')
    ax.axis('off')  # Optionally turn off axis for clearer view

# Turn off the last unused axis (since we only have 7 styles)
if len(styles) < len(axes):
    for ax in axes[len(styles):]:
        ax.axis('off')

plt.tight_layout()
plt.show()
