import matplotlib.pyplot as plt

from b3_Brown_Functions import *


if __name__ == "__main__":
    rows = 20
    cols = 20
    spacing = 5

    styles = ["even", "prime", "diagonal", "border", "checkerboard", "random", "radial"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax, style in zip(axes, styles):
        lattice_matrix = create_crystal_lattice_matrix(rows, cols, spacing, lattice_style=style)
        ax.imshow(lattice_matrix, cmap="viridis", interpolation="nearest")
        ax.set_title(f"{style.capitalize()} Lattice")
        ax.axis("off")

    if len(styles) < len(axes):
        for ax in axes[len(styles):]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()
