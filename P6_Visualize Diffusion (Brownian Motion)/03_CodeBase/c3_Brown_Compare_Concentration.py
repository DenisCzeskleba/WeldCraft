import matplotlib.pyplot as plt
import numpy as np

from b3_Brown_Functions import *


def plot_comparison(file1, file2):
    matrix1, step1 = load_last_snapshot(file1)
    matrix2, step2 = load_last_snapshot(file2)

    profile1 = compute_concentration_profile(matrix1)
    profile2 = compute_concentration_profile(matrix2)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(profile1)), profile1 * 100, label=f"{file1} (Step {step1})", color="#0000FF")
    plt.plot(np.arange(len(profile2)), profile2 * 100, label=f"{file2} (Step {step2})", color="#FF0000")

    plt.xlabel("X Position")
    plt.ylabel("Concentration [%]")
    plt.title("Final Equilibrium Concentration Profiles")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    file1 = results_dir() / "gut mit trap layer_too long.h5"
    file2 = results_dir() / "gut ohne trap layer.h5"

    plot_comparison(file1, file2)
