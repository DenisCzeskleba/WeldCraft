import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time
import matplotlib as mpl
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------- Load Data ---------------------- #
def load_data(h5_filename):
    with h5py.File(h5_filename, 'r') as hf:
        matrices = hf["snapshots"][:]
        saved_steps = hf.attrs["saved_steps"][:]
    return matrices, saved_steps

CODE_DIR = Path(__file__).resolve().parent
P6_ROOT = CODE_DIR.parent
RESULTS_DIR = P6_ROOT / "02_Results"

file_name = RESULTS_DIR / 'random_motion.h5'
output_dir = RESULTS_DIR / "Saved Animations"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "brownian_motion_animation.mp4"
ffmpeg_path = Path(r"C:\Users\DCzes\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.0-full_build\bin\ffmpeg.exe")
if ffmpeg_path.exists():
    mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)

# Load matrix data and saved steps
matrices, saved_steps = load_data(file_name)

# ---------------------- Setup Plot ---------------------- #
def setup_plot(matrix_shape, num_regions):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             gridspec_kw={'width_ratios': [5, 1, 1], 'wspace': 0.3})

    cmap = ListedColormap(['#440154', 'blue', 'red'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    im = axes[0].imshow(np.zeros(matrix_shape), cmap=cmap, norm=norm)
    axes[0].set_title('Diffusion as a result of random motion (Frame: 0)')

    conc_plot, = axes[1].plot([], [], color='blue')
    axes[1].set_title('Concentration Profile')
    axes[1].set_ylim(-1, 101)
    axes[1].set_xlim(-1, matrix_shape[1] + 1)

    # Diffusion Speed Plot (DO NOT UPDATE YET)
    for i in range(num_regions):
        axes[2].plot([0], [0], label=f'Region {i}')  # Fake data
    axes[2].set_title('Diffusion Speed Over Time')
    axes[2].legend()

    return fig, axes, im, conc_plot

# ---------------------- Update Function ---------------------- #
def update(frame, im, matrices, saved_steps, conc_plot, ax):
    h_spots_matrix = matrices[frame]
    im.set_data(h_spots_matrix)
    ax[0].set_title(f'Diffusion as a result of random motion (Step: {saved_steps[frame]})')

    # Compute concentration profile
    total_spots = np.sum(h_spots_matrix > 0, axis=0)
    filled_spots = np.sum(h_spots_matrix == 2, axis=0)
    concentration_profile = np.zeros_like(filled_spots, dtype=float)
    mask = total_spots > 0
    concentration_profile[mask] = filled_spots[mask] / total_spots[mask]

    conc_plot.set_data(np.arange(len(concentration_profile)), concentration_profile * 100)

    return [im, conc_plot]

# ---------------------- Initialize Animation ---------------------- #
fig, axes, im, conc_plot = setup_plot(matrices.shape[1:], 4)

print(f"Converting to .mp4 now. Please wait...\nSaving to: {output_file.resolve()}")

writer = FFMpegWriter(
    fps=12,
    metadata=dict(artist='Denis Czeskleba'),
    bitrate=3000,
    codec="libx264",
    extra_args=["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.1", "-movflags", "+faststart"],
)
with writer.saving(fig, output_file, dpi=120):
    frames = range(len(matrices))
    if tqdm is not None:
        frames = tqdm(frames, desc="Rendering Animation Frames")

    for frame in frames:
        update(frame, im, matrices, saved_steps, conc_plot, axes)
        writer.grab_frame()

plt.close(fig)
