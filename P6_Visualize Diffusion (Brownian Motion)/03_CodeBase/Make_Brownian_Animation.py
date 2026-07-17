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


# ---------------------- Animation Options ---------------------- #
SHOW_MAIN_SIMULATION_PANEL = True
SHOW_CONCENTRATION_PROFILE_PANEL = True
SHOW_DIFFUSION_SPEED_PANEL = True

COLOR_EMPTY = "#440154"
COLOR_AVAILABLE_SPOT = "blue"
COLOR_HYDROGEN = "red"
COLOR_CONCENTRATION_LINE = "blue"
DIFFUSION_SPEED_COLORS = [
    "red",
    "orange",
    "green",
    "purple",
    "brown",
    "cyan",
    "black",
]


# ---------------------- Load Data ---------------------- #
def load_data(h5_filename):
    with h5py.File(h5_filename, 'r') as hf:
        matrices = hf["snapshots"][:]
        saved_steps = hf.attrs["saved_steps"][:]

        region_indices = []
        for key in hf.keys():
            if key.startswith("region_"):
                region_number = key.split("_", 1)[1]
                if region_number.isdigit():
                    region_indices.append(int(region_number))

        diffusion_data = {}
        for region_index in sorted(region_indices):
            group_name = f"region_{region_index}"
            group = hf[group_name]
            if "mean_disp" in group:
                mean_disp = group["mean_disp"][:]
            else:
                mean_disp = np.zeros(len(saved_steps), dtype=float)

            if len(mean_disp) < len(saved_steps):
                mean_disp = np.pad(mean_disp, (0, len(saved_steps) - len(mean_disp)), mode="constant")
            elif len(mean_disp) > len(saved_steps):
                mean_disp = mean_disp[:len(saved_steps)]

            diffusion_data[group_name] = mean_disp

    return matrices, saved_steps, diffusion_data

CODE_DIR = Path(__file__).resolve().parent
P6_ROOT = CODE_DIR.parent
RESULTS_DIR = P6_ROOT / "02_Results"

file_name = RESULTS_DIR / 'random_motion.h5'
output_dir = RESULTS_DIR / "Saved Animations"
if not output_dir.exists():
    raise FileNotFoundError(f"Expected output directory does not exist: {output_dir}")
output_file = output_dir / "brownian_motion_animation.mp4"
ffmpeg_path = Path(r"C:\Users\DCzes\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.0-full_build\bin\ffmpeg.exe")
if ffmpeg_path.exists():
    mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)

# Load matrix data and saved steps
matrices, saved_steps, diffusion_data = load_data(file_name)

# ---------------------- Setup Plot ---------------------- #
def setup_plot(matrix_shape, saved_steps, diffusion_data):
    panels = []
    if SHOW_MAIN_SIMULATION_PANEL:
        panels.append(("main", 5))
    if SHOW_CONCENTRATION_PROFILE_PANEL:
        panels.append(("concentration", 2))
    if SHOW_DIFFUSION_SPEED_PANEL:
        panels.append(("speed", 2))

    if not panels:
        raise ValueError("At least one animation panel must be enabled.")

    width_ratios = [panel[1] for panel in panels]
    fig_width = max(6, sum(width_ratios) * 2.0)
    fig, axes_array = plt.subplots(
        1,
        len(panels),
        figsize=(fig_width, 6),
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.3},
    )

    axes_array = np.atleast_1d(axes_array)
    axes = {panel[0]: axis for panel, axis in zip(panels, axes_array)}

    state = {
        "axes": axes,
        "im": None,
        "conc_plot": None,
        "speed_lines": [],
    }

    if SHOW_MAIN_SIMULATION_PANEL:
        cmap = ListedColormap([COLOR_EMPTY, COLOR_AVAILABLE_SPOT, COLOR_HYDROGEN])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        state["im"] = axes["main"].imshow(np.zeros(matrix_shape), cmap=cmap, norm=norm)
        axes["main"].set_title('Diffusion as a result of random motion (Frame: 0)')

    if SHOW_CONCENTRATION_PROFILE_PANEL:
        state["conc_plot"], = axes["concentration"].plot([], [], color=COLOR_CONCENTRATION_LINE)
        axes["concentration"].set_title('Concentration Profile')
        axes["concentration"].set_ylim(-1, 101)
        axes["concentration"].set_xlim(-1, matrix_shape[1] + 1)

    if SHOW_DIFFUSION_SPEED_PANEL:
        speed_axis = axes["speed"]
        max_speed = 0
        for index, (region_name, mean_disp) in enumerate(diffusion_data.items()):
            color = DIFFUSION_SPEED_COLORS[index % len(DIFFUSION_SPEED_COLORS)]
            line, = speed_axis.plot([], [], label=region_name, color=color)
            state["speed_lines"].append((line, mean_disp))
            if len(mean_disp) > 0:
                max_speed = max(max_speed, float(np.nanmax(mean_disp)))

        speed_axis.set_title('Diffusion Speed Over Time')
        speed_axis.set_xlabel('Step')
        speed_axis.set_ylabel('Mean Displacement')
        if len(saved_steps) > 1:
            speed_axis.set_xlim(saved_steps[0], saved_steps[-1])
        else:
            speed_axis.set_xlim(0, 1)
        speed_axis.set_ylim(0, max(max_speed * 1.1, 1))
        if state["speed_lines"]:
            speed_axis.legend()
        else:
            speed_axis.text(
                0.5,
                0.5,
                "No diffusion speed data found",
                ha="center",
                va="center",
                transform=speed_axis.transAxes,
            )

    return fig, state

# ---------------------- Update Function ---------------------- #
def update(frame, state, matrices, saved_steps):
    h_spots_matrix = matrices[frame]
    artists = []

    if state["im"] is not None:
        state["im"].set_data(h_spots_matrix)
        state["axes"]["main"].set_title(f'Diffusion as a result of random motion (Step: {saved_steps[frame]})')
        artists.append(state["im"])

    if state["conc_plot"] is not None:
        # Compute concentration profile
        total_spots = np.sum(h_spots_matrix > 0, axis=0)
        filled_spots = np.sum(h_spots_matrix == 2, axis=0)
        concentration_profile = np.zeros_like(filled_spots, dtype=float)
        mask = total_spots > 0
        concentration_profile[mask] = filled_spots[mask] / total_spots[mask]

        state["conc_plot"].set_data(np.arange(len(concentration_profile)), concentration_profile * 100)
        artists.append(state["conc_plot"])

    for line, mean_disp in state["speed_lines"]:
        line.set_data(saved_steps[:frame + 1], mean_disp[:frame + 1])
        artists.append(line)

    return artists

# ---------------------- Initialize Animation ---------------------- #
fig, animation_state = setup_plot(matrices.shape[1:], saved_steps, diffusion_data)

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
        update(frame, animation_state, matrices, saved_steps)
        writer.grab_frame()

plt.close(fig)
