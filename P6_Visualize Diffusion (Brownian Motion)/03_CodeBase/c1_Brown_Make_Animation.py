import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
from pathlib import Path

from b3_Brown_Functions import *

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


cfg = load_brown_config()

file_name = results_dir() / cfg.h5_filename
output_dir = results_dir() / cfg.animation_output_folder
if not output_dir.exists():
    raise FileNotFoundError(f"Expected output directory does not exist: {output_dir}")
output_file = output_dir / cfg.animation_filename

ffmpeg_path = Path(cfg.ffmpeg_path)
if ffmpeg_path.exists():
    mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)

matrices, saved_steps, diffusion_data = load_brownian_animation_data(file_name)


def setup_plot(matrix_shape, saved_steps, diffusion_data):
    panels = []
    if cfg.SHOW_MAIN_SIMULATION_PANEL:
        panels.append(("main", 5))
    if cfg.SHOW_CONCENTRATION_PROFILE_PANEL:
        panels.append(("concentration", 2))
    if cfg.SHOW_DIFFUSION_SPEED_PANEL:
        panels.append(("speed", 2))

    if not panels:
        raise ValueError("At least one animation panel must be enabled.")

    width_ratios = [panel[1] for panel in panels]
    fig_width = max(6, sum(width_ratios) * 2.0)
    fig, axes_array = plt.subplots(
        1,
        len(panels),
        figsize=(fig_width, 6),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.3},
    )

    axes_array = np.atleast_1d(axes_array)
    axes = {panel[0]: axis for panel, axis in zip(panels, axes_array)}

    state = {
        "axes": axes,
        "im": None,
        "conc_plot": None,
        "speed_lines": [],
    }

    if cfg.SHOW_MAIN_SIMULATION_PANEL:
        cmap = ListedColormap([cfg.COLOR_EMPTY, cfg.COLOR_AVAILABLE_SPOT, cfg.COLOR_HYDROGEN])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        state["im"] = axes["main"].imshow(np.zeros(matrix_shape), cmap=cmap, norm=norm)
        axes["main"].set_title("Diffusion as a result of random motion (Frame: 0)")

    if cfg.SHOW_CONCENTRATION_PROFILE_PANEL:
        state["conc_plot"], = axes["concentration"].plot([], [], color=cfg.COLOR_CONCENTRATION_LINE)
        axes["concentration"].set_title("Concentration Profile")
        axes["concentration"].set_ylim(-1, 101)
        axes["concentration"].set_xlim(-1, matrix_shape[1] + 1)

    if cfg.SHOW_DIFFUSION_SPEED_PANEL:
        speed_axis = axes["speed"]
        max_speed = 0
        speed_colors = cfg.DIFFUSION_SPEED_COLORS or ["#000000"]

        for index, (region_name, mean_disp) in enumerate(diffusion_data.items()):
            color = speed_colors[index % len(speed_colors)]
            line, = speed_axis.plot([], [], label=region_name, color=color)
            state["speed_lines"].append((line, mean_disp))
            if len(mean_disp) > 0:
                max_speed = max(max_speed, float(np.nanmax(mean_disp)))

        speed_axis.set_title("Diffusion Speed Over Time")
        speed_axis.set_xlabel("Step")
        speed_axis.set_ylabel("Mean Displacement")
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


def update(frame, state, matrices, saved_steps):
    h_spots_matrix = matrices[frame]
    artists = []

    if state["im"] is not None:
        state["im"].set_data(h_spots_matrix)
        state["axes"]["main"].set_title(f"Diffusion as a result of random motion (Step: {saved_steps[frame]})")
        artists.append(state["im"])

    if state["conc_plot"] is not None:
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


fig, animation_state = setup_plot(matrices.shape[1:], saved_steps, diffusion_data)

print(f"Converting to .mp4 now. Please wait...\nSaving to: {output_file.resolve()}")

writer = FFMpegWriter(
    fps=cfg.animation_fps,
    metadata=dict(artist=cfg.animation_artist),
    bitrate=cfg.animation_bitrate,
    codec="libx264",
    extra_args=["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.1", "-movflags", "+faststart"],
)
with writer.saving(fig, output_file, dpi=cfg.animation_dpi):
    frames = range(len(matrices))
    if tqdm is not None:
        frames = tqdm(frames, desc="Rendering Animation Frames")

    for frame in frames:
        update(frame, animation_state, matrices, saved_steps)
        writer.grab_frame()

plt.close(fig)
