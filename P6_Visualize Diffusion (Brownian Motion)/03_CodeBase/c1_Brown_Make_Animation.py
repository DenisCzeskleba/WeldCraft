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
if cfg.MAIN_RENDER_MODE not in ("pixels", "dots"):
    raise ValueError("MAIN_RENDER_MODE must be 'pixels' or 'dots'")
if int(cfg.render_every_nth_frame) < 1:
    raise ValueError("render_every_nth_frame must be 1 or greater")

# ---------------------- Input File ---------------------- #
INPUT_H5_FILENAME = None  # Use None for config controlled filename, or set a string like "random_motion_sparse.h5".


input_h5_filename = cfg.h5_filename if INPUT_H5_FILENAME is None else INPUT_H5_FILENAME
input_h5_path = Path(input_h5_filename)
file_name = input_h5_path if input_h5_path.is_absolute() else results_dir() / input_h5_path
output_dir = results_dir() / cfg.animation_output_folder
if not output_dir.exists():
    raise FileNotFoundError(f"Expected output directory does not exist: {output_dir}")
output_file = output_dir / cfg.animation_filename

ffmpeg_path = Path(cfg.ffmpeg_path)
if ffmpeg_path.exists():
    mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg_path)

matrices, saved_steps, diffusion_data = load_brownian_animation_data(file_name, cfg.render_every_nth_frame)


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
        "available_dots": None,
        "hydrogen_dots": None,
        "conc_plot": None,
        "speed_lines": [],
    }

    if cfg.SHOW_MAIN_SIMULATION_PANEL:
        if cfg.MAIN_RENDER_MODE == "pixels":
            cmap = ListedColormap([cfg.COLOR_EMPTY, cfg.COLOR_AVAILABLE_SPOT, cfg.COLOR_HYDROGEN])
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
            state["im"] = axes["main"].imshow(np.zeros(matrix_shape), cmap=cmap, norm=norm)
        else:
            axes["main"].set_facecolor(cfg.COLOR_EMPTY)
            state["available_dots"] = axes["main"].scatter(
                [],
                [],
                s=cfg.DOT_SIZE_AVAILABLE,
                c=cfg.COLOR_AVAILABLE_SPOT,
                alpha=cfg.DOT_ALPHA_AVAILABLE,
                marker="o",
                edgecolors="none",
            )
            state["hydrogen_dots"] = axes["main"].scatter(
                [],
                [],
                s=cfg.DOT_SIZE_HYDROGEN,
                c=cfg.COLOR_HYDROGEN,
                alpha=cfg.DOT_ALPHA_HYDROGEN,
                marker="o",
                edgecolors="none",
            )
            axes["main"].set_xlim(-0.5, matrix_shape[1] - 0.5)
            axes["main"].set_ylim(matrix_shape[0] - 0.5, -0.5)
            axes["main"].set_aspect("equal")
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
    elif state["available_dots"] is not None and state["hydrogen_dots"] is not None:
        available_y, available_x = np.where(h_spots_matrix == 1)
        hydrogen_y, hydrogen_x = np.where(h_spots_matrix == 2)

        state["available_dots"].set_offsets(np.column_stack((available_x, available_y)))
        state["hydrogen_dots"].set_offsets(np.column_stack((hydrogen_x, hydrogen_y)))
        state["axes"]["main"].set_title(f"Diffusion as a result of random motion (Step: {saved_steps[frame]})")
        artists.extend([state["available_dots"], state["hydrogen_dots"]])

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

print(
    "Converting to .mp4 now. Please wait...\n"
    f"Rendering every {cfg.render_every_nth_frame} saved HDF5 frame(s): {len(matrices)} video frames.\n"
    f"Saving to: {output_file.resolve()}"
)

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
