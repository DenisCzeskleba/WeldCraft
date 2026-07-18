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

# Video has pixel dimensions, not a meaningful print DPI. Keeping rasterization
# at 100 DPI makes Matplotlib's point-sized text predictable; output resolution
# is controlled explicitly by the pixel layout below.
VIDEO_RENDER_DPI = 100

TOP_MARGIN_PX = 60
BOTTOM_MARGIN_PX = 70
LEFT_MARGIN_PX = 70
RIGHT_MARGIN_PX = 30
PANEL_GAP_PX = 70

# ---------------------- Input File ---------------------- #
INPUT_H5_FILENAME = None  # Use None for config controlled filename, or set a string like "random_motion_sparse.h5".


def positive_int(name, value):
    integer_value = int(value)
    if integer_value < 1:
        raise ValueError(f"{name} must be 1 or greater")
    return integer_value


def calculate_layout(matrix_shape, panels):
    matrix_rows, matrix_cols = matrix_shape
    main_pixel_scale = positive_int("animation_main_pixel_scale", cfg.animation_main_pixel_scale)
    side_panel_width_px = positive_int("animation_side_panel_width_px", cfg.animation_side_panel_width_px)

    data_width_px = matrix_cols * main_pixel_scale
    data_height_px = matrix_rows * main_pixel_scale
    panel_widths = {
        "main": data_width_px,
        "concentration": side_panel_width_px,
        "speed": side_panel_width_px,
    }
    total_width_px = LEFT_MARGIN_PX + RIGHT_MARGIN_PX + sum(panel_widths[name] for name in panels)
    total_width_px += PANEL_GAP_PX * (len(panels) - 1)
    total_height_px = TOP_MARGIN_PX + data_height_px + BOTTOM_MARGIN_PX

    # Even frame dimensions keep the layout compatible with conventional H.264
    # encoders too, without changing the data axes or their integer scaling.
    total_width_px += total_width_px % 2
    total_height_px += total_height_px % 2

    return {
        "main_pixel_scale": main_pixel_scale,
        "data_width_px": data_width_px,
        "data_height_px": data_height_px,
        "panel_widths": panel_widths,
        "total_width_px": total_width_px,
        "total_height_px": total_height_px,
    }


def setup_plot(matrix_shape, saved_steps, diffusion_data):
    panels = []
    if cfg.SHOW_MAIN_SIMULATION_PANEL:
        panels.append("main")
    if cfg.SHOW_CONCENTRATION_PROFILE_PANEL:
        panels.append("concentration")
    if cfg.SHOW_DIFFUSION_SPEED_PANEL:
        panels.append("speed")

    if not panels:
        raise ValueError("At least one animation panel must be enabled.")

    layout = calculate_layout(matrix_shape, panels)
    total_width_px = layout["total_width_px"]
    total_height_px = layout["total_height_px"]

    fig = plt.figure(
        figsize=(total_width_px / VIDEO_RENDER_DPI, total_height_px / VIDEO_RENDER_DPI),
        dpi=VIDEO_RENDER_DPI,
    )
    fig.patch.set_facecolor("white")

    axes = {}
    current_left_px = LEFT_MARGIN_PX
    for panel_name in panels:
        axis_width_px = layout["panel_widths"][panel_name]
        axes[panel_name] = fig.add_axes([
            current_left_px / total_width_px,
            BOTTOM_MARGIN_PX / total_height_px,
            axis_width_px / total_width_px,
            layout["data_height_px"] / total_height_px,
        ])
        current_left_px += axis_width_px + PANEL_GAP_PX

    state = {
        "axes": axes,
        "im": None,
        "available_dots": None,
        "hydrogen_dots": None,
        "conc_plot": None,
        "speed_lines": [],
        "layout": layout,
    }

    if cfg.SHOW_MAIN_SIMULATION_PANEL:
        if cfg.MAIN_RENDER_MODE == "pixels":
            cmap = ListedColormap([cfg.COLOR_EMPTY, cfg.COLOR_AVAILABLE_SPOT, cfg.COLOR_HYDROGEN])
            norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
            state["im"] = axes["main"].imshow(
                np.zeros(matrix_shape),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                filternorm=False,
                resample=False,
                aspect="equal",
            )
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
        axes["main"].set_xlim(-0.5, matrix_shape[1] - 0.5)
        axes["main"].set_ylim(matrix_shape[0] - 0.5, -0.5)
        axes["main"].set_aspect("equal", adjustable="box")
        axes["main"].set_title(
            "Diffusion as a result of random motion (Frame: 0)",
            fontsize=cfg.animation_title_font_size,
        )

    if cfg.SHOW_CONCENTRATION_PROFILE_PANEL:
        state["conc_plot"], = axes["concentration"].plot([], [], color=cfg.COLOR_CONCENTRATION_LINE)
        axes["concentration"].set_title(
            "Concentration Profile",
            fontsize=cfg.animation_title_font_size,
        )
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

        speed_axis.set_title("Diffusion Speed Over Time", fontsize=cfg.animation_title_font_size)
        speed_axis.set_xlabel("Step", fontsize=cfg.animation_axis_label_font_size)
        speed_axis.set_ylabel("Mean Displacement", fontsize=cfg.animation_axis_label_font_size)
        if len(saved_steps) > 1:
            speed_axis.set_xlim(saved_steps[0], saved_steps[-1])
        else:
            speed_axis.set_xlim(0, 1)
        speed_axis.set_ylim(0, max(max_speed * 1.1, 1))
        if state["speed_lines"]:
            speed_axis.legend(fontsize=cfg.animation_legend_font_size)
        else:
            speed_axis.text(
                0.5,
                0.5,
                "No diffusion speed data found",
                ha="center",
                va="center",
                fontsize=cfg.animation_axis_label_font_size,
                transform=speed_axis.transAxes,
            )

    for axis in axes.values():
        axis.tick_params(axis="both", labelsize=cfg.animation_tick_font_size)

    return fig, state


def verify_pixel_geometry(fig, state):
    """Fail loudly if a layout change reintroduces non-integer image scaling."""
    fig.canvas.draw()
    layout = state["layout"]
    actual_canvas = tuple(fig.canvas.get_width_height())
    expected_canvas = (layout["total_width_px"], layout["total_height_px"])
    if actual_canvas != expected_canvas:
        raise RuntimeError(f"Animation canvas is {actual_canvas}, expected {expected_canvas} pixels")

    main_axis = state["axes"].get("main")
    if main_axis is None:
        return

    _, _, actual_width, actual_height = main_axis.get_window_extent().bounds
    expected_width = layout["data_width_px"]
    expected_height = layout["data_height_px"]
    if not np.allclose((actual_width, actual_height), (expected_width, expected_height), atol=0.01):
        raise RuntimeError(
            "Main animation panel is not pixel-aligned: "
            f"{actual_width:.3f}x{actual_height:.3f}, expected {expected_width}x{expected_height} pixels"
        )


def update(frame, state, matrices, saved_steps):
    h_spots_matrix = matrices[frame]
    artists = []

    if state["im"] is not None:
        state["im"].set_data(h_spots_matrix)
        state["axes"]["main"].set_title(
            f"Diffusion as a result of random motion (Step: {saved_steps[frame]})",
            fontsize=cfg.animation_title_font_size,
        )
        artists.append(state["im"])
    elif state["available_dots"] is not None and state["hydrogen_dots"] is not None:
        available_y, available_x = np.where(h_spots_matrix == 1)
        hydrogen_y, hydrogen_x = np.where(h_spots_matrix == 2)

        state["available_dots"].set_offsets(np.column_stack((available_x, available_y)))
        state["hydrogen_dots"].set_offsets(np.column_stack((hydrogen_x, hydrogen_y)))
        state["axes"]["main"].set_title(
            f"Diffusion as a result of random motion (Step: {saved_steps[frame]})",
            fontsize=cfg.animation_title_font_size,
        )
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


def main():
    if cfg.MAIN_RENDER_MODE not in ("pixels", "dots"):
        raise ValueError("MAIN_RENDER_MODE must be 'pixels' or 'dots'")
    render_stride = positive_int("render_every_nth_frame", cfg.render_every_nth_frame)

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

    matrices, saved_steps, diffusion_data = load_brownian_animation_data(file_name, render_stride)
    fig, animation_state = setup_plot(matrices.shape[1:], saved_steps, diffusion_data)
    verify_pixel_geometry(fig, animation_state)
    layout = animation_state["layout"]

    print(
        "Converting to video now. Please wait...\n"
        f"Rendering every {render_stride} saved HDF5 frame(s): {len(matrices)} video frames.\n"
        f"Frame size: {layout['total_width_px']}x{layout['total_height_px']} pixels; "
        f"main lattice: {layout['main_pixel_scale']}x{layout['main_pixel_scale']} video pixels per cell.\n"
        f"Saving to: {output_file.resolve()}"
    )

    # RGB lossless H.264 avoids the 4:2:0 chroma averaging that smears isolated
    # one-cell red and blue sites. The preset affects speed/size, not quality.
    writer = FFMpegWriter(
        fps=cfg.animation_fps,
        metadata=dict(artist=cfg.animation_artist),
        codec="libx264rgb",
        extra_args=["-crf", "0", "-preset", "slow", "-pix_fmt", "rgb24", "-movflags", "+faststart"],
    )
    try:
        with writer.saving(fig, output_file, dpi=VIDEO_RENDER_DPI):
            frames = range(len(matrices))
            if tqdm is not None:
                frames = tqdm(frames, desc="Rendering Animation Frames")

            for frame in frames:
                update(frame, animation_state, matrices, saved_steps)
                writer.grab_frame()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
