"""Command-line entry point for the P4 Hydrogen Permeation Flux tool."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from permeation_cases import (
    MODULE_ROOT,
    RUNTIME_CONFIG_PATH,
    build_atlas_cases,
    list_presets,
    load_preset,
    load_settings,
)
from permeation_persistence import load_atlas_hdf5, save_atlas_hdf5


RESULTS_DIR = MODULE_ROOT / "02_Results"


def _plain_result_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "Result name must be a plain filename stem containing letters, numbers, '.', '_' or '-'."
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate normalized 1D hydrogen-permeation flux response figures."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="open the PyQt5 graphical interface")
    mode.add_argument("--cli", action="store_true", help="explicitly use the command-line workflow")
    parser.add_argument("--preset", help="Shipped diagram preset; defaults to overview for new runs.")
    parser.add_argument("--list-presets", action="store_true", help="List shipped presets and exit.")
    parser.add_argument("--config", type=Path, help="Optional Python configuration override.")
    parser.add_argument("--result-name", type=_plain_result_name, default="hydrogen_permeation_flux")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--rerender", type=Path, help="Render an existing P4 HDF5 result without simulating.")
    parser.add_argument(
        "--normalization",
        choices=("common_reference", "per_curve", "physical"),
        help="Override the configured flux display mode.",
    )
    parser.add_argument(
        "--time-axis",
        choices=("reference", "fo", "seconds", "minutes"),
        help="Override the configured time axis.",
    )
    parser.add_argument(
        "--response-metric",
        choices=("t10", "t50", "t90", "time_lag", "peak_flux", "final_flux", "overshoot"),
        help="Metric used for response-map contours.",
    )
    parser.add_argument(
        "--formats",
        help="Comma-separated export formats chosen from pdf,svg,png.",
    )
    parser.add_argument("--show", action="store_true", help="Display generated figures after saving.")
    return parser


def _print_presets() -> None:
    for name, values in list_presets().items():
        print(f"{name:14s} {values.get('description', '')}")


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.gui:
        from permeation_gui import launch_gui

        return launch_gui(
            config_path=args.config or RUNTIME_CONFIG_PATH,
            results_dir=args.output_dir,
        )
    if args.list_presets:
        _print_presets()
        return 0

    from permeation_diagrams import render_figures
    from permeation_gui_support import build_result_metadata

    settings = load_settings(args.config)
    diagram = settings["diagram"]
    normalization = args.normalization or diagram["normalization"]
    time_axis = args.time_axis or diagram["time_axis"]
    response_metric = args.response_metric or diagram["response_metric"]
    formats = (
        [item.strip() for item in args.formats.split(",") if item.strip()]
        if args.formats
        else diagram["formats"]
    )
    diagram.update(
        {
            "normalization": normalization,
            "time_axis": time_axis,
            "response_metric": response_metric,
            "formats": formats,
        }
    )

    if args.rerender:
        results, metadata = load_atlas_hdf5(args.rerender)
        if args.preset:
            figures = load_preset(args.preset)["figures"]
        else:
            figures = metadata.get("figures", ["overview"])
        print(f"Loaded {len(results)} cases from {args.rerender}")
    else:
        preset_name = args.preset or "overview"
        preset = load_preset(preset_name)
        figures = preset["figures"]
        print(f"Simulating preset '{preset_name}' ...")
        results, metadata = build_atlas_cases(settings, figures)
        metadata.update(build_result_metadata(settings, preset_name, figures, args.result_name))
        hdf5_path = args.output_dir / f"{args.result_name}.h5"
        save_atlas_hdf5(hdf5_path, results, metadata)
        print(f"Saved {len(results)} numerical cases to {hdf5_path}")

    figure_paths = render_figures(
        results,
        figures,
        args.output_dir,
        args.result_name,
        normalization=normalization,
        time_axis=time_axis,
        response_metric=response_metric,
        comparison_window_ref=float(diagram.get("comparison_window_ref", 1.25)),
        formats=formats,
        dpi=int(diagram["dpi"]),
        show=args.show,
        style=diagram,
    )
    for path in figure_paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"P4 failed: {exc}", file=sys.stderr)
        raise
