"""Lazy P2 HDF5 indexing, artist reuse, statistics, and animation export."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import h5py
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.figure import Figure


_SNAPSHOT_RE = re.compile(r"^(?P<kind>[uhdt])_snapshot_(?P<index>\d+)$")
DEGREE_C = "\N{DEGREE SIGN}C"


class ResultFormatError(ValueError):
    pass


class ExportCancelled(Exception):
    pass


def _decoded(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


class LazyP2Result:
    """Index a P2 result while reading only the requested 2D frame arrays."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.datasets = {kind: {} for kind in "uhdt"}
        self.metadata = {}
        self.param_config = {}
        self.geometry_config = {}
        self._statistics_cache = None
        self._index()

    def _index(self):
        try:
            with h5py.File(self.path, "r") as handle:
                for key, value in handle.items():
                    if not isinstance(value, h5py.Dataset):
                        continue
                    match = _SNAPSHOT_RE.match(key)
                    if match:
                        self.datasets[match.group("kind")][int(match.group("index"))] = key
                meta = handle.get("/meta")
                if meta is not None:
                    self.metadata = {str(key): _decoded(value) for key, value in meta.attrs.items()}
        except OSError as exc:
            raise ResultFormatError(f"Could not open {self.path.name}: {exc}") from exc
        indices = sorted(set().union(self.datasets["u"], self.datasets["h"], self.datasets["d"]))
        if not indices:
            raise ResultFormatError("No P2 snapshot datasets were found")
        self.indices = indices
        self._parse_metadata_json("param_config_json", "param_config")
        self._parse_metadata_json("geometry_config_json", "geometry_config")
        self.times = self._load_times()
        sample = self.read_frame(0)
        first = next((sample[name] for name in ("temperature", "hydrogen", "diffusivity") if sample[name] is not None), None)
        if first is None or first.ndim != 2:
            raise ResultFormatError("P2 snapshots must be two-dimensional arrays")
        self.shape = first.shape

    def _parse_metadata_json(self, key, attribute):
        raw = self.metadata.get(key)
        if raw is None:
            return
        try:
            value = json.loads(str(raw))
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            setattr(self, attribute, value)

    def _load_times(self):
        result = []
        with h5py.File(self.path, "r") as handle:
            for position, index in enumerate(self.indices):
                key = self.datasets["t"].get(index)
                if key is None:
                    result.append(float(position))
                else:
                    result.append(float(handle[key][()]))
        return np.asarray(result, dtype=float)

    @property
    def frame_count(self):
        return len(self.indices)

    @property
    def run_status(self):
        return str(self.metadata.get("run_status", "legacy/unknown"))

    def _read_optional(self, handle, kind, index):
        key = self.datasets[kind].get(index)
        return None if key is None else handle[key][()]

    def read_frame(self, position):
        position = max(0, min(int(position), self.frame_count - 1))
        index = self.indices[position]
        with h5py.File(self.path, "r") as handle:
            return {
                "position": position,
                "index": index,
                "time": float(self.times[position]),
                "temperature": self._read_optional(handle, "u", index),
                "hydrogen": self._read_optional(handle, "h", index),
                "diffusivity": self._read_optional(handle, "d", index),
            }

    def settings_summary(self):
        settings = self.param_config
        return {
            "Simulation": settings.get("simulation_type", "unknown"),
            "Model": settings.get("model_version", "unknown"),
            "Mesh": f"{settings.get('dx', '?')} x {settings.get('dy', '?')} mm",
            "Frames": self.frame_count,
            "Shape": f"{self.shape[1]} x {self.shape[0]}",
            "Status": self.run_status,
            "Hash": str(self.metadata.get("settings_hash", "legacy/unavailable")),
        }

    def scan_statistics(self, progress_callback=None, stop_event: threading.Event | None = None):
        if self._statistics_cache is not None:
            if progress_callback:
                progress_callback(1.0, "Using cached statistics")
            return {key: value.copy() for key, value in self._statistics_cache.items()}
        max_temperature = []
        mean_temperature = []
        max_hydrogen = []
        mean_hydrogen = []
        for position in range(self.frame_count):
            if stop_event is not None and stop_event.is_set():
                raise ExportCancelled()
            frame = self.read_frame(position)
            temperature = frame["temperature"]
            hydrogen = frame["hydrogen"]
            valid_t = temperature[np.isfinite(temperature) & (temperature >= 0)] if temperature is not None else np.array([])
            valid_h = hydrogen[np.isfinite(hydrogen) & (hydrogen >= 0)] if hydrogen is not None else np.array([])
            max_temperature.append(float(np.max(valid_t)) if valid_t.size else np.nan)
            mean_temperature.append(float(np.mean(valid_t)) if valid_t.size else np.nan)
            max_hydrogen.append(float(np.max(valid_h)) if valid_h.size else np.nan)
            mean_hydrogen.append(float(np.mean(valid_h)) if valid_h.size else np.nan)
            if progress_callback:
                progress_callback((position + 1) / self.frame_count, f"Scanning frame {position + 1}/{self.frame_count}")
        self._statistics_cache = {
            "time": self.times.copy(),
            "max_temperature": np.asarray(max_temperature),
            "mean_temperature": np.asarray(mean_temperature),
            "max_hydrogen": np.asarray(max_hydrogen),
            "mean_hydrogen": np.asarray(mean_hydrogen),
        }
        return {key: value.copy() for key, value in self._statistics_cache.items()}


class P2ResultArtists:
    """Create result artists once and update their data for subsequent frames."""

    def __init__(self, result: LazyP2Result, figure: Figure | None = None):
        self.result = result
        self.figure = figure or Figure(figsize=(12, 7), constrained_layout=True)
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 3, height_ratios=(3.0, 1.35))
        self.temperature_axis = self.figure.add_subplot(grid[0, 0])
        self.hydrogen_axis = self.figure.add_subplot(grid[0, 1])
        self.diffusivity_axis = self.figure.add_subplot(grid[0, 2])
        self.statistics_axis = self.figure.add_subplot(grid[1, :])
        first = result.read_frame(0)
        shape = result.shape
        dx = float(result.param_config.get("dx", 1.0))
        dy = float(result.param_config.get("dy", dx))
        self.extent = [0.0, shape[1] * dx, shape[0] * dy, 0.0]
        blank = np.full(shape, np.nan)
        temperature = first["temperature"] if first["temperature"] is not None else blank
        hydrogen = first["hydrogen"] if first["hydrogen"] is not None else blank
        diffusivity = first["diffusivity"] if first["diffusivity"] is not None else blank
        finite_t = temperature[np.isfinite(temperature)]
        finite_h = hydrogen[np.isfinite(hydrogen)]
        self.temperature_image = self.temperature_axis.imshow(
            temperature, cmap="hot", vmin=0,
            vmax=max(400.0, float(np.max(finite_t)) if finite_t.size else 400.0),
            origin="upper", extent=self.extent, aspect="equal", interpolation="nearest",
        )
        self.hydrogen_image = self.hydrogen_axis.imshow(
            hydrogen, cmap="viridis", vmin=0,
            vmax=max(100.0, float(np.max(finite_h)) if finite_h.size else 100.0),
            origin="upper", extent=self.extent, aspect="equal", interpolation="nearest",
        )
        finite_d = diffusivity[np.isfinite(diffusivity) & (diffusivity >= 0)]
        d_max = float(np.max(finite_d)) if finite_d.size else 1.0
        self.diffusivity_image = self.diffusivity_axis.imshow(
            diffusivity, cmap="magma", vmin=0, vmax=max(d_max, 1e-12),
            origin="upper", extent=self.extent, aspect="equal", interpolation="nearest",
        )
        self.temperature_colorbar = self.figure.colorbar(
            self.temperature_image, ax=self.temperature_axis, label=f"Temperature [{DEGREE_C}]",
        )
        self.hydrogen_colorbar = self.figure.colorbar(
            self.hydrogen_image, ax=self.hydrogen_axis, label="Hydrogen [%]",
        )
        self.diffusivity_colorbar = self.figure.colorbar(
            self.diffusivity_image, ax=self.diffusivity_axis,
            label=r"Thermal diffusivity $D$ [$\mathrm{mm}^2/\mathrm{s}$]",
        )
        self.temperature_axis.set_title("Temperature")
        self.hydrogen_axis.set_title("Hydrogen")
        self.diffusivity_axis.set_title("Thermal diffusivity")
        for axis in (self.temperature_axis, self.hydrogen_axis, self.diffusivity_axis):
            axis.set_xlabel("x [mm]")
            axis.set_ylabel("y [mm]")
        self.statistics_axis.set_xlabel("Simulation time [s]")
        self.statistics_axis.set_ylabel("Value")
        self.statistics_axis.grid(True, alpha=0.25)
        self.max_temperature_line, = self.statistics_axis.plot(
            [], [], color="darkred", label=f"Max temperature [{DEGREE_C}]"
        )
        self.mean_hydrogen_line, = self.statistics_axis.plot([], [], color="steelblue", label="Mean hydrogen [%]")
        self.time_marker = self.statistics_axis.axvline(first["time"], color="black", linestyle="--", alpha=0.65)
        self.statistics_axis.legend(loc="best")
        self.statistics = None
        self.position = 0
        self.update(0)

    def update_statistics(self, statistics):
        self.statistics = statistics
        times = statistics["time"]
        self.max_temperature_line.set_data(times, statistics["max_temperature"])
        self.mean_hydrogen_line.set_data(times, statistics["mean_hydrogen"])
        self.statistics_axis.relim()
        self.statistics_axis.autoscale_view()

    def update(self, position):
        frame = self.result.read_frame(position)
        blank = np.full(self.result.shape, np.nan)
        self.temperature_image.set_data(frame["temperature"] if frame["temperature"] is not None else blank)
        self.hydrogen_image.set_data(frame["hydrogen"] if frame["hydrogen"] is not None else blank)
        self.diffusivity_image.set_data(frame["diffusivity"] if frame["diffusivity"] is not None else blank)
        self.time_marker.set_xdata([frame["time"], frame["time"]])
        self.figure.suptitle(
            f"{self.result.path.name} - frame {frame['position'] + 1}/{self.result.frame_count} - t={frame['time']:.6g} s"
        )
        self.position = frame["position"]
        return frame

    def artist_ids(self):
        return tuple(map(id, (
            self.temperature_image, self.hydrogen_image, self.diffusivity_image,
            self.max_temperature_line, self.mean_hydrogen_line, self.time_marker,
        )))


def export_animation(
    result: LazyP2Result,
    destination,
    *,
    stride=1,
    fps=30,
    dpi=130,
    progress_callback=None,
    stop_event: threading.Event | None = None,
):
    destination = Path(destination)
    temporary_destination = destination.with_name(destination.stem + ".partial" + destination.suffix)
    stride = max(1, int(stride))
    artists = P2ResultArtists(result)
    statistics = result.scan_statistics(stop_event=stop_event)
    artists.update_statistics(statistics)
    positions = list(range(0, result.frame_count, stride))
    if positions[-1] != result.frame_count - 1:
        positions.append(result.frame_count - 1)
    writer = FFMpegWriter(fps=max(1, int(fps)), metadata={"artist": "WeldCraft P2"}, bitrate=1800)
    try:
        if temporary_destination.exists():
            temporary_destination.unlink()
        with writer.saving(artists.figure, str(temporary_destination), dpi=int(dpi)):
            for number, position in enumerate(positions, start=1):
                if stop_event is not None and stop_event.is_set():
                    raise ExportCancelled()
                artists.update(position)
                writer.grab_frame()
                if progress_callback:
                    progress_callback(number / len(positions), f"Rendering frame {number}/{len(positions)}")
        os.replace(temporary_destination, destination)
    except Exception:
        try:
            temporary_destination.unlink()
        except OSError:
            pass
        raise
    return destination

