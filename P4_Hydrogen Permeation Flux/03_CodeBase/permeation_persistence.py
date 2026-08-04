"""HDF5 persistence for reusable P4 numerical atlas results."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Tuple

import h5py
import numpy as np

from permeation_model import SimulationConfig, SimulationResult


FORMAT_NAME = "weldcraft_p4_hydrogen_permeation_atlas"
FORMAT_VERSION = 1


def safe_case_id(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("_.")
    return cleaned or "case"


def save_atlas_hdf5(
    path: Path | str,
    results: Mapping[str, SimulationResult],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.attrs["format"] = FORMAT_NAME
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["metadata_json"] = json.dumps(dict(metadata or {}), sort_keys=True)
        cases = handle.create_group("cases")
        used_ids: set[str] = set()
        for label, result in results.items():
            base_id = safe_case_id(label)
            case_id = base_id
            suffix = 2
            while case_id in used_ids:
                case_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(case_id)
            group = cases.create_group(case_id)
            group.attrs["label"] = label
            group.attrs["config_json"] = json.dumps(result.config.to_dict(), sort_keys=True)
            group.attrs["metrics_json"] = json.dumps(result.metrics, sort_keys=True, allow_nan=True)
            group.attrs["prefill_age_time_ref"] = result.prefill_age_time_ref
            group.attrs["internal_steps"] = result.internal_steps
            for name, values in (
                ("x_ref", result.x_ref),
                ("time_ref", result.time_ref),
                ("mobile_concentration", result.mobile_concentration),
                ("trap_occupancy", result.trap_occupancy),
                ("outlet_flux_common", result.outlet_flux_common),
                ("inlet_concentration", result.inlet_concentration),
                ("total_hydrogen", result.total_hydrogen),
                ("initial_profile", result.initial_profile),
            ):
                group.create_dataset(name, data=np.asarray(values), compression="gzip", shuffle=True)
    return destination


def load_atlas_hdf5(
    path: Path | str,
) -> Tuple[Dict[str, SimulationResult], Dict[str, Any]]:
    source = Path(path)
    results: Dict[str, SimulationResult] = {}
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("format") != FORMAT_NAME:
            raise ValueError(f"Not a P4 Hydrogen Permeation Flux file: {source}")
        if int(handle.attrs.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("Unsupported P4 HDF5 format version.")
        metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
        for group in handle["cases"].values():
            label = str(group.attrs["label"])
            config = SimulationConfig.from_dict(json.loads(group.attrs["config_json"]))
            result = SimulationResult(
                config=config,
                x_ref=np.asarray(group["x_ref"]),
                time_ref=np.asarray(group["time_ref"]),
                mobile_concentration=np.asarray(group["mobile_concentration"]),
                trap_occupancy=np.asarray(group["trap_occupancy"]),
                outlet_flux_common=np.asarray(group["outlet_flux_common"]),
                inlet_concentration=np.asarray(group["inlet_concentration"]),
                total_hydrogen=np.asarray(group["total_hydrogen"]),
                initial_profile=np.asarray(group["initial_profile"]),
                prefill_age_time_ref=float(group.attrs["prefill_age_time_ref"]),
                internal_steps=int(group.attrs["internal_steps"]),
                metrics=json.loads(group.attrs.get("metrics_json", "{}")),
            )
            results[label] = result
    return results, metadata
