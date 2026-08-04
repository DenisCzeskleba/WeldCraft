"""Regression tests for P4 numerical models, persistence, diagrams, and CLI."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from permeation_cases import build_atlas_cases, list_presets, load_preset, load_settings
from permeation_diagrams import render_figures
from permeation_model import (
    PrefillConfig,
    SimulationConfig,
    SurfaceHistory,
    TrapConfig,
    mixed_output_times,
    reaction_exchange_step,
    simulate_case,
)
from permeation_persistence import load_atlas_hdf5, save_atlas_hdf5


def exact_ideal_flux(fourier_number: float, terms: int = 300) -> float:
    """Exact slab result used only as a numerical regression reference."""

    n = np.arange(1, terms + 1, dtype=np.float64)
    return float(1.0 + 2.0 * np.sum((-1.0) ** n * np.exp(-(n * np.pi) ** 2 * fourier_number)))


class IdealSolverTests(unittest.TestCase):
    def test_mixed_output_schedule_has_exact_count_and_endpoints(self):
        for count in (3, 4, 401):
            times = mixed_output_times(4.0, count)
            self.assertEqual(len(times), count)
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 4.0)
            self.assertTrue(np.all(np.diff(times) > 0.0))

    def test_numerical_outlet_flux_matches_exact_slab_series(self):
        times = np.array([0.0, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0])
        result = simulate_case(
            SimulationConfig(n_nodes=201, n_output=len(times), end_time_ref=1.0),
            times,
        )
        expected = np.array([exact_ideal_flux(value) for value in times[1:]])
        np.testing.assert_allclose(result.outlet_flux_common[1:], expected, rtol=0.008, atol=2.0e-5)

    def test_mesh_refinement_reduces_flux_error(self):
        sample_times = [0.0, 0.05, 0.1]
        exact = exact_ideal_flux(0.1)
        errors = []
        for nodes in (51, 101, 201):
            result = simulate_case(
                SimulationConfig(n_nodes=nodes, n_output=3, end_time_ref=0.1),
                sample_times,
            )
            errors.append(abs(result.outlet_flux_common[-1] - exact))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_flux_normalization_and_physical_conversion(self):
        config = SimulationConfig(
            diffusivity_ratio=2.0,
            length_ratio=0.5,
            solubility_ratio=1.5,
            reference_concentration_mol_mm3=2.0e-9,
            n_nodes=51,
            n_output=31,
            end_time_ref=2.0,
        )
        result = simulate_case(config)
        steady = config.steady_flux_common_reference
        self.assertAlmostEqual(result.flux("per_curve")[-1], result.outlet_flux_common[-1] / steady)
        self.assertAlmostEqual(
            result.flux("physical")[-1],
            result.outlet_flux_common[-1] * config.physical_reference_flux,
        )
        self.assertGreater(result.outlet_flux_common[-1], 0.0)


class TrapAndBoundaryTests(unittest.TestCase):
    def test_local_trap_exchange_conserves_mobile_plus_trapped(self):
        mobile, occupancy = 0.7, 0.25
        capacity = 1.4
        before = mobile + capacity * occupancy
        new_mobile, new_occupancy = reaction_exchange_step(
            mobile, occupancy, capacity, 12.0, 0.4, 1.0e-4
        )
        after = new_mobile + capacity * new_occupancy
        self.assertAlmostEqual(before, after, places=14)
        self.assertGreaterEqual(new_mobile, 0.0)
        self.assertGreaterEqual(new_occupancy, 0.0)
        self.assertLessEqual(new_occupancy, 1.0)

    def test_zero_trap_capacity_reproduces_trap_free_flux(self):
        base = SimulationConfig(n_nodes=81, n_output=71, end_time_ref=1.0)
        free = simulate_case(base)
        zero_capacity = simulate_case(
            base.with_changes(
                traps=TrapConfig(enabled=True, capacity_ratio=0.0, release_half_time_ref=1.0)
            )
        )
        np.testing.assert_allclose(
            free.outlet_flux_common, zero_capacity.outlet_flux_common, rtol=0.0, atol=1.0e-13
        )

    def test_slower_release_delays_and_broadens_response(self):
        base = SimulationConfig(n_nodes=81, n_output=151, end_time_ref=5.0)
        shallow = simulate_case(
            base.with_changes(
                traps=TrapConfig(
                    enabled=True,
                    capacity_ratio=1.0,
                    capture_rate_ref=20.0,
                    release_half_time_ref=0.25,
                )
            )
        )
        deep = simulate_case(
            base.with_changes(
                traps=TrapConfig(
                    enabled=True,
                    capacity_ratio=1.0,
                    capture_rate_ref=20.0,
                    release_half_time_ref=2.0,
                )
            )
        )
        self.assertGreater(deep.metrics["t50"], shallow.metrics["t50"])
        self.assertGreater(deep.metrics["t90"], shallow.metrics["t90"])
        self.assertTrue(np.all((deep.trap_occupancy >= -1.0e-12) & (deep.trap_occupancy <= 1.0 + 1.0e-12)))

    def test_surface_history_starts_late_and_approaches_bounded_value(self):
        surface = SurfaceHistory(
            base_concentration=1.0,
            delta_concentration=-0.5,
            onset_time_ref=0.5,
            time_constant_ref=0.2,
        )
        times = np.array([0.0, 0.25, 0.5, 0.7, 2.0])
        result = simulate_case(
            SimulationConfig(
                surface=surface, n_nodes=51, n_output=len(times), end_time_ref=2.0
            ),
            times,
        )
        np.testing.assert_allclose(result.inlet_concentration[:3], 1.0)
        expected_final = surface.value(2.0)
        self.assertAlmostEqual(result.inlet_concentration[-1], expected_final, places=10)
        self.assertGreaterEqual(result.inlet_concentration[-1], 0.5)

        step = SurfaceHistory(
            base_concentration=1.0,
            delta_concentration=0.5,
            onset_time_ref=0.5,
            transition_mode="step",
        )
        self.assertEqual(step.value(0.499), 1.0)
        self.assertEqual(step.value(0.5), 1.5)


class PrefillTests(unittest.TestCase):
    def test_aged_prefill_is_symmetric_zero_at_surfaces_and_hits_target(self):
        result = simulate_case(
            SimulationConfig(
                n_nodes=101,
                n_output=81,
                end_time_ref=1.0,
                prefill=PrefillConfig(
                    enabled=True,
                    initial_fraction=0.20,
                    target_center_fraction=0.10,
                ),
            )
        )
        profile = result.initial_profile
        self.assertAlmostEqual(profile[0], 0.0)
        self.assertAlmostEqual(profile[-1], 0.0)
        self.assertAlmostEqual(profile[len(profile) // 2], 0.10, places=10)
        np.testing.assert_allclose(profile, profile[::-1], atol=2.0e-13)
        self.assertGreater(result.prefill_age_time_ref, 0.0)


class PersistenceAndRenderingTests(unittest.TestCase):
    def test_hdf5_round_trip_preserves_configuration_and_arrays(self):
        result = simulate_case(SimulationConfig(n_nodes=51, n_output=41, end_time_ref=0.5))
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_h5_") as directory:
            path = Path(directory) / "atlas.h5"
            save_atlas_hdf5(path, {"reference": result}, {"figures": ["ideal"]})
            loaded, metadata = load_atlas_hdf5(path)
            self.assertEqual(metadata["figures"], ["ideal"])
            self.assertEqual(loaded["reference"].config, result.config)
            np.testing.assert_allclose(
                loaded["reference"].outlet_flux_common, result.outlet_flux_common
            )

    def test_presets_and_all_figure_export_formats(self):
        presets = list_presets()
        self.assertIn("overview", presets)
        self.assertEqual(load_preset("prefill")["figures"], ["prefill"])
        self.assertEqual(
            load_preset("overview")["figures"], ["ideal", "surface", "trapping"]
        )
        annex_figures = load_preset("annex")["figures"]
        self.assertEqual(
            annex_figures,
            [
                "1.1_trap_capacity_flux",
                "1.2_trap_release_flux",
                "1.3_trap_capture_flux",
                "1.4_combined_trap_flux",
                "2.1_residual_hydrogen_flux",
                "2.2_residual_hydrogen_normalized_flux",
            ],
        )

        settings = deepcopy(load_settings())
        self.assertEqual(settings["diagram"]["formats"], ["png"])
        self.assertEqual(settings["diagram"]["time_axis"], "minutes")
        self.assertAlmostEqual(settings["simulation"]["reference_length_mm"], 0.5)
        self.assertAlmostEqual(
            settings["simulation"]["reference_diffusivity_mm2_s"], 6.0e-5
        )
        settings["simulation"].update({"n_nodes": 51, "n_output": 61})
        settings["ideal"].update(
            {
                "diffusivity_ratios": [0.5, 1.0, 2.0],
                "length_ratios": [0.75, 1.0, 1.5],
                "solubility_ratios": [0.5, 1.0, 1.5],
            }
        )
        settings["trapping"].update(
            {
                "end_time_ref": 4.0,
                "capacity_ratios": [0.0, 0.5, 1.0],
                "release_half_times_ref": [0.2, 0.5, 1.0],
                "map_capacity_ratios": [0.5, 1.0, 1.5],
                "map_release_half_times_ref": [0.2, 0.5, 1.0],
            }
        )
        figures = ["overview", "ideal", "surface", "trapping", "prefill", "response_map"]
        results, _ = build_atlas_cases(settings, figures)
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_figures_") as directory:
            paths = render_figures(
                results,
                figures,
                directory,
                "test_atlas",
                formats=("pdf", "svg", "png"),
                dpi=90,
            )
            self.assertEqual(len(paths), len(figures) * 3)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in paths))
            annex_paths = render_figures(
                results,
                annex_figures,
                directory,
                "test_annex",
                formats=("png",),
                dpi=90,
            )
            self.assertEqual(len(annex_paths), 6)
            self.assertTrue(
                all(path.exists() and path.stat().st_size > 0 for path in annex_paths)
            )


class IntegrationSmokeTests(unittest.TestCase):
    def test_cli_lists_presets(self):
        script = Path(__file__).with_name("permeation_atlas.py")
        completed = subprocess.run(
            [sys.executable, str(script), "--list-presets"],
            cwd=script.parent,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("overview", completed.stdout)
        self.assertIn("trapping", completed.stdout)
        self.assertIn("annex", completed.stdout)

    def test_launcher_slot_four_names_p4_and_stays_informational(self):
        repository = Path(__file__).resolve().parents[2]
        launcher_text = (repository / "P0_Launcher" / "Launcher.py").read_text(encoding="utf-8")
        self.assertIn('"Hydrogen Permeation Atlas"', launcher_text)
        self.assertIn("GUI coming soon", launcher_text)
        self.assertNotIn("P4_Placeholder", launcher_text)


if __name__ == "__main__":
    unittest.main()
