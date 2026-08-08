"""Persistence, job, GUI, and launcher integration tests for P4."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import runpy
import tempfile
import unittest
import contextlib
import io

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtTest, QtWidgets

from permeation_atlas import _build_parser
from permeation_cases import estimate_case_count, load_preset, load_settings
from permeation_diagrams import build_figure
from permeation_gui import MainWindow
from permeation_gui_support import (
    DEFAULT_GUI_STATE,
    create_profile,
    ensure_runtime_state,
    export_loaded_results,
    load_result,
    recover_runtime_defaults,
    result_matches_settings,
    rename_profile,
    run_atlas_job,
    scientific_settings_hash,
    write_runtime_state,
)
from permeation_model import SimulationCancelled


def compact_settings():
    settings = deepcopy(load_settings())
    settings["simulation"].update({"n_nodes": 31, "n_output": 31, "end_time_ref": 2.5})
    settings["ideal"].update({
        "diffusivity_ratios": [1.0],
        "length_ratios": [1.0],
        "solubility_ratios": [1.0],
    })
    settings["diagram"].update({"dpi": 60, "formats": ["png"]})
    return settings


class PersistenceTests(unittest.TestCase):
    def test_atomic_runtime_round_trip_preserves_unknown_code(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_config_") as directory:
            path = Path(directory) / "config.py"
            path.write_text("EXTRA_VALUE = 7\n", encoding="utf-8")
            settings = compact_settings()
            state = deepcopy(DEFAULT_GUI_STATE)
            state["result_name"] = "remembered_result"
            profiles = create_profile("My membrane", settings, {})
            write_runtime_state(settings, state, profiles, path)
            settings["simulation"]["reference_length_mm"] = 0.75
            write_runtime_state(settings, state, profiles, path)
            namespace = runpy.run_path(str(path))
            self.assertEqual(namespace["EXTRA_VALUE"], 7)
            self.assertEqual(namespace["GUI_STATE"]["result_name"], "remembered_result")
            self.assertIn("My membrane", namespace["USER_PROFILES"])
            self.assertEqual(namespace["USER_PROFILES"]["My membrane"]["diagram"]["formats"], [])
            loaded, loaded_state, loaded_profiles = ensure_runtime_state(path)
            self.assertEqual(loaded["simulation"]["reference_length_mm"], 0.75)
            self.assertEqual(loaded_state["result_name"], "remembered_result")
            self.assertIn("My membrane", loaded_profiles)

    def test_profiles_are_renamable_and_shipped_names_are_protected(self):
        profiles = create_profile("Membrane A", compact_settings(), {})
        profiles = rename_profile("Membrane A", "Membrane B", profiles)
        self.assertNotIn("Membrane A", profiles)
        self.assertIn("Membrane B", profiles)
        with self.assertRaises(ValueError):
            create_profile("ideal", compact_settings(), profiles)

    def test_invalid_settings_do_not_replace_runtime_file(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_invalid_") as directory:
            path = Path(directory) / "config.py"
            settings = compact_settings()
            write_runtime_state(settings, DEFAULT_GUI_STATE, {}, path)
            before = path.read_bytes()
            settings["simulation"]["n_nodes"] = 10
            with self.assertRaises(ValueError):
                write_runtime_state(settings, DEFAULT_GUI_STATE, {}, path)
            self.assertEqual(path.read_bytes(), before)

    def test_corrupted_runtime_can_be_recovered_to_defaults(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_recovery_") as directory:
            path = Path(directory) / "config.py"
            path.write_text("CONFIG = {broken", encoding="utf-8")
            with self.assertRaises(SyntaxError):
                ensure_runtime_state(path)
            settings, state, profiles = recover_runtime_defaults(path)
            self.assertEqual(settings["simulation"]["reference_length_mm"], 0.5)
            self.assertEqual(state["preset"], "overview")
            self.assertEqual(profiles, {})
            runpy.run_path(str(path))

    def test_scientific_hash_excludes_presentation(self):
        settings = compact_settings()
        first = scientific_settings_hash(settings)
        settings["diagram"]["font_scale"] = 1.5
        self.assertEqual(scientific_settings_hash(settings), first)
        settings["simulation"]["reference_length_mm"] = 0.7
        self.assertNotEqual(scientific_settings_hash(settings), first)


class JobTests(unittest.TestCase):
    def test_case_count_cancellation_and_transactional_run(self):
        settings = compact_settings()
        figures = load_preset("ideal")["figures"]
        self.assertEqual(estimate_case_count(settings, figures), 3)
        cancelled = np.ones(1, dtype=np.int8)
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_cancel_") as directory:
            with self.assertRaises(SimulationCancelled):
                run_atlas_job(settings, "ideal", "cancelled", [], Path(directory), cancel_flag=cancelled)
            self.assertFalse(any(Path(directory).iterdir()))

        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_cancel_existing_") as directory:
            existing = Path(directory) / "cancelled.h5"
            existing.write_bytes(b"previous completed output")
            with self.assertRaises(SimulationCancelled):
                run_atlas_job(settings, "ideal", "cancelled", [], Path(directory), cancel_flag=cancelled)
            self.assertEqual(existing.read_bytes(), b"previous completed output")

        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_job_") as directory:
            progress = []
            outcome = run_atlas_job(
                settings, "ideal", "gui_test", ["png"], Path(directory),
                progress_callback=lambda fraction, message: progress.append((fraction, message)),
            )
            self.assertTrue(outcome["hdf5_path"].exists())
            self.assertEqual(len(outcome["figure_paths"]), 1)
            self.assertTrue(outcome["figure_paths"][0].exists())
            results, metadata = load_result(outcome["hdf5_path"])
            self.assertEqual(metadata["scientific_settings_hash"], scientific_settings_hash(settings))
            self.assertTrue(result_matches_settings(metadata, settings))
            self.assertEqual(len(results), 3)
            self.assertEqual(progress[-1][0], 1.0)

    def test_shared_figure_builder_and_rerender(self):
        settings = compact_settings()
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_export_") as directory:
            destination = Path(directory)
            outcome = run_atlas_job(settings, "ideal", "source", [], destination)
            results, metadata = load_result(outcome["hdf5_path"])
            styled = deepcopy(settings)
            styled["diagram"].update({"font_scale": 1.2, "title_override": "Custom persistent title"})
            figure = build_figure(results, "ideal", style=styled["diagram"])
            self.assertEqual(figure._suptitle.get_text(), "Custom persistent title")
            paths = export_loaded_results(results, metadata, styled, "rerendered", ["svg"], destination)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_headless_window_persists_valid_edit_and_ready_handshake(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_window_") as directory:
            root = Path(directory)
            ready = root / "ready.flag"
            previous = os.environ.get("WELDCRAFT_STARTUP_READY_FILE")
            os.environ["WELDCRAFT_STARTUP_READY_FILE"] = str(ready)
            try:
                window = MainWindow(compact_settings(), DEFAULT_GUI_STATE, {}, root / "config.py", root)
                window.show()
                QtTest.QTest.qWait(50)
                self.app.processEvents()
                self.assertTrue(ready.exists())
                self.assertEqual(window.preset_combo.count(), 8)
                self.assertEqual(len(window.field_widgets), 50)
                thickness = window.field_widgets["simulation.reference_length_mm"][0]
                thickness.setText("0.8")
                thickness.editingFinished.emit()
                namespace = runpy.run_path(str(root / "config.py"))
                self.assertEqual(namespace["CONFIG"]["simulation"]["reference_length_mm"], 0.8)
                self.assertEqual(namespace["GUI_STATE"]["last_result_path"], "")
                window.close()
            finally:
                if previous is None:
                    os.environ.pop("WELDCRAFT_STARTUP_READY_FILE", None)
                else:
                    os.environ["WELDCRAFT_STARTUP_READY_FILE"] = previous

    def test_window_runs_loads_and_previews_complete_result(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p4_window_run_") as directory:
            root = Path(directory)
            state = deepcopy(DEFAULT_GUI_STATE)
            state.update({"preset": "ideal", "result_name": "window_run", "formats": []})
            window = MainWindow(compact_settings(), state, {}, root / "config.py", root)
            window.show()
            window._start_run()
            self.assertIsNotNone(window.worker)
            finished = QtTest.QSignalSpy(window.worker.finished)
            self.assertTrue(finished.wait(20_000))
            QtTest.QTest.qWait(100)
            self.app.processEvents()
            self.assertTrue((root / "window_run.h5").exists())
            self.assertIsNotNone(window.loaded_results)
            self.assertEqual(len(window.loaded_results), 3)
            self.assertIsNotNone(window.preview_canvas)
            self.assertEqual(window.tabs.currentIndex(), window.results_tab_index)
            window.close()

    def test_cli_modes_are_mutually_exclusive(self):
        parser = _build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.gui)
        self.assertFalse(args.cli)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--gui", "--cli"])


if __name__ == "__main__":
    unittest.main()
