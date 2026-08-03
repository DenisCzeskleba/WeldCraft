"""Focused regression tests for the P6 graphical interface support layer."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import p6_gui_support as support


class StepAndFilenameTests(unittest.TestCase):
    def test_scientific_step_inputs(self):
        for text in ("3000000000", "3_000_000_000", "3e9", "3 × 10^9"):
            self.assertEqual(support.parse_step_count(text), 3_000_000_000)

    def test_scientific_notation_is_valid_for_other_whole_number_fields(self):
        settings = support.load_gui_settings()
        settings.update({
            "x": "1.3e3",
            "y": "9.2e2",
            "SPOT_CENTER_X": "8.66e2",
            "SPOT_CENTER_Y": "4.6e2",
            "animation_fps": "1.2e1",
        })
        checked = support.validate_gui_settings(settings)
        self.assertEqual(checked["x"], 1300)
        self.assertEqual(checked["animation_fps"], 12)

    def test_exact_frame_counts(self):
        self.assertEqual(
            support.frame_summary(3_000_000_000, 10_000_000),
            {"count": 301, "first": 0, "last": 3_000_000_000},
        )
        self.assertEqual(
            support.frame_summary(4_000_000, 1_500_000),
            {"count": 4, "first": 0, "last": 4_000_000},
        )

    def test_gui_progress_record_survives_terminal_redraw_text(self):
        raw = (
            "Event-driven uniformized steps: 93%|#########2| 92800000/100000000\r"
            "P6_GUI_PROGRESS|0.928000000000|92800000|10|Simulating step 92,800,000\n"
        )
        records, diagnostics, cancelled = support.parse_gui_progress_records(raw)
        self.assertEqual(
            records,
            [(0.928, "Simulating step 92,800,000", 92_800_000, 10)],
        )
        self.assertIn("Event-driven uniformized steps", diagnostics[0])
        self.assertFalse(cancelled)

    def test_plain_output_filenames_only(self):
        self.assertEqual(support.validate_filename("result.h5", (".h5",)), "result.h5")
        for value in ("folder/result.h5", r"folder\result.h5", r"C:\result.h5", "../result.h5"):
            with self.assertRaises(support.P6ConfigError):
                support.validate_filename(value, (".h5",))


class ConfigWriterTests(unittest.TestCase):
    def test_atomic_assignment_update_preserves_hidden_settings(self):
        directory = Path(tempfile.mkdtemp(prefix="weldcraft_p6_config_test_"))
        try:
            path = directory / "config.py"
            shutil.copyfile(support.CONFIG_PATH, path)
            original = path.read_text(encoding="utf-8")
            settings = support.load_gui_settings()
            settings["steps"] = 4_000_000
            support.write_gui_settings(settings, path)
            updated = path.read_text(encoding="utf-8")
            self.assertIn("steps = 4000000", updated)
            self.assertIn("max_radius_to_jump = 10", updated)
            self.assertIn("# Number of simulation steps", updated)
            self.assertNotEqual(original, updated)
        finally:
            shutil.rmtree(directory)

    def test_resolved_settings_save_as_new_non_overwriting_preset(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p6_preset_test_") as directory:
            resolved = support.load_diagram_settings("printer_friendly")
            path = support.save_diagram_preset("My Printer Look", resolved, directory)
            self.assertEqual(path.name, "my_printer_look.py")
            source = path.read_text(encoding="utf-8")
            self.assertIn('PRESET_NAME = \'My Printer Look\'', source)
            self.assertIn("RENDER_MODE = 'printer_glyphs'", source)
            self.assertEqual(
                support.list_custom_diagram_presets(directory),
                ["my_printer_look"],
            )
            with self.assertRaises(support.P6ConfigError):
                support.save_diagram_preset("My Printer Look", resolved, directory)
            deleted = support.delete_custom_diagram_preset("my_printer_look", directory)
            self.assertEqual(deleted, path.resolve())
            self.assertFalse(path.exists())

    def test_shipped_style_preset_is_protected_from_gui_deletion(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p6_preset_test_") as directory:
            path = Path(directory) / "shipped.py"
            path.write_text("PRESET_NAME = 'shipped'\n", encoding="utf-8")
            with self.assertRaises(support.P6ConfigError):
                support.delete_custom_diagram_preset("shipped", directory)
            self.assertTrue(path.exists())


class H5CompatibilityTests(unittest.TestCase):
    def _write_file(self, path: Path, schema: str):
        metadata = {"simulation_mode": "event_driven_wiggle", "x": 3, "y": 2}
        with h5py.File(path, "w") as hf:
            hf.create_dataset("snapshots", data=np.ones((2, 2, 3), dtype=np.int8))
            hf.create_dataset("saved_steps", data=np.array([0, 10], dtype=np.int64))
            hf.attrs["frames_written"] = 2
            hf.attrs["run_status"] = "complete"
            meta = hf.create_group("meta")
            meta.attrs["brown_config_json"] = json.dumps(metadata)
            checkpoint = hf.create_group("checkpoint")
            checkpoint.attrs["schema"] = schema
            checkpoint.attrs["complete"] = True
            checkpoint.attrs["snapshot_index"] = 1
            checkpoint.attrs["step"] = 10
            checkpoint.attrs["simulation_mode"] = "event_driven_wiggle"
            checkpoint.attrs["event_pending_wait_steps"] = 0
            checkpoint.attrs["event_total_transition_weight"] = 1.0
            checkpoint.create_dataset("rng_state", data=np.ones(4, dtype=np.uint64))
            checkpoint.create_dataset("ordered_hydrogen_site_ids", data=np.array([0], dtype=np.int32))
            checkpoint.create_dataset("event_fenwick_tree", data=np.ones(2, dtype=np.float64))

    def test_new_exact_checkpoint_is_resumable(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p6_h5_test_") as directory:
            path = Path(directory) / "new.h5"
            self._write_file(path, support.NEW_GUI_CHECKPOINT_SCHEMA)
            source = support.H5FrameSource(path)
            self.assertEqual(source.frame_count, 2)
            self.assertTrue(support.inspect_resume_source(path)["valid"])

    def test_old_checkpoint_remains_viewable_but_not_resumable(self):
        with tempfile.TemporaryDirectory(prefix="weldcraft_p6_h5_test_") as directory:
            path = Path(directory) / "old.h5"
            self._write_file(path, "brownian_exact_restart_v1")
            self.assertEqual(support.H5FrameSource(path).frame_count, 2)
            self.assertFalse(support.inspect_resume_source(path)["valid"])


class DiagramRendererTests(unittest.TestCase):
    def test_printer_glyphs_use_batched_collections(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import c3_Brown_Make_Diagram as diagram

        original_preset = diagram.DIAGRAM_PRESET
        try:
            diagram.apply_diagram_preset("printer_friendly")
            diagram.GLYPH_SHOW_EXPLANATION = False
            figure = Figure()
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            matrix = np.tile(
                np.array([[1, 2], [2, 1]], dtype=np.int8),
                (32, 32),
            )
            diagram.draw_printer_glyphs(axis, matrix, {})
            self.assertLessEqual(len(axis.collections), 4)
            self.assertEqual(len(axis.patches), 0)
        finally:
            diagram.apply_diagram_preset(original_preset)


if __name__ == "__main__":
    unittest.main()
