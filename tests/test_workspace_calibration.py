"""Hardware-independent tests for fixed-camera planar calibration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workspace_calibration import (
    CalibrationNotReadyError,
    DegenerateCalibrationError,
    WorkspaceCalibrationError,
    estimate_planar_homography,
    load_workspace_calibration,
    save_workspace_calibration,
    transform_points,
)


class WorkspaceCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pixels = np.array(
            [
                [100.0, 80.0],
                [640.0, 70.0],
                [1180.0, 90.0],
                [110.0, 360.0],
                [645.0, 350.0],
                [1170.0, 370.0],
                [95.0, 650.0],
                [630.0, 660.0],
                [1190.0, 640.0],
            ],
            dtype=np.float64,
        )
        self.known_homography = np.array(
            [
                [0.45, 0.02, -290.0],
                [-0.015, 0.50, -170.0],
                [0.0001, -0.00004, 1.0],
            ],
            dtype=np.float64,
        )
        self.table = transform_points(self.pixels, self.known_homography)

    def test_exact_homography_maps_calibration_and_unseen_points(self) -> None:
        calibration = estimate_planar_homography(self.pixels, self.table)
        np.testing.assert_allclose(
            transform_points(self.pixels, calibration.table_from_pixel),
            self.table,
            atol=1e-9,
        )
        unseen = np.array([[300.0, 200.0], [900.0, 500.0]], dtype=np.float64)
        np.testing.assert_allclose(
            transform_points(unseen, calibration.table_from_pixel),
            transform_points(unseen, self.known_homography),
            atol=1e-9,
        )
        self.assertLess(calibration.rms_error_mm, 1e-9)

    def test_pixel_table_round_trip(self) -> None:
        calibration = estimate_planar_homography(self.pixels, self.table)
        u, v = 782.5, 413.25
        x_mm, y_mm = calibration.pixel_to_table(u, v)
        round_trip_u, round_trip_v = calibration.table_to_pixel(x_mm, y_mm)
        np.testing.assert_allclose((round_trip_u, round_trip_v), (u, v), atol=1e-9)

    def test_noisy_overdetermined_fit_reports_error(self) -> None:
        noisy_table = self.table.copy()
        noisy_table[:, 0] += np.array([0.2, -0.1, 0.15, 0.0, -0.2, 0.1, -0.1, 0.05, -0.05])
        calibration = estimate_planar_homography(self.pixels, noisy_table)
        self.assertGreater(calibration.rms_error_mm, 0.0)
        self.assertGreaterEqual(calibration.max_error_mm, calibration.rms_error_mm)
        self.assertLess(calibration.max_error_mm, 1.0)

    def test_insufficient_and_collinear_points_are_rejected(self) -> None:
        with self.assertRaises(WorkspaceCalibrationError):
            estimate_planar_homography(self.pixels[:3], self.table[:3])
        line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        with self.assertRaises(DegenerateCalibrationError):
            estimate_planar_homography(line, line)

    def test_placeholder_file_is_not_treated_as_calibrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            placeholder = Path(temporary_directory) / "workspace.json"
            placeholder.write_text(
                json.dumps({"schema_version": 1, "status": "not_calibrated"}),
                encoding="utf-8",
            )
            with self.assertRaises(CalibrationNotReadyError):
                load_workspace_calibration(placeholder)

    def test_save_and_load_preserve_mapping_and_metadata(self) -> None:
        calibration = estimate_planar_homography(self.pixels, self.table)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "workspace.json"
            save_workspace_calibration(
                calibration,
                self.pixels,
                self.table,
                path,
                camera_metadata={
                    "model": "Logitech C270",
                    "camera_index": 2,
                    "resolution": [1280, 720],
                    "rotation_degrees": 0,
                },
                coordinate_frame={"length_unit": "mm"},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "calibrated")
            self.assertEqual(len(payload["correspondences"]), len(self.pixels))
            loaded = load_workspace_calibration(path)
            np.testing.assert_allclose(
                loaded.table_from_pixel,
                calibration.table_from_pixel,
                atol=1e-12,
            )

            checked = load_workspace_calibration(
                path,
                expected_camera_index=2,
                expected_resolution=(1280, 720),
                expected_rotation_degrees=0,
            )
            np.testing.assert_allclose(
                checked.table_from_pixel,
                calibration.table_from_pixel,
                atol=1e-12,
            )

    def test_runtime_camera_mismatch_is_rejected(self) -> None:
        calibration = estimate_planar_homography(self.pixels, self.table)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "workspace.json"
            save_workspace_calibration(
                calibration,
                self.pixels,
                self.table,
                path,
                camera_metadata={
                    "camera_index": 2,
                    "resolution": [1280, 720],
                    "rotation_degrees": 0,
                },
            )
            mismatch_cases = (
                {"expected_camera_index": 1},
                {"expected_resolution": (640, 480)},
                {"expected_rotation_degrees": 180},
            )
            for expectations in mismatch_cases:
                with self.subTest(expectations=expectations):
                    with self.assertRaises(WorkspaceCalibrationError):
                        load_workspace_calibration(path, **expectations)


if __name__ == "__main__":
    unittest.main()
