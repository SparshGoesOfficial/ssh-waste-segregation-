"""Hardware-independent tests for Phase 2 HSV configuration and masking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import (
    HSVRange,
    default_hsv_calibration,
    load_hsv_calibration,
    save_hsv_calibration,
)
from vision import build_hsv_mask


class FakeCV2:
    """Small NumPy implementation used when OpenCV is unavailable in CI."""

    def __init__(self) -> None:
        self.bitwise_or_calls = 0

    @staticmethod
    def inRange(image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        selected = np.all((image >= lower) & (image <= upper), axis=2)
        return np.where(selected, 255, 0).astype(np.uint8)

    def bitwise_or(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        self.bitwise_or_calls += 1
        return np.bitwise_or(first, second)


class HSVCalibrationTests(unittest.TestCase):
    def test_red_wraparound_combines_two_ranges(self) -> None:
        calibration = default_hsv_calibration()
        hsv_pixels = np.array([[[5, 200, 200], [175, 200, 200], [90, 200, 200]]], dtype=np.uint8)
        fake_cv2 = FakeCV2()

        mask = build_hsv_mask(
            hsv_pixels,
            calibration.ranges["red"],
            cv2_module=fake_cv2,
        )

        np.testing.assert_array_equal(mask, np.array([[255, 255, 0]], dtype=np.uint8))
        self.assertEqual(fake_cv2.bitwise_or_calls, 1)

    def test_configuration_round_trip_preserves_ranges_and_status(self) -> None:
        calibration = default_hsv_calibration()
        calibration.calibrated_colors.add("blue")

        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "hsv_ranges.json"
            save_hsv_calibration(calibration, path)
            restored = load_hsv_calibration(path)

        self.assertEqual(restored.ranges, calibration.ranges)
        self.assertEqual(restored.calibrated_colors, {"blue"})

    def test_invalid_hue_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HSVRange((0, 100, 100), (180, 255, 255))

    def test_lower_bound_above_upper_bound_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HSVRange((30, 100, 100), (20, 255, 255))


if __name__ == "__main__":
    unittest.main()
