"""Tests for measured, direction-dependent servo correction."""

from __future__ import annotations

import unittest

from servo_calibration import ServoCalibrationError, load_directional_calibration


class ServoCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.j2 = load_directional_calibration("J2")

    def test_measured_points_round_trip_exactly(self) -> None:
        for direction, points in (
            ("raw_decreasing", ((60.0, 53.0), (70.0, 67.0), (80.0, 78.0), (90.0, 90.0))),
            ("raw_increasing", ((70.0, 60.0), (80.0, 74.0), (90.0, 85.0))),
        ):
            for raw_deg, physical_deg in points:
                self.assertAlmostEqual(
                    self.j2.raw_to_physical(raw_deg, direction), physical_deg
                )
                self.assertAlmostEqual(
                    self.j2.physical_to_raw(physical_deg, direction), raw_deg
                )

    def test_direction_specific_commands_account_for_hysteresis(self) -> None:
        self.assertAlmostEqual(
            self.j2.physical_to_raw(70.0, "raw_decreasing"),
            72.7272727273,
        )
        self.assertAlmostEqual(
            self.j2.physical_to_raw(70.0, "raw_increasing"),
            77.1428571429,
        )

    def test_dh_zero_uses_physical_ninety_degree_reference(self) -> None:
        self.assertAlmostEqual(self.j2.dh_to_raw(0.0, "raw_decreasing"), 90.0)
        with self.assertRaises(ServoCalibrationError):
            self.j2.dh_to_raw(0.0, "raw_increasing")

    def test_unmeasured_extrapolation_is_blocked_by_default(self) -> None:
        with self.assertRaises(ServoCalibrationError):
            self.j2.physical_to_raw(90.0, "raw_increasing")

    def test_descending_approach_uses_measured_curve_and_preload(self) -> None:
        self.assertEqual(self.j2.descending_approach_commands(70.0), (78, 73))
        self.assertEqual(self.j2.descending_approach_commands(90.0), (95, 90))
        with self.assertRaises(ServoCalibrationError):
            self.j2.descending_approach_commands(50.0)


if __name__ == "__main__":
    unittest.main()
