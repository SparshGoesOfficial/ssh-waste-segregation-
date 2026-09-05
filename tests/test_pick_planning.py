"""Tests for the guarded detection-to-IK dry-run integration."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from detection import Detection
from pick_planning import (
    PickGeometry,
    PickGeometryNotReadyError,
    PickPlanningError,
    load_pick_geometry,
    map_detection_to_table,
    plan_detection_pick,
    plan_detection_pick_from_files,
)
from workspace_calibration import estimate_planar_homography


def make_detection(cx: int = 569, cy: int = 219) -> Detection:
    return Detection(
        color="green",
        cx=cx,
        cy=cy,
        area=2500.0,
        bbox=(cx - 25, cy - 25, 50, 50),
        aspect_ratio=1.0,
        extent=1.0,
        solidity=1.0,
        polygon_vertices=4,
        confidence=1.0,
        contour=None,
    )


class PickPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        pixels = np.array(
            [[0.0, 0.0], [1279.0, 0.0], [1279.0, 719.0], [0.0, 719.0]],
            dtype=np.float64,
        )
        self.workspace = estimate_planar_homography(pixels, pixels.copy())
        self.geometry = PickGeometry(
            frame4_pick_z_mm=113.734,
            approach_clearance_mm=30.0,
            tool_pitch_deg=-15.0,
            source_status="synthetic_test",
        )

    def test_detection_is_mapped_without_mutating_original(self) -> None:
        original = make_detection()
        mapped = map_detection_to_table(original, self.workspace)
        self.assertIsNone(original.robot_x)
        self.assertIsNone(original.robot_y)
        self.assertAlmostEqual(mapped.robot_x or 0.0, 569.0, places=9)
        self.assertAlmostEqual(mapped.robot_y or 0.0, 219.0, places=9)

    def test_plan_connects_centroid_mapping_approach_and_both_ik_branches(self) -> None:
        plan = plan_detection_pick(make_detection(), self.workspace, self.geometry)
        np.testing.assert_allclose(plan.table_xy_mm, (569.0, 219.0), atol=1e-9)
        np.testing.assert_allclose(plan.pick_position_mm, (569.0, 219.0, 113.734))
        np.testing.assert_allclose(plan.approach_position_mm, (569.0, 219.0, 143.734))
        self.assertEqual(len(plan.pick_solutions), 2)
        self.assertEqual(len(plan.approach_solutions), 2)
        self.assertEqual(
            plan.provisional_common_branches,
            ("elbow_positive", "elbow_negative"),
        )
        self.assertTrue(plan.to_dict()["dry_run_only"])

    def test_unreachable_mapped_target_is_rejected(self) -> None:
        with self.assertRaises(PickPlanningError):
            plan_detection_pick(make_detection(1200, 650), self.workspace, self.geometry)

    def test_pick_geometry_requires_positive_clearance(self) -> None:
        with self.assertRaises(PickPlanningError):
            PickGeometry(100.0, 0.0, -90.0, "synthetic_test")

    def test_placeholder_geometry_is_rejected(self) -> None:
        path = Path(__file__).resolve().parents[1] / "calibration" / "pick_geometry.json"
        with self.assertRaises(PickGeometryNotReadyError):
            load_pick_geometry(path)

    def test_file_entry_point_rejects_uncalibrated_workspace(self) -> None:
        with self.assertRaises(PickPlanningError):
            plan_detection_pick_from_files(make_detection())


if __name__ == "__main__":
    unittest.main()
