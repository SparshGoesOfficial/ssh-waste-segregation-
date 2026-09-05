"""Synthetic end-to-end centroid-to-IK dry-run acceptance test."""

from __future__ import annotations

import json
import sys

import numpy as np

from detection import Detection
from pick_planning import PickGeometry, PickPlanningError, plan_detection_pick
from workspace_calibration import estimate_planar_homography


def main() -> int:
    pixels = np.array(
        [[0.0, 0.0], [1279.0, 0.0], [1279.0, 719.0], [0.0, 719.0]],
        dtype=np.float64,
    )
    synthetic_table = pixels.copy()
    workspace = estimate_planar_homography(pixels, synthetic_table)
    detection = Detection(
        color="green",
        cx=569,
        cy=219,
        area=2500.0,
        bbox=(544, 194, 50, 50),
        aspect_ratio=1.0,
        extent=1.0,
        solidity=1.0,
        polygon_vertices=4,
        confidence=1.0,
        contour=None,
    )
    geometry = PickGeometry(
        frame4_pick_z_mm=113.734,
        approach_clearance_mm=30.0,
        tool_pitch_deg=-15.0,
        source_status="synthetic_test",
    )
    plan = plan_detection_pick(detection, workspace, geometry)
    print("SYNTHETIC DRY RUN: no physical calibration and no hardware commands")
    print(json.dumps(plan.to_dict(), indent=2))
    if plan.provisional_common_branches != ("elbow_positive", "elbow_negative"):
        raise PickPlanningError("Expected both provisional branches to remain available")
    if any(
        solution.position_error_mm > 1e-6
        for solution in (*plan.approach_solutions, *plan.pick_solutions)
    ):
        raise PickPlanningError("IK/FK round-trip error exceeded tolerance")
    print("PASS: centroid -> table XY -> approach/pick IK integration is consistent")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PickPlanningError as exc:
        print(f"PICK PLANNING TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
