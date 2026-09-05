"""Synthetic acceptance test for fixed-camera pixel/table calibration."""

from __future__ import annotations

import sys

import numpy as np

from workspace_calibration import (
    WorkspaceCalibrationError,
    estimate_planar_homography,
    transform_points,
)


def main() -> int:
    pixel_points = np.array(
        [
            [80.0, 60.0],
            [640.0, 55.0],
            [1200.0, 70.0],
            [75.0, 360.0],
            [640.0, 350.0],
            [1210.0, 365.0],
            [90.0, 670.0],
            [635.0, 665.0],
            [1190.0, 660.0],
        ],
        dtype=np.float64,
    )
    known_table_from_pixel = np.array(
        [
            [0.48, 0.025, -305.0],
            [-0.018, 0.51, -165.0],
            [0.00008, -0.00005, 1.0],
        ],
        dtype=np.float64,
    )
    table_points = transform_points(pixel_points, known_table_from_pixel)
    calibration = estimate_planar_homography(pixel_points, table_points)
    query_pixels = np.array([[320.0, 200.0], [960.0, 520.0]], dtype=np.float64)
    expected_table = transform_points(query_pixels, known_table_from_pixel)
    actual_table = transform_points(query_pixels, calibration.table_from_pixel)
    max_query_error = float(np.max(np.linalg.norm(actual_table - expected_table, axis=1)))

    print("SYNTHETIC ONLY: no physical workspace values were written")
    print("Estimated pixel-to-table homography:")
    print(np.array2string(calibration.table_from_pixel, precision=9, suppress_small=True))
    print(f"Calibration RMS error (mm): {calibration.rms_error_mm:.12f}")
    print(f"Calibration max error (mm): {calibration.max_error_mm:.12f}")
    print(f"Unseen query max error (mm): {max_query_error:.12f}")
    if calibration.max_error_mm > 1e-8 or max_query_error > 1e-8:
        raise WorkspaceCalibrationError("Synthetic calibration exceeded tolerance")
    print("PASS: pixel/table homography and inverse mapping are numerically consistent")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except WorkspaceCalibrationError as exc:
        print(f"WORKSPACE CALIBRATION TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
