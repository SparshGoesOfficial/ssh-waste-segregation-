"""Create and visualize a planar workspace calibration from clicked grid points.

Each ``--point`` supplies one image pixel and its corresponding floor position:
``--point U V X_MM Y_MM``.  Four or more points are required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from workspace_calibration import (
    DEFAULT_WORKSPACE_CALIBRATION_PATH,
    estimate_planar_homography,
    save_workspace_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--point",
        action="append",
        nargs=4,
        type=float,
        metavar=("U", "V", "X_MM", "Y_MM"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_WORKSPACE_CALIBRATION_PATH,
    )
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--camera-index", type=int, default=2)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.point) < 4:
        raise SystemExit("At least four --point values are required")

    pixel_points = [(point[0], point[1]) for point in args.point]
    floor_points = [(point[2], point[3]) for point in args.point]
    calibration = estimate_planar_homography(pixel_points, floor_points)
    save_workspace_calibration(
        calibration,
        pixel_points,
        floor_points,
        args.output,
        camera_metadata={
            "backend": "opencv",
            "camera_index": args.camera_index,
            "resolution": [1280, 720],
            "rotation_degrees": args.rotation,
            "source_image": str(args.image),
        },
        coordinate_frame={
            "name": "temporary_floor_frame",
            "units": "mm",
            "origin": "bottom-left tape centreline intersection",
            "x_axis": "bottom-left to bottom-right",
            "y_axis": "bottom-left to top-left",
            "robot_base_alignment": "not_calibrated",
        },
    )

    if args.overlay:
        _save_overlay(args.image, args.overlay, pixel_points, floor_points)

    print(f"Saved calibration: {args.output}")
    print(f"Points: {calibration.point_count}")
    print(f"RMS fit error: {calibration.rms_error_mm:.3f} mm")
    print(f"Maximum fit error: {calibration.max_error_mm:.3f} mm")
    print("WARNING: provisional until the tape-centre measurements are verified")
    return 0


def _save_overlay(
    image_path: Path,
    output_path: Path,
    pixel_points: list[tuple[float, float]],
    floor_points: list[tuple[float, float]],
) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Overlay output requires OpenCV") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    by_floor = {
        (int(round(x_mm)), int(round(y_mm))): (int(round(u)), int(round(v)))
        for (u, v), (x_mm, y_mm) in zip(pixel_points, floor_points)
    }
    corners = [
        by_floor.get((0, 0)),
        by_floor.get((500, 0)),
        by_floor.get((500, 500)),
        by_floor.get((0, 500)),
    ]
    if all(point is not None for point in corners):
        polygon = np.asarray(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], True, (0, 255, 0), 3)

    for (u, v), (x_mm, y_mm) in zip(pixel_points, floor_points):
        center = (int(round(u)), int(round(v)))
        cv2.circle(image, center, 7, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"({x_mm:.0f},{y_mm:.0f}) mm",
            (center[0] + 10, max(20, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise SystemExit(f"Could not write overlay: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
