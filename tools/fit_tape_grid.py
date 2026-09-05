"""Fit blue-tape centre-lines in a captured 1280x720 calibration frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fit_line(
    mask: np.ndarray,
    rois: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    point_groups: list[np.ndarray] = []
    for x1, y1, x2, y2 in rois:
        ys, xs = np.where(mask[y1:y2, x1:x2] > 0)
        point_groups.append(np.column_stack((xs + x1, ys + y1)))
    points = np.vstack(point_groups).astype(np.float32)
    if len(points) < 100:
        raise ValueError(f"Not enough tape pixels in ROIs {rois}: {len(points)}")
    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L1,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    # ax + by + c = 0
    return np.array([-vy, vx, (vy * x0) - (vx * y0)], dtype=np.float64)


def intersection(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    point = np.cross(first, second)
    if abs(point[2]) < 1e-9:
        raise ValueError("Lines are parallel")
    return float(point[0] / point[2]), float(point[1] / point[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None or image.shape[:2] != (720, 1280):
        raise SystemExit("Expected a readable 1280x720 image")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([80, 28, 25], dtype=np.uint8),
        np.array([130, 255, 250], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )

    height, width = mask.shape
    rois = {
        # Split vertical ROIs around the horizontal strips so their many
        # crossing pixels cannot pull the fitted vertical centre-lines.
        "left": (
            (int(0.20 * width), int(0.13 * height), int(0.33 * width), int(0.42 * height)),
            (int(0.20 * width), int(0.58 * height), int(0.33 * width), int(0.89 * height)),
        ),
        "right": (
            (int(0.75 * width), int(0.12 * height), int(0.86 * width), int(0.42 * height)),
            (int(0.75 * width), int(0.58 * height), int(0.86 * width), int(0.92 * height)),
        ),
        "top": ((int(0.30 * width), 0, int(0.76 * width), int(0.14 * height)),),
        "middle": ((int(0.30 * width), int(0.42 * height), int(0.76 * width), int(0.58 * height)),),
        "bottom": ((int(0.30 * width), int(0.88 * height), int(0.76 * width), height),),
    }
    lines = {name: fit_line(mask, roi) for name, roi in rois.items()}
    points = {
        "TL": intersection(lines["left"], lines["top"]),
        "TR": intersection(lines["right"], lines["top"]),
        "ML": intersection(lines["left"], lines["middle"]),
        "MR": intersection(lines["right"], lines["middle"]),
        "BL": intersection(lines["left"], lines["bottom"]),
        "BR": intersection(lines["right"], lines["bottom"]),
    }
    for name, point in points.items():
        print(f"{name}: ({point[0]:.3f}, {point[1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
