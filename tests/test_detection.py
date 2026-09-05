"""Synthetic, hardware-independent tests for Phase 4 detection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import DetectionConfig, load_detection_config, save_detection_config
from detection import Detection, detect_cubes, select_target
from detection_test import map_detections_to_floor, parse_args, prepare_runtime_args
from workspace_calibration import estimate_planar_homography


def rectangle(x: int, y: int, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + width, y]],
            [[x + width, y + height]],
            [[x, y + height]],
        ],
        dtype=np.int32,
    )


class FakeCV2:
    RETR_EXTERNAL = 1
    CHAIN_APPROX_SIMPLE = 2

    def __init__(self, contours_by_call: list[list[np.ndarray]]) -> None:
        self.contours_by_call = list(contours_by_call)
        self.find_modes: list[tuple[int, int]] = []

    def findContours(
        self,
        _mask: np.ndarray,
        retrieval_mode: int,
        approximation_mode: int,
    ) -> tuple[list[np.ndarray], None]:
        self.find_modes.append((retrieval_mode, approximation_mode))
        return self.contours_by_call.pop(0), None

    @staticmethod
    def contourArea(contour: np.ndarray) -> float:
        points = contour.reshape(-1, 2).astype(float)
        x_values, y_values = points[:, 0], points[:, 1]
        return float(
            abs(
                np.dot(x_values, np.roll(y_values, -1))
                - np.dot(y_values, np.roll(x_values, -1))
            )
            / 2.0
        )

    @staticmethod
    def boundingRect(contour: np.ndarray) -> tuple[int, int, int, int]:
        points = contour.reshape(-1, 2)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)

    @staticmethod
    def convexHull(contour: np.ndarray) -> np.ndarray:
        return contour

    @staticmethod
    def arcLength(contour: np.ndarray, _closed: bool) -> float:
        points = contour.reshape(-1, 2).astype(float)
        differences = points - np.roll(points, -1, axis=0)
        return float(np.sqrt((differences**2).sum(axis=1)).sum())

    @staticmethod
    def approxPolyDP(contour: np.ndarray, _epsilon: float, _closed: bool) -> np.ndarray:
        return contour

    @classmethod
    def moments(cls, contour: np.ndarray) -> dict[str, float]:
        area = cls.contourArea(contour)
        center = contour.reshape(-1, 2).mean(axis=0)
        return {"m00": area, "m10": area * center[0], "m01": area * center[1]}


def make_detection(color: str, cx: int, area: float) -> Detection:
    return Detection(
        color=color,
        cx=cx,
        cy=100,
        area=area,
        bbox=(cx - 10, 90, 20, 20),
        aspect_ratio=1.0,
        extent=0.9,
        solidity=1.0,
        polygon_vertices=4,
        confidence=0.9,
        contour=rectangle(cx - 10, 90, 20, 20),
    )


class DetectionTests(unittest.TestCase):
    def test_headless_cli_defaults_to_finite_sixty_frame_run(self) -> None:
        args = prepare_runtime_args(parse_args(["--headless"]))
        self.assertTrue(args.headless)
        self.assertEqual(args.frames, 60)
        self.assertEqual(args.fourcc, "MJPG")
        self.assertFalse(args.pixel_only)

    def test_pixel_only_cli_explicitly_disables_coordinate_mapping(self) -> None:
        args = parse_args(["--pixel-only"])
        self.assertTrue(args.pixel_only)

    def test_detections_receive_floor_coordinates_without_mutating_inputs(self) -> None:
        pixels = np.array(
            [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            dtype=np.float64,
        )
        floor_mm = pixels * 2.5
        workspace = estimate_planar_homography(pixels, floor_mm)
        original = make_detection("blue", cx=40, area=600)

        mapped = map_detections_to_floor([original], workspace)

        self.assertIsNone(original.robot_x)
        self.assertIsNone(original.robot_y)
        self.assertAlmostEqual(mapped[0].robot_x or 0.0, 100.0, places=8)
        self.assertAlmostEqual(mapped[0].robot_y or 0.0, 250.0, places=8)

    def test_negative_frame_count_is_rejected(self) -> None:
        args = parse_args(["--headless", "--frames", "-1"])
        with self.assertRaises(ValueError):
            prepare_runtime_args(args)

    def test_square_is_detected_and_noise_and_long_region_are_rejected(self) -> None:
        square = rectangle(10, 20, 30, 30)
        noise = rectangle(1, 1, 3, 3)
        long_region = rectangle(50, 20, 80, 10)
        fake_cv2 = FakeCV2([[square, noise, long_region], [], [], []])
        masks = {
            color: np.zeros((200, 200), dtype=np.uint8)
            for color in ("red", "green", "blue", "yellow")
        }

        detections = detect_cubes(
            masks,
            DetectionConfig(min_contour_area=100),
            cv2_module=fake_cv2,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].color, "red")
        self.assertEqual((detections[0].cx, detections[0].cy), (25, 35))
        self.assertAlmostEqual(detections[0].area, 900.0)
        self.assertTrue(all(mode == (1, 2) for mode in fake_cv2.find_modes))

    def test_target_selection_policies_are_deterministic(self) -> None:
        red = make_detection("red", cx=200, area=600)
        blue = make_detection("blue", cx=80, area=900)
        green = make_detection("green", cx=120, area=700)
        detections = [red, blue, green]
        config = DetectionConfig(color_priority=("green", "blue", "red", "yellow"))

        self.assertIs(select_target(detections, config, policy="largest"), blue)
        self.assertIs(select_target(detections, config, policy="leftmost"), blue)
        self.assertIs(select_target(detections, config, policy="first_valid"), red)
        self.assertIs(select_target(detections, config, policy="color_priority"), green)
        self.assertIs(
            select_target(
                detections,
                config,
                policy="nearest",
                reference_xy=(190, 100),
            ),
            red,
        )

    def test_no_detection_returns_no_target(self) -> None:
        self.assertIsNone(select_target([], DetectionConfig()))

    def test_nearest_requires_reference_point(self) -> None:
        with self.assertRaises(ValueError):
            select_target(
                [make_detection("red", 20, 500)],
                DetectionConfig(),
                policy="nearest",
            )

    def test_detection_configuration_round_trip(self) -> None:
        expected = DetectionConfig(
            min_contour_area=250,
            aspect_ratio_min=0.6,
            aspect_ratio_max=1.6,
            target_selection_policy="leftmost",
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "detection_settings.json"
            save_detection_config(expected, path)
            actual = load_detection_config(path)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
