"""Hardware-independent tests for the Phase 3 segmentation pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import (
    ImageROI,
    VisionPreprocessingConfig,
    default_hsv_calibration,
    load_vision_preprocessing_config,
    save_vision_preprocessing_config,
)
from vision import segment_colors


class FakeCV2:
    COLOR_BGR2HSV = 1
    MORPH_ELLIPSE = 2
    MORPH_OPEN = 3
    MORPH_CLOSE = 4

    def __init__(self) -> None:
        self.gaussian_calls: list[tuple[int, int]] = []
        self.morphology_calls: list[tuple[int, tuple[int, int], int]] = []

    @staticmethod
    def cvtColor(image: np.ndarray, _conversion: int) -> np.ndarray:
        # Test pixels are already expressed as HSV values.
        return image.copy()

    def GaussianBlur(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int],
        _sigma: int,
    ) -> np.ndarray:
        self.gaussian_calls.append(kernel_size)
        return image.copy()

    @staticmethod
    def inRange(image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        selected = np.all((image >= lower) & (image <= upper), axis=2)
        return np.where(selected, 255, 0).astype(np.uint8)

    @staticmethod
    def bitwise_or(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.bitwise_or(first, second)

    @staticmethod
    def getStructuringElement(_shape: int, size: tuple[int, int]) -> np.ndarray:
        return np.ones(size, dtype=np.uint8)

    def morphologyEx(
        self,
        image: np.ndarray,
        operation: int,
        kernel: np.ndarray,
        *,
        iterations: int,
    ) -> np.ndarray:
        self.morphology_calls.append((operation, kernel.shape, iterations))
        return image.copy()


class VisionPipelineTests(unittest.TestCase):
    def test_roi_keeps_full_frame_coordinates_and_blacks_outside(self) -> None:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[0, 0] = (5, 200, 200)  # Red, but outside the ROI.
        frame[1, 2] = (5, 200, 200)  # Red and inside the ROI.
        config = VisionPreprocessingConfig(
            roi=ImageROI(x=2, y=1, width=2, height=2),
            gaussian_kernel_size=3,
            morph_open_kernel_size=3,
            morph_close_kernel_size=5,
        )
        fake_cv2 = FakeCV2()

        result = segment_colors(
            frame,
            default_hsv_calibration(),
            config,
            cv2_module=fake_cv2,
        )

        self.assertEqual(result.masks["red"].shape, (4, 6))
        self.assertEqual(result.masks["red"][0, 0], 0)
        self.assertEqual(result.masks["red"][1, 2], 255)
        self.assertEqual(fake_cv2.gaussian_calls, [(3, 3)])

    def test_open_then_close_runs_for_every_color(self) -> None:
        fake_cv2 = FakeCV2()
        config = VisionPreprocessingConfig(
            gaussian_kernel_size=5,
            morph_open_kernel_size=3,
            morph_close_kernel_size=7,
            morph_iterations=2,
        )

        segment_colors(
            np.zeros((3, 3, 3), dtype=np.uint8),
            default_hsv_calibration(),
            config,
            cv2_module=fake_cv2,
        )

        self.assertEqual(len(fake_cv2.morphology_calls), 8)
        for index in range(0, 8, 2):
            self.assertEqual(fake_cv2.morphology_calls[index], (FakeCV2.MORPH_OPEN, (3, 3), 2))
            self.assertEqual(fake_cv2.morphology_calls[index + 1], (FakeCV2.MORPH_CLOSE, (7, 7), 2))

    def test_roi_outside_frame_is_rejected(self) -> None:
        config = VisionPreprocessingConfig(roi=ImageROI(5, 5, 10, 10))
        with self.assertRaises(ValueError):
            segment_colors(
                np.zeros((8, 8, 3), dtype=np.uint8),
                default_hsv_calibration(),
                config,
                cv2_module=FakeCV2(),
            )

    def test_vision_configuration_round_trip(self) -> None:
        expected = VisionPreprocessingConfig(
            roi=ImageROI(10, 20, 300, 200),
            gaussian_kernel_size=7,
            morph_open_kernel_size=3,
            morph_close_kernel_size=9,
            morph_iterations=2,
        )
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "vision_settings.json"
            save_vision_preprocessing_config(expected, path)
            actual = load_vision_preprocessing_config(path)
        self.assertEqual(actual, expected)

    def test_even_kernel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VisionPreprocessingConfig(gaussian_kernel_size=4)


if __name__ == "__main__":
    unittest.main()

