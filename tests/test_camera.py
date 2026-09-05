"""Hardware-independent tests for the Phase 1 camera contract."""

from __future__ import annotations

import unittest

import numpy as np

from camera import (
    Camera,
    CameraBackend,
    CameraError,
    FrameCaptureError,
    _validate_bgr_frame,
)
from config import CameraConfig


class FakeCameraBackend(CameraBackend):
    name = "fake"

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.started = False
        self.stop_calls = 0

    def start(self) -> None:
        self.started = True

    def read(self) -> np.ndarray:
        if not self.started:
            raise CameraError("Fake camera was not started")
        return _validate_bgr_frame(self.frame.copy(), self.name)

    def stop(self) -> None:
        self.started = False
        self.stop_calls += 1


class CameraTests(unittest.TestCase):
    def test_context_manager_captures_and_closes(self) -> None:
        expected = np.zeros((480, 640, 3), dtype=np.uint8)
        backend = FakeCameraBackend(expected)
        camera = Camera(CameraConfig(warmup_seconds=0), backend_override=backend)

        with camera:
            actual = camera.read()
            self.assertEqual(actual.shape, (480, 640, 3))
            self.assertEqual(actual.dtype, np.uint8)

        self.assertFalse(backend.started)
        self.assertEqual(backend.stop_calls, 1)

    def test_rotation_90_is_clockwise(self) -> None:
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = (10, 20, 30)
        backend = FakeCameraBackend(frame)
        camera = Camera(
            CameraConfig(warmup_seconds=0, rotation_degrees=90),
            backend_override=backend,
        )

        with camera:
            rotated = camera.read()

        self.assertEqual(rotated.shape, (3, 2, 3))
        np.testing.assert_array_equal(rotated[0, 1], (10, 20, 30))

    def test_empty_frame_is_rejected(self) -> None:
        with self.assertRaises(FrameCaptureError):
            _validate_bgr_frame(np.array([], dtype=np.uint8), "fake")

    def test_non_three_channel_frame_is_rejected(self) -> None:
        with self.assertRaises(FrameCaptureError):
            _validate_bgr_frame(np.zeros((10, 10), dtype=np.uint8), "fake")

    def test_invalid_rotation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CameraConfig(rotation_degrees=45)  # type: ignore[arg-type]

    def test_opencv_fourcc_is_normalized(self) -> None:
        config = CameraConfig(opencv_fourcc="mjpg")
        self.assertEqual(config.opencv_fourcc, "MJPG")

    def test_invalid_opencv_fourcc_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CameraConfig(opencv_fourcc="MJPEG")


if __name__ == "__main__":
    unittest.main()
