"""Camera abstraction for the fixed-camera cube sorter.

Every backend returns a NumPy array in OpenCV's BGR channel order. Keeping that
contract here prevents color-order assumptions from leaking into later vision
phases.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:  # Support both ``python -m cube_sorter.main`` and ``python main.py``.
    from .config import CameraConfig
except ImportError:  # pragma: no cover - used only by direct script execution
    from config import CameraConfig


LOGGER = logging.getLogger(__name__)
BGRFrame = NDArray[np.uint8]


class CameraError(RuntimeError):
    """Base class for camera initialization and capture errors."""


class CameraUnavailableError(CameraError):
    """Raised when a requested camera backend cannot be opened."""


class FrameCaptureError(CameraError):
    """Raised when the active backend cannot supply a valid frame."""


class CameraBackend(ABC):
    """Small interface implemented by each physical camera adapter."""

    name: str

    @abstractmethod
    def start(self) -> None:
        """Allocate and start the camera."""

    @abstractmethod
    def read(self) -> BGRFrame:
        """Capture one BGR frame or raise ``FrameCaptureError``."""

    @abstractmethod
    def stop(self) -> None:
        """Release all camera resources; safe to call more than once."""


class Picamera2Backend(CameraBackend):
    """Raspberry Pi camera adapter using Picamera2."""

    name = "picamera2"

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._camera: Any | None = None

    @staticmethod
    def _load_module() -> ModuleType:
        try:
            return importlib.import_module("picamera2")
        except ImportError as exc:
            raise CameraUnavailableError(
                "Picamera2 is not installed. On Raspberry Pi OS install "
                "python3-picamera2, or choose --backend opencv."
            ) from exc

    def start(self) -> None:
        if self._camera is not None:
            return

        module = self._load_module()
        camera: Any | None = None
        camera_started = False
        try:
            camera = module.Picamera2()
            # Picamera2's RGB888 buffer is a B,G,R byte triple in memory and is
            # therefore the correct three-channel format for OpenCV.
            video_config = camera.create_video_configuration(
                main={"size": self.config.resolution, "format": "RGB888"},
                controls={"FrameRate": self.config.frame_rate},
            )
            camera.configure(video_config)
            camera.start()
            camera_started = True
            self._camera = camera
            if self.config.warmup_seconds:
                time.sleep(self.config.warmup_seconds)
            LOGGER.info(
                "Picamera2 started at requested resolution %sx%s",
                self.config.width,
                self.config.height,
            )
        except Exception as exc:
            if camera is not None:
                try:
                    if camera_started:
                        camera.stop()
                    camera.close()
                except Exception:
                    LOGGER.debug("Picamera2 cleanup failed", exc_info=True)
            raise CameraUnavailableError(f"Could not start Picamera2: {exc}") from exc

    def read(self) -> BGRFrame:
        if self._camera is None:
            raise CameraError("Picamera2 has not been started")
        try:
            frame = self._camera.capture_array("main")
        except Exception as exc:
            raise FrameCaptureError(f"Picamera2 capture failed: {exc}") from exc
        return _validate_bgr_frame(frame, self.name)

    def stop(self) -> None:
        camera, self._camera = self._camera, None
        if camera is None:
            return
        try:
            camera.stop()
        except Exception:
            LOGGER.warning("Picamera2 stop failed", exc_info=True)
        try:
            camera.close()
        except Exception:
            LOGGER.warning("Picamera2 close failed", exc_info=True)
        LOGGER.info("Picamera2 stopped")


class OpenCVBackend(CameraBackend):
    """USB-webcam adapter using ``cv2.VideoCapture``.

    Linux prefers the direct V4L2 backend.  This avoids GStreamer selecting an
    uncompressed mode that cannot sustain the C270's 1280x720 stream.  The
    requested FOURCC is applied before resolution and frame rate for the same
    reason.
    """

    name = "opencv"

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Any | None = None

    @staticmethod
    def _load_module() -> ModuleType:
        try:
            return importlib.import_module("cv2")
        except ImportError as exc:
            raise CameraUnavailableError(
                "OpenCV is not installed. Install python3-opencv on Raspberry "
                "Pi OS or install requirements.txt on a laptop."
            ) from exc

    def start(self) -> None:
        if self._capture is not None:
            return

        cv2 = self._load_module()
        api_name = "automatic"
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            capture = cv2.VideoCapture(
                self.config.camera_index,
                cv2.CAP_V4L2,
            )
            api_name = "V4L2"
            if not capture.isOpened():
                capture.release()
                LOGGER.warning(
                    "V4L2 could not open camera %s; retrying with OpenCV's "
                    "automatic backend",
                    self.config.camera_index,
                )
                capture = cv2.VideoCapture(self.config.camera_index)
                api_name = "automatic"
        else:
            capture = cv2.VideoCapture(self.config.camera_index)

        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(
                f"OpenCV could not open camera index {self.config.camera_index}"
            )

        if self.config.opencv_fourcc is not None:
            fourcc = cv2.VideoWriter_fourcc(*self.config.opencv_fourcc)
            if not capture.set(cv2.CAP_PROP_FOURCC, fourcc):
                LOGGER.warning(
                    "Camera %s did not accept requested FOURCC %s",
                    self.config.camera_index,
                    self.config.opencv_fourcc,
                )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.frame_rate)

        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        self._capture = capture
        if self.config.warmup_seconds:
            time.sleep(self.config.warmup_seconds)
        LOGGER.info(
            "OpenCV camera %s started via %s; requested=%sx%s@%.1f %s; "
            "negotiated=%sx%s@%.1f %s",
            self.config.camera_index,
            api_name,
            self.config.width,
            self.config.height,
            self.config.frame_rate,
            self.config.opencv_fourcc or "AUTO",
            actual_width,
            actual_height,
            actual_fps,
            actual_fourcc,
        )

    def read(self) -> BGRFrame:
        if self._capture is None:
            raise CameraError("OpenCV camera has not been started")

        for attempt in range(1, self.config.read_attempts + 1):
            success, frame = self._capture.read()
            if success and frame is not None and frame.size > 0:
                return _validate_bgr_frame(frame, self.name)
            LOGGER.warning(
                "OpenCV capture attempt %s/%s failed",
                attempt,
                self.config.read_attempts,
            )
        raise FrameCaptureError(
            f"Camera {self.config.camera_index} failed to return a frame after "
            f"{self.config.read_attempts} attempts"
        )

    def stop(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                LOGGER.warning("OpenCV camera release failed", exc_info=True)
            LOGGER.info("OpenCV camera stopped")


def _decode_fourcc(value: float | int) -> str:
    """Convert OpenCV's packed numeric FOURCC value to readable text."""

    code = int(value)
    text = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
    return text.rstrip("\x00") or "unknown"


def _validate_bgr_frame(frame: Any, source: str) -> BGRFrame:
    """Validate the common backend output contract.

    Input: an object returned by a camera library.
    Output: a non-empty ``uint8`` NumPy array shaped ``(height, width, 3)``.
    Why: later OpenCV stages should fail early instead of processing malformed
    or empty camera data.
    """

    if not isinstance(frame, np.ndarray):
        raise FrameCaptureError(f"{source} returned a non-NumPy frame")
    if frame.size == 0:
        raise FrameCaptureError(f"{source} returned an empty frame")
    if frame.dtype != np.uint8:
        raise FrameCaptureError(
            f"{source} returned dtype {frame.dtype}; expected uint8"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise FrameCaptureError(
            f"{source} returned shape {frame.shape}; expected (height, width, 3)"
        )
    return frame


def _rotate_bgr_frame(frame: BGRFrame, degrees: int) -> BGRFrame:
    """Rotate a BGR frame clockwise to match the physical camera mounting."""

    if degrees == 0:
        return frame
    # np.rot90 rotates counter-clockwise, so 90 degrees clockwise is k=3.
    quarter_turns = {90: 3, 180: 2, 270: 1}[degrees]
    return np.ascontiguousarray(np.rot90(frame, k=quarter_turns))


class Camera:
    """Lifecycle-safe facade that selects and normalizes a camera backend."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        backend_override: CameraBackend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend_override
        self._started = False

    @property
    def backend_name(self) -> str | None:
        """Return the active backend name, or ``None`` before startup."""

        return self._backend.name if self._backend is not None else None

    def _candidates(self) -> list[CameraBackend]:
        if self._backend is not None:
            return [self._backend]
        if self.config.backend == "picamera2":
            return [Picamera2Backend(self.config)]
        if self.config.backend == "opencv":
            return [OpenCVBackend(self.config)]
        return [Picamera2Backend(self.config), OpenCVBackend(self.config)]

    def start(self) -> None:
        """Start the requested backend, with USB fallback in ``auto`` mode."""

        if self._started:
            return

        errors: list[str] = []
        for backend in self._candidates():
            try:
                backend.start()
            except CameraError as exc:
                errors.append(f"{backend.name}: {exc}")
                LOGGER.warning("Camera backend %s unavailable: %s", backend.name, exc)
                continue
            self._backend = backend
            self._started = True
            LOGGER.info("Using %s camera backend", backend.name)
            return

        raise CameraUnavailableError(
            "No camera backend could be started. " + " | ".join(errors)
        )

    def read(self) -> BGRFrame:
        """Capture, validate, and orient one OpenCV-ready BGR frame."""

        if not self._started or self._backend is None:
            raise CameraError("Camera has not been started")
        frame = self._backend.read()
        return _rotate_bgr_frame(frame, self.config.rotation_degrees)

    def stop(self) -> None:
        """Stop the active backend; safe to call repeatedly."""

        backend, self._started = self._backend, False
        if backend is not None:
            backend.stop()

    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
