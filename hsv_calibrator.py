"""Phase 2 live HSV calibration utility.

This utility performs no cube detection and sends no robot commands.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:  # Support both package-style and direct-script execution.
    from .camera import Camera, CameraError
    from .config import (
        DEFAULT_HSV_CONFIG_PATH,
        SUPPORTED_COLORS,
        CameraConfig,
        HSVCalibration,
        HSVRange,
        load_hsv_calibration,
        save_hsv_calibration,
    )
    from .vision import build_hsv_mask
except ImportError:  # pragma: no cover - used by ``python hsv_calibrator.py``
    from camera import Camera, CameraError
    from config import (
        DEFAULT_HSV_CONFIG_PATH,
        SUPPORTED_COLORS,
        CameraConfig,
        HSVCalibration,
        HSVRange,
        load_hsv_calibration,
        save_hsv_calibration,
    )
    from vision import build_hsv_mask


LOGGER = logging.getLogger(__name__)
TRACKBAR_WINDOW = "HSV Controls"
ORIGINAL_WINDOW = "1 - Original"
MASK_WINDOW = "2 - Mask"
MASKED_WINDOW = "3 - Masked Result"
TRACKBARS = (
    ("H lower", 179),
    ("H upper", 179),
    ("S lower", 255),
    ("S upper", 255),
    ("V lower", 255),
    ("V upper", 255),
)
ColorFrame = NDArray[np.uint8]


def _load_cv2() -> ModuleType:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install python3-opencv on Raspberry Pi OS "
            "or install requirements.txt on a laptop."
        ) from exc


class LiveHSVCalibrator:
    """Own the trackbar state and interactive calibration loop."""

    def __init__(
        self,
        calibration: HSVCalibration,
        config_path: Path,
        start_color: str,
        cv2_module: Any,
    ) -> None:
        self.calibration = calibration
        self.config_path = config_path
        self.color = start_color
        self.range_index = 0
        self.cv2 = cv2_module
        self._invalid_bounds_message: str | None = None

    @staticmethod
    def _no_op(_value: int) -> None:
        """Trackbar callback; values are read synchronously each frame."""

    def create_windows(self) -> None:
        """Create the three required views and six HSV controls."""

        self.cv2.namedWindow(TRACKBAR_WINDOW, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(TRACKBAR_WINDOW, 620, 240)
        for name, maximum in TRACKBARS:
            self.cv2.createTrackbar(name, TRACKBAR_WINDOW, 0, maximum, self._no_op)
        self.cv2.namedWindow(ORIGINAL_WINDOW, self.cv2.WINDOW_NORMAL)
        self.cv2.namedWindow(MASK_WINDOW, self.cv2.WINDOW_NORMAL)
        self.cv2.namedWindow(MASKED_WINDOW, self.cv2.WINDOW_NORMAL)
        self._write_range_to_trackbars()

    def _write_range_to_trackbars(self) -> None:
        hsv_range = self.calibration.ranges[self.color][self.range_index]
        values = (
            hsv_range.lower[0],
            hsv_range.upper[0],
            hsv_range.lower[1],
            hsv_range.upper[1],
            hsv_range.lower[2],
            hsv_range.upper[2],
        )
        for (name, _maximum), value in zip(TRACKBARS, values):
            self.cv2.setTrackbarPos(name, TRACKBAR_WINDOW, value)

    def _read_range_from_trackbars(self) -> HSVRange | None:
        values = {
            name: self.cv2.getTrackbarPos(name, TRACKBAR_WINDOW)
            for name, _maximum in TRACKBARS
        }
        try:
            hsv_range = HSVRange(
                (values["H lower"], values["S lower"], values["V lower"]),
                (values["H upper"], values["S upper"], values["V upper"]),
            )
        except ValueError as exc:
            self._invalid_bounds_message = str(exc)
            return None
        self._invalid_bounds_message = None
        return hsv_range

    def _select_color(self, color: str) -> None:
        self.color = color
        self.range_index = 0
        self._invalid_bounds_message = None
        self._write_range_to_trackbars()
        LOGGER.info("Selected %s calibration", color)

    def _toggle_red_range(self) -> None:
        if self.color != "red":
            LOGGER.info("Range toggle applies only to red")
            return
        self.range_index = 1 - self.range_index
        self._invalid_bounds_message = None
        self._write_range_to_trackbars()
        LOGGER.info("Editing red range %s of 2", self.range_index + 1)

    def _save_current_color(self) -> None:
        if self._invalid_bounds_message:
            LOGGER.error("Cannot save invalid bounds: %s", self._invalid_bounds_message)
            return
        self.calibration.calibrated_colors.add(self.color)
        save_hsv_calibration(self.calibration, self.config_path)
        LOGGER.info("Saved %s HSV calibration to %s", self.color, self.config_path)

    def _draw_status(self, frame: ColorFrame) -> ColorFrame:
        preview = frame.copy()
        range_label = (
            f"range {self.range_index + 1}/2" if self.color == "red" else "range 1/1"
        )
        lines = [
            f"COLOR: {self.color.upper()} | {range_label}",
            "1 Red  2 Green  3 Blue  4 Yellow  |  T red range",
            "S save current color  |  Q/Esc quit",
        ]
        if self._invalid_bounds_message:
            lines.append(f"INVALID: {self._invalid_bounds_message}")
        for row, line in enumerate(lines):
            color = (0, 0, 255) if line.startswith("INVALID") else (255, 255, 255)
            self.cv2.putText(
                preview,
                line,
                (10, 25 + row * 25),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                self.cv2.LINE_AA,
            )
        return preview

    def run(self, camera: Camera) -> None:
        """Continuously update masks until the user quits."""

        self.create_windows()
        LOGGER.info("HSV calibrator started; initial color=%s", self.color)
        while True:
            frame = camera.read()
            current_range = self._read_range_from_trackbars()
            if current_range is not None:
                self.calibration.ranges[self.color][self.range_index] = current_range

            hsv_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
            mask = build_hsv_mask(
                hsv_frame,
                self.calibration.ranges[self.color],
                cv2_module=self.cv2,
            )
            masked_result = self.cv2.bitwise_and(frame, frame, mask=mask)

            controls_panel = np.zeros((120, 620, 3), dtype=np.uint8)
            self.cv2.putText(
                controls_panel,
                f"Editing {self.color.upper()} - press S to save",
                (10, 35),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
            self.cv2.imshow(TRACKBAR_WINDOW, controls_panel)
            self.cv2.imshow(ORIGINAL_WINDOW, self._draw_status(frame))
            self.cv2.imshow(MASK_WINDOW, mask)
            self.cv2.imshow(MASKED_WINDOW, masked_result)

            key = self.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return
            if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                self._select_color(SUPPORTED_COLORS[key - ord("1")])
            elif key == ord("t"):
                self._toggle_red_range()
            elif key == ord("s"):
                self._save_current_color()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "picamera2", "opencv"), default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--read-attempts", type=int, default=3)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--color", choices=SUPPORTED_COLORS, default="red")
    parser.add_argument("--config", type=Path, default=DEFAULT_HSV_CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    camera_config = CameraConfig(
        backend=args.backend,
        width=args.width,
        height=args.height,
        camera_index=args.camera_index,
        frame_rate=args.fps,
        warmup_seconds=args.warmup_seconds,
        read_attempts=args.read_attempts,
        rotation_degrees=args.rotation,
    )

    try:
        cv2 = _load_cv2()
        calibration = load_hsv_calibration(args.config)
        calibrator = LiveHSVCalibrator(calibration, args.config, args.color, cv2)
        with Camera(camera_config) as camera:
            calibrator.run(camera)
        return 0
    except (CameraError, RuntimeError, ValueError) as exc:
        LOGGER.error("HSV CALIBRATOR FAILED: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("HSV calibration stopped by user")
        return 0
    finally:
        if "cv2" in locals():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
