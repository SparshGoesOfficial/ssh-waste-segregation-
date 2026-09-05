"""Live Phase 3 preview for ROI and cleaned color masks.

Keys: R selects and saves an ROI, C clears it, and Q/Esc quits. This program
does not find contours or send robot commands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .camera import Camera, CameraError
    from .config import (
        DEFAULT_HSV_CONFIG_PATH,
        DEFAULT_VISION_CONFIG_PATH,
        SUPPORTED_COLORS,
        CameraConfig,
        ImageROI,
        VisionPreprocessingConfig,
        load_hsv_calibration,
        load_vision_preprocessing_config,
        save_vision_preprocessing_config,
    )
    from .vision import _load_cv2, combine_color_masks, segment_colors
except ImportError:  # pragma: no cover - direct script execution
    from camera import Camera, CameraError
    from config import (
        DEFAULT_HSV_CONFIG_PATH,
        DEFAULT_VISION_CONFIG_PATH,
        SUPPORTED_COLORS,
        CameraConfig,
        ImageROI,
        VisionPreprocessingConfig,
        load_hsv_calibration,
        load_vision_preprocessing_config,
        save_vision_preprocessing_config,
    )
    from vision import _load_cv2, combine_color_masks, segment_colors


LOGGER = logging.getLogger(__name__)
ORIGINAL_WINDOW = "Phase 3 - Original and ROI"
MASKS_WINDOW = "Phase 3 - Clean Color Masks"
SEGMENTED_WINDOW = "Phase 3 - Combined Segmentation"
ROI_SELECTION_WINDOW = "Select reachable table ROI, then Enter"
MASK_LABEL_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


class SegmentationPreview:
    """Interactive Phase 3 test display."""

    def __init__(
        self,
        preprocessing: VisionPreprocessingConfig,
        config_path: Path,
        cv2_module: Any,
    ) -> None:
        self.preprocessing = preprocessing
        self.config_path = config_path
        self.cv2 = cv2_module

    def _select_roi(self, frame: np.ndarray) -> None:
        selection = self.cv2.selectROI(
            ROI_SELECTION_WINDOW,
            frame,
            showCrosshair=True,
            fromCenter=False,
        )
        self.cv2.destroyWindow(ROI_SELECTION_WINDOW)
        x, y, width, height = (int(value) for value in selection)
        if width == 0 or height == 0:
            LOGGER.info("ROI selection cancelled; existing ROI kept")
            return
        roi = ImageROI(x=x, y=y, width=width, height=height)
        roi.validate_for_frame(frame.shape[1], frame.shape[0])
        self.preprocessing = replace(self.preprocessing, roi=roi)
        save_vision_preprocessing_config(self.preprocessing, self.config_path)
        LOGGER.info("Saved ROI (%s, %s, %s, %s)", x, y, width, height)

    def _clear_roi(self) -> None:
        self.preprocessing = replace(self.preprocessing, roi=None)
        save_vision_preprocessing_config(self.preprocessing, self.config_path)
        LOGGER.info("Cleared ROI; processing the full camera frame")

    def _original_preview(self, frame: np.ndarray) -> np.ndarray:
        preview = frame.copy()
        roi = self.preprocessing.roi
        if roi is not None:
            self.cv2.rectangle(
                preview,
                (roi.x, roi.y),
                (roi.right - 1, roi.bottom - 1),
                (255, 255, 255),
                2,
            )
            roi_text = f"ROI: x={roi.x} y={roi.y} w={roi.width} h={roi.height}"
        else:
            roi_text = "ROI: full frame (press R to select reachable table area)"

        lines = (
            roi_text,
            "R select/save ROI | C clear ROI | Q/Esc quit",
            f"blur={self.preprocessing.gaussian_kernel_size} "
            f"open={self.preprocessing.morph_open_kernel_size} "
            f"close={self.preprocessing.morph_close_kernel_size}",
        )
        for row, line in enumerate(lines):
            self.cv2.putText(
                preview,
                line,
                (10, 25 + row * 25),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
        return preview

    def _mask_grid(self, masks: dict[str, np.ndarray]) -> np.ndarray:
        labelled_masks: list[np.ndarray] = []
        for color in SUPPORTED_COLORS:
            labelled = self.cv2.cvtColor(masks[color], self.cv2.COLOR_GRAY2BGR)
            self.cv2.putText(
                labelled,
                color.upper(),
                (10, 28),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                MASK_LABEL_COLORS[color],
                2,
                self.cv2.LINE_AA,
            )
            labelled_masks.append(labelled)
        top = np.hstack(labelled_masks[:2])
        bottom = np.hstack(labelled_masks[2:])
        return np.vstack((top, bottom))

    def run(self, camera: Camera, hsv_calibration: Any) -> None:
        LOGGER.info("Phase 3 segmentation preview started")
        while True:
            frame = camera.read()
            result = segment_colors(
                frame,
                hsv_calibration,
                self.preprocessing,
                cv2_module=self.cv2,
            )
            combined_mask = combine_color_masks(result.masks, cv2_module=self.cv2)
            segmented = self.cv2.bitwise_and(frame, frame, mask=combined_mask)

            self.cv2.imshow(ORIGINAL_WINDOW, self._original_preview(frame))
            self.cv2.imshow(MASKS_WINDOW, self._mask_grid(result.masks))
            self.cv2.imshow(SEGMENTED_WINDOW, segmented)

            key = self.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return
            if key in (ord("r"), ord("R")):
                self._select_roi(frame)
            elif key in (ord("c"), ord("C")):
                self._clear_roi()


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
    parser.add_argument("--hsv-config", type=Path, default=DEFAULT_HSV_CONFIG_PATH)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_VISION_CONFIG_PATH)
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
        hsv_calibration = load_hsv_calibration(args.hsv_config)
        missing = set(SUPPORTED_COLORS) - hsv_calibration.calibrated_colors
        if missing:
            LOGGER.warning("Colors not physically calibrated: %s", ", ".join(sorted(missing)))
        preprocessing = load_vision_preprocessing_config(args.vision_config)
        preview = SegmentationPreview(preprocessing, args.vision_config, cv2)
        with Camera(camera_config) as camera:
            preview.run(camera, hsv_calibration)
        return 0
    except (CameraError, RuntimeError, ValueError) as exc:
        LOGGER.error("SEGMENTATION TEST FAILED: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Segmentation test stopped by user")
        return 0
    finally:
        if "cv2" in locals():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())

