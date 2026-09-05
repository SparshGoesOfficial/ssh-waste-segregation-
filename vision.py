"""Phase 3 color-segmentation preprocessing.

This module deliberately stops at binary masks. Contours and object locations
belong to Phase 4.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from .config import (
        SUPPORTED_COLORS,
        HSVCalibration,
        HSVRange,
        ImageROI,
        VisionPreprocessingConfig,
    )
except ImportError:  # pragma: no cover - direct script imports
    from config import (
        SUPPORTED_COLORS,
        HSVCalibration,
        HSVRange,
        ImageROI,
        VisionPreprocessingConfig,
    )


ColorFrame = NDArray[np.uint8]
BinaryMask = NDArray[np.uint8]


def _load_cv2() -> ModuleType:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install python3-opencv on Raspberry Pi OS "
            "or install requirements.txt on a laptop."
        ) from exc


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Full-frame binary masks and the ROI used to produce them."""

    masks: dict[str, BinaryMask]
    roi: ImageROI


def build_hsv_mask(
    hsv_frame: ColorFrame,
    ranges: list[HSVRange],
    *,
    cv2_module: Any | None = None,
) -> BinaryMask:
    """Threshold one or more HSV intervals into a single binary mask.

    Red supplies two intervals because its hue wraps around 0/179. They are
    explicitly combined with ``cv2.bitwise_or``.
    """

    if not ranges:
        raise ValueError("At least one HSV range is required")
    cv2 = cv2_module or _load_cv2()
    masks = [
        cv2.inRange(
            hsv_frame,
            np.asarray(hsv_range.lower, dtype=np.uint8),
            np.asarray(hsv_range.upper, dtype=np.uint8),
        )
        for hsv_range in ranges
    ]
    combined = masks[0]
    for additional_mask in masks[1:]:
        combined = cv2.bitwise_or(combined, additional_mask)
    return combined


def _validate_bgr_frame(frame: ColorFrame) -> None:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Segmentation requires a non-empty NumPy frame")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Segmentation requires a uint8 BGR frame shaped (H, W, 3)")


def _effective_roi(frame: ColorFrame, configured_roi: ImageROI | None) -> ImageROI:
    frame_height, frame_width = frame.shape[:2]
    roi = configured_roi or ImageROI(0, 0, frame_width, frame_height)
    roi.validate_for_frame(frame_width, frame_height)
    return roi


def clean_binary_mask(
    mask: BinaryMask,
    config: VisionPreprocessingConfig,
    *,
    cv2_module: Any | None = None,
) -> BinaryMask:
    """Remove isolated noise with opening, then fill small holes with closing."""

    cv2 = cv2_module or _load_cv2()
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morph_open_kernel_size, config.morph_open_kernel_size),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morph_close_kernel_size, config.morph_close_kernel_size),
    )
    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=config.morph_iterations,
    )
    return cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=config.morph_iterations,
    )


def segment_colors(
    bgr_frame: ColorFrame,
    calibration: HSVCalibration,
    config: VisionPreprocessingConfig,
    *,
    cv2_module: Any | None = None,
) -> SegmentationResult:
    """Run the complete Phase 3 pipeline and return one mask per color.

    The camera abstraction has already normalized its output to BGR. Processing
    occurs only inside the configured ROI, but masks retain full-frame size so
    future centroids remain in original camera pixel coordinates.
    """

    _validate_bgr_frame(bgr_frame)
    cv2 = cv2_module or _load_cv2()
    roi = _effective_roi(bgr_frame, config.roi)
    roi_bgr = bgr_frame[roi.y : roi.bottom, roi.x : roi.right]

    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    blurred_hsv = cv2.GaussianBlur(
        hsv_roi,
        (config.gaussian_kernel_size, config.gaussian_kernel_size),
        0,
    )

    frame_height, frame_width = bgr_frame.shape[:2]
    full_frame_masks: dict[str, BinaryMask] = {}
    for color in SUPPORTED_COLORS:
        raw_roi_mask = build_hsv_mask(
            blurred_hsv,
            calibration.ranges[color],
            cv2_module=cv2,
        )
        clean_roi_mask = clean_binary_mask(
            raw_roi_mask,
            config,
            cv2_module=cv2,
        )
        full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        full_mask[roi.y : roi.bottom, roi.x : roi.right] = clean_roi_mask
        full_frame_masks[color] = full_mask

    return SegmentationResult(masks=full_frame_masks, roi=roi)


def combine_color_masks(
    masks: dict[str, BinaryMask],
    *,
    cv2_module: Any | None = None,
) -> BinaryMask:
    """Combine all supported color masks for a Phase 3 debug preview."""

    cv2 = cv2_module or _load_cv2()
    missing = set(SUPPORTED_COLORS) - set(masks)
    if missing:
        raise ValueError("Missing color masks: " + ", ".join(sorted(missing)))
    combined = masks[SUPPORTED_COLORS[0]].copy()
    for color in SUPPORTED_COLORS[1:]:
        combined = cv2.bitwise_or(combined, masks[color])
    return combined

