"""Phase 4 contour-based cube candidate detection.

Coordinates remain camera pixels. Pixel-to-robot mapping belongs to Phase 7.
"""

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from .config import SUPPORTED_COLORS, DetectionConfig, TargetSelectionPolicy
except ImportError:  # pragma: no cover - direct script imports
    from config import SUPPORTED_COLORS, DetectionConfig, TargetSelectionPolicy


LOGGER = logging.getLogger(__name__)
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
class Detection:
    """One accepted colored cube candidate in full-frame pixel coordinates."""

    color: str
    cx: int
    cy: int
    area: float
    bbox: tuple[int, int, int, int]
    aspect_ratio: float
    extent: float
    solidity: float
    polygon_vertices: int
    confidence: float
    contour: Any = field(repr=False, compare=False)
    robot_x: float | None = None
    robot_y: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a logging/serialization representation without contour data."""

        return {
            "color": self.color,
            "cx": self.cx,
            "cy": self.cy,
            "area": round(self.area, 2),
            "bbox": list(self.bbox),
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "confidence": round(self.confidence, 3),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "extent": round(self.extent, 3),
            "solidity": round(self.solidity, 3),
            "polygon_vertices": self.polygon_vertices,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _quality_score(
    aspect_ratio: float,
    extent: float,
    solidity: float,
    polygon_vertices: int,
    config: DetectionConfig,
) -> float:
    """Calculate a transparent geometric quality heuristic in the range 0-1."""

    aspect_span = max(
        1.0 - config.aspect_ratio_min,
        config.aspect_ratio_max - 1.0,
    )
    aspect_score = _clamp01(1.0 - abs(aspect_ratio - 1.0) / aspect_span)
    extent_score = _clamp01(
        (extent - config.min_extent) / max(1.0 - config.min_extent, 1e-9)
    )
    solidity_score = _clamp01(
        (solidity - config.min_solidity) / max(1.0 - config.min_solidity, 1e-9)
    )
    vertex_score = _clamp01(1.0 - abs(polygon_vertices - 4) / 6.0)
    return _clamp01(
        0.35 * aspect_score
        + 0.25 * extent_score
        + 0.25 * solidity_score
        + 0.15 * vertex_score
    )


def _contour_to_detection(
    contour: Any,
    color: str,
    config: DetectionConfig,
    cv2: Any,
) -> Detection | None:
    area = float(cv2.contourArea(contour))
    if not config.min_contour_area <= area <= config.max_contour_area:
        LOGGER.debug("Skipped %s contour: area %.1f outside limits", color, area)
        return None

    x, y, width, height = (int(value) for value in cv2.boundingRect(contour))
    if width <= 0 or height <= 0:
        LOGGER.debug("Skipped %s contour: empty bounding rectangle", color)
        return None
    aspect_ratio = width / height
    if not config.aspect_ratio_min <= aspect_ratio <= config.aspect_ratio_max:
        LOGGER.debug("Skipped %s contour: aspect ratio %.3f", color, aspect_ratio)
        return None

    extent = area / float(width * height)
    if extent < config.min_extent:
        LOGGER.debug("Skipped %s contour: extent %.3f", color, extent)
        return None

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0
    if solidity < config.min_solidity:
        LOGGER.debug("Skipped %s contour: solidity %.3f", color, solidity)
        return None

    perimeter = float(cv2.arcLength(contour, True))
    approximation = cv2.approxPolyDP(
        contour,
        config.polygon_epsilon_ratio * perimeter,
        True,
    )
    polygon_vertices = len(approximation)
    if not config.polygon_vertices_min <= polygon_vertices <= config.polygon_vertices_max:
        LOGGER.debug("Skipped %s contour: %s polygon vertices", color, polygon_vertices)
        return None

    moments = cv2.moments(contour)
    if abs(float(moments.get("m00", 0.0))) > 1e-9:
        cx = int(round(float(moments["m10"]) / float(moments["m00"])))
        cy = int(round(float(moments["m01"]) / float(moments["m00"])))
    else:
        cx = x + width // 2
        cy = y + height // 2

    confidence = _quality_score(
        aspect_ratio,
        extent,
        solidity,
        polygon_vertices,
        config,
    )
    return Detection(
        color=color,
        cx=cx,
        cy=cy,
        area=area,
        bbox=(x, y, width, height),
        aspect_ratio=aspect_ratio,
        extent=extent,
        solidity=solidity,
        polygon_vertices=polygon_vertices,
        confidence=confidence,
        contour=contour,
    )


def detect_cubes(
    masks: dict[str, BinaryMask],
    config: DetectionConfig,
    *,
    cv2_module: Any | None = None,
) -> list[Detection]:
    """Find and filter external contours from every configured color mask."""

    cv2 = cv2_module or _load_cv2()
    missing = set(SUPPORTED_COLORS) - set(masks)
    if missing:
        raise ValueError("Missing color masks: " + ", ".join(sorted(missing)))

    detections: list[Detection] = []
    for color in SUPPORTED_COLORS:
        mask = masks[color]
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError(f"{color} mask must be a uint8 2D NumPy array")
        contour_result = cv2.findContours(
            mask.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = contour_result[-2]
        for contour in contours:
            detection = _contour_to_detection(contour, color, config, cv2)
            if detection is not None:
                detections.append(detection)

    color_order = {color: index for index, color in enumerate(SUPPORTED_COLORS)}
    detections.sort(key=lambda item: (color_order[item.color], item.cx, item.cy))
    return detections


def select_target(
    detections: list[Detection],
    config: DetectionConfig,
    *,
    policy: TargetSelectionPolicy | None = None,
    reference_xy: tuple[float, float] | None = None,
) -> Detection | None:
    """Select one target deterministically without modifying the detections."""

    if not detections:
        return None
    selected_policy = policy or config.target_selection_policy
    if selected_policy == "largest":
        return max(detections, key=lambda item: (item.area, item.confidence, -item.cx))
    if selected_policy == "leftmost":
        return min(detections, key=lambda item: (item.cx, item.cy, -item.area))
    if selected_policy == "first_valid":
        return detections[0]
    if selected_policy == "color_priority":
        priority = {color: index for index, color in enumerate(config.color_priority)}
        return min(
            detections,
            key=lambda item: (priority[item.color], -item.area, item.cx, item.cy),
        )
    if selected_policy == "nearest":
        if reference_xy is None:
            raise ValueError("nearest target selection requires reference_xy")
        ref_x, ref_y = reference_xy
        return min(
            detections,
            key=lambda item: (
                math.hypot(item.cx - ref_x, item.cy - ref_y),
                -item.area,
            ),
        )
    raise ValueError(f"Unsupported target-selection policy: {selected_policy}")

