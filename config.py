"""Configuration used by the Phase 1 camera test.

Robot geometry, poses, calibration, and motion settings are intentionally not
defined yet. Those values depend on the physical arm and belong to later phases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


CameraBackendName = Literal["auto", "picamera2", "opencv"]


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Settings shared by the Picamera2 and USB-webcam backends.

    Attributes:
        backend: ``auto`` tries Picamera2 first and then OpenCV. The other
            values force a single backend, which is useful while debugging.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        camera_index: Index passed to ``cv2.VideoCapture`` for a USB webcam.
        frame_rate: Requested camera frame rate.
        warmup_seconds: Time for exposure and white balance to settle.
        read_attempts: Consecutive reads allowed before reporting a failure.
        rotation_degrees: Clockwise image rotation applied after capture.
    """

    backend: CameraBackendName = "opencv"
    width: int = 640
    height: int = 480
    camera_index: int = 0
    opencv_fourcc: str | None = "MJPG"
    frame_rate: float = 30.0
    warmup_seconds: float = 1.0
    read_attempts: int = 3
    rotation_degrees: Literal[0, 90, 180, 270] = 0

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "picamera2", "opencv"}:
            raise ValueError(f"Unsupported camera backend: {self.backend}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        if self.camera_index < 0:
            raise ValueError("Camera index cannot be negative")
        if self.opencv_fourcc is not None:
            normalized_fourcc = self.opencv_fourcc.upper()
            if len(normalized_fourcc) != 4 or not normalized_fourcc.isascii():
                raise ValueError("OpenCV FOURCC must contain exactly four ASCII characters")
            object.__setattr__(self, "opencv_fourcc", normalized_fourcc)
        if self.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        if self.warmup_seconds < 0:
            raise ValueError("Warm-up time cannot be negative")
        if self.read_attempts < 1:
            raise ValueError("At least one read attempt is required")
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the camera resolution in the ``(width, height)`` order."""

        return self.width, self.height


DEFAULT_CAMERA_CONFIG = CameraConfig()


SUPPORTED_COLORS = ("red", "green", "blue", "yellow")
DEFAULT_HSV_CONFIG_PATH = Path(__file__).resolve().parent / "calibration" / "hsv_ranges.json"
DEFAULT_VISION_CONFIG_PATH = (
    Path(__file__).resolve().parent / "calibration" / "vision_settings.json"
)
DEFAULT_DETECTION_CONFIG_PATH = (
    Path(__file__).resolve().parent / "calibration" / "detection_settings.json"
)


@dataclass(frozen=True, slots=True)
class HSVRange:
    """One inclusive OpenCV HSV interval.

    Hue uses OpenCV's 0-179 scale. Saturation and value use 0-255.
    """

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        lower = tuple(int(value) for value in self.lower)
        upper = tuple(int(value) for value in self.upper)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("HSV lower and upper bounds must each contain 3 values")

        limits = (179, 255, 255)
        channel_names = ("H", "S", "V")
        for index, (low, high, limit) in enumerate(zip(lower, upper, limits)):
            if not 0 <= low <= limit or not 0 <= high <= limit:
                raise ValueError(
                    f"{channel_names[index]} bounds must be between 0 and {limit}"
                )
            if low > high:
                raise ValueError(
                    f"{channel_names[index]} lower bound cannot exceed its upper bound"
                )

    @classmethod
    def from_dict(cls, value: object) -> "HSVRange":
        if not isinstance(value, dict):
            raise ValueError("Each HSV range must be a JSON object")
        lower = value.get("lower")
        upper = value.get("upper")
        if not isinstance(lower, list) or not isinstance(upper, list):
            raise ValueError("HSV range requires lower and upper JSON arrays")
        return cls(tuple(lower), tuple(upper))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, list[int]]:
        return {"lower": list(self.lower), "upper": list(self.upper)}


@dataclass(slots=True)
class HSVCalibration:
    """HSV intervals plus a record of colors calibrated on real hardware."""

    ranges: dict[str, list[HSVRange]]
    calibrated_colors: set[str]

    def __post_init__(self) -> None:
        if set(self.ranges) != set(SUPPORTED_COLORS):
            raise ValueError(
                "HSV configuration must contain exactly: "
                + ", ".join(SUPPORTED_COLORS)
            )
        for color, ranges in self.ranges.items():
            expected_count = 2 if color == "red" else 1
            if len(ranges) != expected_count:
                raise ValueError(
                    f"{color} must contain {expected_count} HSV range(s)"
                )
        if not self.calibrated_colors.issubset(SUPPORTED_COLORS):
            raise ValueError("Calibrated colors contains an unsupported color")


def default_hsv_calibration() -> HSVCalibration:
    """Return broad starting ranges that must be tuned under actual lighting."""

    return HSVCalibration(
        ranges={
            # Red wraps around the 0/179 end of OpenCV's hue scale.
            "red": [
                HSVRange((0, 100, 80), (10, 255, 255)),
                HSVRange((170, 100, 80), (179, 255, 255)),
            ],
            "green": [HSVRange((35, 70, 60), (85, 255, 255))],
            "blue": [HSVRange((90, 70, 60), (135, 255, 255))],
            "yellow": [HSVRange((20, 80, 80), (35, 255, 255))],
        },
        calibrated_colors=set(),
    )


def load_hsv_calibration(path: Path = DEFAULT_HSV_CONFIG_PATH) -> HSVCalibration:
    """Load and validate HSV calibration JSON, or return defaults if absent."""

    if not path.exists():
        return default_hsv_calibration()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read HSV configuration {path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("HSV configuration must use schema_version 1")
    colors = payload.get("colors")
    calibrated = payload.get("calibrated")
    if not isinstance(colors, dict) or not isinstance(calibrated, dict):
        raise ValueError("HSV configuration requires colors and calibrated objects")

    parsed_ranges: dict[str, list[HSVRange]] = {}
    calibrated_colors: set[str] = set()
    for color in SUPPORTED_COLORS:
        color_value = colors.get(color)
        if not isinstance(color_value, dict) or not isinstance(
            color_value.get("ranges"), list
        ):
            raise ValueError(f"Missing HSV ranges for {color}")
        parsed_ranges[color] = [
            HSVRange.from_dict(item) for item in color_value["ranges"]
        ]
        calibrated_value = calibrated.get(color)
        if not isinstance(calibrated_value, bool):
            raise ValueError(f"Calibration flag for {color} must be true or false")
        if calibrated_value:
            calibrated_colors.add(color)

    return HSVCalibration(parsed_ranges, calibrated_colors)


def save_hsv_calibration(
    calibration: HSVCalibration,
    path: Path = DEFAULT_HSV_CONFIG_PATH,
) -> None:
    """Validate and atomically save HSV ranges as readable JSON."""

    calibration.__post_init__()
    payload = {
        "schema_version": 1,
        "warning": "Starting values only until calibrated under actual lighting.",
        "calibrated": {
            color: color in calibration.calibrated_colors
            for color in SUPPORTED_COLORS
        },
        "colors": {
            color: {"ranges": [hsv_range.to_dict() for hsv_range in calibration.ranges[color]]}
            for color in SUPPORTED_COLORS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        raise ValueError(f"Could not save HSV configuration {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ImageROI:
    """Rectangular image region in full-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y cannot be negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def validate_for_frame(self, frame_width: int, frame_height: int) -> None:
        """Reject an ROI extending beyond the current camera image."""

        if self.right > frame_width or self.bottom > frame_height:
            raise ValueError(
                f"ROI ({self.x}, {self.y}, {self.width}, {self.height}) exceeds "
                f"frame size {frame_width}x{frame_height}"
            )

    @classmethod
    def from_dict(cls, value: object) -> "ImageROI":
        if not isinstance(value, dict):
            raise ValueError("ROI must be a JSON object or null")
        try:
            return cls(
                x=int(value["x"]),
                y=int(value["y"]),
                width=int(value["width"]),
                height=int(value["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("ROI requires integer x, y, width, and height") from exc

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class VisionPreprocessingConfig:
    """Phase 3 ROI, blur, and morphology settings."""

    roi: ImageROI | None = None
    gaussian_kernel_size: int = 5
    morph_open_kernel_size: int = 3
    morph_close_kernel_size: int = 5
    morph_iterations: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("gaussian_kernel_size", self.gaussian_kernel_size),
            ("morph_open_kernel_size", self.morph_open_kernel_size),
            ("morph_close_kernel_size", self.morph_close_kernel_size),
        ):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.morph_iterations < 1:
            raise ValueError("morph_iterations must be at least 1")


def load_vision_preprocessing_config(
    path: Path = DEFAULT_VISION_CONFIG_PATH,
) -> VisionPreprocessingConfig:
    """Load validated Phase 3 settings, or return defaults if absent."""

    if not path.exists():
        return VisionPreprocessingConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read vision configuration {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Vision configuration must use schema_version 1")

    roi_value = payload.get("roi")
    roi = None if roi_value is None else ImageROI.from_dict(roi_value)
    try:
        return VisionPreprocessingConfig(
            roi=roi,
            gaussian_kernel_size=int(payload["gaussian_kernel_size"]),
            morph_open_kernel_size=int(payload["morph_open_kernel_size"]),
            morph_close_kernel_size=int(payload["morph_close_kernel_size"]),
            morph_iterations=int(payload["morph_iterations"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid vision preprocessing settings: {exc}") from exc


def save_vision_preprocessing_config(
    config: VisionPreprocessingConfig,
    path: Path = DEFAULT_VISION_CONFIG_PATH,
) -> None:
    """Atomically save Phase 3 settings as readable JSON."""

    config.__post_init__()
    payload = {
        "schema_version": 1,
        "roi": None if config.roi is None else config.roi.to_dict(),
        "gaussian_kernel_size": config.gaussian_kernel_size,
        "morph_open_kernel_size": config.morph_open_kernel_size,
        "morph_close_kernel_size": config.morph_close_kernel_size,
        "morph_iterations": config.morph_iterations,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        raise ValueError(f"Could not save vision configuration {path}: {exc}") from exc


TargetSelectionPolicy = Literal[
    "largest",
    "nearest",
    "leftmost",
    "first_valid",
    "color_priority",
]


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Broad Phase 4 contour filters pending tuning with the real cubes."""

    min_contour_area: float = 400.0
    max_contour_area: float = 120_000.0
    aspect_ratio_min: float = 0.55
    aspect_ratio_max: float = 1.80
    min_extent: float = 0.45
    min_solidity: float = 0.75
    polygon_epsilon_ratio: float = 0.04
    polygon_vertices_min: int = 4
    polygon_vertices_max: int = 10
    target_selection_policy: TargetSelectionPolicy = "largest"
    color_priority: tuple[str, ...] = SUPPORTED_COLORS

    def __post_init__(self) -> None:
        if self.min_contour_area <= 0:
            raise ValueError("min_contour_area must be positive")
        if self.max_contour_area <= self.min_contour_area:
            raise ValueError("max_contour_area must exceed min_contour_area")
        if not 0 < self.aspect_ratio_min <= 1 <= self.aspect_ratio_max:
            raise ValueError("Aspect-ratio range must be positive and include 1.0")
        if not 0 <= self.min_extent <= 1:
            raise ValueError("min_extent must be between 0 and 1")
        if not 0 <= self.min_solidity <= 1:
            raise ValueError("min_solidity must be between 0 and 1")
        if not 0 < self.polygon_epsilon_ratio < 1:
            raise ValueError("polygon_epsilon_ratio must be between 0 and 1")
        if self.polygon_vertices_min < 3:
            raise ValueError("polygon_vertices_min must be at least 3")
        if self.polygon_vertices_max < self.polygon_vertices_min:
            raise ValueError("polygon vertex maximum cannot be below minimum")
        if self.target_selection_policy not in {
            "largest",
            "nearest",
            "leftmost",
            "first_valid",
            "color_priority",
        }:
            raise ValueError("Unsupported target-selection policy")
        if len(self.color_priority) != len(SUPPORTED_COLORS) or set(
            self.color_priority
        ) != set(SUPPORTED_COLORS):
            raise ValueError("color_priority must contain each supported color once")


def load_detection_config(
    path: Path = DEFAULT_DETECTION_CONFIG_PATH,
) -> DetectionConfig:
    """Load validated contour filters, or return broad defaults if absent."""

    if not path.exists():
        return DetectionConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read detection configuration {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Detection configuration must use schema_version 1")
    try:
        return DetectionConfig(
            min_contour_area=float(payload["min_contour_area"]),
            max_contour_area=float(payload["max_contour_area"]),
            aspect_ratio_min=float(payload["aspect_ratio_min"]),
            aspect_ratio_max=float(payload["aspect_ratio_max"]),
            min_extent=float(payload["min_extent"]),
            min_solidity=float(payload["min_solidity"]),
            polygon_epsilon_ratio=float(payload["polygon_epsilon_ratio"]),
            polygon_vertices_min=int(payload["polygon_vertices_min"]),
            polygon_vertices_max=int(payload["polygon_vertices_max"]),
            target_selection_policy=str(payload["target_selection_policy"]),  # type: ignore[arg-type]
            color_priority=tuple(str(color) for color in payload["color_priority"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid detection settings: {exc}") from exc


def save_detection_config(
    config: DetectionConfig,
    path: Path = DEFAULT_DETECTION_CONFIG_PATH,
) -> None:
    """Atomically save Phase 4 contour and target-selection settings."""

    config.__post_init__()
    payload = {
        "schema_version": 1,
        "warning": "Broad development values; tune with real cubes after the fixed camera is mounted.",
        "min_contour_area": config.min_contour_area,
        "max_contour_area": config.max_contour_area,
        "aspect_ratio_min": config.aspect_ratio_min,
        "aspect_ratio_max": config.aspect_ratio_max,
        "min_extent": config.min_extent,
        "min_solidity": config.min_solidity,
        "polygon_epsilon_ratio": config.polygon_epsilon_ratio,
        "polygon_vertices_min": config.polygon_vertices_min,
        "polygon_vertices_max": config.polygon_vertices_max,
        "target_selection_policy": config.target_selection_policy,
        "color_priority": list(config.color_priority),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        raise ValueError(f"Could not save detection configuration {path}: {exc}") from exc
