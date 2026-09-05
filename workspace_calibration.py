"""Planar calibration for a fixed overhead camera.

The module estimates a homography from image pixels ``(u, v)`` to table-plane
coordinates ``(x, y)`` in millimetres.  It is hardware-free and intentionally
refuses to load the placeholder calibration until real correspondences exist.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_WORKSPACE_CALIBRATION_PATH = (
    Path(__file__).resolve().parent / "calibration" / "workspace_calibration.json"
)
Matrix3 = NDArray[np.float64]
Points2D = NDArray[np.float64]


class WorkspaceCalibrationError(ValueError):
    """Base error for invalid or unusable planar calibration data."""


class CalibrationNotReadyError(WorkspaceCalibrationError):
    """Raised when code tries to use the uncalibrated placeholder."""


class DegenerateCalibrationError(WorkspaceCalibrationError):
    """Raised when point geometry cannot determine a stable homography."""


@dataclass(frozen=True, slots=True)
class PlanarHomography:
    """Bidirectional pixel/table transform plus fit-error diagnostics."""

    table_from_pixel: Matrix3
    pixel_from_table: Matrix3
    point_count: int
    rms_error_mm: float
    max_error_mm: float

    def __post_init__(self) -> None:
        table_from_pixel = _validated_matrix(self.table_from_pixel, "table_from_pixel")
        pixel_from_table = _validated_matrix(self.pixel_from_table, "pixel_from_table")
        if self.point_count < 4:
            raise WorkspaceCalibrationError("A homography requires at least four points")
        if not math.isfinite(self.rms_error_mm) or self.rms_error_mm < 0:
            raise WorkspaceCalibrationError("RMS error must be finite and non-negative")
        if not math.isfinite(self.max_error_mm) or self.max_error_mm < 0:
            raise WorkspaceCalibrationError("Maximum error must be finite and non-negative")
        if self.max_error_mm + 1e-12 < self.rms_error_mm:
            raise WorkspaceCalibrationError("Maximum error cannot be below RMS error")
        identity = table_from_pixel @ pixel_from_table
        identity /= identity[2, 2]
        if not np.allclose(identity, np.eye(3), atol=1e-7, rtol=1e-7):
            raise WorkspaceCalibrationError("Stored homography matrices are not inverses")
        object.__setattr__(self, "table_from_pixel", table_from_pixel)
        object.__setattr__(self, "pixel_from_table", pixel_from_table)

    def pixel_to_table(self, u: float, v: float) -> tuple[float, float]:
        """Map one image pixel to table ``(x, y)`` millimetres."""

        transformed = transform_points(((u, v),), self.table_from_pixel)
        return float(transformed[0, 0]), float(transformed[0, 1])

    def table_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Map one table point back to image ``(u, v)`` pixels."""

        transformed = transform_points(((x_mm, y_mm),), self.pixel_from_table)
        return float(transformed[0, 0]), float(transformed[0, 1])


def estimate_planar_homography(
    pixel_points: Sequence[Sequence[float]] | Points2D,
    table_points_mm: Sequence[Sequence[float]] | Points2D,
) -> PlanarHomography:
    """Fit a normalized DLT homography from pixels to table millimetres."""

    pixels = _validated_points(pixel_points, "pixel_points")
    table = _validated_points(table_points_mm, "table_points_mm")
    if pixels.shape != table.shape:
        raise WorkspaceCalibrationError(
            "Pixel and table point arrays must have matching N-by-2 shapes"
        )
    if len(pixels) < 4:
        raise WorkspaceCalibrationError("At least four point correspondences are required")
    _require_non_collinear(pixels, "pixel points")
    _require_non_collinear(table, "table points")

    normalized_pixels, pixel_normalizer = _normalize_points(pixels)
    normalized_table, table_normalizer = _normalize_points(table)
    rows: list[list[float]] = []
    for (u, v), (x_mm, y_mm) in zip(normalized_pixels, normalized_table):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x_mm * u, x_mm * v, x_mm])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y_mm * u, y_mm * v, y_mm])
    design = np.asarray(rows, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 8:
        raise DegenerateCalibrationError("Point correspondences do not define a homography")

    _, _, right_vectors = np.linalg.svd(design, full_matrices=True)
    normalized_homography = right_vectors[-1].reshape(3, 3)
    table_from_pixel = (
        np.linalg.inv(table_normalizer)
        @ normalized_homography
        @ pixel_normalizer
    )
    scale = table_from_pixel[2, 2]
    if abs(scale) > 1e-12:
        table_from_pixel /= scale
    else:
        table_from_pixel /= np.linalg.norm(table_from_pixel)
    if abs(float(np.linalg.det(table_from_pixel))) < 1e-12:
        raise DegenerateCalibrationError("Estimated homography is singular")
    pixel_from_table = np.linalg.inv(table_from_pixel)
    pixel_from_table /= pixel_from_table[2, 2]

    predicted_table = transform_points(pixels, table_from_pixel)
    errors_mm = np.linalg.norm(predicted_table - table, axis=1)
    return PlanarHomography(
        table_from_pixel=table_from_pixel,
        pixel_from_table=pixel_from_table,
        point_count=len(pixels),
        rms_error_mm=float(np.sqrt(np.mean(errors_mm * errors_mm))),
        max_error_mm=float(np.max(errors_mm)),
    )


def transform_points(
    points: Sequence[Sequence[float]] | Points2D,
    homography: Matrix3,
) -> Points2D:
    """Apply a 3x3 projective transform to an N-by-2 point array."""

    values = _validated_points(points, "points", minimum_count=1)
    matrix = _validated_matrix(homography, "homography")
    homogeneous = np.column_stack((values, np.ones(len(values), dtype=np.float64)))
    mapped = (matrix @ homogeneous.T).T
    denominators = mapped[:, 2]
    if np.any(np.abs(denominators) < 1e-12):
        raise WorkspaceCalibrationError("A transformed point lies at projective infinity")
    result = mapped[:, :2] / denominators[:, np.newaxis]
    if not np.all(np.isfinite(result)):
        raise WorkspaceCalibrationError("Homography produced non-finite coordinates")
    return result


def save_workspace_calibration(
    calibration: PlanarHomography,
    pixel_points: Sequence[Sequence[float]] | Points2D,
    table_points_mm: Sequence[Sequence[float]] | Points2D,
    path: Path = DEFAULT_WORKSPACE_CALIBRATION_PATH,
    *,
    camera_metadata: dict[str, Any] | None = None,
    coordinate_frame: dict[str, Any] | None = None,
) -> None:
    """Atomically save a calibrated homography and its source correspondences."""

    pixels = _validated_points(pixel_points, "pixel_points")
    table = _validated_points(table_points_mm, "table_points_mm")
    if pixels.shape != table.shape or len(pixels) != calibration.point_count:
        raise WorkspaceCalibrationError(
            "Saved correspondences must match the fitted calibration point count"
        )
    payload = {
        "schema_version": 1,
        "status": "calibrated",
        "camera": camera_metadata or {},
        "coordinate_frame": coordinate_frame or {},
        "correspondences": [
            {
                "pixel": [float(pixel[0]), float(pixel[1])],
                "table_mm": [float(point[0]), float(point[1])],
            }
            for pixel, point in zip(pixels, table)
        ],
        "homography_table_from_pixel": calibration.table_from_pixel.tolist(),
        "homography_pixel_from_table": calibration.pixel_from_table.tolist(),
        "quality": {
            "point_count": calibration.point_count,
            "rms_error_mm": calibration.rms_error_mm,
            "max_error_mm": calibration.max_error_mm,
        },
        "warnings": [
            "Recalibrate after any camera, table, resolution, focus, or rotation change."
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)
    except OSError as exc:
        raise WorkspaceCalibrationError(f"Could not save calibration {path}: {exc}") from exc


def load_workspace_calibration(
    path: Path = DEFAULT_WORKSPACE_CALIBRATION_PATH,
    *,
    expected_camera_index: int | None = None,
    expected_resolution: tuple[int, int] | None = None,
    expected_rotation_degrees: int | None = None,
) -> PlanarHomography:
    """Load a completed calibration, optionally checking the active camera setup.

    The optional expectations guard against silently applying pixel coordinates
    captured by a different camera geometry.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkspaceCalibrationError(f"Could not read calibration {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise WorkspaceCalibrationError("Workspace calibration must use schema_version 1")
    if payload.get("status") != "calibrated":
        raise CalibrationNotReadyError(
            "Workspace calibration is not ready; mount the camera and collect real points"
        )
    _validate_camera_metadata(
        payload.get("camera"),
        expected_camera_index=expected_camera_index,
        expected_resolution=expected_resolution,
        expected_rotation_degrees=expected_rotation_degrees,
    )
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        raise WorkspaceCalibrationError("Calibrated file requires quality metrics")
    try:
        return PlanarHomography(
            table_from_pixel=np.asarray(
                payload["homography_table_from_pixel"], dtype=np.float64
            ),
            pixel_from_table=np.asarray(
                payload["homography_pixel_from_table"], dtype=np.float64
            ),
            point_count=int(quality["point_count"]),
            rms_error_mm=float(quality["rms_error_mm"]),
            max_error_mm=float(quality["max_error_mm"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, WorkspaceCalibrationError):
            raise
        raise WorkspaceCalibrationError(f"Invalid workspace calibration: {exc}") from exc


def _validate_camera_metadata(
    camera_metadata: object,
    *,
    expected_camera_index: int | None,
    expected_resolution: tuple[int, int] | None,
    expected_rotation_degrees: int | None,
) -> None:
    """Reject a calibrated map that does not match the active image geometry."""

    expectations_requested = any(
        value is not None
        for value in (
            expected_camera_index,
            expected_resolution,
            expected_rotation_degrees,
        )
    )
    if not expectations_requested:
        return
    if not isinstance(camera_metadata, dict):
        raise WorkspaceCalibrationError(
            "Workspace calibration has no camera metadata to verify"
        )

    if expected_camera_index is not None:
        calibrated_index = camera_metadata.get("camera_index")
        if calibrated_index != expected_camera_index:
            raise WorkspaceCalibrationError(
                "Workspace calibration camera index mismatch: "
                f"calibrated={calibrated_index!r}, active={expected_camera_index}"
            )

    if expected_resolution is not None:
        calibrated_resolution = camera_metadata.get("resolution")
        if not (
            isinstance(calibrated_resolution, list)
            and len(calibrated_resolution) == 2
            and all(isinstance(value, int) for value in calibrated_resolution)
        ):
            raise WorkspaceCalibrationError(
                "Workspace calibration has no valid [width, height] resolution"
            )
        if tuple(calibrated_resolution) != expected_resolution:
            raise WorkspaceCalibrationError(
                "Workspace calibration resolution mismatch: "
                f"calibrated={tuple(calibrated_resolution)}, "
                f"active={expected_resolution}"
            )

    if expected_rotation_degrees is not None:
        calibrated_rotation = camera_metadata.get("rotation_degrees")
        if calibrated_rotation != expected_rotation_degrees:
            raise WorkspaceCalibrationError(
                "Workspace calibration rotation mismatch: "
                f"calibrated={calibrated_rotation!r}, "
                f"active={expected_rotation_degrees}"
            )


def _validated_points(
    points: Sequence[Sequence[float]] | Points2D,
    name: str,
    *,
    minimum_count: int = 0,
) -> Points2D:
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WorkspaceCalibrationError(f"{name} must be numeric") from exc
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < minimum_count:
        raise WorkspaceCalibrationError(
            f"{name} must have shape N-by-2 with at least {minimum_count} rows"
        )
    if not np.all(np.isfinite(values)):
        raise WorkspaceCalibrationError(f"{name} must contain only finite coordinates")
    return values


def _validated_matrix(matrix: Matrix3, name: str) -> Matrix3:
    try:
        value = np.asarray(matrix, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise WorkspaceCalibrationError(f"{name} must be numeric") from exc
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise WorkspaceCalibrationError(f"{name} must be a finite 3x3 matrix")
    if abs(float(np.linalg.det(value))) < 1e-12:
        raise WorkspaceCalibrationError(f"{name} must be invertible")
    value.setflags(write=False)
    return value


def _require_non_collinear(points: Points2D, name: str) -> None:
    augmented = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    if np.linalg.matrix_rank(augmented) < 3:
        raise DegenerateCalibrationError(f"{name} are collinear")


def _normalize_points(points: Points2D) -> tuple[Points2D, Matrix3]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(centered, axis=1)))
    if mean_distance < 1e-12:
        raise DegenerateCalibrationError("Calibration points have no spatial spread")
    scale = math.sqrt(2.0) / mean_distance
    normalizer = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = transform_points(points, normalizer)
    return normalized, normalizer
