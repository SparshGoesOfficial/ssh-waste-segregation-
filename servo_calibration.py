"""Direction-aware conversion between measured joint angles and raw servo commands."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence


DEFAULT_SERVO_MEASUREMENTS_PATH = (
    Path(__file__).resolve().parent / "calibration" / "servo_measurements.json"
)
MovementDirection = Literal["raw_decreasing", "raw_increasing"]


class ServoCalibrationError(ValueError):
    """Raised when measured calibration data or a requested angle is invalid."""


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    raw_servo_deg: float
    physical_joint_deg: float


@dataclass(frozen=True, slots=True)
class DirectionalCurve:
    """Monotonic piecewise-linear raw-to-physical calibration curve."""

    points: tuple[CalibrationPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ServoCalibrationError("A calibration curve needs at least two points")
        raw_values = tuple(point.raw_servo_deg for point in self.points)
        physical_values = tuple(point.physical_joint_deg for point in self.points)
        if not all(math.isfinite(value) for value in (*raw_values, *physical_values)):
            raise ServoCalibrationError("Calibration values must be finite")
        if any(right <= left for left, right in zip(raw_values, raw_values[1:])):
            raise ServoCalibrationError("Raw calibration values must increase strictly")
        if any(right <= left for left, right in zip(physical_values, physical_values[1:])):
            raise ServoCalibrationError("Physical calibration values must increase strictly")

    @property
    def raw_range(self) -> tuple[float, float]:
        return self.points[0].raw_servo_deg, self.points[-1].raw_servo_deg

    @property
    def physical_range(self) -> tuple[float, float]:
        return self.points[0].physical_joint_deg, self.points[-1].physical_joint_deg

    def raw_to_physical(self, raw_deg: float, *, allow_extrapolation: bool = False) -> float:
        """Interpolate a physical joint angle from a raw command."""

        return _interpolate(
            float(raw_deg),
            tuple((point.raw_servo_deg, point.physical_joint_deg) for point in self.points),
            allow_extrapolation=allow_extrapolation,
            label="raw servo angle",
        )

    def physical_to_raw(
        self,
        physical_deg: float,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        """Invert the measured curve to obtain the required raw command."""

        return _interpolate(
            float(physical_deg),
            tuple((point.physical_joint_deg, point.raw_servo_deg) for point in self.points),
            allow_extrapolation=allow_extrapolation,
            label="physical joint angle",
        )


@dataclass(frozen=True, slots=True)
class JointDirectionalCalibration:
    joint_id: str
    physical_reference_deg: float
    dh_reference_deg: float
    raw_decreasing: DirectionalCurve
    raw_increasing: DirectionalCurve

    def curve(self, direction: MovementDirection) -> DirectionalCurve:
        if direction == "raw_decreasing":
            return self.raw_decreasing
        if direction == "raw_increasing":
            return self.raw_increasing
        raise ServoCalibrationError(f"Unsupported movement direction: {direction}")

    def physical_to_raw(
        self,
        physical_deg: float,
        direction: MovementDirection,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        return self.curve(direction).physical_to_raw(
            physical_deg,
            allow_extrapolation=allow_extrapolation,
        )

    def raw_to_physical(
        self,
        raw_deg: float,
        direction: MovementDirection,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        return self.curve(direction).raw_to_physical(
            raw_deg,
            allow_extrapolation=allow_extrapolation,
        )

    def dh_to_raw(
        self,
        theta_deg: float,
        direction: MovementDirection,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        physical_deg = (
            float(theta_deg) - self.dh_reference_deg + self.physical_reference_deg
        )
        return self.physical_to_raw(
            physical_deg,
            direction,
            allow_extrapolation=allow_extrapolation,
        )

    def raw_to_dh(
        self,
        raw_deg: float,
        direction: MovementDirection,
        *,
        allow_extrapolation: bool = False,
    ) -> float:
        physical_deg = self.raw_to_physical(
            raw_deg,
            direction,
            allow_extrapolation=allow_extrapolation,
        )
        return physical_deg - self.physical_reference_deg + self.dh_reference_deg

    def descending_approach_commands(
        self,
        physical_target_deg: float,
        *,
        preload_margin_raw_deg: float = 5.0,
        raw_min_deg: int = 0,
        raw_max_deg: int = 180,
    ) -> tuple[int, int]:
        """Return ``(preload, target)`` commands for a repeatable final descent.

        The target comes from the four measured raw-decreasing points.  The
        preload command intentionally sits above it; positional accuracy at the
        preload point is irrelevant because only the final downward approach is
        used as the calibrated result.
        """

        margin = float(preload_margin_raw_deg)
        if not math.isfinite(margin) or margin <= 0:
            raise ServoCalibrationError("Preload margin must be positive and finite")
        target = int(round(self.physical_to_raw(physical_target_deg, "raw_decreasing")))
        preload = int(math.ceil(target + margin))
        if not raw_min_deg <= target <= raw_max_deg:
            raise ServoCalibrationError("Compensated target exceeds raw servo limits")
        if preload > raw_max_deg:
            raise ServoCalibrationError("Descending preload exceeds raw servo maximum")
        return preload, target


def load_directional_calibration(
    joint_id: str = "J2",
    path: Path = DEFAULT_SERVO_MEASUREMENTS_PATH,
) -> JointDirectionalCalibration:
    """Load one joint's measured, direction-dependent calibration."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServoCalibrationError(f"Could not read servo measurements: {exc}") from exc

    try:
        joint = payload["joints"][joint_id]
        model = joint["calibration_model"]
        if model["type"] != "direction_dependent_piecewise_linear":
            raise ServoCalibrationError("Unsupported servo calibration model")
        reference = model["dh_reference"]
        return JointDirectionalCalibration(
            joint_id=joint_id,
            physical_reference_deg=float(reference["physical_joint_deg"]),
            dh_reference_deg=float(reference["dh_theta_deg"]),
            raw_decreasing=_load_curve(model["raw_decreasing"]),
            raw_increasing=_load_curve(model["raw_increasing"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ServoCalibrationError):
            raise
        raise ServoCalibrationError(
            f"Invalid directional calibration for {joint_id}: {exc}"
        ) from exc


def _load_curve(payload: dict[str, object]) -> DirectionalCurve:
    raw_points = payload["points_raw_to_physical"]
    if not isinstance(raw_points, list):
        raise ServoCalibrationError("Calibration points must be an array")
    points: list[CalibrationPoint] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ServoCalibrationError("Each calibration point must contain two values")
        points.append(CalibrationPoint(float(raw_point[0]), float(raw_point[1])))
    return DirectionalCurve(tuple(points))


def _interpolate(
    value: float,
    points: Sequence[tuple[float, float]],
    *,
    allow_extrapolation: bool,
    label: str,
) -> float:
    if not math.isfinite(value):
        raise ServoCalibrationError(f"{label} must be finite")
    lower, upper = points[0][0], points[-1][0]
    if not allow_extrapolation and not lower <= value <= upper:
        raise ServoCalibrationError(
            f"{label} {value:.3f} is outside the measured range {lower:.3f}..{upper:.3f}"
        )

    if value <= lower:
        left, right = points[0], points[1]
    elif value >= upper:
        left, right = points[-2], points[-1]
    else:
        left, right = next(
            (left, right)
            for left, right in zip(points, points[1:])
            if left[0] <= value <= right[0]
        )
    fraction = (value - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1])
