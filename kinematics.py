"""Hardware-free standard-DH forward kinematics for the four-joint arm.

This module performs mathematics only.  It contains no GPIO, PWM, serial, or
motor-control code.  The supplied DH convention and servo mappings are marked
provisional until physically confirmed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_KINEMATICS_CONFIG_PATH = (
    Path(__file__).resolve().parent / "calibration" / "arm_kinematics.json"
)
Matrix4 = NDArray[np.float64]


class KinematicsError(ValueError):
    """Raised when kinematic inputs or configuration are invalid."""


class UnconfirmedServoMappingError(KinematicsError):
    """Raised when provisional servo calibration is used without opt-in."""


class UnreachableTargetError(KinematicsError):
    """Raised when a requested frame-4 pose is outside the model workspace."""


@dataclass(frozen=True, slots=True)
class DHJoint:
    """One revolute standard-DH row plus its provisional servo mapping."""

    index: int
    name: str
    servo_model: str
    a_mm: float
    alpha_deg: float
    d_mm: float
    dh_theta_offset_deg: float
    servo_raw_min_deg: float
    servo_raw_max_deg: float
    servo_center_deg: float
    servo_theta_at_center_deg: float
    servo_direction: int
    servo_mapping_confirmed: bool

    def __post_init__(self) -> None:
        numeric_values = (
            self.a_mm,
            self.alpha_deg,
            self.d_mm,
            self.dh_theta_offset_deg,
            self.servo_raw_min_deg,
            self.servo_raw_max_deg,
            self.servo_center_deg,
            self.servo_theta_at_center_deg,
        )
        if self.index < 1:
            raise KinematicsError("Joint indices must start at 1")
        if not self.name or not self.servo_model:
            raise KinematicsError("Every joint requires a name and servo model")
        if not all(math.isfinite(value) for value in numeric_values):
            raise KinematicsError("DH and servo values must be finite")
        if self.servo_raw_min_deg >= self.servo_raw_max_deg:
            raise KinematicsError("Servo minimum must be below its maximum")
        if not self.servo_raw_min_deg <= self.servo_center_deg <= self.servo_raw_max_deg:
            raise KinematicsError("Servo center must lie inside its raw range")
        if self.servo_direction not in {-1, 1}:
            raise KinematicsError("Servo direction must be +1 or -1")


@dataclass(frozen=True, slots=True)
class ArmKinematicsConfig:
    """Validated four-joint arm model loaded from JSON."""

    status: str
    dh_convention: str
    joints: tuple[DHJoint, ...]
    gripper_servo_model: str
    gripper_limits_confirmed: bool

    def __post_init__(self) -> None:
        if self.dh_convention != "standard":
            raise KinematicsError(
                "Only standard DH is implemented; modified DH needs a separate model"
            )
        if len(self.joints) != 4:
            raise KinematicsError("This arm model requires exactly four DH joints")
        expected_indices = tuple(range(1, len(self.joints) + 1))
        if tuple(joint.index for joint in self.joints) != expected_indices:
            raise KinematicsError("DH joints must be ordered consecutively from 1")
        if not self.gripper_servo_model:
            raise KinematicsError("A gripper servo model is required")


@dataclass(frozen=True, slots=True)
class ForwardKinematicsResult:
    """Tool transform and each joint-frame origin in base coordinates."""

    transform: Matrix4
    joint_origins_mm: tuple[tuple[float, float, float], ...]

    @property
    def position_mm(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.transform[:3, 3])  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class InverseKinematicsSolution:
    """One position-and-pitch IK branch checked by forward kinematics."""

    branch: str
    theta_degrees: tuple[float, float, float, float]
    provisional_servo_raw_degrees: tuple[float, float, float, float]
    within_provisional_servo_range: bool
    position_error_mm: float


def load_kinematics_config(
    path: Path = DEFAULT_KINEMATICS_CONFIG_PATH,
) -> ArmKinematicsConfig:
    """Load the provisional arm model from a validated JSON document."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KinematicsError(f"Could not read kinematics configuration {path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise KinematicsError("Kinematics configuration must use schema_version 1")
    if payload.get("length_unit") != "mm" or payload.get("angle_unit") != "degrees":
        raise KinematicsError("Kinematics configuration must use millimetres and degrees")

    raw_joints = payload.get("joints")
    gripper = payload.get("gripper")
    if not isinstance(raw_joints, list) or not isinstance(gripper, dict):
        raise KinematicsError("Kinematics configuration requires joints and gripper")

    try:
        joints = tuple(
            DHJoint(
                index=int(item["index"]),
                name=str(item["name"]),
                servo_model=str(item["servo_model"]),
                a_mm=float(item["a_mm"]),
                alpha_deg=float(item["alpha_deg"]),
                d_mm=float(item["d_mm"]),
                dh_theta_offset_deg=float(item["dh_theta_offset_deg"]),
                servo_raw_min_deg=float(item["servo_raw_min_deg"]),
                servo_raw_max_deg=float(item["servo_raw_max_deg"]),
                servo_center_deg=float(item["servo_center_deg"]),
                servo_theta_at_center_deg=float(item["servo_theta_at_center_deg"]),
                servo_direction=int(item["servo_direction"]),
                servo_mapping_confirmed=item["servo_mapping_confirmed"],
            )
            for item in raw_joints
        )
        if any(not isinstance(joint.servo_mapping_confirmed, bool) for joint in joints):
            raise KinematicsError("servo_mapping_confirmed must be Boolean")
        limits_confirmed = gripper["limits_confirmed"]
        if not isinstance(limits_confirmed, bool):
            raise KinematicsError("Gripper limits_confirmed must be Boolean")
        return ArmKinematicsConfig(
            status=str(payload["status"]),
            dh_convention=str(payload["dh_convention"]),
            joints=joints,
            gripper_servo_model=str(gripper["servo_model"]),
            gripper_limits_confirmed=limits_confirmed,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, KinematicsError):
            raise
        raise KinematicsError(f"Invalid kinematics configuration: {exc}") from exc


def standard_dh_transform(
    a_mm: float,
    alpha_deg: float,
    d_mm: float,
    theta_deg: float,
) -> Matrix4:
    """Return ``Rz(theta) Tz(d) Tx(a) Rx(alpha)`` as a 4x4 matrix."""

    values = (a_mm, alpha_deg, d_mm, theta_deg)
    if not all(math.isfinite(value) for value in values):
        raise KinematicsError("DH transform inputs must be finite")

    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a_mm * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a_mm * sin_theta],
            [0.0, sin_alpha, cos_alpha, d_mm],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def forward_kinematics(
    theta_degrees: Sequence[float],
    config: ArmKinematicsConfig | None = None,
) -> ForwardKinematicsResult:
    """Compute frame 4 from four DH joint angles without commanding hardware."""

    model = config or load_kinematics_config()
    if len(theta_degrees) != len(model.joints):
        raise KinematicsError(
            f"Expected {len(model.joints)} joint angles, received {len(theta_degrees)}"
        )
    angles = tuple(float(value) for value in theta_degrees)
    if not all(math.isfinite(value) for value in angles):
        raise KinematicsError("Joint angles must be finite")

    transform = np.eye(4, dtype=np.float64)
    origins: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for joint, theta in zip(model.joints, angles):
        transform = transform @ standard_dh_transform(
            joint.a_mm,
            joint.alpha_deg,
            joint.d_mm,
            theta + joint.dh_theta_offset_deg,
        )
        origins.append(tuple(float(value) for value in transform[:3, 3]))

    return ForwardKinematicsResult(transform=transform, joint_origins_mm=tuple(origins))


def servo_raw_to_dh_angles(
    servo_raw_degrees: Sequence[float],
    config: ArmKinematicsConfig | None = None,
    *,
    allow_unverified: bool = False,
) -> tuple[float, ...]:
    """Convert four raw servo angles, guarded while mappings are provisional."""

    model = config or load_kinematics_config()
    if len(servo_raw_degrees) != len(model.joints):
        raise KinematicsError(
            f"Expected {len(model.joints)} servo angles, received {len(servo_raw_degrees)}"
        )
    if not allow_unverified and not all(
        joint.servo_mapping_confirmed for joint in model.joints
    ):
        raise UnconfirmedServoMappingError(
            "Servo directions/centers are unconfirmed; use DH angles directly or "
            "explicitly allow the provisional mapping for simulation only"
        )

    result: list[float] = []
    for joint, raw_value in zip(model.joints, servo_raw_degrees):
        raw = float(raw_value)
        if not math.isfinite(raw):
            raise KinematicsError("Servo angles must be finite")
        if not joint.servo_raw_min_deg <= raw <= joint.servo_raw_max_deg:
            raise KinematicsError(
                f"{joint.name} raw angle {raw} is outside "
                f"[{joint.servo_raw_min_deg}, {joint.servo_raw_max_deg}]"
            )
        result.append(
            joint.servo_direction * (raw - joint.servo_center_deg)
            + joint.servo_theta_at_center_deg
        )
    return tuple(result)


def dh_angles_to_servo_raw(
    theta_degrees: Sequence[float],
    config: ArmKinematicsConfig | None = None,
    *,
    allow_unverified: bool = False,
    enforce_range: bool = True,
) -> tuple[float, ...]:
    """Convert DH variables to raw servo angles with the same safety guard."""

    model = config or load_kinematics_config()
    if len(theta_degrees) != len(model.joints):
        raise KinematicsError(
            f"Expected {len(model.joints)} joint angles, received {len(theta_degrees)}"
        )
    if not allow_unverified and not all(
        joint.servo_mapping_confirmed for joint in model.joints
    ):
        raise UnconfirmedServoMappingError(
            "Servo directions/centers are unconfirmed; raw output is simulation-only"
        )

    raw_values = _dh_angles_to_servo_raw_unchecked(theta_degrees, model)
    if enforce_range:
        for joint, raw in zip(model.joints, raw_values):
            if not joint.servo_raw_min_deg <= raw <= joint.servo_raw_max_deg:
                raise KinematicsError(
                    f"{joint.name} requires provisional raw angle {raw:.3f}, outside "
                    f"[{joint.servo_raw_min_deg}, {joint.servo_raw_max_deg}]"
                )
    return raw_values


def inverse_kinematics_frame4(
    target_position_mm: Sequence[float],
    tool_pitch_deg: float,
    config: ArmKinematicsConfig | None = None,
) -> tuple[InverseKinematicsSolution, ...]:
    """Solve frame-4 position and radial-plane pitch for both elbow branches.

    ``tool_pitch_deg`` is the angle of link ``a4`` above the horizontal radial
    direction.  The solver returns the positive- and negative-theta3 branches
    when they are distinct.  Servo limits are reported but not enforced because
    the current mappings and collision limits are unconfirmed.
    """

    model = config or load_kinematics_config()
    _validate_supported_ik_structure(model)
    if len(target_position_mm) != 3:
        raise KinematicsError("Target position requires x, y, and z in millimetres")
    target = tuple(float(value) for value in target_position_mm)
    if not all(math.isfinite(value) for value in (*target, tool_pitch_deg)):
        raise KinematicsError("Target position and tool pitch must be finite")

    x_mm, y_mm, z_mm = target
    joint1, joint2, joint3, joint4 = model.joints
    lateral_offset_mm = sum(joint.d_mm for joint in model.joints[1:])
    radius_squared = x_mm * x_mm + y_mm * y_mm
    radial_squared = radius_squared - lateral_offset_mm * lateral_offset_mm
    tolerance = 1e-9
    if radial_squared < -tolerance:
        raise UnreachableTargetError(
            "Target is too close to the base axis for the DH lateral offset"
        )
    radial_mm = math.sqrt(max(0.0, radial_squared))

    q1_effective = math.atan2(y_mm, x_mm) + math.atan2(
        lateral_offset_mm,
        radial_mm,
    )
    planar_radius_mm = radial_mm - joint1.a_mm
    planar_height_mm = z_mm - joint1.d_mm

    pitch = math.radians(float(tool_pitch_deg))
    wrist_radius_mm = planar_radius_mm - joint4.a_mm * math.cos(pitch)
    wrist_height_mm = planar_height_mm - joint4.a_mm * math.sin(pitch)

    denominator = 2.0 * joint2.a_mm * joint3.a_mm
    if denominator <= 0:
        raise KinematicsError("Shoulder and elbow link lengths must be positive")
    cos_q3 = (
        wrist_radius_mm * wrist_radius_mm
        + wrist_height_mm * wrist_height_mm
        - joint2.a_mm * joint2.a_mm
        - joint3.a_mm * joint3.a_mm
    ) / denominator
    if cos_q3 < -1.0 - tolerance or cos_q3 > 1.0 + tolerance:
        raise UnreachableTargetError(
            "Target wrist point is outside the shoulder-elbow workspace"
        )
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_magnitude = math.acos(cos_q3)

    solutions: list[InverseKinematicsSolution] = []
    branch_signs = (("elbow_positive", 1.0), ("elbow_negative", -1.0))
    for branch, sign in branch_signs:
        q3_effective = sign * q3_magnitude
        q2_effective = math.atan2(wrist_height_mm, wrist_radius_mm) - math.atan2(
            joint3.a_mm * math.sin(q3_effective),
            joint2.a_mm + joint3.a_mm * math.cos(q3_effective),
        )
        q4_effective = pitch - q2_effective - q3_effective
        effective_angles = (
            q1_effective,
            q2_effective,
            q3_effective,
            q4_effective,
        )
        theta_degrees = tuple(
            _normalize_degrees(math.degrees(angle) - joint.dh_theta_offset_deg)
            for angle, joint in zip(effective_angles, model.joints)
        )
        theta_tuple = tuple(theta_degrees)  # type: ignore[assignment]
        raw_values = _dh_angles_to_servo_raw_unchecked(theta_tuple, model)
        within_limits = all(
            joint.servo_raw_min_deg <= raw <= joint.servo_raw_max_deg
            for joint, raw in zip(model.joints, raw_values)
        )
        checked_position = np.asarray(
            forward_kinematics(theta_tuple, model).position_mm,
            dtype=np.float64,
        )
        position_error = float(
            np.linalg.norm(checked_position - np.asarray(target, dtype=np.float64))
        )
        if position_error > 1e-6:
            raise KinematicsError(
                f"Internal IK/FK verification failed with {position_error:.6g} mm error"
            )
        solutions.append(
            InverseKinematicsSolution(
                branch=branch,
                theta_degrees=theta_tuple,
                provisional_servo_raw_degrees=raw_values,
                within_provisional_servo_range=within_limits,
                position_error_mm=position_error,
            )
        )
        if math.isclose(q3_magnitude, 0.0, abs_tol=1e-12) or math.isclose(
            q3_magnitude,
            math.pi,
            abs_tol=1e-12,
        ):
            break
    return tuple(solutions)


def _dh_angles_to_servo_raw_unchecked(
    theta_degrees: Sequence[float],
    model: ArmKinematicsConfig,
) -> tuple[float, float, float, float]:
    values: list[float] = []
    for joint, theta_value in zip(model.joints, theta_degrees):
        theta = float(theta_value)
        if not math.isfinite(theta):
            raise KinematicsError("Joint angles must be finite")
        values.append(
            joint.servo_center_deg
            + (theta - joint.servo_theta_at_center_deg) / joint.servo_direction
        )
    return tuple(values)  # type: ignore[return-value]


def _validate_supported_ik_structure(model: ArmKinematicsConfig) -> None:
    """Reject DH geometries that do not match this analytic solver."""

    alpha_values = tuple(joint.alpha_deg for joint in model.joints)
    if not math.isclose(alpha_values[0], 90.0, abs_tol=1e-9) or not all(
        math.isclose(value, 0.0, abs_tol=1e-9) for value in alpha_values[1:]
    ):
        raise KinematicsError(
            "Analytic IK requires alpha=[90, 0, 0, 0] degrees"
        )


def _normalize_degrees(value: float) -> float:
    """Normalize an angle to the interval (-180, 180]."""

    normalized = (value + 180.0) % 360.0 - 180.0
    return 180.0 if math.isclose(normalized, -180.0, abs_tol=1e-12) else normalized
