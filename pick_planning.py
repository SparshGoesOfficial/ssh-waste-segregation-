"""Guarded, hardware-free bridge from vision detection to IK candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

try:
    from .detection import Detection
    from .kinematics import (
        ArmKinematicsConfig,
        InverseKinematicsSolution,
        KinematicsError,
        inverse_kinematics_frame4,
        load_kinematics_config,
    )
    from .workspace_calibration import (
        DEFAULT_WORKSPACE_CALIBRATION_PATH,
        PlanarHomography,
        WorkspaceCalibrationError,
        load_workspace_calibration,
    )
except ImportError:  # pragma: no cover - direct script imports
    from detection import Detection
    from kinematics import (
        ArmKinematicsConfig,
        InverseKinematicsSolution,
        KinematicsError,
        inverse_kinematics_frame4,
        load_kinematics_config,
    )
    from workspace_calibration import (
        DEFAULT_WORKSPACE_CALIBRATION_PATH,
        PlanarHomography,
        WorkspaceCalibrationError,
        load_workspace_calibration,
    )


DEFAULT_PICK_GEOMETRY_PATH = (
    Path(__file__).resolve().parent / "calibration" / "pick_geometry.json"
)


class PickPlanningError(ValueError):
    """Base error for an unsafe or impossible dry-run pick plan."""


class PickGeometryNotReadyError(PickPlanningError):
    """Raised when frame-4 pickup geometry has not been measured."""


@dataclass(frozen=True, slots=True)
class PickGeometry:
    """Explicit frame-4 height and pitch values required for dry-run IK."""

    frame4_pick_z_mm: float
    approach_clearance_mm: float
    tool_pitch_deg: float
    source_status: str

    def __post_init__(self) -> None:
        values = (
            self.frame4_pick_z_mm,
            self.approach_clearance_mm,
            self.tool_pitch_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise PickPlanningError("Pick geometry values must be finite")
        if self.approach_clearance_mm <= 0:
            raise PickPlanningError("Approach clearance must be positive")
        if not self.source_status:
            raise PickPlanningError("Pick geometry requires a source status")


@dataclass(frozen=True, slots=True)
class DetectionPickPlan:
    """Mapped target plus unselected approach and pickup IK candidates."""

    detection: Detection
    table_xy_mm: tuple[float, float]
    approach_position_mm: tuple[float, float, float]
    pick_position_mm: tuple[float, float, float]
    tool_pitch_deg: float
    approach_solutions: tuple[InverseKinematicsSolution, ...]
    pick_solutions: tuple[InverseKinematicsSolution, ...]
    provisional_common_branches: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a contour-free record suitable for dry-run logging."""

        return {
            "color": self.detection.color,
            "pixel": [self.detection.cx, self.detection.cy],
            "table_xy_mm": list(self.table_xy_mm),
            "approach_position_mm": list(self.approach_position_mm),
            "pick_position_mm": list(self.pick_position_mm),
            "tool_pitch_deg": self.tool_pitch_deg,
            "provisional_common_branches": list(self.provisional_common_branches),
            "approach_solutions": [_solution_to_dict(item) for item in self.approach_solutions],
            "pick_solutions": [_solution_to_dict(item) for item in self.pick_solutions],
            "dry_run_only": True,
        }


def load_pick_geometry(path: Path = DEFAULT_PICK_GEOMETRY_PATH) -> PickGeometry:
    """Load measured frame-4 pickup geometry, rejecting the placeholder."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PickPlanningError(f"Could not read pick geometry {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PickPlanningError("Pick geometry must use schema_version 1")
    if payload.get("status") != "measured":
        raise PickGeometryNotReadyError(
            "Pick geometry is not measured; frame-4 height and tool pitch are required"
        )
    try:
        return PickGeometry(
            frame4_pick_z_mm=float(payload["frame4_pick_z_mm"]),
            approach_clearance_mm=float(payload["approach_clearance_mm"]),
            tool_pitch_deg=float(payload["tool_pitch_deg"]),
            source_status=str(payload["status"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PickPlanningError):
            raise
        raise PickPlanningError(f"Invalid pick geometry: {exc}") from exc


def map_detection_to_table(
    detection: Detection,
    workspace: PlanarHomography,
) -> Detection:
    """Return an immutable copy carrying calibrated table coordinates."""

    x_mm, y_mm = workspace.pixel_to_table(detection.cx, detection.cy)
    return replace(detection, robot_x=x_mm, robot_y=y_mm)


def plan_detection_pick(
    detection: Detection,
    workspace: PlanarHomography,
    geometry: PickGeometry,
    arm_model: ArmKinematicsConfig | None = None,
) -> DetectionPickPlan:
    """Map one detection and calculate unselected approach/pick IK branches."""

    model = arm_model or load_kinematics_config()
    mapped = map_detection_to_table(detection, workspace)
    if mapped.robot_x is None or mapped.robot_y is None:  # defensive invariant
        raise PickPlanningError("Calibrated detection did not receive table coordinates")
    x_mm, y_mm = mapped.robot_x, mapped.robot_y
    pick_position = (x_mm, y_mm, geometry.frame4_pick_z_mm)
    approach_position = (
        x_mm,
        y_mm,
        geometry.frame4_pick_z_mm + geometry.approach_clearance_mm,
    )
    try:
        approach_solutions = inverse_kinematics_frame4(
            approach_position,
            geometry.tool_pitch_deg,
            model,
        )
        pick_solutions = inverse_kinematics_frame4(
            pick_position,
            geometry.tool_pitch_deg,
            model,
        )
    except KinematicsError as exc:
        raise PickPlanningError(
            f"Detected {detection.color} cube maps to an unreachable frame-4 target: {exc}"
        ) from exc

    approach_valid = {
        item.branch for item in approach_solutions if item.within_provisional_servo_range
    }
    pick_valid = {
        item.branch for item in pick_solutions if item.within_provisional_servo_range
    }
    branch_order = ("elbow_positive", "elbow_negative")
    common_branches = tuple(
        branch for branch in branch_order if branch in approach_valid & pick_valid
    )
    return DetectionPickPlan(
        detection=mapped,
        table_xy_mm=(x_mm, y_mm),
        approach_position_mm=approach_position,
        pick_position_mm=pick_position,
        tool_pitch_deg=geometry.tool_pitch_deg,
        approach_solutions=approach_solutions,
        pick_solutions=pick_solutions,
        provisional_common_branches=common_branches,
    )


def plan_detection_pick_from_files(
    detection: Detection,
    *,
    workspace_path: Path = DEFAULT_WORKSPACE_CALIBRATION_PATH,
    geometry_path: Path = DEFAULT_PICK_GEOMETRY_PATH,
) -> DetectionPickPlan:
    """Production entry point whose file loaders enforce calibration guards."""

    try:
        workspace = load_workspace_calibration(workspace_path)
    except WorkspaceCalibrationError as exc:
        raise PickPlanningError(f"Workspace calibration unavailable: {exc}") from exc
    geometry = load_pick_geometry(geometry_path)
    return plan_detection_pick(detection, workspace, geometry)


def _solution_to_dict(solution: InverseKinematicsSolution) -> dict[str, object]:
    return {
        "branch": solution.branch,
        "theta_degrees": list(solution.theta_degrees),
        "provisional_servo_raw_degrees": list(
            solution.provisional_servo_raw_degrees
        ),
        "within_provisional_servo_range": solution.within_provisional_servo_range,
        "ik_fk_position_error_mm": solution.position_error_mm,
    }
