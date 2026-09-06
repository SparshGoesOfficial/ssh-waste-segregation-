"""Guarded laptop-to-Arduino serial client for ARM_SERIAL_V1."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .servo_calibration import (
        ServoCalibrationError,
        load_directional_calibration,
    )
except ImportError:  # pragma: no cover - direct script imports
    from servo_calibration import ServoCalibrationError, load_directional_calibration


DEFAULT_SERVO_CONTROLLER_PATH = (
    Path(__file__).resolve().parent / "calibration" / "servo_controller.json"
)


class SerialControllerError(RuntimeError):
    """Base error for transport, protocol, or configuration failures."""


class SerialProtocolError(SerialControllerError):
    """Raised when the Arduino rejects or does not complete a command."""


class ServoSafetyError(SerialControllerError):
    """Raised before sending an unsafe or invalid servo request."""


@dataclass(frozen=True, slots=True)
class JointControllerConfig:
    name: str
    joint_id: str
    pin: int
    minimum_deg: int
    maximum_deg: int
    startup_deg: int

    def __post_init__(self) -> None:
        if not self.name or not self.joint_id:
            raise ServoSafetyError("Every servo requires a name and joint id")
        if self.pin < 0:
            raise ServoSafetyError("Servo pin cannot be negative")
        if not 0 <= self.minimum_deg <= self.startup_deg <= self.maximum_deg <= 180:
            raise ServoSafetyError(
                f"Invalid provisional range for {self.joint_id}: "
                f"{self.minimum_deg}..{self.maximum_deg} with startup {self.startup_deg}"
            )


@dataclass(frozen=True, slots=True)
class ServoControllerConfig:
    status: str
    baud_rate: int
    firmware_protocol: str
    joints: tuple[JointControllerConfig, ...]
    gripper_open_deg: int
    gripper_closed_deg: int
    minimum_duration_ms: int
    maximum_duration_ms: int
    keepalive_interval_ms: int
    firmware_watchdog_ms: int

    def __post_init__(self) -> None:
        if len(self.joints) != 5:
            raise ServoSafetyError("Exactly five servo channels are required")
        if self.baud_rate <= 0:
            raise ServoSafetyError("Baud rate must be positive")
        if self.firmware_protocol != "ARM_SERIAL_V1":
            raise ServoSafetyError("Unsupported Arduino firmware protocol")
        if not 0 < self.minimum_duration_ms <= self.maximum_duration_ms:
            raise ServoSafetyError("Invalid motion-duration range")
        if not 0 < self.keepalive_interval_ms < self.firmware_watchdog_ms:
            raise ServoSafetyError("Keepalive must be faster than the firmware watchdog")
        gripper = self.joints[4]
        for angle in (self.gripper_open_deg, self.gripper_closed_deg):
            if not gripper.minimum_deg <= angle <= gripper.maximum_deg:
                raise ServoSafetyError("Gripper command lies outside the J5 limits")

    @property
    def startup_angles(self) -> tuple[int, ...]:
        return tuple(joint.startup_deg for joint in self.joints)


def load_servo_controller_config(
    path: Path = DEFAULT_SERVO_CONTROLLER_PATH,
) -> ServoControllerConfig:
    """Load and validate the shared host/firmware settings."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SerialControllerError(f"Could not read servo controller config: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SerialControllerError("Servo controller config must use schema_version 1")
    try:
        raw_joints = payload["joints"]
        if not isinstance(raw_joints, list):
            raise TypeError("joints must be an array")
        joints = tuple(
            JointControllerConfig(
                name=str(item["name"]),
                joint_id=str(item["id"]),
                pin=int(item["pin"]),
                minimum_deg=int(item["minimum_deg"]),
                maximum_deg=int(item["maximum_deg"]),
                startup_deg=int(item["startup_deg"]),
            )
            for item in raw_joints
        )
        gripper = payload["gripper"]
        motion = payload["motion"]
        return ServoControllerConfig(
            status=str(payload["status"]),
            baud_rate=int(payload["baud_rate"]),
            firmware_protocol=str(payload["firmware_protocol"]),
            joints=joints,
            gripper_open_deg=int(gripper["open_deg"]),
            gripper_closed_deg=int(gripper["closed_deg"]),
            minimum_duration_ms=int(motion["minimum_duration_ms"]),
            maximum_duration_ms=int(motion["maximum_duration_ms"]),
            keepalive_interval_ms=int(motion["keepalive_interval_ms"]),
            firmware_watchdog_ms=int(motion["firmware_watchdog_ms"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SerialControllerError):
            raise
        raise SerialControllerError(f"Invalid servo controller config: {exc}") from exc


def normalize_servo_angles(
    angles: Iterable[float],
    config: ServoControllerConfig,
) -> tuple[int, ...]:
    """Round five finite angles and enforce every configured limit."""

    values = tuple(angles)
    if len(values) != len(config.joints):
        raise ServoSafetyError("Exactly five servo angles are required")
    normalized: list[int] = []
    for joint, value in zip(config.joints, values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ServoSafetyError(f"{joint.joint_id} angle must be numeric")
        if not math.isfinite(float(value)):
            raise ServoSafetyError(f"{joint.joint_id} angle must be finite")
        rounded = int(round(float(value)))
        if not joint.minimum_deg <= rounded <= joint.maximum_deg:
            raise ServoSafetyError(
                f"{joint.joint_id}={rounded} is outside the provisional "
                f"{joint.minimum_deg}..{joint.maximum_deg} range"
            )
        normalized.append(rounded)
    return tuple(normalized)


def validate_duration(duration_ms: int, config: ServoControllerConfig) -> int:
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
        raise ServoSafetyError("Motion duration must be an integer number of milliseconds")
    if not config.minimum_duration_ms <= duration_ms <= config.maximum_duration_ms:
        raise ServoSafetyError(
            f"Motion duration must be {config.minimum_duration_ms}.."
            f"{config.maximum_duration_ms} ms"
        )
    return duration_ms


class ArmSerialController:
    """Synchronous request/reply client with keepalive during motion."""

    def __init__(
        self,
        stream: Any,
        config: ServoControllerConfig | None = None,
        *,
        response_timeout_s: float = 2.0,
    ) -> None:
        self.stream = stream
        self.config = config or load_servo_controller_config()
        self.response_timeout_s = response_timeout_s
        self._sequence = 0

    @classmethod
    def open(
        cls,
        port: str,
        config: ServoControllerConfig | None = None,
    ) -> "ArmSerialController":
        """Open a real pyserial port and verify the firmware handshake."""

        settings = config or load_servo_controller_config()
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SerialControllerError(
                "pyserial is not installed; run: py -m pip install pyserial"
            ) from exc
        try:
            stream = serial.Serial(
                port=port,
                baudrate=settings.baud_rate,
                timeout=0.1,
                write_timeout=1.0,
            )
        except Exception as exc:
            raise SerialControllerError(f"Could not open Arduino port {port}: {exc}") from exc

        controller = cls(stream, settings)
        try:
            # Uno normally resets when the serial port opens.
            time.sleep(2.0)
            if hasattr(stream, "reset_input_buffer"):
                stream.reset_input_buffer()
            controller.ping()
        except Exception:
            stream.close()
            raise
        return controller

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> "ArmSerialController":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def _write_line(self, line: str) -> None:
        try:
            self.stream.write((line + "\n").encode("ascii"))
            if hasattr(self.stream, "flush"):
                self.stream.flush()
        except Exception as exc:
            raise SerialControllerError(f"Serial write failed: {exc}") from exc

    def _read_line(self) -> str | None:
        try:
            raw = self.stream.readline()
        except Exception as exc:
            raise SerialControllerError(f"Serial read failed: {exc}") from exc
        if not raw:
            return None
        try:
            return raw.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise SerialProtocolError("Arduino returned non-ASCII data") from exc

    def _wait_for_prefix(self, prefix: str, timeout_s: float | None = None) -> str:
        deadline = time.monotonic() + (timeout_s or self.response_timeout_s)
        while time.monotonic() < deadline:
            line = self._read_line()
            if line is None or line.startswith("READY "):
                continue
            if line.startswith("ERR "):
                raise SerialProtocolError(f"Arduino rejected command: {line}")
            if line.startswith(prefix):
                return line
        raise SerialProtocolError(f"Timed out waiting for Arduino response: {prefix}")

    def ping(self) -> str:
        self._write_line("PING")
        line = self._wait_for_prefix("PONG ")
        if line != f"PONG {self.config.firmware_protocol}":
            raise SerialProtocolError(f"Unexpected firmware identity: {line}")
        return line

    def status(self) -> str:
        self._write_line("STATUS")
        return self._wait_for_prefix("STATUS ")

    def limits(self) -> str:
        self._write_line("LIMITS")
        return self._wait_for_prefix("LIMITS ")

    def arm(self, initial_angles: Iterable[float]) -> str:
        angles = normalize_servo_angles(initial_angles, self.config)
        self._write_line("ARM " + " ".join(str(value) for value in angles))
        return self._wait_for_prefix("OK ARMED")

    def stop(self) -> str:
        self._write_line("STOP")
        return self._wait_for_prefix("OK STOPPED")

    def disarm(self) -> str:
        self._write_line("DISARM")
        return self._wait_for_prefix("OK DISARMED")

    def _next_id(self) -> str:
        self._sequence += 1
        return f"m{self._sequence}"

    def _wait_for_motion(self, command_id: str, duration_ms: int) -> str:
        deadline = time.monotonic() + duration_ms / 1000.0 + self.response_timeout_s
        next_keepalive = time.monotonic() + self.config.keepalive_interval_ms / 1000.0
        acknowledged = False
        while time.monotonic() < deadline:
            line = self._read_line()
            if line:
                if line.startswith("ERR "):
                    raise SerialProtocolError(f"Arduino motion failed: {line}")
                if line == f"ACK {command_id}":
                    acknowledged = True
                elif line == f"DONE {command_id}":
                    if not acknowledged:
                        raise SerialProtocolError("Arduino returned DONE before ACK")
                    return line
            if time.monotonic() >= next_keepalive:
                self._write_line("KEEPALIVE")
                next_keepalive = (
                    time.monotonic() + self.config.keepalive_interval_ms / 1000.0
                )
        try:
            self.stop()
        except SerialControllerError:
            pass
        raise SerialProtocolError(f"Motion {command_id} did not complete before timeout")

    def move(self, angles: Iterable[float], duration_ms: int) -> str:
        normalized = normalize_servo_angles(angles, self.config)
        duration = validate_duration(duration_ms, self.config)
        command_id = self._next_id()
        command = "MOVE {} {} {}".format(
            command_id,
            duration,
            " ".join(str(value) for value in normalized),
        )
        self._write_line(command)
        return self._wait_for_motion(command_id, duration)

    def move_joint(self, joint_number: int, angle: float, duration_ms: int) -> str:
        if isinstance(joint_number, bool) or not isinstance(joint_number, int):
            raise ServoSafetyError("Joint number must be an integer")
        if not 1 <= joint_number <= len(self.config.joints):
            raise ServoSafetyError("Joint number must be between 1 and 5")
        joint = self.config.joints[joint_number - 1]
        if not math.isfinite(float(angle)):
            raise ServoSafetyError("Joint angle must be finite")
        normalized = int(round(float(angle)))
        if not joint.minimum_deg <= normalized <= joint.maximum_deg:
            raise ServoSafetyError(
                f"{joint.joint_id}={normalized} is outside "
                f"{joint.minimum_deg}..{joint.maximum_deg}"
            )
        duration = validate_duration(duration_ms, self.config)
        command_id = self._next_id()
        self._write_line(
            f"JOINT {command_id} {joint_number} {normalized} {duration}"
        )
        return self._wait_for_motion(command_id, duration)

    def move_calibrated_shoulder(
        self,
        physical_angle: float,
        duration_ms: int,
        *,
        preload_margin_raw_deg: float = 5.0,
    ) -> tuple[str, str]:
        """Move J2 using its measured curve and a raw-decreasing final approach."""

        shoulder = self.config.joints[1]
        try:
            calibration = load_directional_calibration("J2")
            preload, target = calibration.descending_approach_commands(
                physical_angle,
                preload_margin_raw_deg=preload_margin_raw_deg,
                raw_min_deg=shoulder.minimum_deg,
                raw_max_deg=shoulder.maximum_deg,
            )
        except ServoCalibrationError as exc:
            raise ServoSafetyError(f"J2 calibrated move rejected: {exc}") from exc

        preload_result = self.move_joint(2, preload, duration_ms)
        target_result = self.move_joint(2, target, duration_ms)
        return preload_result, target_result

    def grip(self, opened: bool, duration_ms: int = 500) -> str:
        if not isinstance(opened, bool):
            raise ServoSafetyError("Gripper state must be True (open) or False (closed)")
        duration = validate_duration(duration_ms, self.config)
        command_id = self._next_id()
        state = "OPEN" if opened else "CLOSE"
        self._write_line(f"GRIP {command_id} {state} {duration}")
        return self._wait_for_motion(command_id, duration)
