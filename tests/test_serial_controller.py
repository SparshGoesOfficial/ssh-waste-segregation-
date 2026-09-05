"""Hardware-free tests for ARM_SERIAL_V1 host-side safety and protocol."""

from __future__ import annotations

import unittest
from collections import deque

from serial_controller import (
    ArmSerialController,
    SerialProtocolError,
    ServoSafetyError,
    load_servo_controller_config,
    normalize_servo_angles,
    validate_duration,
)


class FakeSerial:
    def __init__(self, responses: list[str]) -> None:
        self.responses = deque((line + "\n").encode("ascii") for line in responses)
        self.writes: list[str] = []
        self.closed = False

    def write(self, value: bytes) -> None:
        self.writes.append(value.decode("ascii"))

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        return self.responses.popleft() if self.responses else b""

    def close(self) -> None:
        self.closed = True


class SerialControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_servo_controller_config()

    def test_supplied_pin_mapping_and_provisional_limits_are_loaded(self) -> None:
        self.assertEqual(
            tuple(joint.pin for joint in self.config.joints),
            (9, 10, 11, 6, 5),
        )
        self.assertEqual(
            tuple((joint.minimum_deg, joint.maximum_deg) for joint in self.config.joints),
            ((0, 180), (20, 160), (10, 170), (0, 180), (30, 90)),
        )
        self.assertEqual(self.config.startup_angles, (90, 90, 90, 90, 90))

    def test_angles_are_rounded_and_checked_before_serial_write(self) -> None:
        actual = normalize_servo_angles((90.4, 89.6, 90, 90, 90), self.config)
        self.assertEqual(actual, (90, 90, 90, 90, 90))
        with self.assertRaises(ServoSafetyError):
            normalize_servo_angles((90, 0, 90, 90, 90), self.config)
        with self.assertRaises(ServoSafetyError):
            normalize_servo_angles((90, 90, 90, 90), self.config)

    def test_duration_is_guarded(self) -> None:
        self.assertEqual(validate_duration(500, self.config), 500)
        with self.assertRaises(ServoSafetyError):
            validate_duration(50, self.config)
        with self.assertRaises(ServoSafetyError):
            validate_duration(10001, self.config)

    def test_ping_verifies_exact_firmware_identity(self) -> None:
        stream = FakeSerial(["PONG ARM_SERIAL_V1"])
        controller = ArmSerialController(stream, self.config)
        self.assertEqual(controller.ping(), "PONG ARM_SERIAL_V1")
        self.assertEqual(stream.writes, ["PING\n"])

    def test_status_and_limits_are_read_only_commands(self) -> None:
        stream = FakeSerial(
            [
                "STATUS DISARMED IDLE 90 90 90 90 90",
                "LIMITS 0:180 20:160 10:170 0:180 30:90",
            ]
        )
        controller = ArmSerialController(stream, self.config)
        self.assertTrue(controller.status().startswith("STATUS DISARMED"))
        self.assertTrue(controller.limits().startswith("LIMITS 0:180"))
        self.assertEqual(stream.writes, ["STATUS\n", "LIMITS\n"])

    def test_arm_and_motion_require_acknowledgement_and_completion(self) -> None:
        stream = FakeSerial(["OK ARMED", "ACK m1", "DONE m1"])
        controller = ArmSerialController(stream, self.config)
        self.assertEqual(controller.arm(self.config.startup_angles), "OK ARMED")
        self.assertEqual(controller.move((91, 90, 90, 90, 90), 300), "DONE m1")
        self.assertEqual(stream.writes[0], "ARM 90 90 90 90 90\n")
        self.assertEqual(stream.writes[1], "MOVE m1 300 91 90 90 90 90\n")

    def test_joint_and_gripper_commands_are_explicit(self) -> None:
        stream = FakeSerial(["ACK m1", "DONE m1", "ACK m2", "DONE m2"])
        controller = ArmSerialController(stream, self.config)
        self.assertEqual(controller.move_joint(2, 91, 400), "DONE m1")
        self.assertEqual(controller.grip(False, 300), "DONE m2")
        self.assertEqual(stream.writes[0], "JOINT m1 2 91 400\n")
        self.assertEqual(stream.writes[1], "GRIP m2 CLOSE 300\n")

    def test_arduino_error_is_not_silently_ignored(self) -> None:
        stream = FakeSerial(["ERR DISARMED"])
        controller = ArmSerialController(stream, self.config)
        with self.assertRaises(SerialProtocolError):
            controller.move((90, 90, 90, 90, 90), 300)


if __name__ == "__main__":
    unittest.main()
