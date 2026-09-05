"""Tests for the hardware-free standard-DH forward-kinematics model."""

from __future__ import annotations

import math
import unittest

import numpy as np

from kinematics import (
    KinematicsError,
    UnreachableTargetError,
    UnconfirmedServoMappingError,
    dh_angles_to_servo_raw,
    forward_kinematics,
    inverse_kinematics_frame4,
    load_kinematics_config,
    servo_raw_to_dh_angles,
    standard_dh_transform,
)


class KinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_kinematics_config()

    def test_supplied_dh_and_servo_rows_are_loaded(self) -> None:
        self.assertEqual(self.config.dh_convention, "standard")
        self.assertEqual(
            tuple(joint.servo_model for joint in self.config.joints),
            ("DS3218", "DS5160", "DS5160", "MG995"),
        )
        self.assertEqual(self.config.gripper_servo_model, "MG995")
        np.testing.assert_allclose(
            [joint.a_mm for joint in self.config.joints],
            [0.0, 311.221, 288.187, 94.079],
        )
        np.testing.assert_allclose(
            [joint.d_mm for joint in self.config.joints],
            [149.6, 29.56, -40.917, 0.0],
        )

    def test_standard_dh_transform_has_expected_translation(self) -> None:
        transform = standard_dh_transform(10.0, 0.0, 20.0, 90.0)
        np.testing.assert_allclose(transform[:3, 3], [0.0, 10.0, 20.0], atol=1e-9)

    def test_zero_pose_matches_supplied_geometry(self) -> None:
        result = forward_kinematics((0.0, 0.0, 0.0, 0.0), self.config)
        np.testing.assert_allclose(
            result.position_mm,
            [693.487, 11.357, 149.6],
            atol=1e-6,
        )

    def test_base_rotation_rotates_the_chain_about_z(self) -> None:
        result = forward_kinematics((90.0, 0.0, 0.0, 0.0), self.config)
        np.testing.assert_allclose(
            result.position_mm,
            [-11.357, 693.487, 149.6],
            atol=1e-6,
        )

    def test_tool_rotation_is_orthonormal(self) -> None:
        rotation = forward_kinematics((20.0, -30.0, 40.0, 15.0), self.config).transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_servo_mapping_is_blocked_until_confirmed(self) -> None:
        with self.assertRaises(UnconfirmedServoMappingError):
            servo_raw_to_dh_angles((90.0, 90.0, 90.0, 90.0), self.config)

    def test_provisional_servo_mapping_matches_supplied_formula(self) -> None:
        angles = servo_raw_to_dh_angles(
            (0.0, 90.0, 135.0, 180.0),
            self.config,
            allow_unverified=True,
        )
        self.assertEqual(angles, (-90.0, 0.0, 45.0, 90.0))

    def test_inverse_servo_mapping_is_guarded_and_round_trips(self) -> None:
        with self.assertRaises(UnconfirmedServoMappingError):
            dh_angles_to_servo_raw((0.0, 0.0, 0.0, 0.0), self.config)
        raw = dh_angles_to_servo_raw(
            (-45.0, 0.0, 45.0, 90.0),
            self.config,
            allow_unverified=True,
        )
        self.assertEqual(raw, (45.0, 90.0, 135.0, 180.0))
        theta = servo_raw_to_dh_angles(raw, self.config, allow_unverified=True)
        np.testing.assert_allclose(theta, (-45.0, 0.0, 45.0, 90.0))

    def test_ik_returns_both_elbow_branches_and_round_trips_through_fk(self) -> None:
        original_angles = (20.0, -30.0, 60.0, -45.0)
        target = forward_kinematics(original_angles, self.config).position_mm
        solutions = inverse_kinematics_frame4(target, -15.0, self.config)

        self.assertEqual(
            tuple(solution.branch for solution in solutions),
            ("elbow_positive", "elbow_negative"),
        )
        for solution in solutions:
            checked = forward_kinematics(solution.theta_degrees, self.config)
            np.testing.assert_allclose(checked.position_mm, target, atol=1e-6)
            self.assertLessEqual(solution.position_error_mm, 1e-6)
        self.assertTrue(
            any(
                np.allclose(solution.theta_degrees, original_angles, atol=1e-6)
                for solution in solutions
            )
        )

    def test_ik_reports_provisional_servo_range_without_enforcing_it(self) -> None:
        target = forward_kinematics((0.0, 0.0, 0.0, 0.0), self.config).position_mm
        solution = inverse_kinematics_frame4(target, 0.0, self.config)[0]
        self.assertTrue(solution.within_provisional_servo_range)
        np.testing.assert_allclose(
            solution.provisional_servo_raw_degrees,
            (90.0, 90.0, 90.0, 90.0),
            atol=1e-6,
        )

    def test_ik_rejects_unreachable_targets(self) -> None:
        with self.assertRaises(UnreachableTargetError):
            inverse_kinematics_frame4((2000.0, 0.0, 0.0), -90.0, self.config)
        with self.assertRaises(UnreachableTargetError):
            inverse_kinematics_frame4((0.0, 0.0, 149.6), 0.0, self.config)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(KinematicsError):
            forward_kinematics((0.0, 0.0, 0.0), self.config)
        with self.assertRaises(KinematicsError):
            forward_kinematics((0.0, 0.0, math.inf, 0.0), self.config)
        with self.assertRaises(KinematicsError):
            servo_raw_to_dh_angles(
                (90.0, 181.0, 90.0, 90.0),
                self.config,
                allow_unverified=True,
            )


if __name__ == "__main__":
    unittest.main()
