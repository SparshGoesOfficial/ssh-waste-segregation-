"""Print a hardware-free forward-kinematics result for four joint angles."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from kinematics import (
    KinematicsError,
    forward_kinematics,
    inverse_kinematics_frame4,
    load_kinematics_config,
    servo_raw_to_dh_angles,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--joint-angles",
        nargs=4,
        type=float,
        metavar=("J1", "J2", "J3", "J4"),
        default=(0.0, 0.0, 0.0, 0.0),
        help="DH joint angles in degrees",
    )
    mode.add_argument(
        "--servo-angles",
        nargs=4,
        type=float,
        metavar=("J1", "J2", "J3", "J4"),
        help="Raw servo angles in degrees; simulation only while unconfirmed",
    )
    mode.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Solve IK for the frame-4 origin at this base-frame position",
    )
    parser.add_argument(
        "--tool-pitch",
        type=float,
        default=-90.0,
        help="Desired a4 pitch above horizontal in degrees (default: -90)",
    )
    parser.add_argument(
        "--allow-unverified-servo-map",
        action="store_true",
        help="Explicitly permit the provisional raw-servo conversion in simulation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = load_kinematics_config()
        if args.target is not None:
            solutions = inverse_kinematics_frame4(
                args.target,
                args.tool_pitch,
                model,
            )
            _print_ik_solutions(args.target, args.tool_pitch, solutions)
            return 0

        angles = tuple(args.joint_angles)
        if args.servo_angles is not None:
            angles = servo_raw_to_dh_angles(
                args.servo_angles,
                model,
                allow_unverified=args.allow_unverified_servo_map,
            )
        result = forward_kinematics(angles, model)
    except KinematicsError as exc:
        print(f"KINEMATICS TEST FAILED: {exc}", file=sys.stderr)
        return 1

    x_mm, y_mm, z_mm = result.position_mm
    print("DRY RUN: mathematical model only; no hardware commands")
    print(f"Model status: {model.status}; DH convention: {model.dh_convention}")
    print("DH joint angles (deg): " + ", ".join(f"{value:.3f}" for value in angles))
    print(f"Frame-4 origin (mm): x={x_mm:.3f}, y={y_mm:.3f}, z={z_mm:.3f}")
    print("Joint-frame origins (mm):")
    for index, origin in enumerate(result.joint_origins_mm):
        print(f"  frame {index}: ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})")
    print("Frame-4 transform:")
    print(np.array2string(result.transform, precision=6, suppress_small=True))
    return 0


def _print_ik_solutions(
    target: list[float],
    tool_pitch_deg: float,
    solutions: tuple,
) -> None:
    print("DRY RUN: mathematical model only; no hardware commands")
    print(
        "Frame-4 target (mm): "
        f"x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}"
    )
    print(f"Tool pitch (deg): {tool_pitch_deg:.3f}")
    for solution in solutions:
        print(f"{solution.branch}:")
        print(
            "  DH joint angles (deg): "
            + ", ".join(f"{value:.3f}" for value in solution.theta_degrees)
        )
        print(
            "  provisional raw servo angles (deg): "
            + ", ".join(
                f"{value:.3f}" for value in solution.provisional_servo_raw_degrees
            )
        )
        print(
            "  within provisional 0..180 ranges: "
            f"{solution.within_provisional_servo_range}"
        )
        print(f"  IK->FK position error (mm): {solution.position_error_mm:.9f}")


if __name__ == "__main__":
    sys.exit(main())
