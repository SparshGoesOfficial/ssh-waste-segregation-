"""Test ARM_SERIAL_V1 on a real or dummy Arduino Uno."""

from __future__ import annotations

import argparse
import sys

from serial_controller import ArmSerialController, SerialControllerError


SAFETY_CONFIRMATION = "NO_SERVOS_CONNECTED"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Arduino port, for example COM5")
    parser.add_argument(
        "--dummy-motion-test",
        action="store_true",
        help="Exercise ARM/MOVE/JOINT/GRIP only on a board with no servos connected",
    )
    parser.add_argument(
        "--safety-confirmation",
        default="",
        help=f"Required with --dummy-motion-test: {SAFETY_CONFIRMATION}",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    if args.dummy_motion_test and args.safety_confirmation != SAFETY_CONFIRMATION:
        raise SerialControllerError(
            "Dummy motion test refused: confirm that no servo signal wires are connected"
        )

    controller = ArmSerialController.open(args.port)
    try:
        print(controller.ping())
        print(controller.status())
        print(controller.limits())
        if args.dummy_motion_test:
            print("DUMMY BOARD ONLY: exercising commands with no servos connected")
            print(controller.arm(controller.config.startup_angles))
            print(controller.move_joint(1, 91, 300))
            print(controller.move_joint(1, 90, 300))
            print(controller.grip(False, 300))
            print(controller.grip(True, 300))
            print(controller.status())
            print(controller.disarm())
        print("PASS: Arduino serial protocol is responsive")
    finally:
        controller.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
        return 0
    except SerialControllerError as exc:
        print(f"SERIAL LINK TEST FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
