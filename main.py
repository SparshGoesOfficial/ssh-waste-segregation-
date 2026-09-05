"""Phase 1 live camera test.

No robot motion is performed by this program.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

try:  # Support module and direct-script invocation.
    from .camera import Camera, CameraError
    from .config import CameraConfig
except ImportError:  # pragma: no cover - used only by direct script execution
    from camera import Camera, CameraError
    from config import CameraConfig


LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "SNU OpenCV - Phase 1 Camera Test"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse safe camera-test options; there are no robot commands here."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "picamera2", "opencv"),
        default="opencv",
        help="Camera API to use; OpenCV is the Logitech C270 default",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--fourcc",
        type=str.upper,
        choices=("AUTO", "MJPG", "YUYV"),
        default="MJPG",
        help="USB-camera pixel format; MJPG enables C270 HD capture",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--read-attempts", type=int, default=3)
    parser.add_argument(
        "--rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Clockwise image rotation for the physical camera mount",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Capture without opening a display window",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 runs until Q/Esc/Ctrl+C",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        help="Save the last captured BGR frame to this path",
    )
    return parser.parse_args(argv)


def run_camera_test(args: argparse.Namespace) -> int:
    """Run a live preview or a finite headless capture test."""

    if args.headless and args.frames == 0:
        args.frames = 30

    config = CameraConfig(
        backend=args.backend,
        width=args.width,
        height=args.height,
        camera_index=args.camera_index,
        opencv_fourcc=None if args.fourcc == "AUTO" else args.fourcc,
        frame_rate=args.fps,
        warmup_seconds=args.warmup_seconds,
        read_attempts=args.read_attempts,
        rotation_degrees=args.rotation,
    )

    frame_count = 0
    last_frame = None
    started_at = time.monotonic()

    try:
        with Camera(config) as camera:
            LOGGER.info("Camera test running with backend=%s", camera.backend_name)
            while args.frames == 0 or frame_count < args.frames:
                frame = camera.read()
                frame_count += 1
                last_frame = frame
                elapsed = max(time.monotonic() - started_at, 1e-6)
                measured_fps = frame_count / elapsed

                if not args.headless:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"backend={camera.backend_name}  frame={frame_count}  "
                        f"fps={measured_fps:.1f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_NAME, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

        if last_frame is None:
            raise CameraError("The test ended before any frame was captured")

        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_frame), last_frame):
                raise CameraError(f"Could not save frame to {args.save_frame}")
            LOGGER.info("Saved last frame to %s", args.save_frame)

        height, width, channels = last_frame.shape
        LOGGER.info(
            "PASS: captured %s frames; last frame=%sx%s, channels=%s, dtype=%s",
            frame_count,
            width,
            height,
            channels,
            last_frame.dtype,
        )
        return 0
    except CameraError as exc:
        LOGGER.error("CAMERA TEST FAILED: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Camera test stopped by user after %s frames", frame_count)
        return 0
    finally:
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        return run_camera_test(parse_args(argv))
    except ValueError as exc:
        LOGGER.error("Invalid camera configuration: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
