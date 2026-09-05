"""Live Phase 4 contour, centroid, and target-selection test.

This program is vision-only. It never sends a robot command.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from .camera import Camera, CameraError
    from .config import (
        DEFAULT_DETECTION_CONFIG_PATH,
        DEFAULT_HSV_CONFIG_PATH,
        DEFAULT_VISION_CONFIG_PATH,
        CameraConfig,
        load_detection_config,
        load_hsv_calibration,
        load_vision_preprocessing_config,
    )
    from .detection import Detection, detect_cubes, select_target
    from .vision import _load_cv2, combine_color_masks, segment_colors
    from .workspace_calibration import (
        DEFAULT_WORKSPACE_CALIBRATION_PATH,
        PlanarHomography,
        load_workspace_calibration,
    )
except ImportError:  # pragma: no cover - direct script execution
    from camera import Camera, CameraError
    from config import (
        DEFAULT_DETECTION_CONFIG_PATH,
        DEFAULT_HSV_CONFIG_PATH,
        DEFAULT_VISION_CONFIG_PATH,
        CameraConfig,
        load_detection_config,
        load_hsv_calibration,
        load_vision_preprocessing_config,
    )
    from detection import Detection, detect_cubes, select_target
    from vision import _load_cv2, combine_color_masks, segment_colors
    from workspace_calibration import (
        DEFAULT_WORKSPACE_CALIBRATION_PATH,
        PlanarHomography,
        load_workspace_calibration,
    )


LOGGER = logging.getLogger(__name__)
DETECTION_WINDOW = "Phase 4 - Cube Candidates"
MASK_WINDOW = "Phase 4 - Combined Clean Mask"
DRAW_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


def _draw_detection(
    frame: Any,
    detection: Detection,
    selected: bool,
    cv2: Any,
) -> None:
    color = DRAW_COLORS[detection.color]
    x, y, width, height = detection.bbox
    thickness = 4 if selected else 2
    cv2.drawContours(frame, [detection.contour], -1, color, thickness)
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), color, thickness)
    cv2.circle(frame, (detection.cx, detection.cy), 5, (255, 255, 255), -1)
    prefix = "TARGET " if selected else ""
    label = (
        f"{prefix}{detection.color.upper()} px=({detection.cx},{detection.cy}) "
        f"area={detection.area:.0f} conf={detection.confidence:.2f}"
    )
    if detection.robot_x is not None and detection.robot_y is not None:
        label += f" floor=({detection.robot_x:.1f},{detection.robot_y:.1f})mm"
    label_y = max(20, y - 8)
    cv2.putText(
        frame,
        label,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "picamera2", "opencv"), default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fourcc", choices=("AUTO", "MJPG", "YUYV"), default="MJPG")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--read-attempts", type=int, default=3)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--hsv-config", type=Path, default=DEFAULT_HSV_CONFIG_PATH)
    parser.add_argument("--vision-config", type=Path, default=DEFAULT_VISION_CONFIG_PATH)
    parser.add_argument("--detection-config", type=Path, default=DEFAULT_DETECTION_CONFIG_PATH)
    parser.add_argument(
        "--workspace-calibration",
        type=Path,
        default=DEFAULT_WORKSPACE_CALIBRATION_PATH,
        help=(
            "Calibrated pixel-to-floor map (enabled by default); camera index, "
            "resolution, and rotation must match"
        ),
    )
    parser.add_argument(
        "--pixel-only",
        action="store_true",
        help="Disable floor-coordinate mapping and leave robot_x/robot_y empty",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open windows; suitable for a Raspberry Pi over SSH",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many frames; headless mode defaults to 60",
    )
    parser.add_argument("--save-frame", type=Path, help="Save the final annotated frame")
    parser.add_argument("--save-mask", type=Path, help="Save the final combined mask")
    return parser.parse_args(argv)


def prepare_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate finite-run settings and apply the safe headless default."""

    if args.frames < 0:
        raise ValueError("Frame count cannot be negative")
    if args.headless and args.frames == 0:
        args.frames = 60
    return args


def map_detections_to_floor(
    detections: list[Detection],
    workspace: PlanarHomography,
) -> list[Detection]:
    """Return immutable detection copies populated with floor millimetres."""

    mapped: list[Detection] = []
    for detection in detections:
        x_mm, y_mm = workspace.pixel_to_table(detection.cx, detection.cy)
        mapped.append(replace(detection, robot_x=x_mm, robot_y=y_mm))
    return mapped


def load_runtime_workspace(args: argparse.Namespace) -> PlanarHomography | None:
    """Load the map unless pixel-only mode was explicitly requested."""

    if args.pixel_only:
        LOGGER.warning("Floor-coordinate mapping disabled; reporting pixels only")
        return None
    output_resolution = (
        (args.height, args.width)
        if args.rotation in (90, 270)
        else (args.width, args.height)
    )
    workspace = load_workspace_calibration(
        args.workspace_calibration,
        expected_camera_index=args.camera_index,
        expected_resolution=output_resolution,
        expected_rotation_degrees=args.rotation,
    )
    LOGGER.info(
        "Floor-coordinate mapping enabled from %s (RMS fit error %.2f mm)",
        args.workspace_calibration,
        workspace.rms_error_mm,
    )
    return workspace


def run_detection_test(args: argparse.Namespace, cv2: Any) -> None:
    args = prepare_runtime_args(args)

    camera_config = CameraConfig(
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
    hsv_calibration = load_hsv_calibration(args.hsv_config)
    preprocessing = load_vision_preprocessing_config(args.vision_config)
    detection_config = load_detection_config(args.detection_config)
    workspace = load_runtime_workspace(args)
    previous_signature: tuple[tuple[object, ...], ...] | None = None
    frame_count = 0
    started_at = time.monotonic()
    last_preview: Any | None = None
    last_combined_mask: Any | None = None
    last_detections: list[Detection] = []

    with Camera(camera_config) as camera:
        while True:
            frame = camera.read()
            segmentation = segment_colors(
                frame,
                hsv_calibration,
                preprocessing,
                cv2_module=cv2,
            )
            detections = detect_cubes(
                segmentation.masks,
                detection_config,
                cv2_module=cv2,
            )
            if workspace is not None:
                detections = map_detections_to_floor(detections, workspace)
            reference_xy = (frame.shape[1] / 2.0, frame.shape[0] - 1.0)
            target = select_target(
                detections,
                detection_config,
                reference_xy=reference_xy,
            )

            signature = tuple(
                (item.color, item.cx, item.cy, round(item.area, -1))
                for item in detections
            )
            if signature != previous_signature:
                LOGGER.info("Detections: %s", [item.to_dict() for item in detections])
                previous_signature = signature

            preview = frame.copy()
            for detection in detections:
                _draw_detection(preview, detection, detection is target, cv2)

            frame_count += 1
            measured_fps = frame_count / max(time.monotonic() - started_at, 1e-6)
            state_text = (
                f"VISION ONLY | detections={len(detections)} | "
                f"coords={'floor-mm' if workspace is not None else 'pixels-only'} | "
                f"policy={detection_config.target_selection_policy} | fps={measured_fps:.1f}"
            )
            cv2.putText(
                preview,
                state_text,
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            combined_mask = combine_color_masks(segmentation.masks, cv2_module=cv2)
            last_preview = preview
            last_combined_mask = combined_mask
            last_detections = detections

            if not args.headless:
                cv2.imshow(DETECTION_WINDOW, preview)
                cv2.imshow(MASK_WINDOW, combined_mask)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            if args.frames and frame_count >= args.frames:
                break

    if last_preview is None or last_combined_mask is None:
        raise RuntimeError("Detection test ended without processing a frame")
    for output_path, image, label in (
        (args.save_frame, last_preview, "annotated frame"),
        (args.save_mask, last_combined_mask, "combined mask"),
    ):
        if output_path is None:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save {label} to {output_path}")
        LOGGER.info("Saved %s to %s", label, output_path)
    LOGGER.info(
        "PASS: processed %s frames; final detections=%s",
        frame_count,
        [item.to_dict() for item in last_detections],
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    try:
        cv2 = _load_cv2()
        run_detection_test(args, cv2)
        return 0
    except (CameraError, RuntimeError, ValueError) as exc:
        LOGGER.error("DETECTION TEST FAILED: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Detection test stopped by user")
        return 0
    finally:
        if "cv2" in locals() and "args" in locals() and not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
