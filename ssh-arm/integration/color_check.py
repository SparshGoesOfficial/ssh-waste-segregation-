"""Confirm the expected cube colour is present in the PICK zone.

The classifier decides the bin from waste held in INSPECT; this module answers the
separate question "is the matching cube actually sitting in PICK?". It is a
presence check, not a localizer — no centroid is used downstream, nothing is
mapped to robot coordinates, nothing is commanded.

    from color_check import colors_present
    found = colors_present(frame, zones["PICK"])     # -> {"red", "blue"}

Every threshold belongs to the teammate's code. We supply exactly one thing: the
ROI, which is our PICK rect. Specifically:

  HSV ranges      hsv_ranges_local.json  (our rig's copy; never their tracked file)
  contour filters their calibration/detection_settings.json  (broad dev values)
  blur/morphology their calibration/vision_settings.json, with the ROI replaced

Test mode shows the PICK crop, its four masks, and the live colour set:

    python color_check.py --camera-index N

    q/Esc quit    d dump detections    w write a snapshot png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import contract
from teammate_vision import (
    ImageROI,
    VisionPreprocessingConfig,
    detect_cubes,
    load_detection_config,
    load_hsv_calibration,
    load_vision_preprocessing_config,
    segment_colors,
)

ROOT = Path(__file__).resolve().parent
TEAMMATE = ROOT / "ssh-waste-segregation-"

# Our copy. Their calibration/hsv_ranges.json is git-tracked and holds red as
# verified at their venue; it is read-only to us and is never the default here.
HSV_PATH = ROOT / "hsv_ranges_local.json"
THEIR_HSV_PATH = TEAMMATE / "calibration" / "hsv_ranges.json"

# Their tuning, used as-is. These are broad development values by their own
# admission, which is the right starting point — inventing our own filters would
# just be guessing with extra steps.
DETECTION_PATH = TEAMMATE / "calibration" / "detection_settings.json"
VISION_PATH = TEAMMATE / "calibration" / "vision_settings.json"

COLORS = tuple(contract.BIN_TO_COLOR[b] for b in sorted(contract.BIN_TO_COLOR))


@dataclass
class PickZoneResult:
    """What the PICK zone contained on one frame."""

    colors: set[str]                    # colours with >= 1 surviving detection
    detections: list[Any]               # raw teammate Detection objects, full-frame coords
    masks: dict[str, Any]               # full-frame binary mask per colour
    roi: ImageROI

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in COLORS}
        for d in self.detections:
            out[d.color] = out.get(d.color, 0) + 1
        return out


# --- pipeline construction (loaded once, reused every frame) -------------------

_pipeline: dict[str, Any] | None = None


def load_pipeline(hsv_path: Path = HSV_PATH, *, verbose: bool = True) -> dict[str, Any]:
    """Load the teammate's calibration once and report what is actually tuned."""
    global _pipeline
    if _pipeline is not None and _pipeline["hsv_path"] == hsv_path:
        return _pipeline

    if hsv_path.resolve() == THEIR_HSV_PATH.resolve():
        raise ValueError(
            f"refusing to use {THEIR_HSV_PATH} — that file is git-tracked and holds "
            "their venue calibration. Use hsv_ranges_local.json."
        )
    if not hsv_path.exists():
        raise FileNotFoundError(
            f"{hsv_path} not found. Copy their calibration/hsv_ranges.json to the "
            "repo root as hsv_ranges_local.json, then calibrate against it."
        )

    calibration = load_hsv_calibration(hsv_path)
    detection_config = load_detection_config(DETECTION_PATH)
    # Their blur/morphology values are fine; only the ROI is ours to set, and it
    # is replaced per call with the PICK rect.
    vision_base = load_vision_preprocessing_config(VISION_PATH)

    _pipeline = {
        "hsv_path": hsv_path,
        "calibration": calibration,
        "detection_config": detection_config,
        "vision_base": vision_base,
    }
    if verbose:
        report_calibration(calibration, hsv_path)
    return _pipeline


def report_calibration(calibration, hsv_path: Path = HSV_PATH) -> set[str]:
    """Print which colours are calibrated and warn loudly about the rest.

    Returns the uncalibrated set. This is the point of the exercise: a mask tuned
    at their venue under their lamp is not a mask tuned at ours, and an untuned
    one was never right anywhere.
    """
    done = set(calibration.calibrated_colors)
    missing = [c for c in COLORS if c not in done]

    print(f"HSV calibration  : {hsv_path.name}")
    print(f"  calibrated     : {sorted(done) or 'NONE'}")
    print(f"  NOT calibrated : {missing or 'none'}")
    if missing:
        print()
        print(f"  !! WARNING: {len(missing)} of {len(COLORS)} colours are running on")
        print("  !! provisional thresholds that were never tuned on any rig.")
        for color in missing:
            bin_id = contract.COLOR_TO_BIN[color]
            print(f"  !!   {color:<7} -> bin {bin_id} ({contract.BIN_TO_CLASS[bin_id]}) "
                  "will mis-detect under your lamp")
        print("  !! Recalibrate before trusting a MATCH or a NO CUBE from these.")
        print()
    return set(missing)


def _as_roi(pick_rect) -> ImageROI:
    """Accept an ImageROI or a plain {x, y, width, height} dict."""
    if isinstance(pick_rect, ImageROI):
        return pick_rect
    if isinstance(pick_rect, dict):
        return ImageROI.from_dict(pick_rect)
    raise TypeError(f"pick_rect must be an ImageROI or dict, got {type(pick_rect).__name__}")


# --- the check ----------------------------------------------------------------


def check_pick_zone(frame, pick_rect, *, hsv_path: Path = HSV_PATH) -> PickZoneResult:
    """Segment and detect inside the PICK rect. Returns colours plus raw detections."""
    roi = _as_roi(pick_rect)
    roi.validate_for_frame(frame.shape[1], frame.shape[0])
    pipe = load_pipeline(hsv_path)

    # The ROI is the only thing we override — their blur and morphology stand.
    vision_config: VisionPreprocessingConfig = replace(pipe["vision_base"], roi=roi)

    segmentation = segment_colors(frame, pipe["calibration"], vision_config)
    detections = detect_cubes(segmentation.masks, pipe["detection_config"])

    return PickZoneResult(
        colors={d.color for d in detections},
        detections=detections,
        masks=segmentation.masks,
        roi=segmentation.roi,
    )


def colors_present(frame, pick_rect) -> set[str]:
    """Colours with at least one surviving detection inside the PICK rect.

    Thin wrapper over check_pick_zone(); use that when you want the raw
    detections for debugging.
    """
    return check_pick_zone(frame, pick_rect).colors


# --- test mode ----------------------------------------------------------------

TILE_W, TILE_H = 420, 320
SWATCH = {"red": (0, 0, 255), "green": (0, 200, 0), "blue": (255, 0, 0), "yellow": (0, 220, 220)}
WHITE = (255, 255, 255)
GREY = (110, 110, 110)


def _fit(img, w=TILE_W, h=TILE_H):
    """Letterbox into a tile without distorting aspect — masks must read honestly."""
    import cv2
    import numpy as np

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _label(tile, text, color=WHITE):
    import cv2

    cv2.rectangle(tile, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    cv2.putText(tile, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), GREY, 1)
    return tile


def _build_view(frame, result: PickZoneResult, uncalibrated: set[str]):
    """PICK crop with detections, the four masks, and the live colour set."""
    import cv2
    import numpy as np

    roi = result.roi
    crop = frame[roi.y:roi.bottom, roi.x:roi.right].copy()

    # Detections are in full-frame coords; shift them into the crop.
    for d in result.detections:
        bx, by, bw, bh = d.bbox
        cv2.rectangle(crop, (bx - roi.x, by - roi.y), (bx - roi.x + bw, by - roi.y + bh),
                      SWATCH.get(d.color, WHITE), 2)
        cv2.circle(crop, (d.cx - roi.x, d.cy - roi.y), 4, SWATCH.get(d.color, WHITE), -1)
        cv2.putText(crop, f"{d.color} {int(d.area)}", (bx - roi.x, max(12, by - roi.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, SWATCH.get(d.color, WHITE), 2, cv2.LINE_AA)

    counts = result.counts()
    crop_tile = _label(_fit(crop), f"PICK crop {roi.width}x{roi.height} @({roi.x},{roi.y})")

    mask_tiles = []
    for color in COLORS:
        m = result.masks[color][roi.y:roi.bottom, roi.x:roi.right]
        pct = 100.0 * float((m > 0).sum()) / max(1, m.size)
        flag = "" if color not in uncalibrated else "  UNCALIBRATED"
        tile = _label(_fit(m), f"{color}  n={counts[color]}  {pct:4.1f}%{flag}",
                      SWATCH[color] if not flag else (0, 165, 255))
        mask_tiles.append(tile)

    masks_grid = np.vstack([np.hstack(mask_tiles[:2]), np.hstack(mask_tiles[2:])])

    # Status column beside the crop.
    panel = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    found = sorted(result.colors)
    cv2.putText(panel, "colours present:", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, WHITE, 2, cv2.LINE_AA)
    if found:
        for i, c in enumerate(found):
            y = 74 + i * 40
            cv2.rectangle(panel, (14, y - 20), (44, y + 6), SWATCH[c], -1)
            note = "  (uncal.)" if c in uncalibrated else ""
            cv2.putText(panel, f"{c}{note}", (58, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "(none)", (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
    cv2.putText(panel, f"{len(result.detections)} detections", (12, TILE_H - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.putText(panel, "q quit  d dump  w write png", (12, TILE_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
    _label(panel, "live colour set")

    return np.vstack([np.hstack([crop_tile, panel]), masks_grid])


def run_test_mode(camera_index: int, zones_path: Path, hsv_path: Path) -> int:
    import cv2

    import zone_config

    zones = zone_config.load_zones(zones_path)
    pick = zones["PICK"]
    print(f"zones: {zone_config.describe(zones)}")
    print(f"PICK rect used for segmentation: {pick.to_dict()}")
    print()

    pipe = load_pipeline(hsv_path)
    uncalibrated = set(COLORS) - set(pipe["calibration"].calibrated_colors)
    print(f"detection filters: {DETECTION_PATH.name} (their broad dev values)")
    print(f"blur/morphology  : {VISION_PATH.name}, ROI overridden with PICK")
    print()

    win = "PICK zone colour check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last: set[str] | None = None

    with zone_config.open_camera(camera_index) as cam:
        first = True
        while True:
            frame = cam.read()
            if first:
                zone_config.assert_frame_size(frame)
                first = False

            result = check_pick_zone(frame, pick, hsv_path=hsv_path)
            if result.colors != last:            # only on change, not every frame
                shown = sorted(result.colors) or ["none"]
                warn = sorted(result.colors & uncalibrated)
                suffix = f"   (uncalibrated: {warn})" if warn else ""
                print(f"pick zone: {shown}{suffix}")
                last = result.colors

            cv2.imshow(win, _build_view(frame, result, uncalibrated))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                if not result.detections:
                    print("  no detections")
                for d in result.detections:
                    print("  ", d.to_dict())
            elif key == ord("w"):
                out = ROOT / "pick_zone_snapshot.png"
                cv2.imwrite(str(out), _build_view(frame, result, uncalibrated))
                print(f"  wrote {out}")

    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Colour presence check for the PICK zone.")
    ap.add_argument("--camera-index", type=int,
                    help="camera index for live test mode (omit to just print calibration status)")
    ap.add_argument("--zones", type=Path, default=ROOT / "zones.json")
    ap.add_argument("--hsv", type=Path, default=HSV_PATH)
    args = ap.parse_args(argv)

    if args.camera_index is None:
        load_pipeline(args.hsv)
        print("no --camera-index given; calibration status only, camera not opened")
        return 0
    return run_test_mode(args.camera_index, args.zones, args.hsv)


if __name__ == "__main__":
    sys.exit(main())
