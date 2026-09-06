"""The two zones in the frame: INSPECT (real waste) and PICK (the taped cube square).

One JSON at the repo root (zones.json), authored in full-frame 1280x720 pixel
coordinates. Keys are x/y/width/height so each zone feeds ImageROI.from_dict()
straight through and inherits their bounds checking.

    import zone_config
    zones = zone_config.load_zones()      # {"INSPECT": ImageROI, "PICK": ImageROI}

To set the boxes, run this file and drag them:

    python zone_config.py --camera-index N

    i / p        select INSPECT or PICK          tab  cycle
    drag inside  move the selected box           drag a corner  resize
    drag empty   redraw the selected box
    r            reset selected box to default
    s            save to zones.json              q / Esc  quit

Two things this file refuses to do silently:
  - inherit the teammate's 640x480 CameraConfig default (we demand real 1280x720)
  - accept their ROI (205,20,840,700), which was tuned on their rig, not ours
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from teammate_vision import Camera, CameraConfig, ImageROI

ROOT = Path(__file__).resolve().parent
ZONES_PATH = ROOT / "zones.json"

# The frame geometry these zones are authored against. Stored in the JSON and
# re-checked on load, so zones drawn at 1280x720 can never be applied to a
# 640x480 stream without a loud failure.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

ZONE_NAMES = ("INSPECT", "PICK")
MIN_SIDE = 40  # px; smaller than this is a mis-drag, not a zone

# Their rig's ROI, from ssh-waste-segregation-/calibration/vision_settings.json.
# Recorded only so we can reject it if it ever shows up in our zones.json.
THEIR_ROI = {"x": 205, "y": 20, "width": 840, "height": 700}

# Placeholders so the editor has something to grab on a first run. They are a
# left/right split of the frame, not a recommendation — the real numbers depend
# on where the tripod ends up.
DEFAULT_ZONES = {
    "INSPECT": {"x": 60, "y": 140, "width": 480, "height": 440},
    "PICK": {"x": 700, "y": 140, "width": 520, "height": 440},
}


# --- geometry -----------------------------------------------------------------


def _overlap(a: ImageROI, b: ImageROI) -> bool:
    """True if two ROIs share any pixel."""
    return a.x < b.right and b.x < a.right and a.y < b.bottom and b.y < a.bottom


def _overlap_area(a: ImageROI, b: ImageROI) -> int:
    w = max(0, min(a.right, b.right) - max(a.x, b.x))
    h = max(0, min(a.bottom, b.bottom) - max(a.y, b.y))
    return w * h


# --- load / save --------------------------------------------------------------


def load_zones(
    path: Path = ZONES_PATH,
    *,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
) -> dict[str, ImageROI]:
    """Read zones.json and hand back validated ImageROIs.

    Raises rather than asserts: `assert` is stripped under `python -O`, and an
    overlapping INSPECT/PICK pair would then classify the cube it is supposed to
    be checking against. This check must survive every interpreter flag.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the zones have not been set yet. "
            f"Run:  python {Path(__file__).name} --camera-index N"
        )

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    missing = [n for n in ZONE_NAMES if n not in payload]
    if missing:
        raise ValueError(f"{path} is missing zone(s) {missing}; needs {list(ZONE_NAMES)}")

    # Zones are resolution-specific. Refuse to apply 1280x720 boxes to another size.
    frame = payload.get("frame")
    if isinstance(frame, dict):
        authored = (int(frame.get("width", -1)), int(frame.get("height", -1)))
        if authored != (frame_width, frame_height):
            raise ValueError(
                f"{path} was authored for {authored[0]}x{authored[1]} but is being "
                f"loaded for {frame_width}x{frame_height}. Re-run the zone editor."
            )

    zones: dict[str, ImageROI] = {}
    for name in ZONE_NAMES:
        try:
            roi = ImageROI.from_dict(payload[name])
        except ValueError as exc:
            raise ValueError(f"{path}: zone {name} is malformed — {exc}") from exc
        roi.validate_for_frame(frame_width, frame_height)  # their bounds check
        if roi.width < MIN_SIDE or roi.height < MIN_SIDE:
            raise ValueError(
                f"{path}: zone {name} is {roi.width}x{roi.height}, smaller than the "
                f"{MIN_SIDE}px minimum — probably a stray click. Redraw it."
            )
        if roi.to_dict() == THEIR_ROI:
            raise ValueError(
                f"{path}: zone {name} is the teammate's ROI {THEIR_ROI}. That box was "
                "calibrated on their rig under their lighting. Draw your own."
            )
        zones[name] = roi

    inspect, pick = zones["INSPECT"], zones["PICK"]
    if _overlap(inspect, pick):
        raise ValueError(
            f"INSPECT {inspect.to_dict()} and PICK {pick.to_dict()} overlap by "
            f"{_overlap_area(inspect, pick)}px. They must be disjoint: waste held in "
            "INSPECT would otherwise be segmented as a cube in PICK."
        )
    return zones


def save_zones(
    zones: dict[str, ImageROI],
    path: Path = ZONES_PATH,
    *,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
) -> None:
    """Write zones.json, refusing anything load_zones() would reject."""
    inspect, pick = zones["INSPECT"], zones["PICK"]
    if _overlap(inspect, pick):
        raise ValueError(f"refusing to save overlapping zones ({_overlap_area(inspect, pick)}px)")
    for name, roi in zones.items():
        roi.validate_for_frame(frame_width, frame_height)

    payload = {
        "_comment": "Full-frame pixel zones for this rig. Set with zone_config.py.",
        "frame": {"width": frame_width, "height": frame_height},
        "INSPECT": zones["INSPECT"].to_dict(),
        "PICK": zones["PICK"].to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def describe(zones: dict[str, ImageROI]) -> str:
    return "  ".join(
        f"{n}=({r.x},{r.y},{r.width}x{r.height})" for n, r in zones.items()
    )


# --- camera -------------------------------------------------------------------


def open_camera(camera_index: int, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> Camera:
    """Open the camera at an explicitly stated resolution.

    Every field is passed on purpose. CameraConfig() defaults to 640x480 on
    index 0, and inheriting that silently would put the zones in the wrong
    coordinate space.
    """
    config = CameraConfig(
        backend="opencv",
        width=width,
        height=height,
        camera_index=camera_index,
        frame_rate=30.0,
        rotation_degrees=0,
    )
    print(f"camera: index {config.camera_index}, {config.width}x{config.height}, "
          f"backend={config.backend}, fourcc={config.opencv_fourcc}")
    return Camera(config)


def assert_frame_size(frame, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> None:
    """Fail loudly if the camera quietly handed back a different resolution."""
    h, w = frame.shape[:2]
    if (w, h) != (width, height):
        raise SystemExit(
            f"camera delivered {w}x{h}, not the requested {width}x{height}. Zones "
            "would be authored in the wrong coordinate space. Pick a camera index "
            "that supports 720p, or change FRAME_WIDTH/FRAME_HEIGHT together."
        )


# --- editor -------------------------------------------------------------------

HANDLE = 14  # px corner grab radius
COLORS = {"INSPECT": (255, 210, 0), "PICK": (255, 0, 255)}  # BGR
BAD = (0, 0, 255)
WHITE = (255, 255, 255)


class _Editor:
    def __init__(self, zones: dict[str, list[int]]) -> None:
        self.zones = zones                 # name -> [x, y, w, h]
        self.active = "INSPECT"
        self.drag = None                   # (mode, name, payload)
        self.dirty = False

    # -- hit testing
    def _corner_at(self, name, mx, my):
        x, y, w, h = self.zones[name]
        for idx, (cx, cy) in enumerate(((x, y), (x + w, y), (x, y + h), (x + w, y + h))):
            if abs(mx - cx) <= HANDLE and abs(my - cy) <= HANDLE:
                return idx
        return None

    def _inside(self, name, mx, my):
        x, y, w, h = self.zones[name]
        return x <= mx <= x + w and y <= my <= y + h

    def on_mouse(self, event, mx, my, flags, _param):
        import cv2

        if event == cv2.EVENT_LBUTTONDOWN:
            order = [self.active] + [n for n in ZONE_NAMES if n != self.active]
            for name in order:
                corner = self._corner_at(name, mx, my)
                if corner is not None:
                    self.active = name
                    x, y, w, h = self.zones[name]
                    anchor = ((x + w, y + h), (x, y + h), (x + w, y), (x, y))[corner]
                    self.drag = ("resize", name, anchor)
                    return
            for name in order:
                if self._inside(name, mx, my):
                    self.active = name
                    x, y, _, _ = self.zones[name]
                    self.drag = ("move", name, (mx - x, my - y))
                    return
            self.drag = ("draw", self.active, (mx, my))

        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            mode, name, payload = self.drag
            if mode == "move":
                dx, dy = payload
                _, _, w, h = self.zones[name]
                self.zones[name] = self._clamp([mx - dx, my - dy, w, h])
            else:  # resize / draw both rubber-band from an anchor point
                ax, ay = payload
                x0, x1 = sorted((ax, mx))
                y0, y1 = sorted((ay, my))
                self.zones[name] = self._clamp([x0, y0, x1 - x0, y1 - y0])
            self.dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    @staticmethod
    def _clamp(rect):
        x, y, w, h = (int(v) for v in rect)
        w, h = max(MIN_SIDE, w), max(MIN_SIDE, h)
        w, h = min(w, FRAME_WIDTH), min(h, FRAME_HEIGHT)
        x = max(0, min(x, FRAME_WIDTH - w))
        y = max(0, min(y, FRAME_HEIGHT - h))
        return [x, y, w, h]

    def as_rois(self):
        return {n: ImageROI(x=r[0], y=r[1], width=r[2], height=r[3]) for n, r in self.zones.items()}

    def overlapping(self):
        r = self.as_rois()
        return _overlap_area(r["INSPECT"], r["PICK"])


def run_editor(camera_index: int, path: Path = ZONES_PATH) -> int:
    import cv2

    if path.exists():
        try:
            start = {n: [r.x, r.y, r.width, r.height] for n, r in load_zones(path).items()}
            print(f"loaded existing zones from {path.name}")
        except (ValueError, FileNotFoundError) as exc:
            print(f"existing {path.name} unusable ({exc}); starting from defaults")
            start = {n: list(DEFAULT_ZONES[n].values()) for n in ZONE_NAMES}
    else:
        print(f"no {path.name} yet — starting from placeholder boxes; drag them onto your rig")
        start = {n: list(DEFAULT_ZONES[n].values()) for n in ZONE_NAMES}

    ed = _Editor(start)
    win = "zone editor - i/p select, drag, s save, q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(win, ed.on_mouse)

    saved_msg = ""
    with open_camera(camera_index) as cam:
        first = True
        while True:
            frame = cam.read()
            if first:
                assert_frame_size(frame)
                print(f"frame confirmed {frame.shape[1]}x{frame.shape[0]}")
                first = False

            view = frame.copy()
            bad = ed.overlapping()

            for name in ZONE_NAMES:
                x, y, w, h = ed.zones[name]
                is_active = name == ed.active
                col = BAD if bad else COLORS[name]
                cv2.rectangle(view, (x, y), (x + w, y + h), col, 3 if is_active else 2)
                tag = f"{name}{' *' if is_active else ''}  {w}x{h} @({x},{y})"
                cv2.putText(view, tag, (x + 6, max(22, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
                if is_active:
                    for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
                        cv2.rectangle(view, (cx - 7, cy - 7), (cx + 7, cy + 7), col, -1)

            lines = [
                "i/p or tab select | drag inside=move, corner=resize, empty=redraw",
                "r reset selected | s save | q quit",
            ]
            if bad:
                lines.append(f"OVERLAP {bad}px - cannot save until disjoint")
            if saved_msg:
                lines.append(saved_msg)
            for i, text in enumerate(lines):
                cv2.putText(view, text, (14, FRAME_HEIGHT - 70 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            BAD if text.startswith("OVERLAP") else WHITE, 2, cv2.LINE_AA)

            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                if ed.dirty:
                    print("quit without saving — zones.json unchanged")
                break
            elif key == ord("i"):
                ed.active = "INSPECT"
            elif key == ord("p"):
                ed.active = "PICK"
            elif key == 9:  # tab
                ed.active = "PICK" if ed.active == "INSPECT" else "INSPECT"
            elif key == ord("r"):
                ed.zones[ed.active] = list(DEFAULT_ZONES[ed.active].values())
                ed.dirty = True
            elif key == ord("s"):
                try:
                    save_zones(ed.as_rois(), path)
                except ValueError as exc:
                    saved_msg = f"NOT saved: {exc}"
                    print(saved_msg)
                else:
                    ed.dirty = False
                    saved_msg = f"saved to {path.name}"
                    print(f"{saved_msg}: {describe(ed.as_rois())}")

    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Set the INSPECT and PICK zones for this rig.")
    ap.add_argument("--camera-index", type=int, required=True,
                    help="camera index (REQUIRED — no default, so index 0 is never assumed)")
    ap.add_argument("--zones", type=Path, default=ZONES_PATH)
    ap.add_argument("--show", action="store_true", help="print the saved zones and exit")
    args = ap.parse_args(argv)

    if args.show:
        zones = load_zones(args.zones)
        print(f"{args.zones}\n  {describe(zones)}")
        return 0
    return run_editor(args.camera_index, args.zones)


if __name__ == "__main__":
    sys.exit(main())
