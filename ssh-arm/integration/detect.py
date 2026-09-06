"""Live 4-class waste classification, with a colour cross-check on the PICK zone.

One frame, two zones, both read from zones.json:

  INSPECT  real waste is held here; the classifier decides the bin from this crop
  PICK     the taped cube square; the matching cube's colour is confirmed here

The classifier chooses the bin. The colour check only confirms the corresponding
cube is present. Vision only — no serial, no arm motion, no pixel-to-robot mapping.

Keys:  q/Esc quit    SPACE arm one decision (see TEMPORARY guard below)
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

import color_check
import contract
import zone_config

ROOT = Path(__file__).resolve().parent          # cwd-independent: dir name has a trailing space
WEIGHTS = ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"
IMGSZ = 224
DEVICE = "mps"
CONF_HI = 0.70                                   # unchanged gate: green at/above, orange below
DEBOUNCE_FRAMES = 5                              # consecutive agreeing frames before a decision
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
PICK_BOX = (255, 0, 255)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera-index", type=int, required=True,
                    help="camera index (REQUIRED — index 0 is never assumed)")
    ap.add_argument("--zones", type=Path, default=ROOT / "zones.json")
    args = ap.parse_args(argv)

    if not WEIGHTS.exists():
        raise SystemExit(f"weights not found: {WEIGHTS}")

    zones = zone_config.load_zones(args.zones)   # raises if missing, overlapping, or wrong size
    inspect_roi, pick_roi = zones["INSPECT"], zones["PICK"]
    print(f"zones: {zone_config.describe(zones)}")

    model = YOLO(str(WEIGHTS))
    print("class names:", model.names)

    # Fail before the capture loop, not after a demo goes wrong. Ultralytics
    # indexes classes by sorted folder name, so a dataset change silently shifts
    # every index and would mis-bin everything.
    contract.verify_model_names(model.names)
    print("model names verified against contract.CLASS_TO_BIN")

    # Loads their HSV/detection/vision calibration once and prints which colours
    # are actually tuned. Three of four are not, and that matters for every
    # NO CUBE this loop prints.
    color_check.load_pipeline()

    # ─── TEMPORARY: the model has no `nothing` class ──────────────────────────
    # All four classes are waste, so an empty INSPECT zone still returns a best
    # guess — typically `paper 1.00` — which clears both the 0.70 gate and the
    # 5-frame debounce. Left alone the rig idles, emitting a confident false
    # paper decision forever.
    #
    # The guard: a decision is only taken while ARMED. SPACE arms exactly one.
    # This is scaffolding, not a fix. Raising CONF_HI or lengthening the
    # debounce would only hide it — an empty zone genuinely reads 1.00.
    # DELETE this block, `armed`, and the SPACE key handler once Stage 2
    # retrains with a `nothing` class. contract.verify_model_names() already
    # tolerates that class, and the state-change logic below needs no change.
    armed = False
    # ──────────────────────────────────────────────────────────────────────────

    pending_label = None        # label currently accumulating agreement
    agree = 0                   # consecutive frames at >= CONF_HI agreeing with pending_label
    last_decision = None        # (label, frozenset(found), verdict) — print only on change
    banner = ""                 # last decision, redrawn each frame

    times = deque(maxlen=30)
    win = "waste sorter - INSPECT classify + PICK colour check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        with zone_config.open_camera(args.camera_index) as cam:
            first = True
            while True:
                frame = cam.read()
                if first:
                    zone_config.assert_frame_size(frame)   # zones are 1280x720; refuse anything else
                    first = False

                # Extract the crop BEFORE drawing, so the zone boxes are never in the model input.
                model_input = frame[inspect_roi.y:inspect_roi.bottom,
                                    inspect_roi.x:inspect_roi.right].copy()

                t0 = time.perf_counter()
                r = model.predict(model_input, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
                probs = r.probs
                label = model.names[probs.top1]
                conf = float(probs.top1conf)
                times.append(time.perf_counter() - t0)
                fps = len(times) / sum(times) if sum(times) > 0 else 0.0

                # Debounce: N consecutive frames that both clear the gate and agree.
                if conf >= CONF_HI:
                    if label == pending_label:
                        agree += 1
                    else:
                        pending_label, agree = label, 1
                else:
                    pending_label, agree = None, 0

                confirmed = agree >= DEBOUNCE_FRAMES and pending_label is not None

                if confirmed and armed:
                    bin_id = contract.CLASS_TO_BIN[pending_label]
                    expected_color = contract.BIN_TO_COLOR[bin_id]

                    found = color_check.colors_present(frame, pick_roi)
                    verdict = "MATCH" if expected_color in found else "NO CUBE"

                    state = (pending_label, frozenset(found), verdict)
                    if state != last_decision:           # only on a state change, not every frame
                        shown = ", ".join(sorted(found)) if found else "none"
                        print(f"DECISION: {pending_label} -> bin {bin_id} -> expect "
                              f"{expected_color} | pick zone: {shown} | {verdict}")
                        last_decision = state

                    banner = (f"{pending_label} -> bin {bin_id} -> {expected_color} | {verdict}")

                    # TODO(stage 3): arm.write(...) goes here — send bin_id over serial
                    # once the arm is wired. Nothing is commanded in this task.

                    armed = False                        # TEMPORARY: one decision per arming

                # --- draw both zones, labelled -------------------------------------
                expected_now = None
                if confirmed and pending_label in contract.CLASS_TO_BIN:
                    expected_now = contract.BIN_TO_COLOR[contract.CLASS_TO_BIN[pending_label]]

                icol = GREEN if conf >= CONF_HI else ORANGE
                cv2.rectangle(frame, (inspect_roi.x, inspect_roi.y),
                              (inspect_roi.right - 1, inspect_roi.bottom - 1), icol, 3)
                cv2.putText(frame, f"INSPECT  {label} {conf:.2f}  [{agree}/{DEBOUNCE_FRAMES}]",
                            (inspect_roi.x + 6, max(24, inspect_roi.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, icol, 2, cv2.LINE_AA)

                pick_label = "PICK"
                if expected_now:
                    pick_label = f"PICK  expect {expected_now}"
                cv2.rectangle(frame, (pick_roi.x, pick_roi.y),
                              (pick_roi.right - 1, pick_roi.bottom - 1), PICK_BOX, 3)
                cv2.putText(frame, pick_label, (pick_roi.x + 6, max(24, pick_roi.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, PICK_BOX, 2, cv2.LINE_AA)

                arm_txt = "ARMED - hold waste in INSPECT" if armed else "press SPACE to arm one decision"
                cv2.putText(frame, arm_txt, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            GREEN if armed else WHITE, 2, cv2.LINE_AA)
                if banner:
                    cv2.putText(frame, banner, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                GREEN if banner.endswith("MATCH") else RED, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{fps:.1f} FPS   TEMPORARY arm-guard: no 'nothing' class yet",
                            (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            WHITE, 1, cv2.LINE_AA)

                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):                    # TEMPORARY: remove with the guard above
                    armed = True
                    last_decision = None                 # let the next decision print even if identical
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
