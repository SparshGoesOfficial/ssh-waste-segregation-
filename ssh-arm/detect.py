"""Live 4-class waste classification on webcam. Vision only — no serial, no arm.

Inference runs on a centre crop (the model was trained on single objects filling
the frame, so feeding it a wide room shot is off-distribution). The crop box is
drawn on the full frame for aiming.

Keys:  q quit   c toggle crop on/off   + / = grow crop   - shrink crop
"""
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent          # cwd-independent: dir name has a trailing space
WEIGHTS = ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"
CAM = 0
IMGSZ = 224
DEVICE = "mps"
CONF_HI = 0.70                                   # green at/above, orange below
CROP_STEP = 20                                   # px per keypress
CROP_MIN = 100                                   # px floor
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
BOX = (255, 200, 0)

if not WEIGHTS.exists():
    raise SystemExit(f"weights not found: {WEIGHTS}")

model = YOLO(str(WEIGHTS))
print("class names:", model.names)

cap = cv2.VideoCapture(CAM)
if not cap.isOpened():
    raise SystemExit(f"could not open camera {CAM} — check macOS camera permission for your terminal")

ok, probe = cap.read()
if not ok:
    raise SystemExit("could not read a first frame from the camera")
FH, FW = probe.shape[:2]
crop_max = min(FH, FW)                           # never exceed the frame
crop_size = min(crop_max, max(CROP_MIN, crop_max // 2))   # neutral starting point, not a recommendation
use_crop = True

times = deque(maxlen=30)                         # rolling window for FPS
win = "waste classifier - 4 class, public only"
print(f"frame {FW}x{FH} | crop range {CROP_MIN}-{crop_max}px | starting crop {crop_size}px")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed")
            break

        # Extract the crop BEFORE drawing, so the aiming box is never in the model input.
        if use_crop:
            cx, cy = FW // 2, FH // 2
            half = crop_size // 2
            x0, y0 = cx - half, cy - half
            x1, y1 = x0 + crop_size, y0 + crop_size
            model_input = frame[y0:y1, x0:x1].copy()
        else:
            model_input = frame

        t0 = time.perf_counter()
        r = model.predict(model_input, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        probs = r.probs
        label = model.names[probs.top1]
        conf = float(probs.top1conf)
        times.append(time.perf_counter() - t0)

        fps = len(times) / sum(times) if sum(times) > 0 else 0.0
        color = GREEN if conf >= CONF_HI else ORANGE

        if use_crop:
            cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), BOX, 2)
            crop_txt = f"crop {crop_size}px"
        else:
            crop_txt = f"crop OFF (full {FW}x{FH})"

        cv2.putText(frame, f"{label} {conf:.2f}", (16, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS   {crop_txt}", (16, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            use_crop = not use_crop
        elif key in (ord("+"), ord("=")):
            crop_size = min(crop_max, crop_size + CROP_STEP)
            print(f"crop {crop_size}px")
        elif key in (ord("-"), ord("_")):
            crop_size = max(CROP_MIN, crop_size - CROP_STEP)
            print(f"crop {crop_size}px")
finally:
    cap.release()
    cv2.destroyAllWindows()
