# SNU OpenCV

This project will become a colored-cube pick-and-place system using a Logitech
C270 fixed above the table and connected to a Windows laptop. The laptop runs
OpenCV, coordinate mapping, IK, and the autonomous state machine; it commands an
Arduino Uno over USB serial, and the Uno generates the five servo signals.
Before each scan, the arm parks outside the camera ROI and the system locks a
fresh target. The pixel-to-table mapping remains valid while the arm moves, but
must be recalibrated if the camera or table is moved.

**The Windows laptop has physically validated the C270 at index 2 and
1280x720, live color/centroid detection, a fixed 500x500 mm floor map, and the
complete dummy-Uno serial protocol on COM5.** The centre target mapped to
(244.9, 249.5) mm for an expected (250, 250) mm point, and an off-centre target
also tracked stably. HSV values and the tape calibration remain prototype
values that should be refined before the final demo. Robot-base alignment,
real servo calibration, pickup geometry, and autonomous physical motion are
still pending.

## Phase tracker

- [x] Phase 1 - Camera test
- [x] Phase 2 - HSV calibration (functional prototype profile; refinement pending)
- [x] Phase 3 - Cube segmentation (physically tested)
- [x] Phase 4 - Contour / centroid detection (physically tested)
- [ ] Phase 5 - Fixed camera mount and arm-clear capture pose
- [x] Phase 6 - Workspace calibration (500x500 mm floor map; provisional taped rig)
- [ ] Phase 7 - Pixel to robot coordinates (pixel-to-floor works; robot-base alignment pending)
- [x] Phase 8 - Serial communication (full disconnected dummy-Uno test passed on COM5)
- [ ] Phase 9 - Manual arm control
- [ ] Phase 10 - Inverse kinematics / arm coordinate control (provisional FK/IK ready)
- [ ] Phase 11 - Gripper
- [ ] Phase 12 - Pick sequence
- [ ] Phase 13 - Drop zones
- [ ] Phase 14 - State machine
- [ ] Phase 15 - Full autonomous integration
- [ ] Phase 16 - Robustness testing
- [ ] Phase 17 - Hackathon demo mode

## Phase 1 structure

```text
SNU OpenCV/
|-- __init__.py
|-- camera.py          # C270/OpenCV primary adapter plus optional Picamera2
|-- config.py          # Validated camera settings
|-- main.py            # Live/headless camera test
|-- hsv_calibrator.py  # Phase 2 live HSV trackbars and mask previews
|-- vision.py          # Phase 3 ROI, blur, threshold, and mask cleanup
|-- segmentation_test.py # Live Phase 3 mask/ROI acceptance test
|-- detection.py       # Phase 4 contours, centroids, filtering, target policy
|-- detection_test.py  # Live annotated Phase 4 preview
|-- kinematics.py      # Hardware-free standard-DH forward kinematics
|-- kinematics_test.py # DRY_RUN command-line FK inspection
|-- workspace_calibration.py # Fixed-camera pixel/table homography
|-- workspace_calibration_test.py # Synthetic homography acceptance test
|-- pick_planning.py # Guarded detection-to-table-to-IK dry-run bridge
|-- pick_planning_test.py # Synthetic end-to-end planning acceptance test
|-- serial_controller.py # Guarded ARM_SERIAL_V1 laptop client
|-- serial_link_test.py # Read-only or explicitly gated dummy-Uno test
|-- requirements.txt
|-- calibration/
|   |-- hsv_ranges.json
|   |-- vision_settings.json
|   |-- detection_settings.json
|   |-- arm_kinematics.json
|   |-- workspace_calibration.json
|   |-- pick_geometry.json
|   `-- servo_controller.json
|-- tests/
|   |-- __init__.py
|   |-- test_camera.py
|   |-- test_hsv_calibration.py
|   |-- test_kinematics.py
|   |-- test_vision.py
|   |-- test_detection.py
|   |-- test_workspace_calibration.py
|   |-- test_pick_planning.py
|   `-- test_serial_controller.py
|-- arduino/
|   |-- README.md
|   `-- arm_serial_controller/arm_serial_controller.ino
`-- logs/              # Reserved for later runtime logs
```

## What the important Phase 1 functions do

### `Picamera2.create_video_configuration(...)`

- **What:** creates a Picamera2 stream configuration.
- **Why:** requests a predictable 640x480, three-channel stream instead of
  relying on a camera-library default.
- **Input:** image size, `RGB888` format, and requested frame rate.
- **Output:** a configuration object passed to `configure()`.
- **Project connection:** Picamera2's `RGB888` memory layout provides the BGR
  triples OpenCV expects, so future HSV conversion receives correct colors.

### `Picamera2.capture_array("main")`

- **What:** captures the configured stream as a NumPy array.
- **Why:** OpenCV operates directly on NumPy image arrays.
- **Input:** the stream name, `main`.
- **Output:** one frame shaped approximately `(height, width, 3)`.
- **Project connection:** every later segmentation and detection stage begins
  with this frame.

### `cv2.VideoCapture(index)` and `read()`

- **What:** opens a USB/laptop camera and reads `(success, frame)` pairs.
- **Why:** it is the production interface for the fixed Logitech C270.
- **Input:** a camera index such as `0`.
- **Output:** a Boolean success flag and an OpenCV BGR NumPy frame.
- **Project connection:** it obeys the same frame contract as the Pi backend,
  allowing later vision code to remain camera-API independent.

### `Camera.read()`

- **What:** captures, validates, and optionally rotates one frame.
- **Why:** empty data, incorrect array shape, and wrong dtype should fail early
  with a useful error.
- **Input:** none after the `Camera` has been started.
- **Output:** a non-empty `uint8` BGR array shaped `(height, width, 3)`.
- **Project connection:** this is the only interface later vision modules need.

## Raspberry Pi 4 and Logitech C270 setup

Connect the C270 directly to the Pi. Install OpenCV, NumPy, and the V4L2 camera
utilities from Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy v4l-utils
```

Confirm that the USB camera and its two capture formats are visible:

```bash
lsusb
ls -l /dev/video*
v4l2-ctl --device=/dev/video0 --list-formats
```

If you use a virtual environment on Raspberry Pi OS, expose distro camera
packages to it:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

## Automated camera-contract test

Run from inside the `SNU OpenCV` project folder:

```bash
python3 -m unittest discover -s tests -v
```

Expected result: fifty-five tests report `ok`, followed by `OK`. These tests use a
fake frame and do not require or move any hardware.

## Physical C270 test on the Pi

Run from inside the `SNU OpenCV` project folder:

```bash
python3 main.py --backend opencv --camera-index 0 --fourcc MJPG
```

Expected result:

1. A 640x480 live preview opens.
2. Motion is smooth enough for a camera check, and colors look natural.
3. The overlay says `backend=opencv` and the frame count increases.
4. Pressing **Q** or **Esc** closes the camera cleanly.
5. The console ends with a `PASS` line showing a three-channel `uint8` frame.

`Ctrl+C` is also handled safely, but reports that the user stopped the test
instead of reporting `PASS`; use **Q** or **Esc** for the acceptance test.

If the image is physically rotated, add `--rotation 90`, `180`, or `270`.

The tested C270 emits occasional `extraneous bytes before marker` decoder
messages in 1280x720 MJPG mode. The 60-frame hardware test completed and the
saved full-resolution frame had no visual corruption, so these messages are
non-fatal unless capture failures or visible artifacts appear.

For a finite headless test that also saves evidence:

```bash
python3 main.py --backend opencv --camera-index 0 --fourcc MJPG --headless --frames 30 \
  --save-frame logs/phase1_camera.jpg
```

Expected result: the process exits by itself with status 0, logs `PASS`, and
creates a non-empty image with the correct orientation and natural colors.

## Laptop development webcam

Install the portable dependencies on a laptop, then choose the OpenCV backend:

```bash
python -m pip install -r requirements.txt
python main.py --backend opencv --camera-index 0
```

The fixed C270 uses `--backend opencv`; `--backend auto` remains available for
development machines. If the wrong webcam opens, try `--camera-index 1`.

## Phase 1 troubleshooting

- **No camera backend starts:** force one backend so its error is unambiguous.
- **USB webcam unavailable:** close programs already using it and try another
  camera index.
- **C270 HD stream fails:** use `--fourcc MJPG`. Uncompressed YUYV cannot
  sustain every HD/frame-rate combination over USB.
- **Empty-frame failure:** check cable/power stability; the program retries a
  configurable number of reads and then exits instead of waiting forever.
- **Wrong orientation:** pass the matching `--rotation` value. Keep that value
  unchanged after future workspace calibration.
- **Red and blue appear swapped:** confirm the OpenCV backend still returns BGR
  frames before HSV conversion.

## Phase 2: live HSV calibration

The values in `calibration/hsv_ranges.json` are a non-overlapping provisional
profile—not final values. Calibrate with the real cubes and the same camera,
lighting, background, rotation, and viewing geometry intended for the demo.

On Raspberry Pi, run from inside the `SNU OpenCV` folder:

```bash
python3 hsv_calibrator.py --backend opencv --camera-index 0
```

For a laptop USB webcam:

```bash
python hsv_calibrator.py --backend opencv --camera-index 0
```

The utility displays:

1. **Original:** camera feed plus current color and keyboard help.
2. **Mask:** selected pixels are white; rejected pixels are black.
3. **Masked Result:** only image pixels selected by the mask remain visible.

Controls:

- `1`, `2`, `3`, `4`: select red, green, blue, or yellow.
- Six trackbars: set H/S/V lower and upper limits.
- `T`: toggle between red's low-hue and high-hue ranges.
- `S`: save the current color to `calibration/hsv_ranges.json`.
- `Q` or `Esc`: quit safely.

For each color, place a real cube on the intended matte background. Adjust the
hue bounds until the cube is white in the mask, then raise the saturation/value
lower bounds enough to reject gray background, shadows, and glare without
removing the cube. Move the cube to several image positions before saving.
Calibrate both red ranges using `T`; their masks are combined automatically.

### Important Phase 2 functions

#### `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`

- **What:** converts a BGR camera image into HSV.
- **Why:** hue separates color identity more usefully than raw BGR channels.
- **Input:** one three-channel BGR frame.
- **Output:** an HSV frame where H is 0-179 and S/V are 0-255.
- **Project connection:** HSV is the input to every color threshold.

#### `cv2.inRange(hsv, lower, upper)`

- **What:** tests every pixel against inclusive lower and upper HSV bounds.
- **Why:** it isolates one selected color efficiently.
- **Input:** an HSV image and two three-value thresholds.
- **Output:** a binary mask containing 255 for accepted pixels and 0 otherwise.
- **Project connection:** later contour detection will operate on this mask.

#### `cv2.bitwise_or(mask_a, mask_b)`

- **What:** combines two binary masks.
- **Why:** red can occur near both ends of OpenCV's circular hue scale.
- **Input:** the low-red and high-red masks.
- **Output:** one mask containing pixels selected by either red interval.
- **Project connection:** later code sees red as one color despite two ranges.

#### `cv2.bitwise_and(frame, frame, mask=mask)`

- **What:** keeps original pixels only where the mask is white.
- **Why:** it makes calibration mistakes easy to see.
- **Input:** the BGR frame and binary mask.
- **Output:** a color preview of the selected region.
- **Project connection:** this is a diagnostic view, not cube detection.

### Phase 2 acceptance result

Each cube should remain mostly solid white in its own mask across the reachable
image area, while the background and other cube colors remain mostly black.
After pressing `S` for each color, the JSON file should show `true` for red,
green, blue, and yellow under `calibrated`.

## Phase 3: segmentation preprocessing

Run the live test from inside the `SNU OpenCV` folder.

Fixed C270 on Raspberry Pi:

```bash
python3 segmentation_test.py --backend opencv --camera-index 0
```

Laptop USB webcam:

```bash
python segmentation_test.py --backend opencv --camera-index 0
```

Three windows show the camera image, four cleaned color masks, and the combined
segmented result. Press `R` to draw a rectangle around only the arm-reachable
table area, then press Enter. The ROI is saved to
`calibration/vision_settings.json`. Press `C` to clear it and use the full frame.

The ROI is processed as a crop for efficiency, but every returned mask retains
the original 640x480 coordinate system. This prevents a future cube centroid
from being shifted by the ROI's top-left offset.

### Phase 3 pipeline

```text
BGR camera frame
-> reachable-table ROI
-> cv2.cvtColor(..., COLOR_BGR2HSV)
-> cv2.GaussianBlur(...)
-> cv2.inRange(...)
-> cv2.morphologyEx(..., MORPH_OPEN)
-> cv2.morphologyEx(..., MORPH_CLOSE)
-> full-frame binary color mask
```

#### `cv2.GaussianBlur`

- **What:** replaces each pixel with a Gaussian-weighted local average.
- **Why:** reduces small sensor and color variations before thresholding.
- **Input:** the HSV ROI and an odd kernel size such as 5x5.
- **Output:** a smoother HSV image of the same size.
- **Project connection:** produces steadier masks without the extra cost of a
  bilateral filter.

#### `cv2.morphologyEx(..., cv2.MORPH_OPEN, ...)`

- **What:** erodes and then dilates the binary mask.
- **Why:** removes isolated white specks smaller than the selected kernel.
- **Input:** a binary mask and configurable structuring element.
- **Output:** a cleaner mask with small noise removed.
- **Project connection:** prevents Phase 4 from treating specks as objects.

#### `cv2.morphologyEx(..., cv2.MORPH_CLOSE, ...)`

- **What:** dilates and then erodes the binary mask.
- **Why:** fills small black holes inside an otherwise solid cube region.
- **Input:** the opened binary mask and configurable structuring element.
- **Output:** a more solid cube-shaped region.
- **Project connection:** makes later contours and centroids more stable.

The default kernels are stored in `calibration/vision_settings.json`: Gaussian
5, opening 3, closing 5, and one morphology iteration. Kernel sizes must remain
positive odd numbers. These are safe starting values, not mandatory final ones.

### Phase 3 automated and physical tests

```bash
python3 -m unittest discover -s tests -v
```

Expected automated result: fifty-five tests followed by `OK`.

For the physical test, put all four cubes within the selected ROI. Each cube
should appear as one mostly solid white region in its matching mask. The area
outside the ROI must remain black, isolated white specks should be reduced, and
the cube regions must not be eroded away or merged with nearby objects.

## Phase 4: contour and centroid detection

Run the vision-only debug test with any available matte, approximately square,
strongly colored objects. Results remain provisional until tested with the real
cubes.

```bash
python3 detection_test.py --backend opencv --camera-index 0
```

For the headless Raspberry Pi over SSH, capture a finite test and save evidence:

```bash
python3 detection_test.py --backend opencv --camera-index 0 --fourcc MJPG \
  --headless --frames 60 --save-frame logs/detection.jpg \
  --save-mask logs/detection_mask.png
```

The command prints the detected color and pixel centroid, then exits with a
`PASS` message. Copy the two saved images to a computer for visual inspection.

For a laptop webcam:

```bash
python detection_test.py --backend opencv --camera-index 0
```

The annotated window displays every accepted color, contour, bounding box,
centroid `(cx, cy)`, mask area, and confidence. The selected target has a thicker
outline and a `TARGET` label. The default policy is `largest`, so selection is
predictable while testing multiple objects.

### Important Phase 4 functions

#### `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`

- **What:** traces connected white boundaries in a binary mask.
- **Why:** colored-pixel counts alone cannot provide each object's location.
- **Input:** one cleaned color mask.
- **Output:** a list of external contour point arrays.
- **Project connection:** each contour becomes one potential cube candidate.

#### `cv2.contourArea(contour)`

- **What:** measures the pixel area enclosed by a contour.
- **Why:** removes tiny noise and rejects implausibly large regions.
- **Input:** one contour.
- **Output:** area in square pixels.
- **Project connection:** also supports the default largest-target policy.

#### `cv2.boundingRect(contour)`

- **What:** returns the smallest upright `(x, y, width, height)` rectangle.
- **Why:** supplies a display box and a tolerant square-likeness test.
- **Input:** one contour.
- **Output:** four integer pixel values.
- **Project connection:** width divided by height provides the aspect ratio.

#### `cv2.moments(contour)`

- **What:** calculates spatial moments of a contour.
- **Why:** provides a more accurate center than the bounding-box midpoint for
  asymmetric masks.
- **Input:** one contour.
- **Output:** a dictionary including `m00`, `m10`, and `m01`.
- **Project connection:** `cx=m10/m00` and `cy=m01/m00` are the future pick
  coordinates in camera pixels.

Every accepted detection also records extent, solidity, approximate polygon
vertex count, and a transparent 0-1 geometric confidence score. Broad filters
are stored in `calibration/detection_settings.json`; they require physical cube
tuning rather than being treated as final values.

Available target policies are `largest`, `nearest`, `leftmost`, `first_valid`,
and `color_priority`. The live Phase 4 test uses `largest`; robot-coordinate
meaning for `nearest` will be finalized after workspace calibration.

Run the full synthetic test suite:

```bash
python3 -m unittest discover -s tests -v
```

Expected result: fifty-five tests followed by `OK`. The Phase 4 synthetic test
accepts a square, rejects tiny and elongated regions, verifies centroid/area,
and checks every selection policy without requiring a camera or cube.

## Provisional forward-kinematics prework

The supplied four-row DH table and servo assignments are stored in
`calibration/arm_kinematics.json`. Because the source did not identify its
convention, the current implementation explicitly assumes **standard DH**:

```text
A_i = Rz(theta_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
```

Run the zero-angle mathematical test with:

```bash
python3 kinematics_test.py --joint-angles 0 0 0 0
```

The provisional result is frame-4 position
`(x=693.487, y=11.357, z=149.600) mm`. This is the frame-4 origin, not yet the
confirmed gripping point. The gripper is an actuator but is not a fifth arm-pose
DOF.

The image-supplied conversion `theta = servo_raw - 90 degrees` is stored but
blocked by default because all directions, centers, and collision-safe ranges
remain unconfirmed. It can only be inspected in simulation with an explicit
flag:

```bash
python3 kinematics_test.py --servo-angles 90 90 90 90 \
  --allow-unverified-servo-map
```

This module contains no GPIO, PWM, serial, or motor-control code. Forward
kinematics and analytic frame-4 inverse kinematics are software-tested. The IK
solver accepts base-frame `(x, y, z)` plus the desired radial-plane tool pitch,
returns both positive- and negative-elbow branches, reports provisional servo
range compatibility, and verifies every result through FK.

Run a reproducible IK test where both branches are inside the provisional raw
servo ranges:

```bash
python3 kinematics_test.py --target 569.306 219.296 113.734 \
  --tool-pitch -15
```

Both results should report `IK->FK position error (mm): 0.000000000`. Phase 10
still remains incomplete until the DH convention, frame-4-to-gripping-point
offset, per-joint directions/centers, collision-safe limits, and a preferred
branch policy are physically established.

## Fixed-camera workspace-calibration prework

`workspace_calibration.py` implements a normalized DLT homography that maps
fixed-camera pixels `(u, v)` to table-plane coordinates `(x, y)` in millimetres.
It also stores the inverse table-to-pixel transform, point count, RMS error, and
maximum error. At least four non-collinear correspondences are required; nine
or more points spread across the reachable area are preferred for the physical
calibration.

Run the hardware-free numerical acceptance test with:

```bash
python3 workspace_calibration_test.py
```

Expected result: zero numerical error for calibration and unseen synthetic
points, followed by `PASS`. The test does not write synthetic values into the
production calibration file.

`calibration/workspace_calibration.json` intentionally has status
`awaiting_fixed_rig`, empty correspondences, and null homography matrices. The
runtime loader refuses to use it as a real calibration. After mounting, the
physical procedure will be:

1. Lock the camera, table, 1280x720 resolution, and final image rotation.
2. Define and confirm the robot-base table coordinate axes.
3. Mark and measure calibration points across the arm-reachable table area.
4. Record each point's pixel center and measured table `(x, y)` position.
5. Fit the homography and validate it on separate held-out points.
6. Recalibrate after any camera or table movement.

The fitting and transform code is complete, but Phases 6-7 remain physically
incomplete until those real correspondences and an acceptable error threshold
are established.

## Guarded detection-to-IK planning prework

`pick_planning.py` connects one detected cube centroid to the calibrated table
plane and calculates both approach and pickup IK candidates. It deliberately
does not choose an elbow branch, convert the result into trusted servo commands,
or move hardware.

Run its hardware-free integration test with:

```bash
python3 pick_planning_test.py
```

Expected result: the synthetic centroid is mapped to table millimetres, both IK
branches round-trip through FK with effectively zero position error, and the
program ends with `PASS`.

The production entry point has two independent guards. It refuses to plan while
`calibration/workspace_calibration.json` is not a real fixed-rig calibration,
and it refuses to plan while `calibration/pick_geometry.json` has status
`awaiting_measurements`. The latter file needs the measured frame-4 pickup
height, approach clearance, and tool pitch. Those values cannot be inferred
safely from the DH table or camera image because the frame-4-to-gripper-tip
offset and cube height are still unknown.

## Laptop-to-Arduino serial-control prework

The final processing host is the Windows laptop. `serial_controller.py` sends
the `ARM_SERIAL_V1` protocol over USB, while
`arduino/arm_serial_controller/arm_serial_controller.ino` validates each
request and generates smooth servo commands. The firmware boots disarmed and
does not attach a servo output until it receives an explicit valid `ARM`
command.

The supplied manual-sketch mapping is retained provisionally: J1=9, J2=10,
J3=11, J4=6, and J5=5. Its provisional angle ranges are stored in
`calibration/servo_controller.json`. They are software guards, not confirmed
physical collision limits.

Install the host dependency with:

```powershell
py -m pip install pyserial
```

After uploading the firmware to a dummy Uno with no servo signal wires
connected, run the read-only handshake first:

```powershell
py serial_link_test.py --port COM5
```

Replace `COM5` with the port shown on the test laptop. Only after confirming
that no servos are connected may the full dummy protocol exercise be run:

```powershell
py serial_link_test.py --port COM5 --dummy-motion-test `
  --safety-confirmation NO_SERVOS_CONNECTED
```

The dummy test exercises ARM, MOVE, JOINT, GRIP, STATUS, and DISARM without a
motor. On 2026-09-05, both the read-only handshake and complete disconnected
dummy test passed on COM5; the final response was `OK DISARMED`. Real joint
calibration is a separate gated stage.

## Hardware facts needed before robot-dependent phases

Before enabling physical motor control, record: exact arm model; DOF and joint
layout; servo/motor models; controller; link dimensions; gripper mechanism;
final fixed-camera position/orientation; arm-clear capture pose; usable table
dimensions; cube dimensions/colors; power supply; and low-level controller
details. Confirmed hardware includes an Arduino Uno and Logitech C270; a
Raspberry Pi 4 is retained only as a backup. The future camera mount is
approximately 650 mm above the table and points vertically downward. Unknown
dimensions can be measured
in millimetres center-to-center at the joint axes; servo zero angles and limits
will be established through staged, low-risk calibration rather than guessed.

## Safety boundary

The implemented vision stages only read a camera, and the planning stages are
mathematical dry runs. Serial/PWM command code now exists, but the Arduino boots
disarmed and the host rejects values outside provisional limits. It has not yet
been hardware-validated or approved for connected motors. The first physical
test must use a dummy Uno with no servo signal wires. The external servo-power
switch remains the physical emergency stop.
