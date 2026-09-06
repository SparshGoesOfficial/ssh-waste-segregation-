"""Import shim for the teammate's OpenCV vision code.

Their folder is "ssh-waste-segregation-": the trailing hyphen makes it an invalid
module name, and our repo root has a literal trailing space in its name, so
neither a plain import nor a relative one works. This module puts their folder on
sys.path just long enough to pull the vision symbols across, then takes it back
off, and re-exports the result under a name the rest of the project can import.

Scope is deliberately narrow. We classify waste and confirm a cube's colour is
present — we do not localize and we do not command the arm. So nothing here
touches workspace calibration, pick planning, kinematics, or serial. That is
enforced below, not just documented.

Import it for the symbols; run it (`python teammate_vision.py`) for the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import contract

# cwd-independent: the repo dir name ends in a space, so relative paths are a trap.
ROOT = Path(__file__).resolve().parent
TEAMMATE_DIR = ROOT / "ssh-waste-segregation-"

# Their modules that carry arm/serial concerns. We must not pull these in, even
# transitively — importing serial_controller can open a port.
FORBIDDEN_MODULES = ("workspace_calibration", "pick_planning", "kinematics", "serial_controller")

__all__ = [
    "Camera",
    "CameraConfig",
    "load_hsv_calibration",
    "load_vision_preprocessing_config",
    "load_detection_config",
    "segment_colors",
    "detect_cubes",
    "DetectionConfig",
    "ImageROI",
    "VisionPreprocessingConfig",
    "TEAMMATE_DIR",
]

if not TEAMMATE_DIR.is_dir():
    raise ImportError(
        f"teammate vision folder not found at {TEAMMATE_DIR!r}. Expected the "
        "'ssh-waste-segregation-' checkout beside this file (note the trailing "
        "hyphen in the folder name, and the trailing space in the repo root)."
    )

_PATH_ENTRY = str(TEAMMATE_DIR)
_we_inserted = _PATH_ENTRY not in sys.path
if _we_inserted:
    # Front of the path so their 'config' wins over any same-named module.
    sys.path.insert(0, _PATH_ENTRY)

try:
    # config.py holds the dataclasses and the two loaders.
    from config import (  # noqa: E402
        SUPPORTED_COLORS,  # not re-exported; used only for the drift check below
        CameraConfig,
        DetectionConfig,
        ImageROI,
        VisionPreprocessingConfig,
        load_detection_config,
        load_hsv_calibration,
        load_vision_preprocessing_config,
    )
    from camera import Camera  # noqa: E402
    from vision import segment_colors  # noqa: E402
    from detection import detect_cubes  # noqa: E402
except ImportError as exc:
    raise ImportError(
        f"could not import the teammate vision symbols from {TEAMMATE_DIR!r}: {exc}. "
        "Their config/camera/vision/detection modules are expected there, and "
        "numpy + cv2 must be installed in the active venv (~/sih)."
    ) from exc
finally:
    # Take the path entry back out. Their only lazy imports are cv2 and
    # picamera2, which resolve from the venv, so nothing here needs the entry
    # after this point — and removing it means an accidental `import kinematics`
    # elsewhere fails loudly instead of silently working.
    if _we_inserted:
        try:
            sys.path.remove(_PATH_ENTRY)
        except ValueError:
            pass

# Structural check that the narrow scope actually held: if any arm/serial module
# arrived as a transitive import, say so now rather than discovering a port was
# opened mid-demo.
_leaked = sorted(m for m in FORBIDDEN_MODULES if m in sys.modules)
if _leaked:
    raise ImportError(
        f"teammate modules {_leaked} were imported transitively — this task is "
        "vision-only (no localization, no arm motion, no serial). Check their "
        "import graph before continuing."
    )

# Their SUPPORTED_COLORS is the authority; contract.py mirrors it as a literal.
# Fail at import if the two have drifted apart.
contract.verify_supported_colors(SUPPORTED_COLORS)

_SYMBOLS = (
    ("Camera", Camera),
    ("CameraConfig", CameraConfig),
    ("load_hsv_calibration", load_hsv_calibration),
    ("load_vision_preprocessing_config", load_vision_preprocessing_config),
    ("load_detection_config", load_detection_config),
    ("segment_colors", segment_colors),
    ("detect_cubes", detect_cubes),
    ("DetectionConfig", DetectionConfig),
    ("ImageROI", ImageROI),
    ("VisionPreprocessingConfig", VisionPreprocessingConfig),
)


def _report() -> None:
    print(f"teammate vision path : {TEAMMATE_DIR}")
    print(f"path entry           : inserted, then removed (kept clean)")
    for name, obj in _SYMBOLS:
        kind = "class" if isinstance(obj, type) else "func "
        src = getattr(sys.modules.get(obj.__module__), "__file__", obj.__module__)
        src = Path(src).name if src else obj.__module__
        print(f"  ok  {name:<34} {kind}  <- {src}")
    print(f"{len(_SYMBOLS)}/{len(__all__) - 1} symbols imported")
    print(f"SUPPORTED_COLORS     : {SUPPORTED_COLORS}  (matches contract.py)")
    print(f"excluded (not imported): {', '.join(FORBIDDEN_MODULES)}")


_report()

if __name__ == "__main__":
    print()
    print("--- detail ---")
    for name, obj in _SYMBOLS:
        doc = (obj.__doc__ or "").strip().splitlines()
        print(f"{name}: {doc[0] if doc else '(no docstring)'}")
