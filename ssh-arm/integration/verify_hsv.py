"""Confirm hsv_ranges_local.json is fully calibrated on THIS rig.

Run after a calibration session:  python verify_hsv.py
Exits non-zero until all four colours are marked calibrated, so it can gate a demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import contract
import teammate_vision  # noqa: F401  (path shim; also cross-checks SUPPORTED_COLORS)
from teammate_vision import load_hsv_calibration

ROOT = Path(__file__).resolve().parent
LOCAL_HSV = ROOT / "hsv_ranges_local.json"
THEIR_HSV = ROOT / "ssh-waste-segregation-" / "calibration" / "hsv_ranges.json"


def main() -> int:
    if not LOCAL_HSV.exists():
        print(f"FAIL  {LOCAL_HSV.name} not found — copy it from their calibration/ first")
        return 2

    cal = load_hsv_calibration(LOCAL_HSV)
    done = set(cal.calibrated_colors)
    expected = set(contract.BIN_TO_COLOR.values())

    print(f"file             : {LOCAL_HSV}")
    print(f"calibrated_colors: {sorted(done)}")
    print(f"contract needs   : {sorted(expected)}")
    print()
    for bin_id in sorted(contract.BIN_TO_COLOR):
        color = contract.BIN_TO_COLOR[bin_id]
        cls = contract.BIN_TO_CLASS[bin_id]
        mark = "ok " if color in done else "MISSING"
        n = len(cal.ranges[color])
        print(f"  {mark}  bin {bin_id}  {cls:<8} -> {color:<7} ({n} hue range{'s' if n > 1 else ''})")

    # Did their tracked file stay untouched?
    if THEIR_HSV.exists():
        their = load_hsv_calibration(THEIR_HSV)
        if set(their.calibrated_colors) != {"red"}:
            print(f"\nWARN  their tracked file changed — calibrated_colors={sorted(their.calibrated_colors)},"
                  " expected just {'red'}. It must never be written to.")

    missing = expected - done
    if missing:
        print(f"\nFAIL  {sorted(missing)} still uncalibrated. Re-run hsv_calibrator.py"
              " and press s (lowercase) for each.")
        return 1

    stale = done - expected
    print(f"\nPASS  all four colours calibrated on this rig"
          + (f" (plus unused: {sorted(stale)})" if stale else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
