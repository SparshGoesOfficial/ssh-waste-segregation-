"""Keep an Arduino serial connection open and relay line commands from stdin."""

from __future__ import annotations

import sys
import time

import serial


def main() -> int:
    port_name = sys.argv[1] if len(sys.argv) > 1 else "COM8"
    baud_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

    port = serial.Serial(port_name, baud_rate, timeout=0.25)
    time.sleep(2.5)  # Allow the Uno's one reset on initial connection to finish.

    startup = port.read_all().decode(errors="replace").strip()
    if startup:
        print(startup, flush=True)
    print(f"BRIDGE READY {port_name} (connection will remain open)", flush=True)

    try:
        for raw_line in sys.stdin:
            command = raw_line.strip()
            if not command:
                continue
            if command.upper() == "QUIT":
                print("BRIDGE CLOSED", flush=True)
                return 0

            port.write((command + "\n").encode("ascii"))
            port.flush()

            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                response = port.readline().decode(errors="replace").strip()
                if response:
                    print(response, flush=True)
                    break
            else:
                print("BRIDGE TIMEOUT", flush=True)
    finally:
        port.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
