# Arduino Uno servo controller

Open `arm_serial_controller/arm_serial_controller.ino` in Arduino IDE, select
**Arduino Uno** and the dummy board's COM port, then upload. The firmware boots
disarmed and does not attach pins 9, 10, 11, 6, or 5 until an explicit, valid
`ARM` command is received.

## ARM_SERIAL_V1 protocol

Commands and responses are newline-delimited ASCII at 115200 baud.

```text
PING                              -> PONG ARM_SERIAL_V1
STATUS                            -> STATUS <state> <motion> j1 j2 j3 j4 j5
LIMITS                            -> LIMITS min:max ...
ARM j1 j2 j3 j4 j5               -> OK ARMED
MOVE id duration j1 j2 j3 j4 j5  -> ACK id, then DONE id
JOINT id number angle duration    -> ACK id, then DONE id
GRIP id OPEN|CLOSE duration       -> ACK id, then DONE id
STOP                              -> OK STOPPED
DISARM                            -> OK DISARMED
KEEPALIVE                         -> no response
```

During a motion, the host must send `KEEPALIVE` more frequently than the
750 ms firmware watchdog. A watchdog event stops interpolation and holds the
last commanded angle; it does not detach a gravity-loaded arm. The physical
external-power cutoff remains the emergency stop.

The present limits reproduce the supplied manual joystick sketch and are
provisional. Test the protocol on an Uno with no servos connected before any
physical joint calibration.
