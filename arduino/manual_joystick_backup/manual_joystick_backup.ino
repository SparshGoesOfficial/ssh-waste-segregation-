// Backup of the original working manual joystick sketch supplied by the team.
#include <Servo.h>

Servo base, shoulder, elbow, wrist, gripper;
float angBase = 90, angShoulder = 90, angElbow = 90, angWrist = 90;
bool gripperOpen = true;
int lastSW1 = HIGH;
unsigned long lastPressTime = 0;
const unsigned long DEBOUNCE_MS = 250;

const int DEADZONE = 40;
const float MAX_SPEED = 1.5;

const int GRIPPER_OPEN_ANGLE = 90;
const int GRIPPER_CLOSE_ANGLE = 30;

void setup() {
  base.attach(9);
  shoulder.attach(10);
  elbow.attach(11);
  wrist.attach(6);
  gripper.attach(5);
  pinMode(2, INPUT_PULLUP);

  base.write(90);
  shoulder.write(90);
  elbow.write(90);
  wrist.write(90);
  gripper.write(GRIPPER_OPEN_ANGLE);
  delay(500);
}

float readAxis(int pin, float current, float minLim, float maxLim) {
  int raw = analogRead(pin) - 512;
  if (abs(raw) < DEADZONE) return current;
  float speed = (raw / 512.0) * MAX_SPEED;
  current += speed;
  return constrain(current, minLim, maxLim);
}

void loop() {
  angBase = readAxis(A0, angBase, 0, 180);
  angShoulder = readAxis(A1, angShoulder, 20, 160);
  angElbow = readAxis(A2, angElbow, 10, 170);
  angWrist = readAxis(A3, angWrist, 0, 180);

  base.write(angBase);
  shoulder.write(angShoulder);
  elbow.write(angElbow);
  wrist.write(angWrist);

  int sw1 = digitalRead(2);
  unsigned long now = millis();
  if (
    sw1 == LOW && lastSW1 == HIGH &&
    (now - lastPressTime) > DEBOUNCE_MS
  ) {
    gripperOpen = !gripperOpen;
    gripper.write(
      gripperOpen ? GRIPPER_OPEN_ANGLE : GRIPPER_CLOSE_ANGLE
    );
    lastPressTime = now;
  }
  lastSW1 = sw1;

  delay(15);
}
