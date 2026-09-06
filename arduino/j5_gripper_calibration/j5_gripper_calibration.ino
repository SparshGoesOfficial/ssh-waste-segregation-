#include <Servo.h>
#include <stdlib.h>
#include <string.h>

// Temporary, single-servo calibration sketch for J5 only.
// J1-J4 are deliberately never attached.

const uint8_t GRIPPER_PIN = 5;
const int SAFE_MIN_ANGLE = 15;
const int SAFE_MAX_ANGLE = 50;
const int START_ANGLE = 50;
const unsigned long STEP_DELAY_MS = 60;

Servo gripper;
bool attached = false;
int currentAngle = START_ANGLE;

char lineBuffer[48];
uint8_t lineLength = 0;

bool parseAngle(const char *text, int &angle) {
  if (text == NULL || *text == '\0') return false;
  char *end = NULL;
  long value = strtol(text, &end, 10);
  if (end == text || *end != '\0') return false;
  if (value < SAFE_MIN_ANGLE || value > SAFE_MAX_ANGLE) return false;
  angle = static_cast<int>(value);
  return true;
}

void printStatus() {
  Serial.print(F("STATUS J5 "));
  Serial.print(attached ? F("ATTACHED ") : F("DETACHED "));
  Serial.println(currentAngle);
}

void moveSlowlyTo(int targetAngle) {
  if (!attached) {
    Serial.println(F("ERR DETACHED"));
    return;
  }

  while (currentAngle != targetAngle) {
    currentAngle += targetAngle > currentAngle ? 1 : -1;
    gripper.write(currentAngle);
    delay(STEP_DELAY_MS);
  }

  Serial.print(F("OK J5 "));
  Serial.println(currentAngle);
}

void processCommand(char *line) {
  char *command = strtok(line, " \t");
  char *argument = strtok(NULL, " \t");
  char *extra = strtok(NULL, " \t");

  if (command == NULL) return;

  if (strcmp(command, "PING") == 0 && argument == NULL) {
    Serial.println(F("PONG J5_CAL_V1"));
    return;
  }

  if (strcmp(command, "STATUS") == 0 && argument == NULL) {
    printStatus();
    return;
  }

  if (strcmp(command, "ATTACH") == 0 && extra == NULL) {
    int requestedAngle;
    if (!parseAngle(argument, requestedAngle)) {
      Serial.println(F("ERR ANGLE_RANGE_15_50"));
      return;
    }
    if (attached) {
      Serial.println(F("ERR ALREADY_ATTACHED"));
      return;
    }

    currentAngle = requestedAngle;
    gripper.write(currentAngle);
    gripper.attach(GRIPPER_PIN);
    gripper.write(currentAngle);
    attached = true;
    Serial.print(F("OK ATTACHED J5 "));
    Serial.println(currentAngle);
    return;
  }

  if (strcmp(command, "SET") == 0 && extra == NULL) {
    int requestedAngle;
    if (!parseAngle(argument, requestedAngle)) {
      Serial.println(F("ERR ANGLE_RANGE_15_50"));
      return;
    }
    moveSlowlyTo(requestedAngle);
    return;
  }

  if (strcmp(command, "DETACH") == 0 && argument == NULL) {
    gripper.detach();
    attached = false;
    Serial.println(F("OK DETACHED J5"));
    return;
  }

  Serial.println(F("ERR COMMAND"));
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("READY J5_CAL_V1 DETACHED"));
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());
    if (incoming == '\r') continue;
    if (incoming == '\n') {
      lineBuffer[lineLength] = '\0';
      processCommand(lineBuffer);
      lineLength = 0;
      continue;
    }
    if (lineLength < sizeof(lineBuffer) - 1) {
      lineBuffer[lineLength++] = incoming;
    } else {
      lineLength = 0;
      Serial.println(F("ERR LINE_TOO_LONG"));
    }
  }
}
