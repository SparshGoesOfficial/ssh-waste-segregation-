#include <Servo.h>
#include <stdlib.h>
#include <string.h>

// Temporary, single-servo calibration sketch for J4 only.
// J1, J2, J3, and J5 are deliberately never attached.

const uint8_t WRIST_PIN = 6;
const int SAFE_MIN_ANGLE = 0;
const int SAFE_MAX_ANGLE = 180;
const int START_ANGLE = 90;
const unsigned long STEP_DELAY_MS = 60;

Servo wrist;
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
  Serial.print(F("STATUS J4 "));
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
    wrist.write(currentAngle);
    delay(STEP_DELAY_MS);
  }

  Serial.print(F("OK J4 "));
  Serial.println(currentAngle);
}

void processCommand(char *line) {
  char *command = strtok(line, " \t");
  char *argument = strtok(NULL, " \t");
  char *extra = strtok(NULL, " \t");

  if (command == NULL) return;

  if (strcmp(command, "PING") == 0 && argument == NULL) {
    Serial.println(F("PONG J4_CAL_V1"));
    return;
  }

  if (strcmp(command, "STATUS") == 0 && argument == NULL) {
    printStatus();
    return;
  }

  if (strcmp(command, "ATTACH") == 0 && extra == NULL) {
    int requestedAngle;
    if (!parseAngle(argument, requestedAngle)) {
      Serial.println(F("ERR ANGLE_RANGE_0_180"));
      return;
    }
    if (attached) {
      Serial.println(F("ERR ALREADY_ATTACHED"));
      return;
    }

    currentAngle = requestedAngle;
    wrist.write(currentAngle);
    wrist.attach(WRIST_PIN);
    wrist.write(currentAngle);
    attached = true;
    Serial.print(F("OK ATTACHED J4 "));
    Serial.println(currentAngle);
    return;
  }

  if (strcmp(command, "SET") == 0 && extra == NULL) {
    int requestedAngle;
    if (!parseAngle(argument, requestedAngle)) {
      Serial.println(F("ERR ANGLE_RANGE_0_180"));
      return;
    }
    moveSlowlyTo(requestedAngle);
    return;
  }

  if (strcmp(command, "DETACH") == 0 && argument == NULL) {
    wrist.detach();
    attached = false;
    Serial.println(F("OK DETACHED J4"));
    return;
  }

  Serial.println(F("ERR COMMAND"));
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("READY J4_CAL_V1 DETACHED"));
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
