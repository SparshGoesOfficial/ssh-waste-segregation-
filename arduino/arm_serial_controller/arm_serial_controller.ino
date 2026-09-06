#include <Servo.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// ARM_SERIAL_V1
// Laptop/Pi sends newline-delimited ASCII commands. The controller starts
// DISARMED: no Servo object is attached until a valid ARM command arrives.

const uint8_t JOINT_COUNT = 5;
const uint8_t SERVO_PINS[JOINT_COUNT] = {9, 10, 11, 6, 5};
const int MIN_ANGLES[JOINT_COUNT] = {0, 20, 10, 0, 15};
const int MAX_ANGLES[JOINT_COUNT] = {180, 160, 170, 180, 50};
const int STARTUP_ANGLES[JOINT_COUNT] = {90, 90, 90, 90, 50};
const int GRIPPER_OPEN_ANGLE = 50;
const int GRIPPER_CLOSED_ANGLE = 15;

const unsigned long MIN_DURATION_MS = 100;
const unsigned long MAX_DURATION_MS = 10000;
const unsigned long MOTION_WATCHDOG_MS = 750;
const unsigned long SERVO_UPDATE_MS = 20;

Servo servos[JOINT_COUNT];
float currentAngles[JOINT_COUNT];
float motionStartAngles[JOINT_COUNT];
float targetAngles[JOINT_COUNT];

bool armed = false;
bool moving = false;
unsigned long motionStartedAt = 0;
unsigned long motionDurationMs = 0;
unsigned long lastServoUpdateAt = 0;
unsigned long lastValidCommandAt = 0;
char activeCommandId[16] = "";

char lineBuffer[112];
uint8_t lineLength = 0;
bool lineOverflow = false;

void replyError(const __FlashStringHelper *code) {
  Serial.print(F("ERR "));
  Serial.println(code);
}

bool tokenToLong(const char *token, long &value) {
  if (token == NULL || *token == '\0') return false;
  char *end = NULL;
  value = strtol(token, &end, 10);
  return end != token && *end == '\0';
}

bool validCommandId(const char *token) {
  if (token == NULL) return false;
  size_t length = strlen(token);
  if (length == 0 || length >= sizeof(activeCommandId)) return false;
  for (size_t i = 0; i < length; ++i) {
    if (!isalnum(static_cast<unsigned char>(token[i])) && token[i] != '_') return false;
  }
  return true;
}

bool angleIsSafe(uint8_t joint, long angle) {
  return angle >= MIN_ANGLES[joint] && angle <= MAX_ANGLES[joint];
}

void writeCurrentAngles() {
  if (!armed) return;
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    servos[i].write(static_cast<int>(currentAngles[i] + 0.5f));
  }
}

void attachServos() {
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    servos[i].attach(SERVO_PINS[i]);
  }
  armed = true;
  writeCurrentAngles();
}

void detachServos() {
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    servos[i].detach();
  }
  armed = false;
  moving = false;
  activeCommandId[0] = '\0';
}

void printAngles() {
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    Serial.print(' ');
    Serial.print(static_cast<int>(currentAngles[i] + 0.5f));
  }
}

void printStatus() {
  Serial.print(F("STATUS "));
  Serial.print(armed ? F("ARMED ") : F("DISARMED "));
  Serial.print(moving ? F("MOVING") : F("IDLE"));
  printAngles();
  Serial.println();
}

void printLimits() {
  Serial.print(F("LIMITS"));
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    Serial.print(' ');
    Serial.print(MIN_ANGLES[i]);
    Serial.print(':');
    Serial.print(MAX_ANGLES[i]);
  }
  Serial.println();
}

bool parseFiveAngles(char *tokens[], uint8_t startIndex, long parsed[]) {
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    if (!tokenToLong(tokens[startIndex + i], parsed[i])) return false;
    if (!angleIsSafe(i, parsed[i])) return false;
  }
  return true;
}

void beginMotion(const char *commandId, const long requested[], unsigned long durationMs) {
  strncpy(activeCommandId, commandId, sizeof(activeCommandId) - 1);
  activeCommandId[sizeof(activeCommandId) - 1] = '\0';
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    motionStartAngles[i] = currentAngles[i];
    targetAngles[i] = requested[i];
  }
  motionStartedAt = millis();
  lastServoUpdateAt = 0;
  motionDurationMs = durationMs;
  moving = true;
  Serial.print(F("ACK "));
  Serial.println(activeCommandId);
}

void stopMotion(const __FlashStringHelper *reason) {
  if (moving) {
    for (uint8_t i = 0; i < JOINT_COUNT; ++i) targetAngles[i] = currentAngles[i];
    moving = false;
    activeCommandId[0] = '\0';
  }
  Serial.println(reason);
}

void updateMotion() {
  if (!moving) return;
  unsigned long now = millis();
  if (now - lastValidCommandAt > MOTION_WATCHDOG_MS) {
    stopMotion(F("ERR WATCHDOG"));
    return;
  }
  if (lastServoUpdateAt != 0 && now - lastServoUpdateAt < SERVO_UPDATE_MS) return;
  lastServoUpdateAt = now;

  unsigned long elapsed = now - motionStartedAt;
  float fraction = elapsed >= motionDurationMs ? 1.0f : static_cast<float>(elapsed) / motionDurationMs;
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    currentAngles[i] = motionStartAngles[i] + (targetAngles[i] - motionStartAngles[i]) * fraction;
  }
  writeCurrentAngles();

  if (fraction >= 1.0f) {
    moving = false;
    Serial.print(F("DONE "));
    Serial.println(activeCommandId);
    activeCommandId[0] = '\0';
  }
}

uint8_t splitTokens(char *line, char *tokens[], uint8_t capacity) {
  uint8_t count = 0;
  char *token = strtok(line, " \t");
  while (token != NULL && count < capacity) {
    tokens[count++] = token;
    token = strtok(NULL, " \t");
  }
  return count;
}

void processCommand(char *line) {
  char *tokens[10] = {NULL};
  uint8_t count = splitTokens(line, tokens, 10);
  if (count == 0) return;
  lastValidCommandAt = millis();

  if (strcmp(tokens[0], "PING") == 0 && count == 1) {
    Serial.println(F("PONG ARM_SERIAL_V1"));
    return;
  }
  if (strcmp(tokens[0], "KEEPALIVE") == 0 && count == 1) return;
  if (strcmp(tokens[0], "STATUS") == 0 && count == 1) {
    printStatus();
    return;
  }
  if (strcmp(tokens[0], "LIMITS") == 0 && count == 1) {
    printLimits();
    return;
  }
  if (strcmp(tokens[0], "STOP") == 0 && count == 1) {
    stopMotion(F("OK STOPPED"));
    return;
  }
  if (strcmp(tokens[0], "DISARM") == 0 && count == 1) {
    if (moving) {
      replyError(F("BUSY"));
      return;
    }
    detachServos();
    Serial.println(F("OK DISARMED"));
    return;
  }
  if (strcmp(tokens[0], "ARM") == 0) {
    if (count != 6) {
      replyError(F("SYNTAX"));
      return;
    }
    if (armed) {
      replyError(F("ALREADY_ARMED"));
      return;
    }
    long requested[JOINT_COUNT];
    if (!parseFiveAngles(tokens, 1, requested)) {
      replyError(F("ANGLE_LIMIT"));
      return;
    }
    for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
      currentAngles[i] = requested[i];
      targetAngles[i] = requested[i];
    }
    attachServos();
    Serial.println(F("OK ARMED"));
    return;
  }
  if (strcmp(tokens[0], "MOVE") == 0) {
    if (count != 8 || !validCommandId(tokens[1])) {
      replyError(F("SYNTAX"));
      return;
    }
    if (!armed) {
      replyError(F("DISARMED"));
      return;
    }
    if (moving) {
      replyError(F("BUSY"));
      return;
    }
    long duration;
    long requested[JOINT_COUNT];
    if (!tokenToLong(tokens[2], duration) || duration < static_cast<long>(MIN_DURATION_MS) ||
        duration > static_cast<long>(MAX_DURATION_MS)) {
      replyError(F("DURATION"));
      return;
    }
    if (!parseFiveAngles(tokens, 3, requested)) {
      replyError(F("ANGLE_LIMIT"));
      return;
    }
    beginMotion(tokens[1], requested, static_cast<unsigned long>(duration));
    return;
  }
  if (strcmp(tokens[0], "JOINT") == 0) {
    if (count != 5 || !validCommandId(tokens[1])) {
      replyError(F("SYNTAX"));
      return;
    }
    if (!armed) {
      replyError(F("DISARMED"));
      return;
    }
    if (moving) {
      replyError(F("BUSY"));
      return;
    }
    long jointNumber;
    long angle;
    long duration;
    if (!tokenToLong(tokens[2], jointNumber) || jointNumber < 1 || jointNumber > JOINT_COUNT ||
        !tokenToLong(tokens[3], angle) || !angleIsSafe(jointNumber - 1, angle) ||
        !tokenToLong(tokens[4], duration) || duration < static_cast<long>(MIN_DURATION_MS) ||
        duration > static_cast<long>(MAX_DURATION_MS)) {
      replyError(F("RANGE"));
      return;
    }
    long requested[JOINT_COUNT];
    for (uint8_t i = 0; i < JOINT_COUNT; ++i) requested[i] = static_cast<long>(currentAngles[i] + 0.5f);
    requested[jointNumber - 1] = angle;
    beginMotion(tokens[1], requested, static_cast<unsigned long>(duration));
    return;
  }
  if (strcmp(tokens[0], "GRIP") == 0) {
    if (count != 4 || !validCommandId(tokens[1])) {
      replyError(F("SYNTAX"));
      return;
    }
    if (!armed) {
      replyError(F("DISARMED"));
      return;
    }
    if (moving) {
      replyError(F("BUSY"));
      return;
    }
    long duration;
    if (!tokenToLong(tokens[3], duration) || duration < static_cast<long>(MIN_DURATION_MS) ||
        duration > static_cast<long>(MAX_DURATION_MS)) {
      replyError(F("DURATION"));
      return;
    }
    int gripperAngle;
    if (strcmp(tokens[2], "OPEN") == 0) gripperAngle = GRIPPER_OPEN_ANGLE;
    else if (strcmp(tokens[2], "CLOSE") == 0) gripperAngle = GRIPPER_CLOSED_ANGLE;
    else {
      replyError(F("GRIP_STATE"));
      return;
    }
    long requested[JOINT_COUNT];
    for (uint8_t i = 0; i < JOINT_COUNT; ++i) requested[i] = static_cast<long>(currentAngles[i] + 0.5f);
    requested[4] = gripperAngle;
    beginMotion(tokens[1], requested, static_cast<unsigned long>(duration));
    return;
  }

  replyError(F("UNKNOWN_COMMAND"));
}

void readSerialLines() {
  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());
    if (incoming == '\r') continue;
    if (incoming == '\n') {
      if (lineOverflow) {
        replyError(F("LINE_TOO_LONG"));
      } else {
        lineBuffer[lineLength] = '\0';
        processCommand(lineBuffer);
      }
      lineLength = 0;
      lineOverflow = false;
      continue;
    }
    if (lineLength < sizeof(lineBuffer) - 1) lineBuffer[lineLength++] = incoming;
    else lineOverflow = true;
  }
}

void setup() {
  for (uint8_t i = 0; i < JOINT_COUNT; ++i) {
    currentAngles[i] = STARTUP_ANGLES[i];
    targetAngles[i] = STARTUP_ANGLES[i];
  }
  Serial.begin(115200);
  lastValidCommandAt = millis();
  Serial.println(F("READY ARM_SERIAL_V1"));
}

void loop() {
  readSerialLines();
  updateMotion();
}
