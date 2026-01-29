#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create PCA9685 object
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo channels on PCA9685
#define THUMB  0
#define INDEX  1
#define MIDDLE 2
#define RING   3
#define PINKY  4

// Servo pulse length limits (adjust based on your servos)
#define SERVOMIN  125  // Min pulse length (0 degrees)
#define SERVOMAX  575  // Max pulse length (180 degrees)

void setup() {
  Serial.begin(115200);
  
  // Initialize PCA9685
  pwm.begin();
  pwm.setPWMFreq(50);  // Servos run at 50Hz
  
  delay(10);
  
  // Initialize servos to open position (0 degrees)
  setServoAngle(THUMB, 0);
  setServoAngle(INDEX, 0);
  setServoAngle(MIDDLE, 0);
  setServoAngle(RING, 0);
  setServoAngle(PINKY, 0);
  
  Serial.println("ESP32 + PCA9685 Servo Hand Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    
    // Expected format: "T:90,I:45,M:60,R:30,P:75"
    int thumbAngle = parseValue(data, "T:");
    int indexAngle = parseValue(data, "I:");
    int middleAngle = parseValue(data, "M:");
    int ringAngle = parseValue(data, "R:");
    int pinkyAngle = parseValue(data, "P:");
    
    // Move servos if valid angles received
    if (thumbAngle >= 0) setServoAngle(THUMB, constrain(thumbAngle, 0, 180));
    if (indexAngle >= 0) setServoAngle(INDEX, constrain(indexAngle, 0, 180));
    if (middleAngle >= 0) setServoAngle(MIDDLE, constrain(middleAngle, 0, 180));
    if (ringAngle >= 0) setServoAngle(RING, constrain(ringAngle, 0, 180));
    if (pinkyAngle >= 0) setServoAngle(PINKY, constrain(pinkyAngle, 0, 180));
  }
}

// Convert angle (0-180) to pulse length and set servo
void setServoAngle(uint8_t channel, int angle) {
  int pulse = map(angle, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel, 0, pulse);
}

// Parse value from string like "T:90"
int parseValue(String data, String key) {
  int keyIndex = data.indexOf(key);
  if (keyIndex == -1) return -1;
  
  int startIndex = keyIndex + key.length();
  int endIndex = data.indexOf(',', startIndex);
  if (endIndex == -1) endIndex = data.length();
  
  String valueStr = data.substring(startIndex, endIndex);
  return valueStr.toInt();
}