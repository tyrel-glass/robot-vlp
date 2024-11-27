//=========== Encoder Pins ===========
#define ENCA 2         // Encoder A Signal A (Motor A) - INT0
#define ENCB 3         // Encoder A Signal B (Motor A) - INT1

#define ENCA2 A0       // Encoder B Signal A (Motor B) - PCINT8
#define ENCB2 A1       // Encoder B Signal B (Motor B) - PCINT9

//=========== Motor Pins ===========
#define PWMA 5
#define AIN2 6
#define AIN1 7
#define STBY 8
#define BIN1 9
#define BIN2 10
#define PWMB 11

//=========== Global Variables ===========
volatile int encoderCountA = 0;  // Encoder A count
volatile int encoderCountB = 0;  // Encoder B count (Motor B)
int lastCountA = 0;              // Previous count for A
int lastCountB = 0;              // Previous count for B
float speedA = 0;                // Calculated speed for A
float speedB = 0;                // Calculated speed for B
int pwmA = 100;                  // Initial PWM for Motor A
int pwmB = 100;                  // Initial PWM for Motor B
float counts_per_degree = 10.0;  // Adjust based on encoder and gear ratio
int pwm = 100;                   // Base motor speed
float Kp = 2.0;                  // Proportional constant for speed correction

//=========== Interrupt Handlers for Motor A ===========
void encoderA_ISR() {
  bool stateA = digitalRead(ENCA);
  bool stateB = digitalRead(ENCB);

  encoderCountA += (stateA == stateB) ? -1 : 1;  // Inline direction logic
}

//=========== Pin Change Interrupt Handler for Motor B ===========
ISR(PCINT1_vect) {  // Handles PCINT[8:14] (PORTC pins)
  static bool lastStateA2 = LOW;
  static bool lastStateB2 = LOW;

  bool stateA2 = digitalRead(ENCA2);
  bool stateB2 = digitalRead(ENCB2);

  // Determine direction of rotation
  if (stateA2 != lastStateA2) {
    encoderCountB += (stateA2 == stateB2) ? 1 : -1;  // Reverse logic for Motor B
  }

  lastStateA2 = stateA2;
  lastStateB2 = stateB2;
}

//=========================================================================
// SETUP
//=========================================================================

void setup() {
  Serial.begin(9600);

  // Motor Pins
  pinMode(PWMA, OUTPUT);
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMB, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT);

  // Encoder Pins
  pinMode(ENCA, INPUT_PULLUP);
  pinMode(ENCB, INPUT_PULLUP);
  pinMode(ENCA2, INPUT_PULLUP);
  pinMode(ENCB2, INPUT_PULLUP);

  // Attach Hardware Interrupts for Motor A
  attachInterrupt(digitalPinToInterrupt(ENCA), encoderA_ISR, CHANGE);

  // Enable Pin Change Interrupts for Motor B (PORTC)
  PCICR |= (1 << PCIE1);             // Enable pin change interrupts for PORTC
  PCMSK1 = (1 << PCINT8) | (1 << PCINT9);  // Enable PCINT for A0 (ENCA2) and A1 (ENCB2)


  move(0,50,0);
  move(1,50,0);
  Serial.println("SETUP COMPLETE");
}

//=========================================================================
// MAIN LOOP
//=========================================================================

void loop() {
  static unsigned long lastTime = 0;
  unsigned long currentTime = millis();

  // Calculate speed and adjust PWM every 100 ms
  if (currentTime - lastTime >= 100) {
    lastTime = currentTime;

    calculateSpeed();           // Calculate encoder speeds
    adjustPWM(60);              // Adjust PWM to maintain target speed of 20 counts/interval
  }


  // Optional: Print speeds and counts for debugging
  Serial.print("Speed A: ");
  Serial.print(speedA);
  Serial.print(" | Speed B: ");
  Serial.print(speedB);
  Serial.print(" | EncA_Count: ");
  Serial.print(encoderCountA);
  Serial.print(" | EncB_Count: ");
  Serial.print(encoderCountB);
  Serial.print(" | PWM A: ");
  Serial.print(pwmA);
  Serial.print(" | PWM B: ");
  Serial.println(pwmB);

  delay(100);
}

//=========================================================================
// HELPER FUNCTIONS
//=========================================================================

void calculateSpeed() {
  noInterrupts(); // Disable interrupts to read encoder counts safely
  int currentCountA = encoderCountA;
  int currentCountB = encoderCountB;
  interrupts();   // Re-enable interrupts

  // Calculate speeds (counts per interval)
  speedA = currentCountA - lastCountA;
  speedB = currentCountB - lastCountB;

  // Update last counts
  lastCountA = currentCountA;
  lastCountB = currentCountB;
}

void adjustPWM(float targetSpeed) {
  // Calculate corrections based on speed differences
  float errorA = targetSpeed - speedA;
  float errorB = targetSpeed - speedB;

  // Adjust PWM using proportional control
  pwmA += Kp * errorA;
  pwmB += Kp * errorB;

  // Constrain PWM values to valid range (0-255)
  pwmA = constrain(pwmA, 0, 255);
  pwmB = constrain(pwmB, 0, 255);

  // Apply adjusted PWM values to motors
  analogWrite(PWMA, pwmA);
  analogWrite(PWMB, pwmB);
}

//=========================================================================
// MOVEMENT CONTROL
//=========================================================================

void rotate(float degrees) {
  int targetCount = abs(degrees) * counts_per_degree;

  noInterrupts();
  int startCountA = encoderCountA;
  int startCountB = encoderCountB;
  interrupts();

  if (degrees > 0) {
    // Rotate right
    move(1, pwmA, 1);
    move(2, pwmB, 0);
    while ((encoderCountA - startCountA < targetCount) &&
           (encoderCountB - startCountB < targetCount)) {
      calculateSpeed();
      adjustPWM(10); // Adjust for rotational consistency
    }
  } else {
    // Rotate left
    move(1, pwmA, 0);
    move(2, pwmB, 1);
    while ((encoderCountA - startCountA > -targetCount) &&
           (encoderCountB - startCountB > -targetCount)) {
      calculateSpeed();
      adjustPWM(10); // Adjust for rotational consistency
    }
  }

  stop();
}


void stop() {
  digitalWrite(STBY, LOW);  // Stop the motors
}

void move(int motor, int speed, int direction) {
  digitalWrite(STBY, HIGH); // Disable standby

  boolean inPin1 = LOW;
  boolean inPin2 = HIGH;

  if (direction == 1) {
    inPin1 = HIGH;
    inPin2 = LOW;
  }

  if (motor == 1) {
    digitalWrite(AIN1, inPin1);
    digitalWrite(AIN2, inPin2);
    analogWrite(PWMA, speed);
  } else {
    digitalWrite(BIN1, inPin1);
    digitalWrite(BIN2, inPin2);
    analogWrite(PWMB, speed);
  }
}
