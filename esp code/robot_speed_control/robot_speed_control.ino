/*
===============================================================================
                              ROBOT CONTROL SYSTEM
===============================================================================
  This program controls a two-motor robotic platform using encoder feedback 
  to provide precise movement. Serial commands are used to control the robot, 
  with support for forward movement, rotations, PID tuning, and debug options.

------------------------------------------------------------------------------
                              SERIAL COMMANDS
------------------------------------------------------------------------------
1. Movement Commands:
   - MOVE:<encoder_counts>
     Moves the robot forward for the specified encoder count distance.

   - TURN:<encoder_counts>
     Rotates the robot by the specified encoder count distance.

   - FORWARD:<time_in_ms>
     Drives both motors forward at a constant power for the specified time 
     in milliseconds. Power is determined by the global variable `forwardPower`.

   - ROTATE:<time_in_ms>
     Rotates the robot in place. Positive values rotate clockwise for the 
     specified time in milliseconds. Negative values rotate counterclockwise. 
     Power is determined by the global variable `rotatePower`.

   - STOP
     Stops both motors immediately.

2. Setting Parameters:
   - SET:KP:<value>
     Updates the proportional constant `Kp` used in speed control.

   - SET:TS:<value>
     Updates the target speed for the PID controller in encoder counts 
     per second.

   - SET:FPOWER:<value>
     Updates the forward driving power (PWM) for the `FORWARD` command.

   - SET:RPOWER:<value>
     Updates the rotation power (PWM) for the `ROTATE` command.

3. Debugging:
   - DEBUG:ON
     Enables debugging mode, providing encoder and speed updates on Serial.

   - DEBUG:OFF
     Disables debugging mode.

------------------------------------------------------------------------------
                             FUNCTIONALITY OVERVIEW
------------------------------------------------------------------------------
1. Encoder Feedback:
   The program tracks encoder counts for both motors to provide real-time 
   feedback for precise movement.

2. PID Speed Control:
   Implements a simple proportional control loop to maintain consistent 
   motor speeds based on the target speed.

3. Motor States:
   Motors can be controlled independently for movement (both forward and 
   backward), as well as for rotational commands.

4. Adjustable PWM Power:
   Global power values for forward motion (`forwardPower`) and rotation 
   (`rotatePower`) are adjustable at runtime via the `SET` commands.

------------------------------------------------------------------------------
*/


#define LED_PIN 13

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
volatile int encoderCountB = 0;  // Encoder B count
int lastCountA = 0;              // Previous count for A
int lastCountB = 0;              // Previous count for B
float speedA = 0;                // Speed for Motor A (encoder counts per second)
float speedB = 0;                // Speed for Motor B (encoder counts per second)
int pwmA = 50;                  // PWM for Motor A
int pwmB = 50;                  // PWM for Motor B
int pwm = 50;                   // Base motor speed
int forwardPower = 50;           // Default power for FORWARD command
int rotatePower = 50;            // Default power for ROTATE command
float Kp = 0.1;                  // Proportional constant
float targetSpeed = 200.0;       // Target speed in encoder counts per second
bool debugEnabled = false;       // Debug flag

unsigned long lastSpeedUpdateTime = 0; // Last time speed was updated (ms)

//=========== Interrupt Handlers ===========
void encoderA_ISR() {
  bool stateA = digitalRead(ENCA);
  bool stateB = digitalRead(ENCB);
  encoderCountA += (stateA == stateB) ? -1 : 1;
}

ISR(PCINT1_vect) {  // Handles PCINT[8:14] (PORTC pins)
  static bool lastStateA2 = LOW;
  static bool lastStateB2 = LOW;
  bool stateA2 = digitalRead(ENCA2);
  bool stateB2 = digitalRead(ENCB2);
  if (stateA2 != lastStateA2) {
    encoderCountB += (stateA2 == stateB2) ? 1 : -1;
  }
  lastStateA2 = stateA2;
  lastStateB2 = stateB2;
}

//=========================================================================
// SETUP
//=========================================================================
void setup() {
  Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

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

  attachInterrupt(digitalPinToInterrupt(ENCA), encoderA_ISR, CHANGE);
  PCICR |= (1 << PCIE1);             // Enable pin change interrupts for PORTC
  PCMSK1 = (1 << PCINT8) | (1 << PCINT9);  // Enable PCINT for A0 (ENCA2) and A1 (ENCB2)
  Serial.println("SETUP COMPLETED!");

  // setMotorState(1, 140, 0);
  // setMotorState(0, 140, 0);
}

//=========================================================================
// MAIN LOOP
//=========================================================================
void loop() {
  processSerialInput(); // Process incoming serial commands
  debugPrint();         // Output debug information at regular intervals
}

//=========================================================================
// SERIAL COMMAND PROCESSING
//=========================================================================
void processSerialInput() {
  static String commandBuffer = "";
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      commandBuffer.trim();
      processCommand(commandBuffer);
      commandBuffer = "";
    } else {
      commandBuffer += c;
    }
  }
}

void processCommand(String command) {
  command.trim();
  if (command.startsWith("MOVE:")) {
    int targetCounts = command.substring(5).toInt();
    Serial.print("Moving forward ");
    Serial.print(targetCounts);
    Serial.println(" encoder counts...");
    move(targetCounts);
    Serial.println("Move done");
  } else if (command.startsWith("TURN:")) {
    int targetCounts = command.substring(5).toInt();
    Serial.print("Turning ");
    Serial.print(targetCounts);
    Serial.println(" encoder counts...");
    rotate(targetCounts);
    Serial.println("Turn done");
  } else if (command.startsWith("SET:KP:")) {
    Kp = command.substring(7).toFloat();
    Serial.print("Updated Kp to: ");
    Serial.println(Kp);
  } else if (command.startsWith("SET:TS:")) {
    targetSpeed = command.substring(7).toFloat();
    Serial.print("Updated targetSpeed to: ");
    Serial.println(targetSpeed);
  } else if (command.startsWith("SET:FPOWER:")) {
    forwardPower = command.substring(11).toInt();
    Serial.print("Updated forwardPower to: ");
    Serial.println(forwardPower);
  } else if (command.startsWith("FORWARD:")) {
    int time = command.substring(8).toInt();
    Serial.print("Driving forward for ");
    Serial.print(time);
    Serial.println(" ms...");
    forward(time);
    Serial.println("Forward done");
  } else if (command.startsWith("SET:RPOWER:")) {
    rotatePower = command.substring(11).toInt();
    Serial.print("Updated rotatePower to: ");
    Serial.println(rotatePower);
  } else if (command.startsWith("ROTATE:")) {
    int time = command.substring(7).toInt();
    Serial.print("Rotating for ");
    Serial.print(time);
    Serial.println(" ms...");
    timedRotate(time);
    Serial.println("Rotate done");
  } else if (command == "DEBUG:ON") {
    debugEnabled = true;
    Serial.println("Debugging enabled.");
  } else if (command == "DEBUG:OFF") {
    debugEnabled = false;
    Serial.println("Debugging disabled.");
  } else if (command == "STOP") {
    stop();
    Serial.println("Motors stopped.");
  } else if (command.startsWith("RUN:")) {
    // Format: RUN:<motor>:<power>:<duration>
    // Example: RUN:1:100:1000 or RUN:2:-120:1500
    int firstColon = command.indexOf(':', 4);
    int secondColon = command.indexOf(':', firstColon + 1);

    if (firstColon == -1 || secondColon == -1) {
      Serial.println("Invalid RUN command format.");
      return;
    }

    int motor = command.substring(4, firstColon).toInt();
    int power = command.substring(firstColon + 1, secondColon).toInt();
    int duration = command.substring(secondColon + 1).toInt();

    if (motor != 1 && motor != 2) {
      Serial.println("Invalid motor number. Use 1 or 2.");
      return;
    }

    int absPower = abs(power);
    int direction = (power >= 0) ? 1 : 0;

    Serial.print("Running Motor ");
    Serial.print(motor);
    Serial.print(" at power ");
    Serial.print(power);
    Serial.print(" for ");
    Serial.print(duration);
    Serial.println(" ms");

    setMotorState(motor, absPower, direction);
    delay(duration);
    stop();

    Serial.println("Done.");
  } else {
    Serial.println("Unknown command.");
  }
}

//=========================================================================
// FORWARD FUNCTION
//=========================================================================
void forward(int timeInMs) {
  setMotorState(1, forwardPower, 0); // Motor A forward
  setMotorState(2, forwardPower, 0); // Motor B forward

  delay(timeInMs); // Wait for the specified time

  stop(); // Stop motors after delay
}







//=========================================================================
// TIMED ROTATE FUNCTION
//=========================================================================
void timedRotate(int timeInMs) {
  if (timeInMs > 0) {
    // Clockwise rotation
    setMotorState(1, rotatePower, 1); // Motor A forward
    setMotorState(2, rotatePower, 0); // Motor B backward
  } else {
    // Counterclockwise rotation
    setMotorState(1, rotatePower, 0); // Motor A backward
    setMotorState(2, rotatePower, 1); // Motor B forward
  }

  delay(abs(timeInMs)); // Wait for the specified time

  stop(); // Stop motors after delay
}

//=========================================================================
// HELPER FUNCTIONS
//=========================================================================
void debugPrint() {
  static unsigned long lastDebugTime = 0;
  unsigned long currentTime = millis();
  if (debugEnabled && (currentTime - lastDebugTime >= 100)) {

    noInterrupts();
    int currentCountA = encoderCountA;
    int currentCountB = encoderCountB;
    interrupts();

    Serial.print(currentCountA);
    Serial.print(",");
    Serial.print(currentCountB);
    Serial.print(",");
    Serial.print(speedA);
    Serial.print(",");
    Serial.print(speedB);
    Serial.print(",");
    Serial.print(pwmA);
    Serial.print(",");
    Serial.println(pwmB);
    lastDebugTime = currentTime;
  }
}



void calculateSpeed() {
  noInterrupts();
  int currentCountA = encoderCountA;
  int currentCountB = encoderCountB;
  interrupts();

  unsigned long currentTime = millis();
  unsigned long deltaTime = currentTime - lastSpeedUpdateTime;
  lastSpeedUpdateTime = currentTime;

  if (deltaTime > 0) {
    // Calculate absolute speed (encoder counts per second)
    speedA = abs((float)(currentCountA - lastCountA) * 1000.0 / deltaTime);
    speedB = abs((float)(currentCountB - lastCountB) * 1000.0 / deltaTime);
  //   speedA = abs((float)(currentCountA - lastCountA) );
  //   speedB = abs((float)(currentCountB - lastCountB) );
  }

  lastCountA = currentCountA;
  lastCountB = currentCountB;
}



void adjustPWM(bool isTurning = false) {
  // Calculate encoder difference (positional error)
  int encoderDifference = encoderCountA - encoderCountB;

  if (isTurning) {
    // For turning, use absolute encoder differences for control
    encoderDifference = abs(encoderCountA) - abs(encoderCountB);
  }

  // Adjust target speeds based on positional error
  float adjustedTargetSpeedA = targetSpeed - (Kp * 15) * encoderDifference;
  float adjustedTargetSpeedB = targetSpeed + (Kp * 15) * encoderDifference;

  // Calculate speed errors
  float speedErrorA = adjustedTargetSpeedA - speedA;
  float speedErrorB = adjustedTargetSpeedB - speedB;

  // Update PWM values based on speed errors
  pwmA += Kp * speedErrorA;
  pwmB += Kp * speedErrorB;

  // Constrain PWM values to a safe range
  pwmA = constrain(pwmA, 0, 100);
  pwmB = constrain(pwmB, 0, 100);


  // Apply PWM values to the motors
  analogWrite(PWMA, pwmA);
  analogWrite(PWMB, pwmB);
}




//=========================================================================
// MOVEMENT CONTROL
//=========================================================================
void move(int targetCounts) {
  zeroEncoders(); // Reset encoder counts

  // Start both motors moving forward
  setMotorState(1, pwmA, 0);  // Motor A forward
  setMotorState(2, pwmB, 0);  // Motor B forward

  bool motorAStopped = false;
  bool motorBStopped = false;

  while (!motorAStopped || !motorBStopped) {
    noInterrupts();
    int currentCountA = encoderCountA;
    int currentCountB = encoderCountB;
    interrupts();

    // Stop Motor A if it reaches its target
    if (!motorAStopped && (currentCountA >= targetCounts)) {
      setMotorState(1, 0, 0); // Stop Motor A
      motorAStopped = true;
    }

    // Stop Motor B if it reaches its targetF
    if (!motorBStopped && (currentCountB >= targetCounts)) {
      setMotorState(2, 0, 0); // Stop Motor B
      motorBStopped = true;
    }

    // Update speed and PWM every 50ms
    unsigned long currentTime = millis();
    if (currentTime - lastSpeedUpdateTime >= 50) {
      calculateSpeed();
      adjustPWM(); // Use updated adjustPWM() for proportional control
      debugPrint();
    }
  }

  stop(); // Ensure both motors are stopped
}





void rotate(int targetCounts) {
  zeroEncoders(); // Reset encoder counts

  // Determine rotation direction based on the sign of targetCounts
  if (targetCounts > 0) {
    // Clockwise rotation
    setMotorState(1, pwmA, 1);  // Motor A forward
    setMotorState(2, pwmB, 0);  // Motor B backward
  } else {
    // Counterclockwise rotation
    setMotorState(1, pwmA, 0);  // Motor A backward
    setMotorState(2, pwmB, 1);  // Motor B forward
  }

  bool motorAStopped = false;
  bool motorBStopped = false;

  while (!motorAStopped || !motorBStopped) {
    noInterrupts();
    int currentCountA = encoderCountA;
    int currentCountB = encoderCountB;
    interrupts();

    // Stop Motor A if it reaches its target
    if (!motorAStopped && (abs(currentCountA) >= abs(targetCounts))) {
      setMotorState(1, 0, 0); // Stop Motor A
      motorAStopped = true;
    }

    // Stop Motor B if it reaches its target
    if (!motorBStopped && (abs(currentCountB) >= abs(targetCounts))) {
      setMotorState(2, 0, 0); // Stop Motor B
      motorBStopped = true;
    }

    // Update speed and PWM every 50ms
    unsigned long currentTime = millis();
    if (currentTime - lastSpeedUpdateTime >= 50) {
      calculateSpeed();
      adjustPWM(true); // Use updated adjustPWM() with turning flag
      debugPrint();
    }
  }

  stop(); // Ensure both motors are stopped
}


void zeroEncoders() {
  noInterrupts();
  encoderCountA = 0;
  encoderCountB = 0;
  interrupts();
}


//=========================================================================
void setMotorState(int motor, int speed, int direction) {
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

void stop() {
  digitalWrite(STBY, LOW);
}