#include <WiFi.h>
#include <ArduinoOTA.h>

// Define relay pins for motors
#define RELAY_LEFT_MOTOR_PIN 2  // Pin D2 controls the left motor
#define RELAY_RIGHT_MOTOR_PIN 4 // Pin D4 controls the right motor

// Define Serial2 pins for sensor communication
#define RX2_PIN 16 // RX2 pin (connect to sensor TX)
#define TX2_PIN 17 // TX2 pin (connect to sensor RX)

// Wi-Fi credentials
const char* ssid = "Glasshouse";
const char* password = "Jagerhouse";

// Server configuration
WiFiServer server(8080); // Port 8080 for communication

// Function prototypes
void fetchSensorReading(WiFiClient& client);
void processCommand(const String& command, WiFiClient& client);
void executeTurn(int angle);
void executeMove(int distance);

void setup() {
  // Initialize Serial Communication
  Serial.begin(115200);

  // Connect to Wi-Fi
  Serial.println("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Start OTA setup
  ArduinoOTA.setHostname("ESP32-OTA"); // Optional: Set a hostname
  ArduinoOTA.setPassword("esp32pass"); // Optional: Set an OTA password

  ArduinoOTA.onStart([]() {
    String type = (ArduinoOTA.getCommand() == U_FLASH) ? "sketch" : "filesystem";
    Serial.println("Start updating " + type);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nOTA Update Complete.");
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("OTA Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
    else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
    else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
    else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
    else if (error == OTA_END_ERROR) Serial.println("End Failed");
  });
  ArduinoOTA.begin();
  Serial.println("OTA ready.");

  // Start server
  server.begin();
  Serial.println("Server started, waiting for clients...");

  // Initialize relay pins
  pinMode(RELAY_LEFT_MOTOR_PIN, OUTPUT);
  digitalWrite(RELAY_LEFT_MOTOR_PIN, HIGH); // Default off

  pinMode(RELAY_RIGHT_MOTOR_PIN, OUTPUT);
  digitalWrite(RELAY_RIGHT_MOTOR_PIN, HIGH); // Default off

  // Initialize Serial2 for sensor communication
  Serial2.begin(460800, SERIAL_8N1, RX2_PIN, TX2_PIN);
  Serial.println("UART communication with sensor initialized.");
}

void loop() {
  // Handle OTA updates
  ArduinoOTA.handle();

  // Check for incoming client connections
  WiFiClient client = server.available();
  if (client) {
    Serial.println("Client connected.");
    while (client.connected()) {
      if (client.available()) {
        String command = client.readStringUntil('\n');
        command.trim();
        Serial.println("Received command: " + command);
        processCommand(command, client);
      }
    }
    client.stop();
    Serial.println("Client disconnected.");
  }
}

void processCommand(const String& command, WiFiClient& client) {
  if (command.equalsIgnoreCase("FETCH")) {
    fetchSensorReading(client);
  } else if (command.startsWith("TURN")) {
    int angle = command.substring(5).toInt();
    client.println("Executing: TURN");
    executeTurn(angle);
  } else if (command.startsWith("MOVE")) {
    int distance = command.substring(5).toInt();
    client.println("Executing: MOVE");
    executeMove(distance);
  } else {
    client.println("Invalid Command! Use 'FETCH', 'TURN,angle', or 'MOVE,distance'.");
  }
}

void fetchSensorReading(WiFiClient& client) {
  // Send 'b' command to start sensor readings
  Serial2.print("b\r\n");
  Serial.println("Sent 'b' command to sensor.");
  
  // Flush Serial2 input
  while (Serial2.available()) {
    Serial2.read();
  }

  String adcData = "";

  // Wait for "[DEBUG] ADC Data:["
  while (true) {
    if (Serial2.available()) {
      char ch = Serial2.read();
      adcData += ch;

      if (adcData.endsWith("[DEBUG] ADC Data:[")) {
        adcData = ""; // Clear buffer to start collecting ADC values
        break;
      }
    }
  }

  Serial.println("Found ADC Data start.");

  // Read and parse ADC values until "]"
  while (true) {
    if (Serial2.available()) {
      char ch = Serial2.read();
      if (ch == ']') {
        break; // End of ADC data
      }
      adcData += ch;
    }
  }

  // Split and parse the ADC data
  Serial.println("Parsing ADC Data...");
  client.println("ADC Data:");
  int currentIndex = 0;
  while (currentIndex < adcData.length()) {
    int nextComma = adcData.indexOf(',', currentIndex);
    if (nextComma == -1) nextComma = adcData.length();

    String value = adcData.substring(currentIndex, nextComma);
    client.println(value); // Send each value back to the client
    currentIndex = nextComma + 1;
  }

  // Send 's' command to stop the sensor
  Serial2.print("s\n");
  Serial.println("Sent 's' command to sensor.");

  // Clear Serial2 buffer
  while (Serial2.available()) {
    Serial2.read();
  }

  Serial.println("Sensor reading completed.");
}

void executeTurn(int angle) {
  if (angle > 0) {
    // Positive angle: turn right (activate left motor)
    Serial.println("Turning Right: Activating Left Motor");
    digitalWrite(RELAY_LEFT_MOTOR_PIN, LOW); // Activate left motor
    delay(abs(angle) * 10); // Duration proportional to angle
    digitalWrite(RELAY_LEFT_MOTOR_PIN, HIGH); // Deactivate left motor
  } else if (angle < 0) {
    // Negative angle: turn left (activate right motor)
    Serial.println("Turning Left: Activating Right Motor");
    digitalWrite(RELAY_RIGHT_MOTOR_PIN, LOW); // Activate right motor
    delay(abs(angle) * 10); // Duration proportional to angle
    digitalWrite(RELAY_RIGHT_MOTOR_PIN, HIGH); // Deactivate right motor
  } else {
    Serial.println("No Turn: Angle is zero.");
  }
}

void executeMove(int distance) {
  // Activate both motors to move forward
  Serial.println("Moving Forward: Activating Both Motors");
  digitalWrite(RELAY_LEFT_MOTOR_PIN, LOW); // Activate left motor
  digitalWrite(RELAY_RIGHT_MOTOR_PIN, LOW); // Activate right motor
  delay(distance * 10); // Duration proportional to distance
  digitalWrite(RELAY_LEFT_MOTOR_PIN, HIGH); // Deactivate left motor
  digitalWrite(RELAY_RIGHT_MOTOR_PIN, HIGH); // Deactivate right motor
}
