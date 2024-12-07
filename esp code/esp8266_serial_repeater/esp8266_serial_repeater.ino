

// #include <WiFi.h>
// #include <ArduinoOTA.h>
#include <ESP8266WiFi.h>


// Replace these with your network credentials
const char* ssid = "phd";        // Your WiFi network SSID
const char* password = "Jagerhouse"; // Your WiFi network password

// #define LED_PIN 2 // Define the LED pin (use GPIO 2 for the onboard LED)

// Communication with Arduino Nano
#define BAUD_RATE 9600

// #define LED_PIN 18

#define LED_PIN 2

WiFiServer server(8080); // Start a server on port 8080

void setup() {
  // Initialize the default serial port for communication with Arduino Nano
  Serial.begin(BAUD_RATE);
  Serial.print("turned on!");
  pinMode(LED_PIN, OUTPUT);

  // Start connecting to WiFi
  WiFi.begin(ssid, password);


while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, HIGH); // Blink ON÷
    delay(250);
    digitalWrite(LED_PIN, LOW);  // Blink OFF
    delay(250);
}
digitalWrite(LED_PIN, HIGH); // Solid ON after connected



  // Print the IP address over Serial
  Serial.println("WiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  // Start the server
  server.begin();

  // // Set up OTA
  // ArduinoOTA.setHostname("ESP8266_OTA_Device"); // Optional: Set a custom hostname for OTA
  // ArduinoOTA.onStart([]() {});
  // ArduinoOTA.onEnd([]() {});
  // ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {});
  // ArduinoOTA.onError([](ota_error_t error) {});
  // ArduinoOTA.begin();
}

void loop() {
  // Handle OTA updates
  // ArduinoOTA.handle();

  // Handle client-server communication
  WiFiClient client = server.available();

  if (client) {
    while (client.connected()) {
      if (client.available()) {
        // Read command from the client
        String command = client.readStringUntil('\n'); // Read until newline
        command.trim(); // Remove any extra spaces or newline characters

        // Check if the command length is sufficient to send to the Nano
        if (command.length() >= 2) {
          // Forward the command to the Arduino Nano
          Serial.println(command);
        }

        // Read response from the Nano
        String nanoResponse = readNanoResponse();
        if (nanoResponse.length() > 0) {
          client.print(nanoResponse);
        }
      }
    }

    client.stop();
  }
}

/**
 * Reads a response from the Arduino Nano, ending with '\n' or timing out.
 *
 * @return The response string from the Nano, or an empty string if timed out.
 */
String readNanoResponse() {
  String response = "";
  unsigned long startTime = millis(); // Record the start time

  while (millis() - startTime < 30000) { // 30-second timeout
    if (Serial.available()) {
      char c = Serial.read();
      response += c;
      if (c == '\n') { // End of message detected
        return response;
      }
    }
    delay(1); // Small delay to avoid busy looping
  }

  // Timeout reached
  return ""; // Return an empty string if no '\n' is received
}
