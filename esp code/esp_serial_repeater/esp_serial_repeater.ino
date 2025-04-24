#include <WiFi.h>
#include <ArduinoOTA.h>

// Replace these with your network credentials
const char* ssid = "phd";        // Your WiFi network SSID
const char* password = "Jagerhouse"; // Your WiFi network password

// Communication with Arduino Nano
#define BAUD_RATE 9600

// LED Pin (use GPIO 2 for onboard LED or another GPIO for external LED)
#define LED_PIN 2

WiFiServer server(8080); // Start a server on port 8080

void setup() {
  // Initialize Serial2 for communication with Arduino Nano
  Serial2.begin(BAUD_RATE, SERIAL_8N1, 16, 17); // GPIO 16 = RX2, GPIO 17 = TX2

  // Initialize debugging Serial
  Serial.begin(115200);

  // Initialize the LED pin
  pinMode(LED_PIN, OUTPUT);

  // Start connecting to WiFi
  Serial.println("[ESP UART1] Connecting to WiFi...");
  WiFi.begin(ssid, password);

  // Blink the LED while connecting to WiFi
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LED_PIN, HIGH); // Turn LED on
    delay(250);                  // Wait 250ms
    digitalWrite(LED_PIN, LOW);  // Turn LED off
    delay(250);                  // Wait 250ms
    Serial.print(".");
  }

  // Turn the LED solid ON once connected
  digitalWrite(LED_PIN, HIGH);
  Serial.println("\n[ESP UART1] WiFi Connected");
  Serial.print("[ESP UART1] IP Address: ");
  Serial.println(WiFi.localIP());

  // Start the server
  server.begin();
  Serial.println("[ESP UART1] Server started");

  // Set up OTA
  ArduinoOTA.setHostname("ESP32_OTA_Device"); // Optional: Set a custom hostname for OTA
  ArduinoOTA.onStart([]() {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH) {
      type = "sketch";
    } else { // U_SPIFFS
      type = "filesystem";
    }
    Serial.println("[OTA] Start updating " + type);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\n[OTA] Update complete!");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("[OTA] Progress: %u%%\r", (progress * 100) / total);
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("[OTA] Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) {
      Serial.println("Auth Failed");
    } else if (error == OTA_BEGIN_ERROR) {
      Serial.println("Begin Failed");
    } else if (error == OTA_CONNECT_ERROR) {
      Serial.println("Connect Failed");
    } else if (error == OTA_RECEIVE_ERROR) {
      Serial.println("Receive Failed");
    } else if (error == OTA_END_ERROR) {
      Serial.println("End Failed");
    }
  });
  ArduinoOTA.begin();
  Serial.println("[ESP UART1] OTA Ready");
}

void loop() {
  // Handle OTA updates
  ArduinoOTA.handle();

  // Handle client-server communication
  WiFiClient client = server.available();

  if (client) {
    Serial.println("[ESP UART1] Client Connected");

    while (client.connected()) {
      if (client.available()) {
        // Read command from the client
        String command = client.readStringUntil('\n');
        command.trim(); // Remove any extra spaces or newline characters
        Serial.println("[ESP UART1] Received Command: " + command);

        // Forward the command to the Arduino Nano via Serial2
        Serial2.println(command);
        Serial.println("[ESP UART2] Forwarded Command to Nano: " + command);
      }

      if (Serial2.available()) {
        // Relay Arduino response back to client
        String nanoResponse = "";
        while (Serial2.available()) {
          nanoResponse += char(Serial2.read());
        }
        client.print(nanoResponse);
        Serial.println("[Nano] Response: " + nanoResponse);
      }
    }

    client.stop();
    Serial.println("[ESP UART1] Client Disconnected");
  }
}
