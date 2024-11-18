# File: send_commands_to_esp32.py

import serial
import time

# Configure the serial connection (adjust port and baudrate as needed)
SERIAL_PORT = '/dev/tty.usbserial-0001'  # Replace with your port (e.g., '/dev/ttyUSB0' on Linux or 'COMx' on Windows)
BAUD_RATE = 115200

def send_command(ser, command):
    """
    Send a command string to the ESP32 via serial.
    """
    ser.write((command + '\n').encode('utf-8'))  # Append newline for ESP32 parsing
    print(f"Sent: {command}")

    # Read the response (optional, for acknowledgment)
    time.sleep(0.5)  # Give ESP32 time to respond
    while ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').strip()
        print(f"ESP32: {response}")

def main():
    try:
        # Open the serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

        # Allow the user to send commands in a loop
        while True:
            print("\nEnter a command (e.g., 'TURN,90' or 'MOVE,50'), or 'exit' to quit:")
            command = input("> ").strip()

            if command.lower() == 'exit':
                print("Exiting program.")
                break

            # Validate command format (basic check)
            if command.startswith("TURN,") or command.startswith("MOVE,"):
                send_command(ser, command)
            else:
                print("Invalid command format. Use 'TURN,angle' or 'MOVE,distance'.")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()
