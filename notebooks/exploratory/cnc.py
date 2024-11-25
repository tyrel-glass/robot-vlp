import csv
import serial
import numpy as np
import scipy.fftpack
from xgboost import XGBRegressor
import time
import os
import math
import random

# CNC Code for movement
def stream_gcode():
    with open(filepath_gcode, 'r') as gcode:
        for line in gcode:
            l = line.strip()
            print('Sending: ' + l)
            blackbox.write(str.encode(l) + b"\n")
            grbl_out = blackbox.readline()
            print("GRBL Response: ", grbl_out.strip())

def relative_movement(x, y):
    blackbox.write(str.encode("G91") + b"\n")  # G91: Relative positioning mode
    blackbox.readline()  # Wait for CNC response
    gcode_command = f"G1 X{x} Y{y} F1000"
    with open(filepath_gcode, "w") as f:
        f.write(gcode_command)  # Writing movement to G-code file
    print(f"Generated G-code: {gcode_command}")  # Print the generated G-code
    stream_gcode()  # Send the G-code to CNC
    blackbox.write(str.encode("G90") + b"\n")  # G90: Return to absolute positioning mode
    blackbox.readline()  # Wait for CNC response

# Sensor Code
def connectToSensor(port, rate):
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = rate
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = 1
    try:
        ser.open()
    except Exception:
        print("Failure to open port")
        exit()
    return ser

def read_SPD(com_port):
    ser = connectToSensor(com_port, 460800)
    ser.write('b\n'.encode(encoding='ascii'))
    line = ser.readline()
    ser.readline()
    ser.write('s\n'.encode(encoding='ascii'))
    ser.close()
    raw_data = line.decode('ascii')
    raw_data = raw_data.replace('[DEBUG] ADC Data:[', '')
    raw_data = raw_data.replace(']\r\n', '')
    str_data = raw_data.split(',')

    int_data = [int(num) for num in str_data]

    return int_data

def process_signal(signal_data):
    signal = np.array(signal_data)
    signal = signal - np.mean(signal)
    signal = signal / np.max(signal)
    han = np.hanning(len(signal))
    signal = han * signal
    fft = scipy.fftpack.fft(signal)
    fft = fft[:int(len(fft) / 2)]
    fft = np.abs(fft)
    return fft

def extract_led_rss_for_prediction(fft):
    led_ranges = [
        range(8, 12),  
        range(29, 33), 
        range(49, 53),  
        range(70, 74)   
    ]
    led_rss_values = [np.max(fft[led_range]) for led_range in led_ranges]
    return led_rss_values

# Load the trained ML model
def load_model():
    model = XGBRegressor()
    model.load_model('E:/_INDIV/Individual Code/Jupyter/ML_VLP_CNC/50mm_grid/ML_MODEL_50mm.json')  # Updated with your model path
    return model

# Function to estimate the current position based on RSS readings
def estimate_position_average(model, com_port, num_readings=5):
    rss_values_list = []

    # Take multiple readings to average out the noise
    for _ in range(num_readings):
        sensor_data = read_SPD(com_port)
        fft = process_signal(sensor_data)
        led_rss_values = extract_led_rss_for_prediction(fft)
        log_rss_values = [np.log(np.abs(rss) + 1) for rss in led_rss_values]
        rss_values_list.append(log_rss_values)

    # Average the RSS values
    averaged_rss_values = np.mean(rss_values_list, axis=0)

    # Prepare model input (RSS values only)
    model_input = np.array(averaged_rss_values).reshape(1, -1)

    # Predict coordinates using averaged RSS values
    predicted_coords = model.predict(model_input)

    return predicted_coords[0][0], predicted_coords[0][1]

# Calculate heading based on two estimated positions, with the positive Y-axis as 0 degrees
def calculate_heading(position1, position2):
    delta_x = position2[0] - position1[0]  
    delta_y = position2[1] - position1[1]
    
    # Calculate the angle relative to the Y-axis
    heading = math.atan2(delta_x, delta_y) * 180 / np.pi
    
    if heading < 0:
        heading += 360  # Ensure positive angles
    
    return heading

# Function to calculate distance between two positions
def calculate_distance(position1, position2):
    return math.sqrt((position2[0] - position1[0])**2 + (position2[1] - position1[1])**2)

# Main function to move based on estimated coords, log actual positions for ground truth
def main():
    step_size = 25
    com_port = 'COM3'  # Update with your sensor's COM port
    global blackbox, filepath_gcode

    # Load the pre-trained model
    model = load_model()

    # Starting actual position (450, 450)
    actual_position = [800, 150]
    actual_heading = None

    # Define the square path target points
    target_positions = [(150, 150)]

    # Open the CNC serial connection
    blackbox = serial.Serial('COM8', 115200)
    filepath_gcode = os.getcwd() + "/CNC/gcode_generation/generateGCode.gcode"

    # Wake up GRBL and zero CNC
    blackbox.write(b"\r\n\r\n")
    time.sleep(2)
    blackbox.flushInput()
    blackbox.write(str.encode("$X") + b"\n")
    blackbox.readline()  # Wait for CNC response
    blackbox.write(str.encode("G10 P0 L20 X0 Y0 Z0") + b"\n")
    blackbox.readline()  # Zero CNC response

    # Open CSV file to log actual positions, estimated positions, and headings
    csv_file_path = "CNC_position_log.csv"
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Actual_X", "Actual_Y", "Estimated_X", "Estimated_Y", "Estimated_Heading", "Actual_Heading"])

        # Take the initial RSS reading at (450, 450), but no headings
        print("Taking the initial RSS reading at (450, 450)...")
        input("Press Enter to take the RSS reading...")
        estimated_position1 = estimate_position_average(model, com_port)
        print(f"Initial Estimated Position: {estimated_position1}")

        # Log the initial position (without headings) to the CSV
        csv_writer.writerow([actual_position[0], actual_position[1], estimated_position1[0], estimated_position1[1], None, None])

        # Move step_size in a random direction based on estimated position
        random_angle = random.uniform(0, 2 * math.pi)
        random_x = step_size * math.cos(random_angle)
        random_y = step_size * math.sin(random_angle)
        relative_movement(random_x, random_y)

        # Update the actual heading based on the first random movement
        actual_heading = math.degrees(random_angle)
        if actual_heading < 0:
            actual_heading += 360

        input("Press Enter to take the RSS reading...")
        estimated_position2 = estimate_position_average(model, com_port)

        # Update the actual position based on CNC movement
        actual_position[0] += random_x
        actual_position[1] += random_y
        print(f"Position after random move: {estimated_position2}")

        # Calculate heading after random movement
        estimated_heading = calculate_heading(estimated_position1, estimated_position2)
        print(f"Estimated Heading: {estimated_heading} degrees, Actual Heading: {actual_heading} degrees")

        # Log the readings after random movement to the CSV
        csv_writer.writerow([actual_position[0], actual_position[1], estimated_position2[0], estimated_position2[1], estimated_heading, actual_heading])

        # Now move purely based on estimated coordinates in 200mm increments
        for target_position in target_positions:
            while True:
                # Calculate the distance from the current estimated position to the next target
                distance_to_target = calculate_distance(estimated_position2, target_position)
                print(f"Distance to target: {distance_to_target} mm")

                if distance_to_target <= step_size:
                    # Move the remaining distance if it's less than step_size
                    move_distance = distance_to_target
                else:
                    # Otherwise, move step_size
                    move_distance = step_size

                # Calculate estimated heading based on the two most recent readings
                estimated_heading = calculate_heading(estimated_position1, estimated_position2)
                print(f"Estimated Heading: {estimated_heading} degrees")

                # Calculate target heading based on the most recent reading and the target position
                target_heading = calculate_heading(estimated_position2, target_position)
                print(f"Target Heading: {target_heading} degrees")

                # Adjust the actual heading based on the difference between estimated and target heading
                angle_to_rotate = target_heading - estimated_heading
                actual_heading += angle_to_rotate
                if actual_heading < 0:
                    actual_heading += 360
                elif actual_heading >= 360:
                    actual_heading -= 360

                print(f"Actual Heading after adjustment: {actual_heading} degrees")

                # Convert distance and heading into relative X and Y movements
                relative_x = move_distance * math.sin(math.radians(actual_heading))
                relative_y = move_distance * math.cos(math.radians(actual_heading))

                print(f"Moving relatively by X: {relative_x} mm, Y: {relative_y} mm")
                relative_movement(relative_x, relative_y)

                # Update the actual position based on CNC movement
                actual_position[0] += relative_x
                actual_position[1] += relative_y
                print("Waiting for 3 seconds to take the RSS reading...")
                time.sleep(3)
                new_estimated_position = estimate_position_average(model, com_port)
                print(f"New Estimated Position: {new_estimated_position}")

                # Log the new readings to the CSV
                csv_writer.writerow([actual_position[0], actual_position[1], new_estimated_position[0], new_estimated_position[1], estimated_heading, actual_heading])

                # Update the previous and current estimated positions for the next step
                estimated_position1 = estimated_position2
                estimated_position2 = new_estimated_position

                # Break the loop if we've reached the target
                if distance_to_target <= 50:
                    break

    # Close the CNC serial connection after completing all movements
    blackbox.close()

if __name__ == "__main__":
    main()