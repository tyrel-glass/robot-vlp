import matplotlib.pyplot as plt
import socket
import time
import serial
import time
import robot_vlp.data.triad_openvr.triad_openvr as vr
import pandas as pd
import numpy as np
import csv
import openvr
import os


# ---- VLP functions ----

def FFT_win(clip):
    signal = np.array(clip)
    signal = signal - np.mean(signal)
    signal = signal/np.max(signal)
    win = np.hanning(len(signal))
    signal = signal*win
    
    fft =  np.fft.fft(signal)
    fft = fft[:int(len(fft)/2)]
    fft = np.abs(fft)
    fre = np.linspace(0,25200,int(len(fft)))

    return fft, fre

def take_mean_fft(n):
    ffts = []
    print('taking vlp readings: ',end = ' ')
    for i in range(n):
        print(i, end = ' ')
        fft, fre = FFT_win(read_vlp())
        ffts.append(fft)
        time.sleep(0.1)

    return np.mean(ffts, axis = 0), fre



def read_vlp(ESP32_IP="192.168.10.100", max_retries=3, timeout=3):
    ESP32_PORT = 8080
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Attempt to connect to the ESP32
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)  # Set timeout for the connection
            s.connect((ESP32_IP, ESP32_PORT))
            s.sendall(b"FETCH\n")  # Request ADC data

            adc_data = []  # List to store parsed ADC values
            buffer = ""  # Buffer to accumulate partial data
            start_received = False  # Flag to check if START marker is processed

            while True:
                chunk = s.recv(1024).decode().strip()  # Receive and decode a chunk
                buffer += chunk  # Accumulate the chunk into the buffer

                # Process the buffer
                if "START ADC DATA" in buffer:
                    start_received = True
                    buffer = buffer.split("START ADC DATA", 1)[1]  # Discard the marker

                if "END ADC DATA" in buffer:
                    # Process up to the "END ADC DATA" marker
                    data_section, _, buffer = buffer.partition("END ADC DATA")

                    if start_received:
                        data_section = data_section.replace('\r\n', '')  # Remove newlines
                        try:
                            # Split the string into integers, skipping empty or invalid values
                            adc_data = [int(val) for val in data_section.split(',') if val.strip().isdigit()]
                        except ValueError as e:
                            print(f"Error parsing adc data: {data_section} - {e}")  # Debugging invalid lines

                    break  # Exit the loop once the end marker is processed

            s.close()  # Close the socket connection
            return adc_data

        except (socket.timeout, ConnectionError) as e:
            retry_count += 1
            print(f"Connection attempt {retry_count} failed: {e}")
            time.sleep(1)  # Wait 1 second before retrying

    print("Failed to connect to ESP32 after multiple attempts.")
    return None  # Return None if all retries fail


# ---- LOGGING FUNCTION -----
def vive_robot_log_clear(log_file = 'robot_vive_data_log.csv' ):
    """Clears the log file and writes headers with a pipe delimiter."""
    with open(log_file, 'w') as f:
        f.write("vive_data|last_cmd\n")


def vive_robot_log_write(vive_data, cmd, log_file = 'robot_vive_data_log.csv'):
    """Writes a single data point to the log file."""
    vive_data = str(vive_data).replace('\n', '')  # Remove newline characters
    last_cmd = str(cmd).replace('\n', '')    # Remove newline characters
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([vive_data, cmd])



# ----- VIVE FUNCTIONS ------
def vive_setup():
    # Initialize OpenVR
    try:
        ovr = openvr.init(openvr.VRApplication_Scene)
    except Exception as e:
        print(f"Failed to initialize OpenVR: {e}")
        exit() 

    v = vr.triad_openvr()
    print(v.devices)
    return v


def read_vive(vive, n_readings = 10 , transformer = None):
    readings = []
    for _ in range(n_readings):
        readings.append(np.array(vive.devices["tracker_1"].get_pose_euler()))
        time.sleep(0.1)
    mean_readings = np.mean(readings, axis = 0)
    if transformer is None:
        return mean_readings
    else:
        return np.concatenate([transformer.transform_point(mean_readings[:3]), mean_readings[3:]])
    

def take_vive_cal_point(point_no, log_file, vive, raw = True):
    vive_data = read_vive(vive)
    cmd = 'CAL:'+str(point_no)
    vive_robot_log_write(vive_data, cmd, log_file= log_file)

def get_last_vive_position(log_file):
    # Load the log file
    df = pd.read_csv(log_file, delimiter='|', header=0, names=['vive_data', 'last_cmd'])
    
    # Extract the vive_data from the last row
    last_row_vive_data = df.iloc[-1]['vive_data']
    
    # Parse the vive_data string into a numpy array and extract the first two values
    vive_array = np.fromstring(last_row_vive_data.strip('[]'), sep=' ')
    return vive_array[:2]


# ----- ROBOT SERIAL FUNCTIONS -----
def open_robot_port(com_port = 'COM10'):
    robot_port = serial.Serial(com_port, 9600, timeout= 20)
    print(robot_port.readline())
    return robot_port

def send_robot_cmd(robot_ser, cmd):
    robot_ser.write((cmd + '\n').encode())
    print(robot_ser.readline())
    print(robot_ser.readline())



########################  CONTROL PROCESSING ##########################

def process_move(cmd, robot_ser, log_file, vive, transformer):

    send_robot_cmd(robot_ser = robot_ser, cmd = cmd)
    vive_data = read_vive(vive, transformer= transformer)
    vive_robot_log_write(vive_data, cmd = cmd, log_file = log_file)


######################## VIVE TRANSLANTION CODE #########################
class ViveToRobotTransform:
    def __init__(self):
        self.translation = None
        self.rotation_matrix = None

    def derive_transform(self, df):
        """
        Derive the translation vector and rotation matrix using the calibration points.
        """
        # Extract calibration points
        calibration_points = df[df['last_cmd'].str.startswith('CAL')].head(3)
        assert len(calibration_points) == 3, "Insufficient calibration points for alignment."

        # Known robot coordinates for calibration points
        robot_coords = np.array([
            [0, 1, 0],  # CAL:1
            [0, 0, 0],  # CAL:2
            [1, 0, 0]   # CAL:3
        ])

        # Extract Vive coordinates for calibration points
        vive_positions = calibration_points['vive_data'].apply(
            lambda v: np.fromstring(v.strip('[]'), sep=' ')[:3]
        )
        vive_coords = np.stack(vive_positions.to_list())  # Extract x, y, z coordinates

        # Compute the translation to align Vive CAL:2 with Robot CAL:2
        self.translation = robot_coords[1] - vive_coords[1]

        # Translate Vive calibration points
        vive_coords_translated = vive_coords + self.translation

        # Compute the rotation matrix to align Vive CAL:1 and CAL:3 with the robot frame
        # Vive frame basis vectors
        vive_x = vive_coords_translated[2] - vive_coords_translated[1]  # CAL:3 - CAL:2
        vive_y = vive_coords_translated[0] - vive_coords_translated[1]  # CAL:1 - CAL:2
        vive_z = np.cross(vive_x, vive_y)  # Orthogonal vector (right-hand rule)
        vive_x = vive_x / np.linalg.norm(vive_x)
        vive_y = vive_y / np.linalg.norm(vive_y)
        vive_z = vive_z / np.linalg.norm(vive_z)

        # Robot frame basis vectors
        robot_x = robot_coords[2] - robot_coords[1]  # CAL:3 - CAL:2
        robot_y = robot_coords[0] - robot_coords[1]  # CAL:1 - CAL:2
        robot_z = np.cross(robot_x, robot_y)  # Orthogonal vector (right-hand rule)
        robot_x = robot_x / np.linalg.norm(robot_x)
        robot_y = robot_y / np.linalg.norm(robot_y)
        robot_z = robot_z / np.linalg.norm(robot_z)

        # Construct the rotation matrix
        vive_basis = np.stack([vive_x, vive_y, vive_z], axis=1)
        robot_basis = np.stack([robot_x, robot_y, robot_z], axis=1)
        self.rotation_matrix = robot_basis @ vive_basis.T

    def transform_point(self, point):
        """
        Transform a single point using the derived translation and rotation matrix.
        """
        if self.translation is None or self.rotation_matrix is None:
            raise ValueError("Transformation not yet derived. Call `derive_transform` first.")

        point = np.array(point[:3])
        transformed_point = (point + self.translation) @ self.rotation_matrix.T
        return transformed_point

def build_transformer(log_file):
    df = pd.read_csv(log_file, delimiter='|', header=0, nrows = 3)
    df.columns = ['vive_data', 'last_cmd']
    transformer = ViveToRobotTransform()

    # Derive the transformation
    transformer.derive_transform(df)
    return transformer


################### CNC Control code #####################
def relative_movement(x, y, cnc_serial):
    """Perform relative movement on X and Y axes."""
    try:
        # Switch to relative positioning mode
        cnc_serial.write(b"G91\n")  # G91: Relative positioning
        cnc_serial.readline()  # Wait for CNC response
        
        # Send the movement command directly
        gcode_command = f"G1 X{x} Y{y} F1000"
        print(f"Generated G-code: {gcode_command}")
        cnc_serial.write(str.encode(gcode_command) + b"\n")
        grbl_out = cnc_serial.readline().decode().strip()
        print("GRBL Response: ", grbl_out)
        
        # Return to absolute positioning mode
        cnc_serial.write(b"G90\n")  # G90: Absolute positioning
        cnc_serial.readline()  # Wait for CNC response
    except Exception as e:
        print(f"Error during relative movement: {e}")

def absolute_movement(x, y, cnc_serial, feedrate=1000):
    """
    Moves the CNC to an absolute location (X, Y) at the specified feedrate.

    Args:
        x (float): Absolute X-coordinate.
        y (float): Absolute Y-coordinate.
        feedrate (int): Feedrate for the movement in units per minute. Default is 1000.

    Returns:
        None
    """
    try:
        # Ensure absolute positioning mode
        cnc_serial.write(b"G90\n")  # G90: Absolute positioning mode
        cnc_serial.readline()  # Wait for CNC response
        
        # Create and send the absolute movement command
        gcode_command = f"G1 X{x} Y{y} F{feedrate}"
        print(f"Sending command: {gcode_command}")
        cnc_serial.write(str.encode(gcode_command) + b"\n")
        
        # Wait for CNC's acknowledgment
        grbl_out = cnc_serial.readline().decode().strip()
        print("GRBL Response: ", grbl_out)
    except Exception as e:
        print(f"Error during absolute movement: {e}")



def init_cnc(cnc_serial):
    # Wake up GRBL and zero CNC
    cnc_serial.write(b"\r\n\r\n")
    time.sleep(2)
    cnc_serial.flushInput()
    cnc_serial.write(str.encode("$X") + b"\n")
    cnc_serial.readline()  # Wait for CNC response
    cnc_serial.write(str.encode("G10 P0 L20 X0 Y0 Z0") + b"\n")
    cnc_serial.readline()  # Zero CNC response


def cnc_log_clear(file_name):
    """Clears the log file and writes headers with a pipe delimiter."""
    with open(file_name, 'w') as f:
        f.write("position_id|timestamp|vive_data|vlp_data|cnc_data\n")

def cnc_log_write(file_name, reading_no, x, y, vive_data, vlp_data):
    """Writes a single data point to the log file."""
    with open(file_name, 'a') as f:
        f.write(f'{reading_no}|')
        f.write(f'{time.time()}|')
        f.write(f'"{vive_data}"|')  # Enclose in quotes to handle special characters
        f.write(f'"{vlp_data}"|')  # Enclose in quotes to handle special characters
        f.write(f'{x},{y}\n')


def get_last_logged_point(log_file):
    """Returns the last recorded position_id if log file exists, otherwise returns -1."""
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        data = pd.read_csv(log_file, sep='|')
        if not data.empty:
            return data['position_id'].iloc[-1]  # Get the last position_id
    return -1

def generate_scan_points(step=50, width=900, height=1000):
    """
    Generates a set of coordinate points to scan over a 2D area with a zigzag pattern.
    """
    points = []
    for y in range(0, height + step, step):  # Increment along the height
        if y % (2 * step) == 0:
            # Move left to right
            row_points = [(x, y) for x in range(0, width + step, step)]
        else:
            # Move right to left
            row_points = [(x, y) for x in range(width, -step, -step)]
        points.extend(row_points)
    return np.array(points)