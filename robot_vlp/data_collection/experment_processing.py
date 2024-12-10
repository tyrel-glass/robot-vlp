import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Calculate absolute 2D distance traveled
def calculate_distance_2d(df):
    """
    Calculates the 2D distance traveled between consecutive points.
    """
    x = df['x'].diff().values
    y = df['y'].diff().values
    y[0]= 0
    x[0] = 0

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    df['distance_traveled'] = np.sqrt(x**2 + y**2)

    return df


def extract_original_orientation(df):
    """
    Extract the original pitch, yaw, and roll from the vive_data column
    and add them as separate columns to the DataFrame.
    """
    # Extract Vive orientation data
    vive_data = df['vive_data'].apply(lambda v: np.fromstring(v.strip('[]'), sep=' '))
    orientations = np.stack(vive_data.to_list())[:, 3:]  # Extract yaw, pitch, roll
    positions = np.stack(vive_data.to_list())[:, :3]
    
    # Add columns for the original orientation
    df[['yaw', 'pitch', 'roll']] = orientations
    df[['x', 'y', 'z']] = positions
    return df

def transform_positions_to_robot_frame(df):
    """
    Transforms positional data (x, y, z) from the Vive frame to the robot frame.
    """
    # Extract Vive positions
    vive_data = df['vive_data'].apply(lambda v: np.fromstring(v.strip('[]'), sep=' '))
    positions = np.stack(vive_data.to_list())[:, :3]

    # Fit a plane to the Vive positions
    def fit_plane(points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]
        return normal, centroid

    # Fit the plane and compute the rotation matrix
    normal, centroid = fit_plane(positions)
    target_normal = np.array([0, 0, 1])  # Robot's x-y plane (z-aligned)
    rotation_axis = np.cross(normal, target_normal)
    rotation_angle = np.arccos(np.dot(normal, target_normal) / np.linalg.norm(normal))
    plane_alignment_rotation = R.from_rotvec(rotation_axis * rotation_angle).as_matrix()

    # Apply the rotation to align positions
    aligned_positions = (positions - centroid) @ plane_alignment_rotation.T
    aligned_positions -= aligned_positions[0]  # Map the first position to the origin

    # Update the DataFrame with transformed positions
    df[['x', 'y', 'z']] = aligned_positions

    return df

def transform_orientation_with_three_reference_points(df):
    """
    Aligns all Vive points to the robot frame using the derived transform.
    """
    # Instantiate the transformation class
    transformer = c.ViveToRobotTransform()

    # Derive the transformation
    transformer.derive_transform(df)

    # Transform all points in the DataFrame
    vive_positions = df['vive_data'].apply(
        lambda v: np.fromstring(v.strip('[]'), sep=' ')
    )
    transformed_positions = vive_positions.apply(transformer.transform_point)

    # Replace original positions with transformed ones
    df[['x', 'y', 'z']] = np.stack(transformed_positions.to_list())

    return df

def transform_vive_to_robot_frame(df):
    """
    Transform both positional and orientation data from the Vive frame to the robot frame.
    """
    df = extract_original_orientation(df)
    # df = transform_positions_to_robot_frame(df)
    # df = transform_orientation_with_three_reference_points(df)
    return df

def calculate_turn_angles(df):
    """
    Calculate the angle formed by vectors originating at the current row
    (for rows with 'TURN:-1250' in the 'last_cmd' column) and spanning to the
    x-y locations 2 rows back and 1 row forward.
    """
    # Ensure the DataFrame is sorted by position or time
    df = df.reset_index(drop=True)

    # Initialize a column for the calculated angles
    df['turn_angle'] = np.nan

    for i in range(2, len(df) - 1):
        if 'TURN'  in df.loc[i, 'last_cmd']:
            # Coordinates for the current row
            x_curr, y_curr = df.loc[i, ['x', 'y']]

            # Coordinates 2 rows back
            x_back_2, y_back_2 = df.loc[i - 2, ['x', 'y']]

           # Coordinates 2 rows back
            x_back_1, y_back_1 = df.loc[i - 1, ['x', 'y']]


            # Coordinates 1 row forward
            x_forward, y_forward = df.loc[i + 1, ['x', 'y']]

            # Vectors originating from the current row
            vector_back = np.array([x_back_1 - x_back_2, y_back_1 - y_back_2])
            vector_forward = np.array([x_forward - x_curr, y_forward - y_curr])

            # Normalize the vectors
            norm_back = np.linalg.norm(vector_back)
            norm_forward = np.linalg.norm(vector_forward)

            if norm_back > 0 and norm_forward > 0:
                vector_back = vector_back / norm_back
                vector_forward = vector_forward / norm_forward

                # Calculate the angle using the dot product
                dot_product = np.clip(np.dot(vector_back, vector_forward), -1.0, 1.0)
                angle = np.arccos(dot_product)  # Angle in radians

                # Convert to degrees and store in the DataFrame
                df.loc[i, 'turn_angle'] = np.degrees(angle)

    return df

# Process 'MOVE' and 'TURN' commands into separate columns
def parse_command(cmd):
    if cmd.startswith("MOVE:"):
        return int(float(cmd.split(":")[1])), 0
    elif cmd.startswith("TURN:"):
        return 0, int(cmd.split(":")[1])
    return 0, 0




def assess_move_errs(df):

    move_df = df[df['move_count']> 0 ]
    move_df['counts_per_cm'] = move_df['move_count']/(move_df['distance_traveled']*100)
    
    print('mean: ', move_df['counts_per_cm'][1:].mean())
    return move_df


def assess_turn_errs(df):
    turn_df = df[df['turn_count'].abs()> 0 ]
    turn_df['counts_per_degree'] = turn_df['turn_count'].abs()/turn_df['turn_angle'].abs()

    print('mean: ', turn_df[(turn_df['counts_per_degree'] < 18)]['counts_per_degree'].mean())
    return turn_df



# ------------------------------------------------ Functions to process VLP data -------------------------------------

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    

def parse_vlp(data):
    if type(data) != str:
        return data
    else:
        fft = [float(val) for val in data.replace('\r\n','').split(' ') if is_float(val)]
        fft, fre = eval(data.replace('\n', '').replace('array', 'np.array')[1:-1])
        return fft


# --------Return only a list of the peaks------------
def calc_pks(fft, width = 5):
    if type(fft) != np.ndarray:
        return [None, None, None, None]

    light_frequencys  = {
    'l1':1000,
    'l2':3000,
    'l3':5000,
    'l4':7000,
    }
    fft = np.array(fft)

    fre = np.linspace(0,25000,int(len(fft)))
    intensitys = []

    for light in light_frequencys.keys():
        cen_fre = light_frequencys[light]
        cen_ind = len(fre[fre<cen_fre])
        lower = cen_ind - width
        upper = cen_ind + width
        index = fft[lower:upper].argmax() + lower
        value = fft[index]
        intensitys.append(value)
    return intensitys
