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
