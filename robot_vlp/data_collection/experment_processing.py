import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import robot_vlp.data_collection.communication as c


from pathlib import Path

import typer
from loguru import logger

from robot_vlp.config import INTERIM_DATA_DIR, RAW_DATA_DIR, VLP_MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    
    files = ['exp01','exp02','exp03','exp04']
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing experment dataset")

    input_file = RAW_DATA_DIR / 'experiments/Robot/' / files[0]


    vlp_models_path = VLP_MODELS_DIR / "CNC/CNC_vlp_models.pkl"
    vlp_models =  pickle.load(open(vlp_models_path, 'rb'))


    output_file = INTERIM_DATA_DIR / 'exp_vive_navigated_paths/' / files[0].split('.')[0] /'.pkl'

    for vlp_name, vlp_model in vlp_models.items():
        for filename in files:
            logger.info(f"processing {filename} with {vlp_name}")
            input_file = RAW_DATA_DIR / 'experiments/Robot/' / (filename+'.csv')
            output_filename = f'{filename}_{vlp_name}.pkl'
            output_file = INTERIM_DATA_DIR / 'exp_vive_navigated_paths/' /output_filename

            run_data_dic = process_robot_exp_file(input_file, vlp_model)

            pickle.dump(run_data_dic, open(output_file , 'wb'))




    logger.success("Processing dataset complete.")
    # -----------------------------------------



def process_robot_exp_file(input_file, vlp_model):
    logger.success("opending file: ", input_file)
    df= pd.read_csv(input_file, delimiter = '|')
    df = c.parse_vive(df)
    df = c.transform_vive_df(df)

    df = df[~df['last_cmd'].str.contains('CAL:')]#remove cal points

    df = c.process_vlp(df)  

    df['x_hist'] = df['vive_x'] +0.067*np.sin(df['vive_yaw']/180*np.pi)
    df['y_hist'] = df['vive_z'] +0.067*np.cos(df['vive_yaw']/180*np.pi)
    df['heading_hist'] = [c.normalize_angle(a) for a in (df['vive_yaw'] + 180)]

    df[['vlp_x_hist', 'vlp_y_hist']] = vlp_model.predict(df[['L1', 'L2', 'L3', 'L4']].values)/1000 #cnc in mm

    df = calculate_encoder_data(df)

    df = df[~df['last_cmd'].str.contains('TURN')] # drop rows after a turn (before move)
    df.reset_index(inplace = True)

    df = calc_vlp_heading(df)

    targets = ['x_hist', 'y_hist','heading_hist','vlp_x_hist', 'vlp_y_hist','vlp_heading_hist', 'encoder_x_hist','encoder_y_hist', 'encoder_heading_hist']
    df = df[targets]


    data_dic = convert_df_to_dic(df)

    return data_dic

    




def convert_df_to_dic(df):
    X_data = np.array([
        df['encoder_x_hist'].to_list(), 
        df['encoder_y_hist'].to_list(), 
        df['encoder_heading_hist'].to_list(), 
        df['vlp_x_hist'].to_list(), 
        df['vlp_y_hist'].to_list(), 
        df['vlp_heading_hist'].to_list()
        ]).T

    y_data = np.array([
        df['x_hist'].to_list(),
        df['y_hist'].to_list(),
        df['heading_hist'].to_list()
        ]).T

    data_dic = {'X':X_data, 'y':y_data}
    return data_dic






# Calculate absolute 2D distance traveled
def calculate_distance_2d(df):
    """
    Calculates the 2D distance traveled between consecutive points.
    """
    x = df['vive_x'].diff().values
    y = df['vive_z'].diff().values
    y[0]= 0
    x[0] = 0

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    df['distance_traveled'] = np.sqrt(x**2 + y**2)

    return df
def calculate_move_heading(df):
    df['move_heading'] = np.nan
    for i in range(2, len(df) - 1):
        if 'MOVE'  in df.loc[i, 'last_cmd']:
            # Coordinates for the current row
            x_cur, y_cur = df.loc[i, ['vive_x', 'vive_z']]

           # Coordinates 1 row back
            x_back_1, y_back_1 = df.loc[i - 1, ['vive_x', 'vive_z']]


            x_d =  x_cur - x_back_1
            y_d =  y_cur - y_back_1
            new_heading = np.arctan2(x_d, y_d)*180/np.pi

            # Convert to degrees and store in the DataFrame
            df.loc[i, 'move_heading'] = new_heading

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
            x_curr, y_curr = df.loc[i, ['vive_x', 'vive_z']]

            # Coordinates 2 rows back
            x_back_2, y_back_2 = df.loc[i - 2, ['vive_x', 'vive_z']]

           # Coordinates 2 rows back
            x_back_1, y_back_1 = df.loc[i - 1, ['vive_x', 'vive_z']]


            # Coordinates 1 row forward
            x_forward, y_forward = df.loc[i + 1, ['vive_x', 'vive_z']]

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



def plot_surface(df):

    # Creating 4 subplots for each target peak column
    fig = plt.figure(figsize=(16, 12))

    target_peaks = ['peak_1000Hz', 'peak_3000Hz', 'peak_5000Hz', 'peak_7000Hz']

    # Example additional data (add to match the structure of your actual dataframe)


    # Iterate through each target peak column to create a subplot
    for i, peak in enumerate(target_peaks):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')  # 2x2 grid of subplots

        # Prepare the grid for surface plotting
        X, Y = np.meshgrid(df["cnc_x"].unique(), df["cnc_y"].unique())
        Z = df.pivot_table(index='cnc_y', columns='cnc_x', values=peak).values

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Label the axes
        ax.set_title(f'{peak}')
        ax.set_xlabel('X Location')
        ax.set_ylabel('Y Location')
        ax.set_zlabel(peak)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



def calculate_encoder_data(input_df):
    df = input_df.copy()
    df.reset_index(inplace = True)
    enc_per_degree = 11.34
    enc_per_cm = 89.08

    robot_moves = df['last_cmd'].to_list()

    # set encoder val to current vive
    encoder_x_hist = [df['x_hist'].iloc[0]]
    encoder_y_hist = [df['y_hist'].iloc[0]]
    encoder_heading_hist = [c.normalize_angle(df['vive_yaw'].iloc[0] + 180)]

    for move in robot_moves[1:]:

        if 'TURN:' in move:
            encoder_count = int(float(move.split(':')[1]))
            angle_turned = encoder_count / enc_per_degree
            cur_heading = encoder_heading_hist[-1]
            encoder_heading_hist.append(cur_heading + angle_turned)
            encoder_x_hist.append(encoder_x_hist[-1])
            encoder_y_hist.append(encoder_y_hist[-1])
        if 'MOVE:' in move:
            encoder_count = int(float(move.split(':')[1]))
            distance_moved = encoder_count / enc_per_cm
            cur_x = encoder_x_hist[-1]
            cur_y = encoder_y_hist[-1]
            cur_heading = encoder_heading_hist[-1]

            new_x = cur_x + (distance_moved/100)*np.sin(cur_heading /180 *np.pi)
            new_y = cur_y + (distance_moved/100)*np.cos(cur_heading/180*np.pi)

            encoder_x_hist.append(new_x)
            encoder_y_hist.append(new_y)
            encoder_heading_hist.append(encoder_heading_hist[-1])

    df['encoder_heading_hist'] = [c.normalize_angle(a) for a in encoder_heading_hist]
    df['encoder_x_hist'] = encoder_x_hist
    df['encoder_y_hist'] = encoder_y_hist

    return df

def calc_vlp_heading(df):
    # calculate heading by looking at vector from last to cur point
    df['vlp_heading_hist'] = np.arctan2(df['vlp_x_hist'].diff(1) , df['vlp_y_hist'].diff(1)) *180/np.pi
    
    # populate initial heading 
    df.loc[0,'vlp_heading_hist'] = df['encoder_heading_hist'].iloc[0]

    return df

def plot_path(df, s, l):
    df_1 = df.iloc[s:s+l]
    plt.plot(df_1['x_hist'], df_1['y_hist'], marker = '.', label = 'actual')
    plt.plot(df_1['vlp_x_hist'], df_1['vlp_y_hist'], marker = '.', label = 'vlp')
    plt.plot(df_1['encoder_x_hist'], df_1['encoder_y_hist'], marker = '.', label = 'encoder')
    plt.legend()




if __name__ == "__main__":
    app()