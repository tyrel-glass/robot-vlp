import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import robot_vlp.data_collection.communication as c

import robot_vlp.modeling.gen_cnc_vlp_model as vlp

from pathlib import Path

import typer
from loguru import logger

from robot_vlp.config import INTERIM_DATA_DIR, RAW_DATA_DIR, VLP_MODELS_DIR

app = typer.Typer()



enc_per_degree = 11.34/2
enc_per_cm = 89.08/2


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # input_path: Path = RAW_DATA_DIR / "dataset.csv",
    # output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    
    files = ['exp01','exp02','exp03', 'exp07', 'exp04','exp05']
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing experment dataset")

    input_file = RAW_DATA_DIR / 'experiments/Robot/' / files[0]


    # vlp_models_path = VLP_MODELS_DIR / "CNC/CNC_vlp_models.pkl"
    # vlp_models =  pickle.load(open(vlp_models_path, 'rb'))
    vlp_models = vlp.load_vlp_models()


    output_file = INTERIM_DATA_DIR / 'exp_vive_navigated_paths/' / files[0].split('.')[0] /'.pkl'

    for vlp_name, vlp_model in vlp_models.items():
        for filename in files:
            logger.info(f"processing {filename} with {vlp_name}")
            input_file = RAW_DATA_DIR / 'experiments/Robot/' / (filename+'.csv')
            output_filename = f'{filename}_{vlp_name}.csv'
            output_file = INTERIM_DATA_DIR / 'exp_vive_navigated_paths/' /output_filename

            df = process_robot_exp_file(input_file, vlp_model)
            run_data_dic = convert_df_to_dic(df)
      

            # pickle.dump(run_data_dic, open(output_file , 'wb'))
            df.to_csv(output_file )




    logger.success("Processing dataset complete.")
    # -----------------------------------------



def process_robot_exp_file(input_file, vlp_model):
    logger.success("opending file: ", input_file)
    df= pd.read_csv(input_file, delimiter = '|')
    df = c.parse_vive(df)
    df = c.transform_vive_df(df)

    #======================== Drop the calibration rows ============================
    df = df[~df['last_cmd'].str.contains('CAL:')]#remove cal points
    #========================================================================


    # parse the move cmds before dropping the move rows
    def parse_last_turn(cmd):
        if 'TURN:' in cmd:
            return -int(float(cmd.split('TURN:')[1]))
        else:
            return np.nan
    
    df['encoder_heading_change_step'] = df['last_cmd'].apply(parse_last_turn)
    no_turn_filt =  df['last_cmd'].str.contains('MOVE:') & df['last_cmd'].shift(1).str.contains('MOVE:')
    df.loc[no_turn_filt, 'encoder_heading_change_step'] = 0
    df['encoder_heading_change_step'] = df['encoder_heading_change_step'].ffill() 
    df['encoder_heading_change'] = df['encoder_heading_change_step'] / enc_per_degree


    #======================== Drop the move rows ============================
    #========================================================================
    df = df[df['last_cmd'].str.contains('MOVE')]  #remove turn datasteps
    df.reset_index(inplace = True)
    #========================================================================
    #========================================================================


    # extract rss values for each light
    df = c.process_vlp(df)  
    # predict new vlp locations
    df[['vlp_x_hist', 'vlp_y_hist']] = vlp_model.predict(df[['L1', 'L2', 'L3', 'L4']].values)/1000 #cnc in mm

    # map between robot center and vive tracker
    df['x_hist'] = df['vive_x'] +0.067*np.sin(df['vive_yaw']/180*np.pi) 
    df['y_hist'] = df['vive_z'] +0.067*np.cos(df['vive_yaw']/180*np.pi) 

    # take heading from vive (offset by 180 degrees)
    df['heading_hist'] = [c.normalize_angle(a) for a in (df['vive_yaw'] + 180)]

    


        # ============================ CALCULATE HEADING CHANGE STATS========================
    df['vlp_heading_hist'] = np.arctan2(df['vlp_x_hist'].diff(1) , df['vlp_y_hist'].diff(1)) *180/np.pi

    df['vive_heading_change'] = df['heading_hist'].diff().apply(c.normalize_angle)
    df['vlp_heading_change'] = df['vlp_heading_hist'].diff().apply(c.normalize_angle)



    # ----------------- Compute errs  ----------------------------------
    df['encoder_heading_change_err'] = df['vive_heading_change'] - df['encoder_heading_change']
    df['vlp_heading_change_err'] = df['vive_heading_change'] - df['vlp_heading_change']

    #===========================================================================

    # ============================ CALCULATE STEP CHANGE STATS========================
    def parse_last_move(cmd):
        if 'MOVE:' in cmd:
            return int(float(cmd.split('MOVE:')[1]))
        else:
            return np.nan
    df['encoder_location_change'] = df['last_cmd'].apply(parse_last_move) / enc_per_cm /100

    df['vive_location_change'] = np.sqrt(np.square(df['x_hist'].diff(1)) + np.square(df['y_hist'].diff(1)))
    
    df['vlp_location_change'] = np.sqrt(np.square(df['vlp_x_hist'].diff(1)) + np.square(df['vlp_y_hist'].diff(1))) 

    df.loc[:,'encoder_location_change_err'] = df['vive_location_change'] - df['encoder_location_change']
    df['vlp_location_change_err'] = df['vive_location_change'] - df['vlp_location_change']


    df = calc_encoder_heading_hist(df)

    df = calc_encoder_xy_hist(df)
    df.reset_index(drop = True, inplace = True)
    return df

    

def convert_df_to_dic(df):
    X_data = np.array([
        # df['encoder_x_hist'].to_list(), 
        # df['encoder_y_hist'].to_list(), 
        # df['encoder_heading_hist'].to_list(), 
        # df['encoder_location_change'].to_list(), 
        # df['encoder_location_change'].to_list(), 
        
        df['vlp_x_hist'].to_list(), 
        df['vlp_y_hist'].to_list(), 
        df['encoder_heading_change'].to_list(), 
        # df['vlp_heading_hist'].to_list()
        ]).T

    y_data = np.array([
        df['x_hist'].to_list(),
        df['y_hist'].to_list(),
        df['heading_hist'].to_list()
        ]).T

    data_dic = {'X':X_data, 'y':y_data}
    return data_dic


# ------------------------------------------------ Functions to process VLP data -------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_surface_irregular(df, cmap='viridis', grid_resolution=100):
    # Creating 4 subplots for each target peak column
    fig = plt.figure(figsize=(16, 12))
    target_peaks = ['L1', 'L2', 'L3', 'L4']

    # Generate a regular grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(df["cnc_x"].min(), df["cnc_x"].max(), grid_resolution),
        np.linspace(df["cnc_y"].min(), df["cnc_y"].max(), grid_resolution)
    )

    for i, peak in enumerate(target_peaks):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Interpolate Z values onto the regular grid
        grid_z = griddata(
            points=(df["cnc_x"], df["cnc_y"]),
            values=df[peak],
            xi=(grid_x, grid_y),
            method='linear'
        )

        # Plot the surface
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap)

        # Label the axes
        ax.set_title(f'{peak} Surface Plot')
        ax.set_xlabel('X Location')
        ax.set_ylabel('Y Location')
        ax.set_zlabel(peak)

        # Add color bar for reference
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()




def calc_encoder_heading_hist(df):
    # -------------  Calculate  --------
    ang_deltas = (df['encoder_heading_change']).to_list()[1:] #values are to get to current step
    new_lst = []
    for i in range(len(ang_deltas)+1):
        new_lst.append(c.normalize_angle(np.sum(ang_deltas[:i]))) 

    df['encoder_heading_hist'] = np.array(new_lst) + df['heading_hist'].iloc[0] #map encoder to initial heading
    df['encoder_heading_hist'] = df['encoder_heading_hist'].apply(c.normalize_angle)
    
    return df

def calc_encoder_xy_hist(df):

    heading = df['encoder_heading_hist'].to_numpy()
    dx = np.sin(heading/180*np.pi) * df['encoder_location_change']
    dy = np.cos(heading/180*np.pi) * df['encoder_location_change']



    dx_sum = [sum(dx[1:i]) for i in range(1,len(dx)+1)]
    dy_sum = [sum(dy[1:i]) for i in range(1,len(dy)+1)]


    df['encoder_x_hist'] = np.array(dx_sum) + df['x_hist'].iloc[0]
    df['encoder_y_hist'] = np.array(dy_sum) + df['y_hist'].iloc[0]

    return df

if __name__ == "__main__":
    app()