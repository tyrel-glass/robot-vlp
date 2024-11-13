""""
This code creates a target paths, then generates robots with various error parameters to navigate these paths.
The resulting datasets are then stored in the INERM_DATA_DIR
"""

from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from robot_vlp.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, VLP_MODELS_DIR

app = typer.Typer()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import pickle
import robot_vlp.robot as r


    # Constants
ERROR_VALUES = {'err_1': 0.01, 'err_2': 0.05, 'err_3': 0.1}
GENERATED_PATHS_DIR = INTERIM_DATA_DIR/'generated_paths.plk'
PATH_REPEATS = 4

@app.command()
def main():
    
    # -------------------------------------------------------------
    logger.info("Reading in VLP dataset")
    vlp_dataset_path = EXTERNAL_DATA_DIR / "vlp_dataset.csv"
    df = pd.read_csv(vlp_dataset_path, index_col=0)
    logger.success('Pulled in VLP dataset')

    folder_path = INTERIM_DATA_DIR / 'odometer_path_data'
    folder_path.mkdir(parents = False, exist_ok = True)

    with open(GENERATED_PATHS_DIR, 'rb') as f:
        generated_paths = pickle.load(f)

    vlp_models = read_vlp_models()

    # err_val_dic = {
    #     'err_1':0.01,
    #     'err_2':0.05,
    #     'err_3':0.1
    # }


    for vlp_name, vlp_model in vlp_models.items():

        for err_name, err_val in ERROR_VALUES.items():

            for run_no in range(PATH_REPEATS):
                
                for path_name, path_coordinates in generated_paths.items():
                            name = path_name +f'_{vlp_name}_{err_name}_run{run_no}'

                            robot = r.Robot(x=path_coordinates[0,0], y=path_coordinates[0,1], heading = 0, step_err = err_val, turn_err = err_val, df = df, vlp_mod = vlp_model)  


                            for i in range(3):
                                for j in range(len(path_coordinates)):
                                    navigate_to_target(robot,path_coordinates[j,0],path_coordinates[j,1])

                            X_data = np.array([robot.encoder_x_hist, robot.encoder_y_hist, robot.encoder_heading_hist, robot.vlp_x_hist, robot.vlp_y_hist, robot.vlp_heading_hist]).T
                            y_data = np.array([robot.x_hist, robot.y_hist, robot.heading_hist]).T

                            data_dic = {'X':X_data, 'y':y_data}

                            file_path = folder_path / name
                            pickle.dump(data_dic, open(file_path , 'wb')) 

                      
    # -----------------------------------------


def read_vlp_models():
    """Reads in the vlp model"""
    
    vlp_models_path = VLP_MODELS_DIR / "vlp_models.pkl"
    return pickle.load(open(vlp_models_path, 'rb'))

def in_bounds(robot,x_lim, y_lim):
    return (robot.x > x_lim[0])&(robot.x < x_lim[1])&(robot.y > y_lim[0])&(robot.y < y_lim[1])

def navigate_to_target(robot,x,y,step_size = 0.1, target_threshold = 0.1):
    robot.target_x = x 
    robot.target_y = y
    while (robot.calc_distance_to_target() > target_threshold) & in_bounds(robot,x_lim=(0.5,6.5), y_lim = (0.5,5.7)):
        robot.correct_heading()
        robot.step(step_size)

X_labels = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist", "vlp_x_hist", "vlp_y_hist", "vlp_heading_hist"]
y_labels = ["x_hist", "y_hist", "heading_hist"]

def create_poly_targets(n, r = 2):
    """
    Creates a list of n equally spaces points that lie on the circumfrence of a circle
    """

    c = 3.5,3
    target_points = np.array([[r*np.cos(ang)+c[0], r*np.sin(ang)+c[1]] for ang in np.linspace(0,2*np.pi, n+1)[1:]+np.pi/4])
    return target_points



if __name__ == "__main__":
    app()
