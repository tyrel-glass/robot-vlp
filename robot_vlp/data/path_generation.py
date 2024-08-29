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



@app.command()
def main():
    
    # -------------------------------------------------------------
    logger.info("Reading in VLP dataset")
    vlp_dataset_path = EXTERNAL_DATA_DIR / "vlp_dataset.csv"
    df = pd.read_csv(vlp_dataset_path, index_col=0)
    logger.success('Pulled in VLP dataset')

    vlp_model_dic = read_vlp_models()

    err_val_dic = {
        'err_1':0.01,
        'err_2':0.05,
        'err_3':0.1
    }

    for vlp_name, vlp_model in vlp_model_dic.items():
    
        for err_name, err_val in err_val_dic.items():

            for direction in ['clockwise','anticlockwise', 'shuffle']:
                for run_no in range(4):

                    for n in range(2,11):
                        if direction == 'clockwise':
                            targets = create_poly_targets(n)
                        elif direction =='anticlockwise':
                            targets = create_poly_targets(n)[::-1]
                        elif direction == 'shuffle':
                            targets = create_poly_targets(n)
                            np.random.shuffle(targets)
                            
                        name = 'n_'+str(n)+'_polygon_'+vlp_name+'_'+err_name+'_'+direction+'_run'+str(run_no)

                        robot = r.Robot(x=2, y=1.5, heading = 0, step_err = err_val, turn_err = err_val, df = df, vlp_mod = vlp_model)  

                        for i in range(3):
                            for j in range(n):
                                navigate_to_target(robot,targets[j,0],targets[j,1])

                        X_data = np.array([robot.encoder_x_hist, robot.encoder_y_hist, robot.encoder_heading_hist, robot.vlp_x_hist, robot.vlp_y_hist]).T
                        y_data = np.array([robot.x_hist, robot.y_hist, robot.heading_hist]).T

                        data_dic = {'X':X_data, 'y':y_data}

                        folder_path = '../../data/interim/'
                        file_path = INTERIM_DATA_DIR / name
                        pickle.dump(data_dic, open(file_path , 'wb')) 

                        # plot_path(data = data_dic, name = name)
    # -----------------------------------------


def read_vlp_models():
    """Reads in the vlp model"""
    
    vlp_models_path = VLP_MODELS_DIR / "vlp_models"
    return pickle.load(open(vlp_models_path, 'rb'))

def in_bounds(robot,x_lim, y_lim):
    return (robot.x > x_lim[0])&(robot.x < x_lim[1])&(robot.y > y_lim[0])&(robot.y < y_lim[1])

def navigate_to_target(robot,x,y,step_size = 0.1, target_threshold = 0.1):
    robot.target_x = x 
    robot.target_y = y
    while (robot.calc_distance_to_target() > target_threshold) & in_bounds(robot,x_lim=(0.5,6.5), y_lim = (0.5,5.7)):
        robot.correct_heading()
        robot.step(step_size)

X_labels = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist", "vlp_x_hist", "vlp_y_hist"]
y_labels = ["x_hist", "y_hist", "heading_hist"]

def create_poly_targets(n):
    r = 2
    c = 3.5,3
    target_points = np.array([[r*np.cos(ang)+c[0], r*np.sin(ang)+c[1]] for ang in np.linspace(0,2*np.pi, n+1)[1:]])
    return target_points

def plot_path(data, name):
    plt.figure()
    plt.plot(data['y'][:,0], data['y'][:,1], label = 'Actual path')

    plt.plot(data['X'][:,0], data['X'][:,1], label = 'Encoder path')

    plt.legend()

    plt.title('Training path')
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlim(1,6)
    plt.ylim(0.5,5.5)

    # plt.savefig(fig_path+name+'.png')




if __name__ == "__main__":
    app()
