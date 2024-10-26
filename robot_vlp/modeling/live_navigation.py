from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import keras
import numpy as np

from robot_vlp.config import MODELS_DIR, PROCESSED_DATA_DIR

import robot_vlp.data.preprocessing as p
import robot_vlp.data.path_generation as pg

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    # features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    # logger.info("Performing inference for model...")
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    # logger.success("Inference complete.")
    # # -----------------------------------------

    X_scaler = p.build_scaler()
    vlp_model_dic = pg.read_vlp_models()



    def ang_loss_fn(y_true, y_pred):
        return keras.losses.cosine_similarity(y_true, y_pred) + 1
    model = keras.models.load_model(MODELS_DIR / 'model_01.keras',custom_objects={"ang_loss_fn": ang_loss_fn})

def make_prediction(X):
    return model.predict(X_scaler.transform(X), verbose = None))

def navigate_to_target(robot,x,y,step_size = 0.1, target_threshold = 0.1):
    robot.target_x = x 
    robot.target_y = y
    while (robot.calc_distance_to_target() > target_threshold) & pg.in_bounds(robot,x_lim=(0.5,6.5), y_lim = (0.5,5.7)):
        robot.correct_heading()
        robot.step(step_size)
        X_data = np.array([robot.encoder_x_hist, robot.encoder_y_hist, robot.encoder_heading_hist, robot.vlp_x_hist, robot.vlp_y_hist]).T
        pre = make_prediction(X_data)

        robot.model_x = pre[0,0]
        robot.model_y = pre[0,1]
        robot.model_heading = pre[0,2]
        robot.model_x_hist.append(robot.model_x)
        robot.model_y_hist.append(robot.model_y)
        robot.model_heading_hist.append(robot.model_heading)

if __name__ == "__main__":
    app()
