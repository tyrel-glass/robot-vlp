from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import os
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

import robot_vlp.data_collection.communication as c

from robot_vlp.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, VLP_MODELS_DIR

model_training_samples_dic = {
    'low_acc':0.1,
    'high_acc':0.75
}
app = typer.Typer()

@app.command()
def main(
    vlp_dataset_path: Path = RAW_DATA_DIR / "experiments/CNC/cnc_fingerprint_01.csv",
    output_path: Path = VLP_MODELS_DIR / 'CNC/'
):
    # -----------------------------------------
    logger.info("Reading vlp dataset")

    df= pd.read_csv(vlp_dataset_path, delimiter = '|')
    df = c.process_cnc(df)
    df = c.process_vlp(df)

    model_dic = {}
    for model_name, train_size in model_training_samples_dic.items():
        model_dic[model_name] = build_model(df= df, train_size = train_size)

    models_filename = output_path / "CNC_vlp_models.pkl"
    pickle.dump(model_dic, open(models_filename, "wb"))
    logger.success("Created VLP models")
    # -----------------------------------------


def build_model(df, train_size: float):
    X = df[['L1', 'L2', 'L3', 'L4']]
    y = df[['cnc_x', 'cnc_y']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    regr = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(max_iter=10000))])
    regr.fit(X_train.values, y_train.values)

    y_pre = regr.predict(X_test.values)
    errs = np.sqrt(
        np.square(y_pre[:, 0] - y_test.values[:, 0]) + np.square(y_pre[:, 1] - y_test.values[:, 1])
    )
    logger.info("mean error of: " + str(errs.mean()) + str(train_size)+" training samples")

    return regr



if __name__ == "__main__":
    app()
