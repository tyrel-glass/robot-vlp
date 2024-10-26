from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import keras
from keras import ops
import random

from sklearn.preprocessing import MinMaxScaler
from robot_vlp.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

import robot_vlp.data.preprocessing as p
import typer
from loguru import logger
from tqdm import tqdm

from robot_vlp.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR

def train_mlp(
):
    # pull in processed dataset
    with open(PROCESSED_DATA_DIR/'data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    def ang_loss_fn(y_true, y_pred):
        return keras.losses.cosine_similarity(y_true, y_pred) + 1
    

    # ------------------------------------------------------------------
    input_ = keras.layers.Input(shape=(data['X_train'].shape[1], 5))
    flat_input = keras.layers.Flatten()(input_)
    hidden1 = keras.layers.Dense(10)(flat_input)
    hidden2 = keras.layers.Dense(20)(hidden1)
    hidden3 = keras.layers.Dense(20)(hidden2)
    out1 = keras.layers.Dense(2, name='loss1')(hidden3)
    out2 = keras.layers.Dense(2, name='loss2')(hidden3)

    model = keras.Model(inputs = [input_], outputs = [out1,out2])


  

    model.compile(optimizer='adam',
                loss = ['mse',ang_loss_fn],
                #   loss_weights = [1.],
                )
    
    logger.info("Training the model")
    history = model.fit(
        x = data['X_train'], 
        y = [data['y_train'][:,[0,1]], p.ang_to_vector(data['y_train'][:,2], unit = 'degrees').numpy()],
        epochs = 200,
        validation_data =    (data['X_valid'], [data['y_valid'][:,[0,1]], p.ang_to_vector(data['y_valid'][:,2], unit = 'degrees').numpy()])                               
    )
    logger.success("Modeling training complete.")
    # -----------------------------------------
    model.save(MODELS_DIR / 'model_01.keras')
    # plot(history)

    

def plot(history):
    plt.plot(history.history['loss'], label = 'total loss')
    plt.plot(history.history['loss1_loss'], label = 'loss 1')
    plt.plot(history.history['loss2_loss'], label = 'loss 2')
    plt.plot(history.history['loss3_loss'], label = 'loss 3')

    plt.plot(history.history['val_loss'], label = 'valid total loss')
    plt.plot(history.history['val_loss1_loss'],"-x", label = 'valid loss 1')
    plt.plot(history.history['val_loss2_loss'],"-x", label = 'valid loss 2')
    plt.plot(history.history['val_loss3_loss'],"-x",label = 'valid loss 3')
    plt.legend()
    plt.savefig(FIGURES_DIR/'mlp_training.png')

if __name__ == "__main__":
    train_mlp()
