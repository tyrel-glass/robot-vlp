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
    
    with open(PROCESSED_DATA_DIR/'model_train_test_data.pickle', 'rb') as handle:
        data_dic = pickle.load(handle)

    train_files = data_dic['train_files']
    valid_files = data_dic['valid_files']
    test_files = data_dic['test_files']

    X_train_data = data_dic['X_train_data']
    X_test_data = data_dic['X_test_data']
    X_valid_data = data_dic['X_valid_data']

    y_train_data = data_dic['y_train_data']
    y_test_data = data_dic['y_test_data']
    y_valid_data = data_dic['y_valid_data']

    X_train = data_dic['X_test']
    X_valid = data_dic['X_valid']
    X_test = data_dic['X_test']

    y_train = data_dic['y_test']
    y_valid = data_dic['y_valid']
    y_test = data_dic['y_test']

    def ang_loss_fn(y_true, y_pred):
        return keras.losses.cosine_similarity(y_true, y_pred) + 1

    input_ = keras.layers.Input(shape=(10, 5))
    flat_input = keras.layers.Flatten()(input_)
    hidden1 = keras.layers.Dense(10)(flat_input)
    hidden2 = keras.layers.Dense(20)(hidden1)
    out1 = keras.layers.Dense(1, name='loss1')(hidden2)
    out2 = keras.layers.Dense(1, name='loss2')(hidden2)
    out3 = keras.layers.Dense(2, name='loss3')(hidden2)

    model = keras.Model(inputs = [input_], outputs = [out1,out2,out3])

    model.compile(optimizer='adam',
                loss = ['mse','mse',ang_loss_fn],
                #   loss_weights = [1.],
                )
    
    logger.info("Training some model...")
    history = model.fit(
        x = X_train, 
        y = [y_train[:,0], y_train[:,1], p.ang_to_vector(y_train[:,2], unit = 'degrees').numpy()],
        epochs = 50,
        validation_data =    (X_valid, [y_valid[:,0], y_valid[:,1], p.ang_to_vector(y_valid[:,2], unit = 'degrees').numpy()])                               
    )
    logger.success("Modeling training complete.")
    # -----------------------------------------

    plot(history)

    model.save(MODELS_DIR / 'model_01.keras')

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

    plt.savefig(FIGURES_DIR/'mpl_traing.png')

train_mlp()