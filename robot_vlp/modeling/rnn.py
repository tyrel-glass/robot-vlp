from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import keras
import keras_tuner as kt
from keras import ops
from time import strftime
import random

from sklearn.preprocessing import MinMaxScaler


import robot_vlp.data.preprocessing as p

import typer
from loguru import logger
from tqdm import tqdm

from robot_vlp.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TRAINING_LOGS_DIR, INTERIM_DATA_DIR

def ang_loss_fn(y_true, y_pred):
    return tf.add(keras.losses.cosine_similarity(y_true, y_pred) , 1)


app = typer.Typer()
@app.command()
def main(
        model_name
):

    build_default_model(model_name, save = True)


def get_run_logdir():
    return TRAINING_LOGS_DIR /'tensorboard'/ strftime("run_%Y_%m_%d_%H_%M_%S")

def build_default_model(model_name, save = False):
    model = build_model(kt.HyperParameters())
    if save:
        model.save(MODELS_DIR / model_name)
    return model

def build_model(hp):
    n_hidden = hp.Int('n_hidden', min_value = 1, max_value = 10, default = 3)
    n_neurons = hp.Int('n_neurons', min_value = 5, max_value = 250, default = 50, step = 5)
    learning_rate = hp.Float('learning_rate', min_value = 4e-4, max_value = 4e-3,default = 9e-4,step = 1e-4)
    optimizer = hp.Choice('optimizer', values = ['adam','sgd'])
    layer_type = hp.Choice('layer_type', values = ['simple', 'lstm', 'gru'])

    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate)

    input_ = keras.layers.Input(shape=(None, 6))

    next_input = input_

    for i in range(n_hidden - 1):
        if layer_type == 'simple':
            hidden_layer = keras.layers.SimpleRNN(n_neurons, return_sequences= True)(next_input)
        elif layer_type == 'lstm':
            hidden_layer = keras.layers.LSTM(n_neurons, return_sequences= True)(next_input)
        elif layer_type == 'gru':
            hidden_layer = keras.layers.GRU(n_neurons, return_sequences= True)(next_input)
        next_input = hidden_layer

    last_hidden_layer = keras.layers.SimpleRNN(n_neurons, return_sequences= False)(next_input)
 
    out1 = keras.layers.Dense(2, name='pos')(last_hidden_layer)
    out2 = keras.layers.Dense(2, name='heading')(last_hidden_layer)

    model = keras.Model(inputs = [input_], outputs = [out1,out2])

    model.compile(optimizer=optimizer,
            loss = ['mse',ang_loss_fn],
                loss_weights = [1., 1.],
            )
    return model


def retrain_model(model_name , dataset_name):

    with open(PROCESSED_DATA_DIR/dataset_name, 'rb') as handle:
        data = pickle.load(handle)
    model = keras.models.load_model(MODELS_DIR / model_name,custom_objects={"ang_loss_fn": ang_loss_fn})

    run_logdir = get_run_logdir()  
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights = True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('rnn_checkpoints.weights.h5',save_weights_only = True)

    history = model.fit(
        x = data['X_train'], 
        y = [data['y_train'][:,[0,1]],  p.ang_to_vector(data['y_train'][:,2], unit = 'degrees').numpy()],
        epochs = 2000,
        batch_size = 32,
        validation_data = (data['X_valid'], [data['y_valid'][:,[0,1]], p.ang_to_vector(data['y_valid'][:,2], unit = 'degrees').numpy()]), 
        callbacks = [tensorboard_cb, early_stopping_cb, checkpoint_cb]    
        )
    
    model.save(MODELS_DIR / model_name)
    logger.success("Modeling training complete.")
    

if __name__ == "__main__":
    app()
