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
from robot_vlp.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

import robot_vlp.data.preprocessing as p

import typer
from loguru import logger
from tqdm import tqdm

from robot_vlp.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TRAINING_LOGS_DIR

def ang_loss_fn(y_true, y_pred):
    return keras.losses.cosine_similarity(y_true, y_pred) + 1

def main(
):
    retrain_model(model_name = 'model_02.keras')


def get_run_logdir():
    return TRAINING_LOGS_DIR /'tensorboard'/ strftime("run_%Y_%m_%d_%H_%M_%S")


def build_model(hp):
    n_hidden = hp.Int('n_hidden', min_value = 1, max_value = 4, default = 2)
    n_neurons = hp.Int('n_neurons', min_value = 1, max_value = 50)
    learning_rate = hp.Float('learning_rate', min_value = 1e-4, max_value = 1e-2, sampling = 'log')
    optimizer = hp.Choice('optimizer', values = ['sgd', 'adam'])

    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate)

    input_ = keras.layers.Input(shape=(None, 5))

    next_input = input_

    for i in range(n_hidden - 1):
        hidden_layer = keras.layers.SimpleRNN(n_neurons, return_sequences= True)(next_input)
        next_input = hidden_layer

    last_hidden_layer = keras.layers.SimpleRNN(n_neurons, return_sequences= False)(next_input)
 

    # hidden1 = keras.layers.SimpleRNN(20, return_sequences=True)(input_)
    # hidden2 = keras.layers.SimpleRNN(20)(hidden1)

    out1 = keras.layers.Dense(2, name='pos')(last_hidden_layer)
    out2 = keras.layers.Dense(2, name='heading')(last_hidden_layer)

    model = keras.Model(inputs = [input_], outputs = [out1,out2])

    model.compile(optimizer=optimizer,
            loss = ['mse',ang_loss_fn],
                loss_weights = [1., 1.],
            )
    return model


def run_hyperparameter_tuner():

    with open(PROCESSED_DATA_DIR/'data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    random_search_tuner = kt.Hyperband(
        build_model, 
        objective='val_loss', 
        max_epochs = 20, 
        overwrite = True,
        factor = 10,
        hyperband_iterations= 1, 
        directory = TRAINING_LOGS_DIR, 
        project_name = "rnn_rnd_search", 
        seed = 42 
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(TRAINING_LOGS_DIR / 'tensorboard')
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)

    random_search_tuner.search(x = data['X_train'],
                               y = [data['y_train'][:,[0,1]],  p.ang_to_vector(data['y_train'][:,2], unit = 'degrees').numpy()],
                               epochs = 10,
                               validation_data = (data['X_valid'], [data['y_valid'][:,[0,1]], p.ang_to_vector(data['y_valid'][:,2], unit = 'degrees').numpy()]), 
                               callbacks = [early_stopping_cb, tensorboard_cb]
                               )

    best_model = random_search_tuner.get_best_models(num_models=1)[0]
    best_model.save(MODELS_DIR / 'model_02.keras')


def retrain_model(model_name = 'model_02.keras'):

    with open(PROCESSED_DATA_DIR/'data.pickle', 'rb') as handle:
        data = pickle.load(handle)
    model = keras.models.load_model(MODELS_DIR / model_name,custom_objects={"ang_loss_fn": ang_loss_fn})


    run_logdir = get_run_logdir()  


    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights = True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('rnn_checkpoints.weights.h5',save_weights_only = True)


    history = model.fit(
        x = data['X_train'], 
        y = [data['y_train'][:,[0,1]],  p.ang_to_vector(data['y_train'][:,2], unit = 'degrees').numpy()],
        epochs = 2000,
        batch_size = 128,
        validation_data = (data['X_valid'], [data['y_valid'][:,[0,1]], p.ang_to_vector(data['y_valid'][:,2], unit = 'degrees').numpy()]), 
        callbacks = [tensorboard_cb, early_stopping_cb, checkpoint_cb]    
        )
    
    model.save(MODELS_DIR / 'model_02.keras')
    logger.success("Modeling training complete.")
    

if __name__ == "__main__":
    main()
