from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import keras
import keras_tuner as kt
from keras import ops
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

def train_rnn(
):
    # pull in processed dataset
    with open(PROCESSED_DATA_DIR/'data.pickle', 'rb') as handle:
        data = pickle.load(handle)


    random_search_tuner = kt.Hyperband(
        build_model, 
        objective='val_loss', 
        max_epochs = 200, 
        overwrite = False,
        factor = 3,
        hyperband_iterations= 2, 
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

    history = best_model.fit(
        x = data['X_train'], 
        y = [data['y_train'][:,[0,1]],  p.ang_to_vector(data['y_train'][:,2], unit = 'degrees').numpy()],
        epochs = 20,
        validation_data = (data['X_valid'], [data['y_valid'][:,[0,1]], p.ang_to_vector(data['y_valid'][:,2], unit = 'degrees').numpy()]), 
        callbacks = [early_stopping_cb, tensorboard_cb]    
    )

    best_model.save(MODELS_DIR / 'model_02.keras')


    # random_search_tuner.get_best_hyperparameters()[0].values
    # random_search_tuner.oracle.get_best_trials(num_trials = 6)[-1].summary()

    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])


    logger.success("Modeling training complete.")
    # -----------------------------------------

   

    

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

if __name__ == "__main__":
    train_rnn()
