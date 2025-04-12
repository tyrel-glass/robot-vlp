import numpy as np
import tensorflow as tf
import keras
import keras_tuner as kt
from kerastuner import HyperParameters
from keras import ops
from keras import layers
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

def expand(y):
    y_rad = y[:,2] * np.pi / 180
    y_angles = np.column_stack((np.sin(y_rad), np.cos(y_rad)))
    return [y[:,:2], y_angles]  # return both position and angle lisths



def slice_last_n_timesteps(n):
    """Returns a function that slices the last `n` timesteps dynamically"""
    return layers.Lambda(lambda t: t[:, -n:, :])

def build_architecture_model(hp):
    # Fix less critical parameters
    fixed_dropout = 0.0
    fixed_lr = 1e-3
    fixed_optimizer = "adam"
    fixed_scheduler = "constant"
    
    # Tune sequence length and architecture:
    seq_length = hp.Fixed("sequence_length", 25)
    inputs = keras.layers.Input(shape=(50, 8))
    x = slice_last_n_timesteps(seq_length)(inputs)
    
    num_recurrent_layers = hp.Int("num_recurrent_layers", min_value=1, max_value=4, step=1)
    recurrent_layer_type = hp.Fixed("recurrent_layer_type", "LSTM")
    
    for i in range(num_recurrent_layers):
        units = hp.Choice(f"recurrent_units_{i}", values=[2, 4, 8, 16, 32, 64, 128, 256])

        return_sequences = i < (num_recurrent_layers - 1)
        
        x = layers.LSTM(units, return_sequences=return_sequences, 
                        dropout=fixed_dropout, recurrent_dropout=fixed_dropout, 
                        name=f'lstm_{i}')(x)
    
    loc_output = layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_output = layers.Dense(2, activation='tanh', name='angle_output')(x)
    
    model = Model(inputs=inputs, outputs=[loc_output, angle_output])
    optimizer = keras.optimizers.Adam(learning_rate=fixed_lr)
    model.compile(optimizer=optimizer, loss={'loc_output': 'mse', 'angle_output': 'mse'})
    return model




def build_regularization_model(hp):
    # Assume these are the best values from stage 1
    fixed_sequence_length = 26
    fixed_num_recurrent_layers = 2
    fixed_recurrent_layer_type = "LSTM"
    fixed_recurrent_units = [32, 32]  # Best units for layer 0 and 1, respectively

    inputs = keras.layers.Input(shape=(50, 8))
    x = slice_last_n_timesteps(fixed_sequence_length)(inputs)

    # Stage 2: Tune regularization for each layer
    for i in range(fixed_num_recurrent_layers):
        # Set tunable dropout rates (from 0.0 to 0.5 with 0.1 step)
        dropout_rate = hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)
        recurrent_dropout_rate = hp.Float(f"recurrent_dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)

        # Determine if the layer should return sequences (all except the last layer)
        return_sequences = i < (fixed_num_recurrent_layers - 1)
        
        if fixed_recurrent_layer_type == "LSTM":
            x = keras.layers.LSTM(
                fixed_recurrent_units[i],
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                name=f'lstm_{i}'
            )(x)
        else:  # If you were to use GRU
            x = keras.layers.GRU(
                fixed_recurrent_units[i],
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                name=f'gru_{i}'
            )(x)

        # Optionally tune normalization settings
        if hp.Boolean(f"batch_norm_{i}"):
            x = keras.layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        if hp.Boolean(f"layer_norm_{i}"):
            x = keras.layers.LayerNormalization(name=f'layer_norm_{i}')(x)

    # Output layers remain unchanged
    loc_output = keras.layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_output = keras.layers.Dense(2, activation='tanh', name='angle_output')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=[loc_output, angle_output])
    
    # Fix the optimizer and learning rate for now
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss={'loc_output': 'mse', 'angle_output': 'mse'},
        loss_weights={'loc_output': 1.0, 'angle_output': 1.0}
    )
    
    return model



def build_optimization_model(hp):
    # Fixed values from stage 1 and 2:
    fixed_sequence_length = 25
    fixed_num_recurrent_layers = 2
    fixed_recurrent_layer_type = "LSTM"
    fixed_recurrent_units = [32, 32]  # Best units for layer 0 and layer 1
    # Fixed regularization (from stage 2 best hyperparameters)
    fixed_dropout_0 = 0.0
    fixed_recurrent_dropout_0 = 0.0
    use_batch_norm_0 = True
    fixed_dropout_1 = 0.0
    fixed_recurrent_dropout_1 = 0.0
    use_batch_norm_1 = False

    # Input and slicing
    inputs = keras.layers.Input(shape=(50, 8))
    x = slice_last_n_timesteps(fixed_sequence_length)(inputs)
    
    # First recurrent layer
    x = keras.layers.LSTM(
        fixed_recurrent_units[0],
        return_sequences=True,
        dropout=fixed_dropout_0,
        recurrent_dropout=fixed_recurrent_dropout_0,
        name='lstm_0'
    )(x)
    if use_batch_norm_0:
        x = keras.layers.BatchNormalization(name='batch_norm_0')(x)
    
    # Second recurrent layer
    x = keras.layers.LSTM(
        fixed_recurrent_units[1],
        return_sequences=False,
        dropout=fixed_dropout_1,
        recurrent_dropout=fixed_recurrent_dropout_1,
        name='lstm_1'
    )(x)
    if use_batch_norm_1:
        x = keras.layers.BatchNormalization(name='batch_norm_1')(x)

    # Output layers
    loc_output = keras.layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_output = keras.layers.Dense(2, activation='tanh', name='angle_output')(x)
    
    model = keras.models.Model(inputs=inputs, outputs=[loc_output, angle_output])

    # Stage 3: Tune the optimization hyperparameters
    # Learning rate (tuned on a logarithmic scale)
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="LOG")
    
    # Learning rate scheduler selection
    lr_scheduler_type = hp.Choice("lr_scheduler", values=["constant", "exponential", "cosine"])
    if lr_scheduler_type == "exponential":
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr, decay_steps=10000, decay_rate=0.96, staircase=True
        )
    elif lr_scheduler_type == "cosine":
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=10000
        )
    else:  # Constant learning rate
        lr_schedule = lr

    # Optimizer selection
    optimizer_choice = hp.Choice("optimizer", values=["adam", "rmsprop", "nadam", "sgd"])
    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer_choice == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
    elif optimizer_choice == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    else:  # "nadam"
        optimizer = keras.optimizers.Nadam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss={'loc_output': 'mse', 'angle_output': 'mse'},
        loss_weights={'loc_output': 1.0, 'angle_output': 1.0}
    )
    return model





def build_stage4_model(hp):
    # Stage 4: Fine-tuning additional hyperparameters

    # Tune sequence length from a set of candidate values.
    seq_length = hp.Choice("sequence_length", values=[10, 20, 30, 40, 50])
    
    # Fixed values from previous stages:
    fixed_num_recurrent_layers = 2
    fixed_recurrent_units = [32, 32]         # Best units from stage 1
    fixed_recurrent_layer_type = "LSTM"       
    
    # Fixed regularization (from stage 2):
    fixed_dropout_0 = 0.0
    fixed_recurrent_dropout_0 = 0.0
    use_batch_norm_0 = True
    fixed_dropout_1 = 0.0
    fixed_recurrent_dropout_1 = 0.0
    use_batch_norm_1 = False
    
    # Build the model:
    inputs = keras.layers.Input(shape=(50, 8))
    x = slice_last_n_timesteps(seq_length)(inputs)
    
    # First recurrent layer:
    x = keras.layers.LSTM(
        fixed_recurrent_units[0],
        return_sequences=True,
        dropout=fixed_dropout_0,
        recurrent_dropout=fixed_recurrent_dropout_0,
        name="lstm_0"
    )(x)
    if use_batch_norm_0:
        x = keras.layers.BatchNormalization(name="batch_norm_0")(x)
        
    # Second recurrent layer:
    x = keras.layers.LSTM(
        fixed_recurrent_units[1],
        return_sequences=False,
        dropout=fixed_dropout_1,
        recurrent_dropout=fixed_recurrent_dropout_1,
        name="lstm_1"
    )(x)
    if use_batch_norm_1:
        x = keras.layers.BatchNormalization(name="batch_norm_1")(x)
    
    # Output layers:
    loc_output = keras.layers.Dense(2, activation="linear", name="loc_output")(x)
    angle_output = keras.layers.Dense(2, activation="tanh", name="angle_output")(x)
    
    model = keras.models.Model(inputs=inputs, outputs=[loc_output, angle_output])

    # Use the best optimization settings from stage 3:
    # (Best stage 3 hyperparameters were: lr=0.005663272327474776, lr_scheduler='cosine', optimizer='adam')
    lr = 0.005663272327474776
    lr_schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr, decay_steps=10000)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss={'loc_output': 'mse', 'angle_output': 'mse'},
        loss_weights={'loc_output': 1, 'angle_output': 1},
    )
    return model


def build_final_model():
    
    # Architecture hyperparameters (from stage 1):
    fixed_num_recurrent_layers = 2
    fixed_recurrent_units = [32, 32]  # Best units from stage 1
    fixed_recurrent_layer_type = "LSTM"  # Assuming LSTM is preferred
    
    # Regularization hyperparameters (from stage 2):
    fixed_dropout_0 = 0.0
    fixed_recurrent_dropout_0 = 0.0
    use_batch_norm_0 = True
    fixed_dropout_1 = 0.0
    fixed_recurrent_dropout_1 = 0.0
    use_batch_norm_1 = False
    
    # Build the model:
    inputs = keras.layers.Input(shape=(20, 8))


    # Preprocessing: Normalization layer to scale the data.
    # Note: Call `model.get_layer("normalization").adapt(X_train)` before training.
    x = keras.layers.Normalization(axis=-1, name="normalization")(inputs)

    
    # First recurrent layer:
    x = keras.layers.LSTM(
        fixed_recurrent_units[0],
        return_sequences=True,
        dropout=fixed_dropout_0,
        recurrent_dropout=fixed_recurrent_dropout_0,
        name="lstm_0"
    )(x)
    if use_batch_norm_0:
        x = keras.layers.BatchNormalization(name="batch_norm_0")(x)
        
    # Second recurrent layer:
    x = keras.layers.LSTM(
        fixed_recurrent_units[1],
        return_sequences=False,
        dropout=fixed_dropout_1,
        recurrent_dropout=fixed_recurrent_dropout_1,
        name="lstm_1"
    )(x)
    if use_batch_norm_1:
        x = keras.layers.BatchNormalization(name="batch_norm_1")(x)
    
    # Output layers:
    loc_output = keras.layers.Dense(2, activation="linear", name="loc_output")(x)
    angle_output = keras.layers.Dense(2, activation="tanh", name="angle_output")(x)
    
    model = keras.models.Model(inputs=inputs, outputs=[loc_output, angle_output])
    
    # Optimization hyperparameters (from stage 3):
    lr = 0.005663272327474776
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr, decay_steps=10000
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile the model with fixed loss weights:
    model.compile(
        optimizer=optimizer,
        loss={'loc_output': 'mse', 'angle_output': 'mse'},
        loss_weights={'loc_output': 1, 'angle_output': 1}
    )
    return model





