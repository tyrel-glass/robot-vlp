import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

def build_final_model():
    # Hardcoded configuration values
    SEQUENCE_LENGTH = 20
    INPUT_FEATURES = 8

    # Architecture
    NUM_LAYERS = 2
    RECURRENT_UNITS = [32, 8]

    # Regularization
    DROPOUT = [0.0, 0.01]
    RECURRENT_DROPOUT = [0.025, 0.05]
    BATCH_NORM = [False, False]
    LAYER_NORM = [True, False]

    # Optimization
    LEARNING_RATE = 0.0033727437553468355
    OPTIMIZER = keras.optimizers.Nadam(LEARNING_RATE)

    # Model input
    inputs = keras.layers.Input(shape=(SEQUENCE_LENGTH, INPUT_FEATURES), name='input')

    # Optional: replace with actual normalization logic if needed
    x = tf.keras.layers.LayerNormalization()(inputs)  # acts like global_norm()

    # LSTM stack
    for i in range(NUM_LAYERS):
        return_seq = (i < NUM_LAYERS - 1)
        x = layers.LSTM(
            RECURRENT_UNITS[i],
            return_sequences=return_seq,
            dropout=DROPOUT[i],
            recurrent_dropout=RECURRENT_DROPOUT[i],
            name=f'lstm_{i}'
        )(x)
        if BATCH_NORM[i]:
            x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        if LAYER_NORM[i]:
            x = layers.LayerNormalization(name=f'layer_norm_{i}')(x)

    # Outputs
    loc_out = layers.Dense(2, activation='linear', name='loc_output')(x)
    angle_out = layers.Dense(2, activation='tanh', name='angle_output')(x)

    # Compile model
    model = Model(inputs=inputs, outputs=[loc_out, angle_out])
    model.compile(
        optimizer=OPTIMIZER,
        loss={'loc_output': 'mse', 'angle_output': 'mse'}
    )
    return model

# 1. Build and export the model
model = build_final_model()

model(tf.random.uniform([1, 20, 8]))
MODEL_DIR = "keras_lstm"
model.export(MODEL_DIR)

# 2. Convert to TFLite with fixes
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.experimental_enable_resource_variables = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# 3. Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as 'model.tflite'")