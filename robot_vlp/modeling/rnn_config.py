# config.py
import json, os

# Centralized hyperparameter configuration dictionary
_DEFAULT = {
    # Architecture parameters determined after stage 1
    "best_architecture": {
        "num_layers": 3,
        "recurrent_units": [16, 16, 8]
    },
    # Sequence lengths for slicing
    "sequence_length": {
        "input_length": 50,
        "use_length": 25
    },
    # Default regularization settings
    "regularization_defaults": {
        "dropout": [0.0, 0.1, 0.1],
        "recurrent_dropout": [0.1, 0.1, 0.1],
        "batch_norm": [True, True, False],
        "layer_norm": [False, False, False] 
    },
    # Default optimization settings for final model
    "optimization_defaults": {
        "lr": 0.005663272327474776,
        "scheduler": "cosine",
        "decay_steps": 10000
    }
}


if os.path.isfile("config.json"):
    with open("config.json") as f:
        GLOBAL_CONFIG = json.load(f)
else:
    GLOBAL_CONFIG = _DEFAULT
