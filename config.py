# File: config.py
# Author: Ronil Pancholia
# Date: 3/7/19
# Time: 2:51 AM

import torch

### Learning Parameters
BASE_LR = 1e-4
TRAIN_EPOCHS = 10
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 10

### Dataset Config
DATA_DIR = "data/TrainingData"
ALLOWED_CLASSES = ["0", "1"]
NUM_CLASSES = len(ALLOWED_CLASSES)
MODEL_DIR = "results"

### Miscellaneous Config
MODEL_PREFIX = "cnn_v3"
BATCH_SIZE = 32
RANDOM_SEED = 629
TIME_WINDOW = 3 # in seconds
TRAIN_OVERLAP = 0.1
TEST_OVERLAP = 1
FREQUENCY = 50
LSTM_IN_SIZE = 24
HIDDEN_DIM = 64

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")