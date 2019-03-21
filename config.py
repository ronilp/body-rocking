# File: config.py
# Author: Ronil Pancholia
# Date: 3/7/19
# Time: 2:51 AM

import torch

### Learning Parameters
BASE_LR = 1e-6
TRAIN_EPOCHS = 5
EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_PATIENCE = 10

### Dataset Config
DATA_DIR = "data/TrainingData"
ALLOWED_CLASSES = ["0", "1"]
NUM_CLASSES = len(ALLOWED_CLASSES)

### Miscellaneous Config
MODEL_PREFIX = "model_name"
BATCH_SIZE = 64
RANDOM_SEED = 629

### GPU SETTINGS
CUDA_DEVICE = 0  # GPU device ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")