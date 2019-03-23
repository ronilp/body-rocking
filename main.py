# File: main.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 5:24 PM
import os

import numpy as np
import torch
from torch import nn
import pickle

from config import RANDOM_SEED, TRAIN_EPOCHS, device, MODEL_DIR, MODEL_PREFIX
from dataset_utils import load_datasets, load_testset
from rocking_dataset import RockingDataset
from training_utils import get_model, fit

torch.manual_seed(RANDOM_SEED)


def save_loss_acc(train_loss, train_acc, val_loss, val_acc):
    pickle.dump(train_loss, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "train_loss"), "wb"))
    pickle.dump(train_acc, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "train_acc"), "wb"))
    pickle.dump(val_loss, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "val_loss"), "wb"))
    pickle.dump(val_acc, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "val_acc"), "wb"))


def test_model(model):
    test_dataloader = load_testset(RockingDataset)
    model.eval()
    results = []
    corrects = 0
    for image, label in test_dataloader:
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        preds = torch.argmax(outputs.data, 1)
        corrects += torch.sum(preds == label).item()
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    print("Testing accuracy :", 100 * corrects / len(test_dataloader.dataset))


dataset_loaders, dataset_sizes = load_datasets(RockingDataset)
train_dataloader = dataset_loaders['train']
val_dataloader = dataset_loaders['val']

model, opt = get_model()
criterion = nn.CrossEntropyLoss()

train_loss, train_acc, val_loss, val_acc, model = fit(TRAIN_EPOCHS, model, criterion, opt, train_dataloader, val_dataloader)

save_loss_acc(train_loss, train_acc, val_loss, val_acc)

test_model(model)
