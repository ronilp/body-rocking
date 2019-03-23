# File: main.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 5:24 PM

import numpy as np
import torch
from torch import nn

from config import RANDOM_SEED, TRAIN_EPOCHS, device
from dataset_utils import load_datasets, load_testset
from rocking_dataset import RockingDataset
from training_utils import get_model, fit

torch.manual_seed(RANDOM_SEED)

dataset_loaders, dataset_sizes = load_datasets(RockingDataset)
train_dataloader = dataset_loaders['train']
val_dataloader = dataset_loaders['val']

model, opt = get_model()
criterion = nn.CrossEntropyLoss()

train_loss, train_acc, val_loss, val_acc = fit(TRAIN_EPOCHS, model, criterion, opt, train_dataloader, val_dataloader)

def test_model():
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

test_model()