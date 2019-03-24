# File: main.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 5:24 PM
import os
import pickle
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import RANDOM_SEED, TRAIN_EPOCHS, device, MODEL_DIR, MODEL_PREFIX, FREQUENCY, TIME_WINDOW
from dataset_utils import load_datasets, load_testset
from rocking_dataset import RockingDataset
from training_utils import get_model, fit

torch.manual_seed(RANDOM_SEED)


def save_loss_acc(train_loss, train_acc, val_loss, val_acc):
    pickle.dump(train_loss, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_train_loss.pkl"), "wb"))
    pickle.dump(train_acc, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_train_acc.pkl"), "wb"))
    pickle.dump(val_loss, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_val_loss.pkl"), "wb"))
    pickle.dump(val_acc, open(os.path.join(MODEL_DIR, MODEL_PREFIX + "_val_acc.pkl"), "wb"))


def test_model(model):
    test_dataloader, test_datasize = load_testset(RockingDataset)
    model.eval()
    results = []
    corrects = 0
    for image, label in tqdm(test_dataloader['test'], file=sys.stdout):
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        raw_preds = torch.argmax(outputs.data, 1)
        preds = []
        for pred in raw_preds:
            for j in range(FREQUENCY * TIME_WINDOW):
                preds.append(pred)

        preds = torch.Tensor(np.asarray(preds).astype(np.long))
        labels = []
        for l in label:
            labels.extend(l)
        labels = torch.Tensor(np.asarray(labels).astype(np.long))
        corrects += torch.sum(preds == labels).item()
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    print("Testing accuracy :", (100.0 * corrects) / (len(test_dataloader['test'].dataset) * FREQUENCY * TIME_WINDOW))


dataset_loaders, dataset_sizes = load_datasets(RockingDataset)
train_dataloader = dataset_loaders['train']
val_dataloader = dataset_loaders['val']

model, opt = get_model()
criterion = nn.CrossEntropyLoss()

# from LRFinder import LRFinder
# lr_finder = LRFinder(model, opt, criterion, device="cpu")
# lr_finder.range_test(train_dataloader, val_dataloader, end_lr=0.1, num_iter=100)
# lr_finder.plot()

train_loss, train_acc, val_loss, val_acc, model = fit(TRAIN_EPOCHS, model, criterion, opt, train_dataloader, val_dataloader)

save_loss_acc(train_loss, train_acc, val_loss, val_acc)

test_model(model)
