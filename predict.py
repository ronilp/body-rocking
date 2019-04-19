# File: predict.py
# Author: Ronil Pancholia
# Date: 3/25/19
# Time: 10:34 PM

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from config import RANDOM_SEED, MODEL_DIR, device
from dataset_utils import load_testset
from rocking_dataset import RockingDataset
from training_utils import get_model

torch.manual_seed(RANDOM_SEED)

def test_model(model, sessions):
    test_dataloader, test_datasize = load_testset(RockingDataset, sessions=sessions)
    model.eval()
    results = []
    for image, label in tqdm(test_dataloader['test'], file=sys.stdout):
        image = image.to(device)
        outputs = model(image)
        preds = torch.argmax(outputs.data, 1)
        preds = preds.cpu()
        results.extend(np.asarray(preds))

    return results

model_name = sys.argv[2]
model, opt = get_model(model_name)
model_ckpt = sys.argv[1]
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_ckpt)))

sessions = ["Session01", "Session02", "Session03", "Session05", "Session06", "Session07", "Session12", "Session13", "Session15", "Session16", "Session08", "Session09", "Session11", "Session17"]

if not os.path.exists(model_name):
    os.makedirs(model_name)

for session in sessions:
    results = test_model(model, [session])
    with open(model_name + "/" + session + ".txt", "w") as f:
        for r in results:
            f.write(str(r) + "\n")