# File: training_utils.py
# Author: Ronil Pancholia
# Date: 3/20/19
# Time: 5:22 PM
import copy
import os
import sys
import time

import torch
from torch import optim
from tqdm import tqdm

import config
from config import device, BASE_LR, MODEL_PREFIX, MODEL_DIR, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_ENABLED, BATCH_SIZE
from models.cnn_lstm_model import CnnLSTMModel
from models.cnn_paper import Cnn_Model3
from models.lstm_v2 import LSTM_v2
from models.lstm_v3 import LSTM_v3
from models.lstm_v4 import LSTM_v4
from models.lstm_v5 import LSTM_v5
from models.lstm_v6 import LSTM_v6
from models.lstm_v7 import LSTM_v7


def loss_batch(model, criterion, x, y, opt=None):
    outputs = model(x)
    loss = criterion(outputs, y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    preds = torch.argmax(outputs.data, 1)
    corrects = torch.sum(preds == y)
    return loss.item(), len(x), corrects


def fit(num_epochs, model, criterion, opt, train_dataloader, val_dataloader=None):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_loss = sys.maxsize
    best_acc = 0
    patience = 0

    for epoch in range(num_epochs):
        print("\nEpoch " + str(epoch + 1))

        running_loss = 0.0
        model.train()
        running_corrects = 0
        for image, label in tqdm(train_dataloader, file=sys.stdout):
            image, label = image.to(device), label.to(device)
            losses, nums, corrects = loss_batch(model, criterion, image, label, opt)
            running_loss += losses
            running_corrects += corrects

        train_loss.append(running_loss / (len(train_dataloader.dataset) / BATCH_SIZE))
        train_acc.append(running_corrects.item() / (len(train_dataloader.dataset)))
        print('Training loss: {:6f}, accuracy: {:4f}'.format(train_loss[-1], train_acc[-1]))

        if val_dataloader == None:
            model_name = MODEL_PREFIX + "_" + str(epoch) + "_" + str(time.time()) + ".pt"
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))
            print("Saved model :", model_name)
            continue

        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for image, label in tqdm(val_dataloader, file=sys.stdout):
                image, label = image.to(device), label.to(device)
                losses, nums, corrects = loss_batch(model, criterion, image, label)
                running_loss += losses
                running_corrects += corrects

        val_loss.append(running_loss / (len(train_dataloader.dataset) / BATCH_SIZE))
        val_acc.append(running_corrects.item() / (len(val_dataloader.dataset)))
        print('Validation loss: {:6f}, accuracy: {:4f}'.format(val_loss[-1], val_acc[-1]))

        if val_loss[-1] < best_loss:
            patience = 0
            best_acc = val_acc[-1]
            best_loss = val_loss[-1]
            best_model = copy.deepcopy(model)
            print('Best accuracy: {:4f}, loss: {:6f}'.format(best_acc, best_loss))
            model_name = MODEL_PREFIX + "_" + str(epoch) + "_" + str(time.time()) + ".pt"
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_name))
            print("Saved model :", model_name)
        else:
            patience += 1
            print("Loss did not improve, patience: " + str(patience))

        # Early stopping
        if patience > EARLY_STOPPING_PATIENCE and EARLY_STOPPING_ENABLED:
            print("Early Stopping!")
            break

    print('Best accuracy: {:4f}, loss: {:6f}'.format(best_acc, best_loss))

    return train_loss, train_acc, val_loss, val_acc


def get_model(model_='CNN'):
    if model_ == 'CNN':
        model = Cnn_Model3()
    elif model_ == "CNN_LSTM":
        model = CnnLSTMModel(input_dim=config.LSTM_IN_SIZE, hidden_dim=config.HIDDEN_DIM, batch_size=config.BATCH_SIZE, output_dim=config.NUM_CLASSES)
    elif model_ == "lstm_v2":
        model = LSTM_v2()
    elif model_ == "lstm_v3":
        model = LSTM_v3()
    elif model_ == "lstm_v4":
        model = LSTM_v4()
    elif model_ == "lstm_v5":
        model = LSTM_v5()
    elif model_ == "lstm_v6":
        model = LSTM_v6()
    elif model_ == "lstm_v7":
        model = LSTM_v7()

    model.to(device)
    return model, optim.RMSprop(model.parameters(), lr=BASE_LR)