# File: cnn_model.py
# Author: Ronil Pancholia
# Date: 3/23/19
# Time: 5:52 PM

from torch import nn
from config import FREQUENCY, TIME_WINDOW

class Cnn_Model(nn.Module):
    def __init__(self):
        super(Cnn_Model, self).__init__()
        self.conv1 = nn.Conv2d(12, 24, kernel_size=(3, 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=(3, 1), stride=1, padding=0)
        self.act = nn.ReLU()
        self.adaptivePool = nn.AdaptiveAvgPool2d((48, 1))
        self.fc1 = nn.Linear(2304, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.input_dropout = nn.Dropout(p=0.2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, 12, FREQUENCY * TIME_WINDOW, 1)
        x = self.input_dropout(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.dropout(x)
        x = self.adaptivePool(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x)
        return x
