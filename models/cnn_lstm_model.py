# File: cnn_lstm_model.py
# Author: Ronil Pancholia
# Date: 4/16/19
# Time: 6:35 PM
# File: cnn_lstm_model.py
# Author: Vincent Tompkins
# Date: 4/15/19
# Time: Now

from torch import nn
from config import FREQUENCY, TIME_WINDOW


class CnnLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size,
                 output_dim=1, num_layers=1):
        super(CnnLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(12, 96, kernel_size=(6, 1), stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96*2, kernel_size=(4, 1), stride=1, padding=1)
        self.conv3 = nn.Conv2d(96*2, 96*4, kernel_size=(3, 1), stride=2, padding=0)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(192)
        self.bn2 = nn.BatchNorm2d(96*4)
        self.adaptivePool = nn.AdaptiveAvgPool2d((24, 1))
        self.lstm1 = nn.LSTM(288, self.hidden_dim, self.num_layers, bias=True)
        self.fc2 = nn.Linear(2048, 2)
        self.dropout = nn.Dropout()


    def forward(self, x):
        x = x.view(-1, 12, FREQUENCY * TIME_WINDOW, 1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.bn1(x)
        x = self.act(self.conv3(x))
        x = self.adaptivePool(x)
        x = self.bn2(x)
        x, self.hidden = self.lstm1(x.view(x.size(0), self.batch_size, -1))
        x = self.fc2(x.view(x.size(0), -1))
        return x