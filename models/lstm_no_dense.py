# File: lstm_no_dense.py
# Author: Ronil Pancholia
# Date: 4/18/19
# Time: 10:08 PM

from torch import nn
import torch
from config import FREQUENCY, TIME_WINDOW


class LSTMnodense(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size,
                 output_dim, num_layers):
        super(LSTMnodense, self).__init__()
        # self.input_dim = input_dim #12
        # self.hidden_dim = hidden_dim #100
        # self.batch_size = batch_size #64
        # self.num_layers = num_layers #2
        # self.output_dim = output_dim #2
        # self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bias=True)

        self.lstm = nn.LSTM(input_size=12, hidden_size=200, num_layers=4, bias=True, bidirectional=True)
        self.fc2 = nn.Linear(60000, 2)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        x, self.hidden = self.lstm(x.view(len(x), -1, 12))
        x = self.fc2(x.view(x.size(0), -1))
        return x