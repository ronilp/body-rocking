# File: lstm_v7.py
# Author: Ronil Pancholia
# Date: 4/19/19
# Time: 12:44 AM

from torch import nn

class LSTM_v7(nn.Module):
    def __init__(self):
        super(LSTM_v7, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=200, num_layers=4, bias=True, bidirectional=True, dropout=0.2)
        self.fc2 = nn.Linear(30000, 2)
        self.input_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.input_dropout(x)
        x, self.hidden = self.lstm(x.view(x.size(0), -1, 12))
        x = self.fc2(x.view(x.size(0), -1))
        x = self.dropout(x)
        return x