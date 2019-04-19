# File: lstm_v2.py
# Author: Ronil Pancholia
# Date: 4/18/19
# Time: 10:08 PM

from torch import nn

class LSTM_v2(nn.Module):
    def __init__(self):
        super(LSTM_v2, self).__init__()
        self.lstm = nn.LSTM(input_size=120, hidden_size=200, num_layers=4, bias=True, bidirectional=False)
        self.fc2 = nn.Linear(3000, 2)
        self.input_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x, self.hidden = self.lstm(x.view(x.size(0), -1, 120))
        x = self.fc2(x.view(x.size(0), -1))
        return x