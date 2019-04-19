# File: lstm_v4.py
# Author: Ronil Pancholia
# Date: 4/18/19
# Time: 10:50 PM

from torch import nn

class LSTM_v4(nn.Module):
    def __init__(self):
        super(LSTM_v4, self).__init__()
        self.lstm = nn.LSTM(input_size=120, hidden_size=200, num_layers=4, bias=True, bidirectional=False, dropout=0.5)
        self.fc2 = nn.Linear(3000, 2)
        self.input_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.input_dropout()
        x, self.hidden = self.lstm(x.view(x.size(0), -1, 120))
        x = self.fc2(x.view(x.size(0), -1))
        x = self.dropout()
        return x