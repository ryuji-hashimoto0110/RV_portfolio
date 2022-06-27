from torch import nn
import torch.nn.functional as F

#---
# Multi-Task model using LSTM
#---

class LSTM_RV_PF(nn.Module):
    def __init__(self,
                 input_size=5,
                 hidden_size=124,
                 output_size=5):
        super(LSTM_RV_PF, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=2, dropout=0.5,
                           batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc21 = nn.Linear(hidden_size, output_size+1)
        self.fc22 = nn.Linear(hidden_size, output_size+1)

    def forward(self, x):
        b, _, _    = x.shape
        out, _     = self.rnn(x, None)
        rv_preds   = self.fc1(out[:,-1,:])
        out1       = self.fc21(out[:,-1,:])
        out2       = self.fc22(out[:,-1,:])
        out1       = F.softmax(out1, dim=-1)
        out2       = F.softmax(out2, dim=-1)
        portfolio  = 2*out1 - out2
        return rv_preds, portfolio