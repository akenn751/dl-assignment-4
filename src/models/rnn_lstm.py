# ---------------------------------------------------------------------------------------
# rnn_lstm.py
#
# LSTM RNN using PyTorch optimized LSTM kernel. For use with train.py script.
# Configured for char level & word level
# ---------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)

        # Output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, h0=None):
        # inputs: (T, B) long
        # convert to one-hot (T, B, D)
        one_hot = F.one_hot(inputs, num_classes=self.input_size).float()

        if h0 is None:
            h0 = torch.zeros(1, inputs.size(1), self.hidden_size, device=inputs.device)
            c0 = torch.zeros(1, inputs.size(1), self.hidden_size, device=inputs.device)
        else:
            h0, c0 = h0

        outputs, (hn, cn) = self.lstm(one_hot, (h0, c0))
        logits = self.fc(outputs)  # (T, B, vocab)

        return logits, (hn, cn)
