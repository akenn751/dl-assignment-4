# ------------------------------------------------------------
# rnn_tiny.py
#
# A small vanilla RNN to be used for gradient checking
# ------------------------------------------------------------

# src/tests/rnn_tiny.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVanillaRNN(nn.Module):
    """
    Minimal, unbatched Vanilla RNN for gradient checking.

    h_t = tanh(U x_t + W h_{t-1} + b_h)
    y_t = V h_t + b_y

    U: (H, D)
    W: (H, H)
    V: (O, H)
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size   # D
        self.hidden_size = hidden_size # H
        self.output_size = output_size # O

        self.U = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)   # (H, D)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # (H, H)
        self.V = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)  # (O, H)

        self.bh = nn.Parameter(torch.zeros(hidden_size))   # (H,)
        self.by = nn.Parameter(torch.zeros(output_size))   # (O,)

    def forward(self, inputs, h0=None):
        """
        inputs: LongTensor (T,)
        h0: (H,)

        Returns:
            logits: (T, O)
            h_T: (H,)
        """
        T = inputs.size(0)

        if h0 is None:
            h = torch.zeros(self.hidden_size)
        else:
            h = h0

        logits = []

        for t in range(T):
            # x_t: (D,)
            x_t = F.one_hot(inputs[t], num_classes=self.input_size).float()

            # U x_t: (H, D) @ (D,) -> (H,)
            Ux = self.U @ x_t

            # recurrence
            h = torch.tanh(Ux + self.W @ h + self.bh)  # (H,)

            # output
            y = self.V @ h + self.by  # (O,)

            logits.append(y)

        logits = torch.stack(logits, dim=0)  # (T, O)
        return logits, h
