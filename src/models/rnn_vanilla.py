# ---------------------------------------------------------------------------------------
# rnn_vanilla.py
#
# Vanilla RNN with U, V, W. Using PyTorch autograd for backprop.
# Configured for char level & word level
# ---------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    """
    Tanh RNN:
        h_t = tanh(U x_t + W h_{t-1} + b_h)
        y_t = V h_t + b_y

    U: (H, D)
    W: (H, H)
    V: (O, H)
    """

    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.U = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)   # (H, D)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # (H, H)
        self.V = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)  # (O, H)

        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))  # (H, 1)
        self.by = nn.Parameter(torch.zeros(output_size, 1))  # (O, 1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.hidden_size, batch_size, device=self.device)

    def forward(self, inputs, h0=None):
        """
        inputs: (T, B) LongTensor
        h0: (H, B)
        returns: logits (T, B, O), h_T (H, B)
        """
        T, B = inputs.shape

        if h0 is None:
            h = self.init_hidden(B)
        else:
            h = h0

        logits = []

        for t in range(T):
            # one-hot: (B, D)
            x_t = F.one_hot(inputs[t], num_classes=self.input_size).float()

            # U @ x_t^T: (H, D) @ (D, B) -> (H, B)
            Ux = self.U @ x_t.T

            # recurrence
            h = torch.tanh(Ux + self.W @ h + self.bh)  # (H, B)

            # output
            y = self.V @ h + self.by  # (O, B)

            logits.append(y.transpose(0, 1))  # (B, O)

        logits = torch.stack(logits, dim=0)  # (T, B, O)
        return logits, h