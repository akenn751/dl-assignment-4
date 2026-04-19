# ---------------------------------------------------------------------------------------
# rnn_vanilla.py
#
# Vanilla RNN with U, V, W. Using PyTorch autograd for backprop.
# Configured for char level & word level
# ---------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """
    Vectorized Vanilla RNN using U, W, V math and one-hot encoding.
    
    inputs (T, B) LongTensor -> logits (T, B, O), h_T (H, B).
    """

    def __init__(self, input_size, hidden_size, output_size, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        # Parameter shapes
        self.U = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)   # (H, D)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)  # (H, H)
        self.V = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)  # (O, H)

        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))  # (H, 1)
        self.by = nn.Parameter(torch.zeros(output_size, 1))  # (O, 1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.hidden_size, batch_size, device=self.U.device)

    def forward(self, inputs, h0=None):
        """
        inputs: (T, B) LongTensor of token indices
        returns: logits (T, B, O), h_T (H, B)
        """
        T, B = inputs.shape

        if h0 is None:
            h = self.init_hidden(B)   # (H, B)
        else:
            h = h0.to(self.U.device)

        # Ensure inputs are on same device as U for indexing
        if inputs.device != self.U.device:
            inputs = inputs.to(self.U.device)

        # Vectorized projection:
        # U[:, inputs] -> (H, T, B)
        # Permute to (T, B, H) so Ux[t] is (B, H)
        U_cols = self.U[:, inputs]            # (H, T, B)
        Ux = U_cols.permute(1, 2, 0).contiguous()  # (T, B, H)

        logits_list = []
        for t in range(T):
            # Ux[t]: (B, H) -> transpose to (H, B) to match W @ h
            Ux_t = Ux[t].transpose(0, 1)  # (H, B)

            # recurrence: h = tanh(Ux_t + W @ h + bh)
            h = torch.tanh(Ux_t + self.W @ h + self.bh)  # (H, B)

            # output: y = V @ h + by  -> (O, B)
            y = self.V @ h + self.by

            # append as (B, O)
            logits_list.append(y.transpose(0, 1))  # (B, O)

        logits = torch.stack(logits_list, dim=0)  # (T, B, O)
        return logits, h
