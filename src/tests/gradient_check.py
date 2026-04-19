# ---------------------------------------------------------
# gradient_check.py
#
# Gradient check feature, builds a small VanillaRNN, creates a small input sequence,
# computes the loss, computes the autograd gradients, computes the numerical gradients, 
# and compares them to validate that the gradients are correct.
# ---------------------------------------------------------

import torch
import torch.nn as nn
import copy

from src.tests.rnn_tiny import TinyVanillaRNN


def compute_loss(model, x, targets):
    """
    Computes loss for unbatched TinyVanillaRNN.
    x: (T,)
    targets: (T,)
    """
    logits, _ = model(x)  # (T, O)
    T, O = logits.shape
    logits = logits.view(T, O)
    targets = targets.view(T)
    return nn.CrossEntropyLoss()(logits, targets)


def numerical_gradient(model, x, targets, param_name, eps=1e-5):
    """
    Numerical gradient with model cloning to avoid state contamination.
    """
    param = dict(model.named_parameters())[param_name]
    num_grad = torch.zeros_like(param)

    flat = param.view(-1)

    for i in range(flat.shape[0]):
        old_val = flat[i].item()

        # +eps
        model_pos = copy.deepcopy(model)
        with torch.no_grad():
            dict(model_pos.named_parameters())[param_name].view(-1)[i] = old_val + eps
        loss_pos = compute_loss(model_pos, x, targets)

        # -eps
        model_neg = copy.deepcopy(model)
        with torch.no_grad():
            dict(model_neg.named_parameters())[param_name].view(-1)[i] = old_val - eps
        loss_neg = compute_loss(model_neg, x, targets)

        # numerical derivative
        num_grad.view(-1)[i] = (loss_pos - loss_neg) / (2 * eps)

    return num_grad


def gradient_check():
    torch.manual_seed(0)

    vocab_size = 5
    hidden_size = 4
    seq_length = 3

    model = TinyVanillaRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size,
    )

    x = torch.tensor([0, 1, 2], dtype=torch.long)        # (T,)
    targets = torch.tensor([1, 2, 3], dtype=torch.long)  # (T,)

    # autograd gradients
    model.zero_grad()
    loss = compute_loss(model, x, targets)
    loss.backward()

    print("Running Gradient Check...\n")

    for name, param in model.named_parameters():
        print(f"Checking {name}...")

        autograd_grad = param.grad.clone()
        num_grad = numerical_gradient(model, x, targets, name)

        diff = torch.norm(autograd_grad - num_grad)
        denom = torch.norm(autograd_grad) + torch.norm(num_grad)
        rel_error = (diff / denom).item()

        print(f"Relative error: {rel_error:.6e}")
        if rel_error < 1e-4:
            print("PASS\n")
        else:
            print("FAIL\n")

    print("Gradient Check Complete.")


if __name__ == "__main__":
    gradient_check()
