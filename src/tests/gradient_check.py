# ---------------------------------------------------------
# gradient_check.py
#
# Gradient check feature, builds a small VanillaRNN, creates a small input sequence,
# computes the loss, computes the autograd gradients, computes the numerical gradients, 
# and compares them to validate that the gradients are correct.
# ---------------------------------------------------------

import torch
import torch.nn as nn

from src.models.rnn_vanilla import VanillaRNN

def numerical_gradient(model, x, targets, param_name, eps=1e-5):
    param = dict(model.named_parameters())[param_name]
    num_grad = torch.zeros_like(param)

    flat = param.view(-1)

    for i in range(flat.shape[0]):
        old_val = flat[i].item()

        # f(theta + eps)
        with torch.no_grad():
            flat[i] = old_val + eps
        loss_pos = compute_loss(model, x, targets)

        # f(theta - eps)
        with torch.no_grad():
            flat[i] = old_val - eps
        loss_neg = compute_loss(model, x, targets)

        # numerical derivative
        num_grad.view(-1)[i] = (loss_pos - loss_neg) / (2 * eps)

        # restore original value
        with torch.no_grad():
            flat[i] = old_val

    return num_grad

def compute_loss(model, x, targets):
    """
    Computes the loss
    """
    logits, _ = model(x)
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    return nn.CrossEntropyLoss()(logits, targets)

def gradient_check():
    """
    Performs the gradient check
    """
    torch.manual_seed(0)

    vocab_size = 5
    hidden_size = 4
    seq_length = 3

    model = VanillaRNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size,
        device="cpu"
    )

    # Small synthetic input for testing
    x = torch.tensor([[0, 1, 2]], dtype=torch.long).t()   # shape (T=3, B=1)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.long).t()

    # Compute autograd gradients
    loss = compute_loss(model, x, targets)
    loss.backward()

    print("Running Gradient Check...")
    for name, param in model.named_parameters():
        print(f"\nChecking {name}...")

        autograd_grad = param.grad.clone()
        num_grad = numerical_gradient(model, x, targets, name)

        # Compute relative error
        diff = torch.norm(autograd_grad - num_grad)
        denom = torch.norm(autograd_grad) + torch.norm(num_grad)
        rel_error = diff / denom

        print(f"Relative error: {rel_error.item():.6e}")

        if rel_error < 1e-4:
            print("PASS")
        else:
            print("FAIL")
    
    print("Gradient Check Complete.")


if __name__ == "__main__":
    gradient_check()
