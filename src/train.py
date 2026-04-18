# ----------------------------------------------------------------------------------------
# train.py
#
# Training script used for
#       char level vanilla RNN
#       word level vanilla RNN
#       char level LSTM
#       word level LSTM
#
# Switch use cases with flags: --mode char|word, -- model rnn|lstm
# ----------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from src.models.rnn_vanilla import VanillaRNN
# from src.models.rnn_lstm import LSTMRNN
from src.utils.dataloader import SequenceDataLoader
from src.utils.tokenizer import CharTokenizer, WordTokenizer

def train(
    mode="char",
    model_type="rnn",
    hidden_size=128,
    seq_length=50,
    batch_size=32,
    epochs=3,
    lr=1e-2,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):

    print("Using device:", device)

    # -----------------------------
    # Tokenizer & DataLoader
    # -----------------------------
    loader_obj = SequenceDataLoader(mode=mode, seq_length=seq_length)
    loader = loader_obj.get_loader(batch_size)
    tokenizer = loader_obj.tokenizer
    vocab_size = loader_obj.vocab_size

    # -----------------------------
    # Model selection
    # -----------------------------
    if model_type == "rnn":
        model = VanillaRNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            device=device,
        ).to(device)

    elif model_type == "lstm":
        raise NotImplementedError("LSTM not yet built")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0

        for batch_inputs, batch_targets in tqdm(loader, desc=f"Epoch {epoch}"):
            batch_inputs = batch_inputs.transpose(0, 1).to(device, non_blocking=True)
            batch_targets = batch_targets.transpose(0, 1).to(device, non_blocking=True)
            T, B = batch_inputs.shape

            optimizer.zero_grad()

            logits, _ = model(batch_inputs)
            logits_flat = logits.reshape(T * B, vocab_size)
            targets_flat = batch_targets.reshape(T * B)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

        print("Sample:", sample(model, tokenizer, mode, device=device))
        print()


def sample(model, tokenizer, mode, device="cpu", seed_text="The ", length=200):
    model.eval()
    with torch.no_grad():
        idx = tokenizer.encode(seed_text)[0]
        h = None
        indices = [idx]

        for _ in range(length):
            x = torch.tensor([[idx]], dtype=torch.long, device=device)
            logits, h = model(x, h0=h)
            probs = torch.softmax(logits[-1, 0], dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            indices.append(idx)

        return tokenizer.decode(indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["char", "word"], default="char")
    parser.add_argument("--model", choices=["rnn", "lstm"], default="rnn")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seq", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)

    args = parser.parse_args()

    train(
        mode=args.mode,
        model_type=args.model,
        hidden_size=args.hidden,
        seq_length=args.seq,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )