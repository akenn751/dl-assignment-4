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

import os
import json
import csv
import time
import random
from datetime import datetime, timezone
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.models.rnn_vanilla import VanillaRNN
# from src.models.rnn_lstm import LSTMRNN
from src.utils.dataloader import SequenceDataLoader
from src.utils.tokenizer import CharTokenizer, WordTokenizer

# ----------------------------
# Helper functions for outputs
# ----------------------------

def makedirs(path):
    """
    Make directories
    """
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state, path):
    """
    Save checkpoints
    """
    makedirs(os.path.dirname(path))
    torch.save(state, path)

# def append_loss_csv(csv_path, row):
#     """
#     Append loss calculations to CSV
#     """
#     makedirs(os.path.dirname(csv_path))
#     write_header = not os.path.exists(csv_path)
#     with open (csv_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(["timestamp", "global_step", "epoch", "iter", "loss"])
#         writer.writerow(row)

def append_epoch_csv(csv_path, row):
    """
    Append epoch loss calculations to CSV
    """
    makedirs(os.path.dirname(csv_path))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "epoch", "avg_loss"])
        writer.writerow(row)


def save_sample_text(text, path):
    """"
    Save the sample text produced.
    """
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as  f:
        f.write(text)

def save_metadata(meta, path):
    """
    Save metadata produced
    """
    makedirs(os.path.dirname(path))
    with open (path, "w") as f:
        json.dump(meta, f, indent=2)

def plot_loss(csv_path, out_png):
    """
    Plot losses against global steps and save.
    """
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.plot(df['global_step'], df['loss'], linewidth=1)
    plt.xlabel('Global step')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.grid(True)
    makedirs(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

def plot_epoch_loss(epoch_csv, out_png):
    """
    Plot losses against epochs and save.
    """
    if not os.path.exists(epoch_csv):
        return
    df = pd.read_csv(epoch_csv)
    if df.empty:
        return
    plt.figure(figsize=(8,4))
    plt.plot(df['epoch'], df['avg_loss'], marker='o', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Avg Loss')
    plt.title('Training loss vs Epoch')
    plt.grid(True)
    makedirs(os.path.dirname(out_png))
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()


def train(
    mode="char",
    model_type="rnn",
    hidden_size=128,
    seq_length=50,
    batch_size=32,
    epochs=3,
    lr=1e-2,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # artifact/config options
    exp_name="exp_default",
    save_dir="artifacts",
    save_every=1,
    breakpoints=None,
    breakpoint_interval=0,
    sample_length=200,
    max_steps=0,
    seed=42,
    seed_text="The ",
    save_checkpoints=False,
    save_final_only=False,
):
    
    # normalize breakpoints into a set of epoch integers
    if breakpoints is None:
        breakpoints = set()
    else:
        breakpoints = set(breakpoints)

    # if an interval is provided, compute epoch numbers up to `epochs`
    if breakpoint_interval and breakpoint_interval > 0:
        interval_set = set(range(breakpoint_interval, epochs + 1, breakpoint_interval))
        # interval overrides explicit list
        breakpoints = interval_set


    print("Using device:", device)

    # ---------------------------------
    # Artifact Directories and Metadata
    # ---------------------------------
    base_dir = save_dir
    ckpt_dir = os.path.join(base_dir, "checkpoints", exp_name)
    logs_dir = os.path.join(base_dir, "logs", exp_name)
    samples_dir = os.path.join(base_dir, "samples", exp_name)
    plots_dir = os.path.join(base_dir, "plots", exp_name)
    meta_dir = os.path.join(base_dir, "metadata", exp_name)

    makedirs(ckpt_dir); makedirs(logs_dir); makedirs(samples_dir); makedirs(plots_dir); makedirs(meta_dir)

    meta = {
        "exp_name": exp_name,
        "mode": mode,
        "model_type": model_type,
        "hidden": hidden_size,
        "batch": batch_size,
        "seq": seq_length,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    save_metadata(meta, os.path.join(meta_dir, "meta.json"))

    # ---------------------------------
    # Seeds
    # ---------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    # Autograd sanity check (one forward/backward pass)
    # -----------------------------
    autograd_sanity_path = os.path.join(logs_dir, "autograd_sanity.txt")
    try:
        model.zero_grad()
        xb, yb = next(iter(loader))
        xb = xb.transpose(0,1).to(device, non_blocking=True)
        yb = yb.transpose(0,1).to(device, non_blocking=True)
        T, B = xb.shape
        logits, _ = model(xb)
        logits_flat = logits.reshape(T * B, vocab_size)
        targets_flat = yb.reshape(T * B)
        loss0 = criterion(logits_flat, targets_flat)
        loss0.backward()
        with open(autograd_sanity_path, "w") as f:
            for name, p in model.named_parameters():
                grad_sample = p.grad.view(-1)[:8].cpu().numpy().tolist() if p.grad is not None else None
                f.write(f"{name} grad_sample: {grad_sample}\n")
    except Exception as e:
        with open(autograd_sanity_path, "w") as f:
            f.write("Autograd sanity check failed: " + str(e) + "\n")

    # -----------------------------
    # Training loop
    # -----------------------------
    epoch_losses = []
    global_step = 0
    # loss_csv = os.path.join(logs_dir, "loss.csv")
    last_epoch_loss = None
    stop_training = False

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for iter_idx, (batch_inputs, batch_targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
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
            global_step += 1

            # append per-iteration loss
            # append_loss_csv(loss_csv, [datetime.now(timezone.utc).isoformat(), global_step, epoch, iter_idx, float(loss.item())])

            # stop early if indicated
            if max_steps and global_step >= max_steps:
                print(f"Reached max_steps={max_steps}; stopping early")
                stop_training = True
                break

        avg_loss = total_loss / max(1, steps)
        epoch_losses.append(avg_loss)
        last_epoch_loss = avg_loss
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

        # write per-epoch CSV row
        epoch_csv = os.path.join(logs_dir, "epoch_loss.csv")
        append_epoch_csv(epoch_csv, [datetime.now(timezone.utc).isoformat(), epoch, float(avg_loss)])

        # Save checkpoint every save_every epochs if requested
        if save_every and (epoch % save_every == 0):
            if save_checkpoints:
                ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}.pt")
                save_checkpoint({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "args": meta
                }, ckpt_path)
                print(f"Saved checkpoint {ckpt_path}")
            else:
                # write a tiny marker so you can still see epoch progress without large files
                marker = os.path.join(ckpt_dir, f"epoch{epoch}.done")
                makedirs(os.path.dirname(marker))
                with open(marker, "w") as f:
                    f.write(f"epoch {epoch} completed at {datetime.now(timezone.utc).isoformat()}\n")


        # Save a single sample at breakpoints (simple behavior)
        if epoch in breakpoints:
            print(f"\n--- Sample at epoch {epoch} ---")
            sample_text = sample(model, tokenizer, mode, device=device, seed_text=seed_text, length=sample_length)
            sample_path = os.path.join(samples_dir, f"sample_epoch{epoch}.txt")
            save_sample_text(sample_text, sample_path)
            print(f"Saved sample -> {sample_path}\n")

        if stop_training:
            break

    # Save a final model state_dict
    final_dir = os.path.join(save_dir, exp_name)
    makedirs(final_dir)
    final_path = os.path.join(final_dir, "final_model.pt")
    # If user requested final-only or checkpoints were not enabled, write final state_dict
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model state_dict to {final_path}")


    # After training: plot loss and write summary
    # plot_loss(loss_csv, os.path.join(plots_dir, "loss_curve.png"))
    # Plot epoch-level loss
    epoch_csv = os.path.join(logs_dir, "epoch_loss.csv")
    plot_epoch_loss(epoch_csv, os.path.join(plots_dir, "loss_vs_epoch.png"))

    final_loss_str = f"{last_epoch_loss:.4f}" if last_epoch_loss is not None else "NA"
    summary_path = os.path.join(base_dir, "results_summary.txt")
    with open(summary_path, "a") as f:
        f.write(f"{exp_name}: finished epochs={epochs}, final_loss={final_loss_str}, checkpoints={ckpt_dir}, samples={samples_dir}\n")

    return epoch_losses

def sample(model, tokenizer, mode, device="cpu", seed_text="The ", length=200):
    """
    Sampling function to produce sample outputs.
    """
    model.eval()
    with torch.no_grad():
        # tokenizer.encode returns list of token ids for the string
        # pick first token as start
        encoded = tokenizer.encode(seed_text)
        if isinstance(encoded, list) and len(encoded) > 0:
            idx = encoded[0]
        else:
            # fallback to a random token if encoding fails
            idx = 0
        h = None
        indices = [idx]

        for _ in range(length):
            x = torch.tensor([[idx]], dtype=torch.long, device=device)
            logits, h = model(x, h0=h)
            probs = torch.softmax(logits[-1, 0], dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            indices.append(idx)

        return tokenizer.decode(indices)

# ----------------------------------
# Command-line interface options 
# ----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["char", "word"], default="char")
    parser.add_argument("--model", choices=["rnn", "lstm"], default="rnn")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seq", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)

    # artifact and reproducibility flags
    parser.add_argument("--exp-name", type=str, default="exp_default")
    parser.add_argument("--save-dir", type=str, default="artifacts")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--breakpoints", type=str, default="1,3,5,7,10",
                        help="comma list of epochs to save samples, e.g. 2,4,6,8,10")
    parser.add_argument("--breakpoint-interval", type=int, default=0,
                        help="if >0, run breakpoint sampling every K epochs (overrides list)")
    parser.add_argument("--sample-length", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-text", type=str, default="The ")
    parser.add_argument("--save-checkpoints", action="store_true", help="save per-epoch checkpoints (default: False)")
    parser.add_argument("--save-final-only", action="store_true", help="save only final model state_dict")


    args = parser.parse_args()

    # parse breakpoints list (backwards compatible)
    breakpoints = [int(x) for x in args.breakpoints.split(",") if x.strip()] if args.breakpoints else []
    breakpoint_interval = args.breakpoint_interval if args.breakpoint_interval and args.breakpoint_interval > 0 else 0

    train(
        mode=args.mode,
        model_type=args.model,
        hidden_size=args.hidden,
        seq_length=args.seq,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        save_every=args.save_every,
        breakpoints=breakpoints,
        breakpoint_interval=breakpoint_interval,
        sample_length=args.sample_length,
        max_steps=args.max_steps,
        seed=args.seed,
        seed_text=args.seed_text,
        save_checkpoints=args.save_checkpoints,
        save_final_only=args.save_final_only,
    )