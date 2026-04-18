# -------------------------------------------------------------------------------------------
# dataloader.py
# 
# Creates training sequences for RNN & LSTM models.
#
# Loads corpus (chars or words)
# Converts corpus to indices using tokenizer
# Slices into fixed-length sequences
# Produces input_seq & target_seq pairs
# Optionally batches
# -------------------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.tokenizer import CharTokenizer, WordTokenizer, load_char_corpus, load_word_corpus

# -----------------------------------------------
# Sequence Dataset
# 
# PyTorch dataset that returns:
#   x: input sequence of length seq_len
#   y: target sequence (shifted by 1)
# 
# added to replace Python-loop get_sequence() logic (far too slow)
# -----------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        # Number of valid input/target pairs
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

# -------------------------------------------------------
# Sequence DataLoader
#
# Updated to be handled by PyTorch for speed improvement
# -------------------------------------------------------

class SequenceDataLoader:
    """
    Sequence loader for char-level and word-level corpuses.

    Parameters: 
        mode: "char" or "word"
        seq_length: length of each training sequence
    """

    def __init__(self, mode="char", seq_length=50):
        assert mode in ("char", "word"), "mode must be either 'char' or 'word'"
        self.mode = mode
        self.seq_length = seq_length

        # Load corpus and tokenizer depending on mode
        if mode == "char":
            self.tokenizer = CharTokenizer()
            raw = load_char_corpus()
            encoded = self.tokenizer.encode(raw)

        else:
            self.tokenizer = WordTokenizer()
            raw = load_word_corpus()
            encoded = self.tokenizer.encode(" ".join(raw))

        # Store as a single long tensor
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.vocab_size = self.tokenizer.vocab_size

    def get_loader(self, batch_size):
        """
        Returns a PyTorch DataLoader that creates batches of:
            inputs: (batch_size, seq_len)
            targets: (batch_size, seq_len)

        Replaces the original batch generation functions to improve speeds.
        """

        dataset = SequenceDataset(self.data, self.seq_length)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,      # Parallel workers to improve speed
            pin_memory=True,    # Pinning memory for faster GPU xfer
        )
