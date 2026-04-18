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

import numpy as np
from src.utils.tokenizer import CharTokenizer, WordTokenizer, load_char_corpus, load_word_corpus

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
            self.data = self.tokenizer.encode(raw)

        # Word-level
        else:
            self.tokenizer = WordTokenizer()
            raw = load_word_corpus()
            # since raw is already a list of words, we now encode each one
            self.data = self.tokenizer.encode(" ".join(raw))

        self.data = np.array(self.data, dtype=np.int32)
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        """
        Provides number of training samples available.
        Each sample is a pair (input_seq, target_seq)
        """
        return len(self.data) - self.seq_length
    
    def get_sequence(self, idx):
        """
        Return a single (input_seq, target_seq) pair.

        input_seq: [x0, x1, x2, ..., x_{L-1}]
        target_seq: [x1, x2, x3, ..., x_L]
        """

        assert 0 <= idx < len(self), "Index is out of range"

        seq = self.data[idx : idx + self.seq_length + 1]

        input_seq = seq[:-1]
        target_seq = seq[1:]

        return input_seq, target_seq
    
    def batch(self, batch_size):
        """
        Provides batches of (inputs, targets) as numpy arrays
        
        inputs: (batch_size, seq_size)
        targets: (batch_size, seq_length)
        """

        for start in range(0, len(self), batch_size):
            end = start + batch_size
            if end >= len(self):
                break

            batch_inputs = []
            batch_targets = []

            for i in range(start, end):
                inp, tgt = self.get_sequence(i)
                batch_inputs.append(inp)
                batch_targets.append(tgt)

                yield (
                    np.array(batch_inputs, dtype=np.int32),
                    np.array(batch_targets, dtype=np.int32),
                )