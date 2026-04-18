# ------------------------------------------------------------------------------
# tokenizer.py
#
# Provides tokenization utilities for character-level and word-level modeling
# Builds index mappings (token to index, index to token)
# Converts text to sequence of indices
# Converts indices to one-hot vectors
# Converts sequences of indices back to text
# -------------------------------------------------------------------------------

import json
import numpy as np
import string
import os

# Set directory
PROCESSED_DIR = "data/processed"

# -------------------------
# Character-level tokenizer
# --------------------------

class CharTokenizer:
    """
    Character-level tokenizer using fixed 256 extended ASCII
    
    Builds char-to_idx & idx_to char mappings
    Converts text to list of integer indices
    Converts index to one-hot vector of length 256
    """

    def __init__(self):
        # Build the 256-character vocabulary
        self.vocab_size = 256

        # Create mappings of index to character & character to index
        self.idx_to_char = {i: chr(i) for i in range(256)}
        self.char_to_idx = {chr(i): i for i in range(256)}

    def encode(self, text):
        """
        Convert a string into a list of integer indices.
        Characters outside ASCII range are replaced with 0.
        """
        return [self.char_to_idx.get(ch, 0) for ch in text]
        
    def decode(self, indices):
        """
        Convert a list of integer indices back into a string.
        """
        return "".join(self.idx_to_char[i] for i in indices)
    
    def one_hot(self, index):
        """
        Convert a single index into a one-hot vector of length 256.
        """
        vec = np.zeros((self.vocab_size, 1))
        vec[index] = 1.0
        return vec
    
# -------------------------
# Word-level tokenizer
# --------------------------

class WordTokenizer:
    """
    Word-level tokenizer using the top-5000 vocabulary produced via preprocess.py
    
    Load vocab_words.json (word to index)
    Build the reverse mapping (index to word)
    Encode text into word indices
    Decode indices back into text
    Produce one-hot vectors of length vocab_size

    Words not in the vocabulary are mapped to index 0 - <UNK>
    """

    def __init__(self, vocab_path=os.path.join(PROCESSED_DIR, "vocab_words.json")):
        # Load the vocabulary created during preprocessing
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.word_to_idx = json.load(f)

        # Add an explicit unknown token if not present
        if "<UNK>" not in self.word_to_idx:
            self.word_to_idx["<UNK>"] = 0

        # Build reverse mapping: index -> word
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Vocabulary size (after adding <UNK>)
        self.vocab_size = len(self.word_to_idx)

    def encode(self, text):
        """
        Convert a string into a list of word indices.
        Unknown words map to index 0.
        """
        words = text.split()
        return [self.word_to_idx.get(w, 0) for w in words]
    
    def decode(self, indices):
        """
        Convert a list of word indices back into a space separated string
        """
        return " ".join(self.idx_to_word.get(i, "<UNK>") for i in indices)
    
    def one_hot(self, index):
        """
        Convert a single index into a one-hot vector of length vocab_size.
        """

        vec = np.zeros((self.vocab_size, 1))
        vec[index] = 1.0
        return vec
    
# -------------------------------------
# Helper functions for loading corpuses
# --------------------------------------

def load_char_corpus():
    """
    Load the character-level corpus (chars.text) as a raw string.
    """
    path = os.path.join(PROCESSED_DIR, "chars.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_word_corpus():
    """
    Load the word-level corpus (words.txt) as a list of words.
    """
    path = os.path.join(PROCESSED_DIR, "words.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split()
        