# -----------------------------------------------
# preprocess.py
#
# Script performs the following actions
# - Loads all .txt data files
# - Strips out Gutenberg standard headers/footers
# - Normalizes white-space
# - Creates:
# -- A character-level corpus - chars.txt
# -- A word-level corpus - words.txt
# -- A word frequency list
# -- A top-5000 vocabulary JSON file
# 
# ------------------------------------------------

import os
import re
import json
from collections import Counter

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def strip_header_footer(text):
    """
    Removes the standard Project Gutenberg headers and footers
    """

    # Look for start marker & end marker
    start = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG", text)
    end = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG", text)

    # If markers exist, slice text between them
    if start and end:
        return text[start.end():end.start()]
    
    # if markers are missing, fall back to returning normal text
    return text 

def clean_text(text):
    """
    Normalizes whitespace and line breaks.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Replace any whitespace sequence with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    return text.strip()

def load_all_books():
    """
    Load every .txt file in data/raw, strip the header & footer, clean whitespace,
    return a list of cleaned book texts.
    """
    texts = []

    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()

                stripped = strip_header_footer(raw)
                cleaned = clean_text(stripped)

                texts.append(cleaned)
    
    return texts

def build_char_corpus(texts):
    """
    Concatenate all books into a single long string and save it. Used for 
    character-level RNN training
    """

    corpus = " ".join(texts)

    with open(os.path.join(PROCESSED_DIR, "chars.txt"), "w", encoding="utf-8") as f:
        f.write(corpus)

def build_word_corpus_and_vocab(texts, vocab_size=5000):
    """
    Build the word-level corpus and vocabulary
    
    Concatenate all books into one long string, split into words using whitespace, 
    count word frequencies, keep the top vocab_size (most common words),
    save words.txt (full corpus) and vocab_words.json (word to index mapping)
    """
        
    corpus = " ".join(texts)
    words = corpus.split()

    # Count word frequencies
    counter = Counter(words)

    # Select the top N words
    most_common = counter.most_common(vocab_size)

    # Build word to index mapping
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

    # Save vocabulary
    with open(os.path.join(PROCESSED_DIR, "vocab_words.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    # Save the full word corpus
    with open(os.path.join(PROCESSED_DIR, "words.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(words))

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Beginning preprocessing (preprocess.py)...")

    print("Loading books...")
    texts = load_all_books()

    print("Building character-level corpus...")
    build_char_corpus(texts)

    print("Building word-level corpus and vocabulary...")
    build_word_corpus_and_vocab(texts)

    print("Preprocessing completed (end preprocess.py)!")
    
if __name__ == "__main__":
    main()