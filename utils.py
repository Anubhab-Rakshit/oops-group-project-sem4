"""
utils.py
--------
Shared helper utilities used across all modules.

Responsibilities:
  - Text cleaning / normalisation
  - Token counting
  - Sentence tokenisation (with NLTK fallback)
  - Logging setup
"""

import re
import logging
from typing import List


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger with a standard format.

    Usage
    -----
    from utils import get_logger
    logger = get_logger(__name__)
    logger.info("Processing 423 sentences...")
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove control characters from a document.

    Steps
    -----
    1. Replace Windows line endings with Unix line endings.
    2. Collapse runs of more than two newlines into two (preserve paragraphs).
    3. Strip trailing whitespace from each line.
    4. Remove null bytes and other control characters.

    Parameters
    ----------
    text : str
        Raw document text.

    Returns
    -------
    str
        Cleaned text.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null bytes and control characters (but keep \n and \t)
    text = re.sub(r"[^\S\n\t ]+", " ", text)  # collapse unusual whitespace
    text = re.sub(r"\x00", "", text)           # remove null bytes

    # Strip trailing spaces per line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines (more than 2 in a row → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def count_words(text: str) -> int:
    """Count whitespace-delimited word tokens in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Return approximate sentence count (used for progress estimates)."""
    return len(tokenise_sentences(text))


# ---------------------------------------------------------------------------
# Sentence tokenisation
# ---------------------------------------------------------------------------

def tokenise_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences.

    Strategy
    --------
    1. Try NLTK's Punkt sentence tokeniser (high quality, language-aware).
    2. Fall back to a simple regex split on '. ', '! ', '? ' if NLTK is
       unavailable or its data is not downloaded.

    Returns
    -------
    List[str]
        Non-empty sentences as stripped strings.
    """
    try:
        import nltk
        # Download quietly if not already present
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r"(?<=[.!?])\s+", text)

    # Remove empty strings and strip each sentence
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Token window utilities (for overlap chunker)
# ---------------------------------------------------------------------------

def split_words(text: str) -> List[str]:
    """Split text into word tokens (whitespace split — no NLTK dependency)."""
    return text.split()


def join_words(words: List[str]) -> str:
    """Rejoin a list of word tokens into a string."""
    return " ".join(words)
