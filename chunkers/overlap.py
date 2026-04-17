"""
chunkers/overlap.py
--------------------
Overlap-Based (Sliding Window) Chunking.

How it works
------------
The document is tokenised into words. A sliding window of `chunk_size`
words moves through the document, advancing `stride = chunk_size - overlap`
words at each step. Every window position becomes one chunk.

Visual example (chunk_size=10, overlap=3):
  Window 1:  words[0:10]
  Window 2:  words[7:17]   ← 3 words shared with window 1
  Window 3:  words[14:24]  ← 3 words shared with window 2
  ...

Why overlap matters
-------------------
Without overlap, information that spans a chunk boundary (e.g., a
definition that starts at the end of chunk 5 and continues into chunk 6)
can only be retrieved by the chunk that contains most of it — and that
might not score high enough to appear in the top-k results.

With overlap, every span of text is fully contained within at least one
chunk (as long as the span is shorter than chunk_size). This makes
overlap chunking a robust fallback: it never loses information at
boundaries.

Trade-off: overlap increases the total number of chunks, which increases
embedding time. We default to 20% overlap (overlap = 0.2 × chunk_size),
which is the industry standard for RAG systems.
"""

from __future__ import annotations

from typing import List
from utils import get_logger, clean_text, split_words, join_words

logger = get_logger(__name__)


class OverlapChunker:
    """
    Sliding-window word-overlap chunker.

    Parameters
    ----------
    chunk_size : int
        Number of words per chunk. Default: 400.
        (A 200-page textbook ≈ 100,000 words → ~250 chunks at size 400.)
    overlap : int
        Number of words shared between consecutive chunks. Default: 80
        (= 20% of chunk_size).
    min_chunk_words : int
        Minimum words for the final (possibly shorter) tail chunk.
        If the tail is too short, it is merged into the previous chunk.
        Default: 30.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 80,
        min_chunk_words: int = 30,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_words = min_chunk_words
        self.stride = chunk_size - overlap  # how far the window moves each step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[str]:
        """
        Slide a fixed-size window over the document's words.

        Parameters
        ----------
        text : str
            The full document text.

        Returns
        -------
        List[str]
            A list of overlapping chunk strings.
        """
        text = clean_text(text)
        words = split_words(text)
        n = len(words)

        if n == 0:
            return []

        if n <= self.chunk_size:
            # Document fits in a single chunk
            return [text]

        chunks: List[str] = []
        start = 0

        while start < n:
            end = min(start + self.chunk_size, n)
            chunk_words = words[start:end]
            chunks.append(join_words(chunk_words))

            if end == n:
                break  # We've reached the end of the document

            start += self.stride  # slide the window forward

        # Handle very short tail chunk
        if len(chunks) >= 2:
            last_words = split_words(chunks[-1])
            if len(last_words) < self.min_chunk_words:
                # Absorb the tail into the previous chunk
                second_last = split_words(chunks[-2])
                chunks[-2] = join_words(second_last + last_words)
                chunks.pop()

        logger.info(
            f"[OverlapChunker] Produced {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.overlap}, "
            f"stride={self.stride}) from {n} words."
        )
        return chunks

    # ------------------------------------------------------------------
    # Informational helpers
    # ------------------------------------------------------------------

    def estimated_chunk_count(self, word_count: int) -> int:
        """
        Estimate how many chunks a document of `word_count` words will produce.

        Useful for pre-allocating arrays or estimating embedding time.
        """
        if word_count <= self.chunk_size:
            return 1
        return max(1, (word_count - self.overlap) // self.stride)

    def coverage_ratio(self) -> float:
        """
        How much of each chunk overlaps with its neighbours (as a fraction).

        Example: chunk_size=400, overlap=80 → coverage_ratio = 0.20 (20%)
        """
        return self.overlap / self.chunk_size
