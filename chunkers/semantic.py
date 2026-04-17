"""
chunkers/semantic.py
---------------------
Semantic Chunking — splits a document at topic-shift boundaries detected
by measuring cosine distance between consecutive sentence embeddings.

Algorithm
---------
1. Tokenise the document into sentences using NLTK.
2. Embed every sentence using a lightweight SBERT model
   (all-MiniLM-L6-v2 by default; fast enough for 200-page docs on CPU).
3. Compute cosine distance between each pair of adjacent sentences:
       distance[i] = 1 - cosine_similarity(embed[i], embed[i+1])
4. Find the breakpoint_percentile-th percentile of these distances.
   Any gap whose distance exceeds this threshold is a chunk boundary.
5. Merge small fragments if min_chunk_size is set.
6. Force-split oversized chunks if max_chunk_size is set.

Why this works
--------------
Sentences that discuss the same topic have very similar embedding vectors
(distance ≈ 0). When the topic changes, embeddings diverge sharply
(distance spikes). This makes the algorithm robust to different document
styles — whether the author signals topic changes explicitly or not.

Analogy: Imagine plotting the embedding similarity as a terrain. Topic
shifts are valleys. We drop a boundary at every valley that is deep enough.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from utils import get_logger, tokenise_sentences, clean_text

logger = get_logger(__name__)


class SemanticChunker:
    """
    Embedding-distance-based semantic chunker.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model to use for encoding sentences.
        'all-MiniLM-L6-v2' is fast (22M params) and accurate for English.
    breakpoint_percentile : int
        The Nth percentile of sentence-to-sentence distances used as the
        split threshold. Higher → fewer, larger chunks. Lower → more, finer
        chunks. Range: 50–99. Default: 95.
    min_chunk_size : int
        Minimum number of words a chunk must contain. Smaller chunks are
        merged with the next chunk.
    max_chunk_size : int
        Maximum number of words a chunk may contain. Larger chunks are
        force-split at their midpoint.
    batch_size : int
        Number of sentences to encode at once. Increase for GPU, keep small
        for CPU with limited RAM.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_percentile: int = 95,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.batch_size = batch_size
        self._model = None  # lazy-loaded on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[str]:
        """
        Split `text` into semantically coherent chunks.

        Parameters
        ----------
        text : str
            The full document text (cleaned).

        Returns
        -------
        List[str]
            A list of chunk strings.
        """
        text = clean_text(text)
        sentences = tokenise_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return [sentences[0]]

        logger.info(
            f"[SemanticChunker] Encoding {len(sentences)} sentences "
            f"with model '{self.model_name}'..."
        )

        # Step 1: Embed all sentences
        embeddings = self._encode(sentences)

        # Step 2: Compute pairwise cosine distances between adjacent sentences
        distances = self._compute_distances(embeddings)

        # Step 3: Determine split threshold
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        logger.info(
            f"[SemanticChunker] Distance threshold ({self.breakpoint_percentile}th pct): "
            f"{threshold:.4f}"
        )

        # Step 4: Find breakpoints where distance exceeds threshold
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        # Step 5: Assemble chunks from breakpoints
        chunks = self._assemble_chunks(sentences, breakpoints)

        # Step 6: Post-process — merge small, split large
        chunks = self._merge_small_chunks(chunks)
        chunks = self._split_large_chunks(chunks)

        logger.info(
            f"[SemanticChunker] Produced {len(chunks)} chunks "
            f"(avg {int(sum(len(c.split()) for c in chunks)/max(len(chunks),1))} words each)"
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"[SemanticChunker] Loading model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encode a list of sentences into L2-normalised embedding vectors.

        Returns
        -------
        np.ndarray
            Shape: (n_sentences, embedding_dim). Rows are unit vectors.
        """
        model = self._get_model()
        # show_progress_bar=True gives a tqdm bar for long documents
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=len(sentences) > 200,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
            convert_to_numpy=True,
        )
        return embeddings

    def _compute_distances(self, embeddings: np.ndarray) -> List[float]:
        """
        Compute cosine distance between each consecutive pair of embeddings.

        Since embeddings are L2-normalised, cosine similarity = dot product,
        and cosine distance = 1 − dot product.

        Returns
        -------
        List[float]
            Length: n_sentences - 1. Value range: [0, 2].
        """
        # Matrix multiply: each row i dotted with row i+1
        similarities = np.einsum(
            "ij,ij->i",
            embeddings[:-1],  # sentences 0..n-2
            embeddings[1:],   # sentences 1..n-1
        )
        distances = 1.0 - similarities
        return distances.tolist()

    def _assemble_chunks(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """
        Group sentences into chunks using breakpoint indices.

        `breakpoints[i]` is the index of the last sentence in chunk i.
        (A breakpoint at index j means sentences j and j+1 belong to
        different chunks.)

        Returns
        -------
        List[str]
            Each element is a chunk formed by joining consecutive sentences.
        """
        chunks: List[str] = []
        start = 0
        for bp in breakpoints:
            end = bp + 1  # inclusive
            chunk_text = " ".join(sentences[start:end])
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start = end

        # Don't forget the last segment after the final breakpoint
        if start < len(sentences):
            tail = " ".join(sentences[start:])
            if tail.strip():
                chunks.append(tail.strip())

        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are below min_chunk_size words with the next chunk.

        Rationale: A chunk of 3 words gives an embedding that is too noisy
        for reliable similarity comparison.
        """
        if not chunks:
            return chunks

        merged: List[str] = []
        buffer = chunks[0]

        for next_chunk in chunks[1:]:
            if len(buffer.split()) < self.min_chunk_size:
                # Merge current buffer into next chunk
                buffer = buffer + " " + next_chunk
            else:
                merged.append(buffer)
                buffer = next_chunk

        merged.append(buffer)
        return merged

    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """
        Force-split any chunk that exceeds max_chunk_size words.

        Uses a midpoint split so we don't have to recurse deeply.
        This is a safety valve for edge cases (e.g., a 5-page preamble
        with no topic shift).
        """
        result: List[str] = []
        for chunk in chunks:
            words = chunk.split()
            if len(words) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Recursive split at midpoint until all sub-chunks are small enough
                mid = len(words) // 2
                result.extend(
                    self._split_large_chunks(
                        [" ".join(words[:mid]), " ".join(words[mid:])]
                    )
                )
        return result
