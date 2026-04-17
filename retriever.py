"""
retriever.py
------------
Semantic retrieval engine.

Responsibilities:
  1. Embed a query string into a vector.
  2. Embed a list of chunk strings into a matrix.
  3. Compute cosine similarity between the query vector and every chunk vector.
  4. Return the top-k chunks along with their scores.

Why cosine similarity over SBERT embeddings?
--------------------------------------------
Two alternatives were considered:

  BM25 (Okapi BM25)
    Pros: Fast, no GPU needed, great for keyword-heavy queries
    Cons: Fails on paraphrase — "What is gradient descent?" will not match
          chunks discussing "parameter optimisation via derivatives" even
          though they mean the same thing.

  ROUGE / BLEU
    These are generation metrics (measuring overlap between a reference and
    a hypothesis). They are designed for summarisation evaluation, not
    retrieval.

  Cosine similarity over SBERT embeddings ← CHOSEN
    Pros: Captures semantic meaning, handles synonyms and paraphrase,
          state-of-the-art for passage retrieval, works with the same model
          already loaded by SemanticChunker (shared weight loading).
    Cons: Slower than BM25 for very large corpora (tens of millions of
          chunks) — but at document scale (hundreds to low thousands of
          chunks) this is negligible.

Model reuse
-----------
The Retriever accepts an optional pre-loaded SentenceTransformer model so
that SemanticChunker and Retriever share the same loaded model, saving
memory and load time.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Any
from utils import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Encode chunks and queries, then rank by cosine similarity.

    Parameters
    ----------
    model_name : str
        SBERT model identifier. Should match what SemanticChunker uses so
        the same model weights are reused.
    top_k : int
        Number of top-scoring chunks to return per query.
    batch_size : int
        Encoding batch size. Larger batches are faster on GPU.
    model : Any, optional
        A pre-loaded SentenceTransformer model. If provided, model_name is
        ignored and no new model is loaded.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        batch_size: int = 64,
        model: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = model  # may be injected from SemanticChunker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, chunks: List[str]
    ) -> Tuple[List[str], List[float], float]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Parameters
        ----------
        query : str
            The search query.
        chunks : List[str]
            All chunks produced by a chunking strategy.

        Returns
        -------
        top_chunks : List[str]
            The top-k chunk strings, ordered by descending similarity.
        scores : List[float]
            Cosine similarity scores for each top_chunk (same order).
        avg_score : float
            Mean similarity score across the top-k results.
            This is the primary evaluation metric for strategy comparison.
        """
        if not chunks:
            return [], [], 0.0

        effective_k = min(self.top_k, len(chunks))

        # Encode query (shape: [embedding_dim])
        query_vec = self._encode_query(query)

        # Encode all chunks (shape: [n_chunks, embedding_dim])
        chunk_vecs = self._encode_chunks(chunks)

        # Compute cosine similarity for all chunks in one vectorised operation
        # Since both query_vec and chunk_vecs are L2-normalised, cos_sim = dot product
        similarities = chunk_vecs @ query_vec  # shape: [n_chunks]

        # Get top-k indices (argsort descending)
        top_indices = np.argsort(similarities)[::-1][:effective_k]

        top_chunks = [chunks[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]
        avg_score = float(np.mean(top_scores)) if top_scores else 0.0

        logger.info(
            f"[Retriever] Retrieved top-{effective_k} chunks. "
            f"avg_score={avg_score:.4f}, best={top_scores[0]:.4f}"
        )

        return top_chunks, top_scores, avg_score

    def retrieve_with_scores(
        self, query: str, chunks: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Return all chunks with their similarity scores (not just top-k).

        Useful for analysis, visualisation, and debugging which chunks
        score poorly.

        Returns
        -------
        List[Tuple[str, float]]
            All (chunk, score) pairs, sorted by descending score.
        """
        if not chunks:
            return []

        query_vec = self._encode_query(query)
        chunk_vecs = self._encode_chunks(chunks)
        similarities = chunk_vecs @ query_vec

        ranked = sorted(
            zip(chunks, similarities.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(c, float(s)) for c, s in ranked]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the SentenceTransformer model if not already available."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"[Retriever] Loading SBERT model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a normalised embedding vector.

        Returns
        -------
        np.ndarray
            Shape: (embedding_dim,). Unit vector.
        """
        model = self._get_model()
        vec = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec[0]  # shape: (embedding_dim,)

    def _encode_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Encode a list of chunk strings into a normalised embedding matrix.

        Returns
        -------
        np.ndarray
            Shape: (n_chunks, embedding_dim). Each row is a unit vector.
        """
        model = self._get_model()
        vecs = model.encode(
            chunks,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(chunks) > 100,
        )
        return vecs  # shape: (n_chunks, embedding_dim)
