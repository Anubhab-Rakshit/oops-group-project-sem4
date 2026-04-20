"""
retriever.py
------------
Hybrid Retrieval Engine.

Now implements Hybrid Search:
  1. Semantic Search (SBERT): Captures meaning and context.
  2. Keyword Search (BM25): Captures exact technical terms and names.
  3. Combination (Reciprocal Rank Fusion): Merges results for optimal accuracy.

This dual-engine approach is significantly more robust than semantic search alone,
especially for technical notebooks where specific variable names or function calls
might be the primary search targets.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from utils import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Hybrid retriever combining Semantic (SBERT) and Lexical (BM25) search.

    Parameters
    ----------
    model_name : str
        SBERT model identifier.
    top_k : int
        Number of top-scoring chunks to return.
    batch_size : int
        Encoding batch size.
    model : Any, optional
        Pre-loaded SentenceTransformer model.
    alpha : float
        Weight for semantic score in hybrid retrieval (0.0 to 1.0).
        1.0 = Pure Semantic, 0.0 = Pure BM25. Default 0.7.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        batch_size: int = 64,
        model: Optional[Any] = None,
        alpha: float = 0.7,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self._model = model
        self.alpha = alpha
        
        # BM25 state
        self._bm25 = None
        self._last_chunks_for_bm25 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, chunks: List[str], precomputed_embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], List[float], float]:
        """
        Retrieve chunks using Hybrid Search.
        """
        if not chunks:
            return [], [], 0.0

        effective_k = min(self.top_k, len(chunks))

        # 1. Get Semantic Scores
        semantic_scores = self._get_semantic_scores(query, chunks, precomputed_embeddings)

        # 2. Get BM25 Scores
        bm25_scores = self._get_bm25_scores(query, chunks)

        # 3. Hybrid Merge (Weighted Average)
        # Normalise BM25 scores to [0, 1] range
        if np.max(bm25_scores) > np.min(bm25_scores):
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_norm = bm25_scores

        hybrid_scores = (self.alpha * semantic_scores) + ((1 - self.alpha) * bm25_norm)

        # 4. Rank and Return
        top_indices = np.argsort(hybrid_scores)[::-1][:effective_k]
        
        top_chunks = [chunks[i] for i in top_indices]
        top_scores = [float(hybrid_scores[i]) for i in top_indices]
        avg_score = float(np.mean(top_scores)) if top_scores else 0.0

        return top_chunks, top_scores, avg_score

    # ------------------------------------------------------------------
    # Internal Search Engines
    # ------------------------------------------------------------------

    def _get_semantic_scores(self, query: str, chunks: List[str], precomputed: Optional[np.ndarray]) -> np.ndarray:
        query_vec = self._encode_query(query)
        
        if precomputed is not None:
            chunk_vecs = precomputed
        else:
            chunk_vecs = self.encode_chunks(chunks)
            
        return chunk_vecs @ query_vec

    def _get_bm25_scores(self, query: str, chunks: List[str]) -> np.ndarray:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Falling back to semantic only.")
            return np.zeros(len(chunks))

        # Re-initialise BM25 if chunks changed
        if self._bm25 is None or chunks != self._last_chunks_for_bm25:
            tokenised_corpus = [doc.lower().split() for doc in chunks]
            self._bm25 = BM25Okapi(tokenised_corpus)
            self._last_chunks_for_bm25 = chunks

        tokenised_query = query.lower().split()
        return np.array(self._bm25.get_scores(tokenised_query))

    # ------------------------------------------------------------------
    # Embedding Helpers (Public so Evaluator can use them for caching)
    # ------------------------------------------------------------------

    def encode_chunks(self, chunks: List[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(
            chunks,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(chunks) > 100,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vec[0]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
