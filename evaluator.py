"""
evaluator.py
-------------
Orchestrates all three chunking strategies, runs retrieval for each,
and determines the best-performing strategy.

This module is the "brain" of the system. It:
  1. Runs all three chunkers on the same document.
  2. Runs the Retriever on the chunks from each strategy.
  3. Compares strategies by their avg_score.
  4. Returns the exact output format specified in the assignment.

Output format (guaranteed)
--------------------------
{
  "semantic": {
    "top_chunks": List[str],
    "similarity_scores": List[float],
    "avg_score": float,
    "num_chunks": int,           # extra diagnostic
  },
  "structure_aware": { ... },
  "overlap": { ... },
}

Plus three top-level fields:
  best_strategy : str     -- name of the winning strategy
  best_chunks   : List[str] -- top_chunks from the best strategy
  final_score   : float   -- avg_score of the best strategy

Design decisions
----------------
- Model sharing: the Retriever is initialised once and shared across all
  three strategy evaluations. This means the SBERT model is loaded exactly
  once, saving ~200MB of RAM and several seconds of startup time.
- SemanticChunker also shares the same model by passing `model` to the
  Retriever constructor.
- chunkers run sequentially (not in parallel) to avoid saturating RAM on
  large documents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

from chunkers import SemanticChunker, StructureAwareChunker, OverlapChunker
from retriever import Retriever
from utils import get_logger, count_words

logger = get_logger(__name__)


# Type alias for the per-strategy result dict
StrategyResult = Dict[str, Any]

# Type alias for the full results dict
ResultsDict = Dict[str, StrategyResult]


class ChunkingEvaluator:
    """
    Runs all three chunking strategies and selects the best one.

    Parameters
    ----------
    model_name : str
        SBERT model for both SemanticChunker and Retriever.
    top_k : int
        Number of top chunks to retrieve per strategy.
    semantic_percentile : int
        Passed to SemanticChunker.breakpoint_percentile.
    overlap_chunk_size : int
        Passed to OverlapChunker.chunk_size.
    overlap_amount : int
        Passed to OverlapChunker.overlap.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        semantic_percentile: int = 95,
        overlap_chunk_size: int = 400,
        overlap_amount: int = 80,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.semantic_percentile = semantic_percentile
        self.overlap_chunk_size = overlap_chunk_size
        self.overlap_amount = overlap_amount

        # Initialise chunkers (model not loaded yet — lazy)
        self._semantic_chunker = SemanticChunker(
            model_name=model_name,
            breakpoint_percentile=semantic_percentile,
        )
        self._structure_chunker = StructureAwareChunker()
        self._overlap_chunker = OverlapChunker(
            chunk_size=overlap_chunk_size,
            overlap=overlap_amount,
        )

        # Retriever is shared — model loaded once
        self._retriever: Optional[Retriever] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, document: str, query: str) -> Tuple[ResultsDict, str, List[str], float]:
        """
        Run all three strategies and return comparison results.

        Parameters
        ----------
        document : str
            The full document text.
        query : str
            The user's search query.

        Returns
        -------
        results : ResultsDict
            Mapping of strategy name → StrategyResult dict.
            Keys: "semantic", "structure_aware", "overlap".
        best_strategy : str
            Name of the strategy with the highest avg_score.
        best_chunks : List[str]
            top_chunks from the best strategy.
        final_score : float
            avg_score of the best strategy.
        """
        word_count = count_words(document)
        logger.info(
            f"[Evaluator] Starting evaluation. Document: {word_count:,} words. "
            f'Query: "{query[:80]}{"..." if len(query) > 80 else ""}"'
        )

        results: ResultsDict = {}

        # -----------------------------------------------------------
        # Strategy 1: Semantic Chunking
        # -----------------------------------------------------------
        results["semantic"] = self._run_strategy(
            name="semantic",
            chunker=self._semantic_chunker,
            document=document,
            query=query,
        )

        # After semantic chunker has loaded the model, re-use it in Retriever
        # to avoid loading the model a second time.
        if self._semantic_chunker._model is not None and self._retriever is None:
            self._retriever = Retriever(
                model_name=self.model_name,
                top_k=self.top_k,
                model=self._semantic_chunker._model,
            )

        # -----------------------------------------------------------
        # Strategy 2: Structure-Aware Chunking
        # -----------------------------------------------------------
        results["structure_aware"] = self._run_strategy(
            name="structure_aware",
            chunker=self._structure_chunker,
            document=document,
            query=query,
        )

        # -----------------------------------------------------------
        # Strategy 3: Overlap-Based Chunking
        # -----------------------------------------------------------
        results["overlap"] = self._run_strategy(
            name="overlap",
            chunker=self._overlap_chunker,
            document=document,
            query=query,
        )

        # -----------------------------------------------------------
        # Select best strategy
        # -----------------------------------------------------------
        best_strategy, best_chunks, final_score = self._select_best(results)

        logger.info(
            f"[Evaluator] ✓ Best strategy: '{best_strategy}' "
            f"(avg_score={final_score:.4f})"
        )

        return results, best_strategy, best_chunks, final_score

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_retriever(self) -> Retriever:
        """Return a Retriever, creating it if it doesn't exist yet."""
        if self._retriever is None:
            self._retriever = Retriever(
                model_name=self.model_name,
                top_k=self.top_k,
            )
        return self._retriever

    def _run_strategy(
        self,
        name: str,
        chunker: Any,
        document: str,
        query: str,
    ) -> StrategyResult:
        """
        Run a single chunking strategy from text → chunks → retrieval result.

        Returns
        -------
        StrategyResult
            Dict with keys: top_chunks, similarity_scores, avg_score,
            num_chunks, chunk_time_s, retrieval_time_s.
        """
        logger.info(f"\n[Evaluator] ──── Strategy: {name!r} ────")

        # --- Chunk ---
        t0 = time.perf_counter()
        chunks = chunker.chunk(document)
        chunk_time = time.perf_counter() - t0

        if not chunks:
            logger.warning(f"[Evaluator] Strategy '{name}' produced zero chunks!")
            return {
                "top_chunks": [],
                "similarity_scores": [],
                "avg_score": 0.0,
                "num_chunks": 0,
                "chunk_time_s": chunk_time,
                "retrieval_time_s": 0.0,
            }

        logger.info(
            f"[Evaluator] {name}: {len(chunks)} chunks produced in {chunk_time:.2f}s"
        )

        # --- Retrieve ---
        retriever = self._get_retriever()
        t1 = time.perf_counter()
        top_chunks, scores, avg_score = retriever.retrieve(query, chunks)
        retrieval_time = time.perf_counter() - t1

        return {
            "top_chunks": top_chunks,
            "similarity_scores": [round(s, 6) for s in scores],
            "avg_score": round(avg_score, 6),
            "num_chunks": len(chunks),
            "chunk_time_s": round(chunk_time, 3),
            "retrieval_time_s": round(retrieval_time, 3),
        }

    def _select_best(
        self, results: ResultsDict
    ) -> Tuple[str, List[str], float]:
        """
        Identify the strategy with the highest avg_score.

        Tie-breaking rule: If two strategies tie on avg_score, the one with
        the higher top-1 score wins (more confident first answer is better).

        Returns
        -------
        best_strategy : str
        best_chunks : List[str]
        final_score : float
        """
        def sort_key(item):
            name, res = item
            top_score = res["similarity_scores"][0] if res["similarity_scores"] else 0.0
            return (res["avg_score"], top_score)

        best_name, best_result = max(results.items(), key=sort_key)

        return (
            best_name,
            best_result["top_chunks"],
            best_result["avg_score"],
        )
