"""
evaluator.py
-------------
Orchestrates chunking strategies with integrated Embedding Caching.

Enhancements:
  - Integrated CacheManager for instant repeated queries.
  - Efficient model weight sharing across all components.
  - Performance profiling for each strategy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

from chunkers import SemanticChunker, StructureAwareChunker, OverlapChunker
from retriever import Retriever
from cache_manager import CacheManager
from utils import get_logger, count_words

logger = get_logger(__name__)

class ChunkingEvaluator:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        semantic_percentile: int = 95,
        overlap_chunk_size: int = 400,
        overlap_amount: int = 80,
        use_cache: bool = True
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.use_cache = use_cache
        
        # Init components
        self._semantic_chunker = SemanticChunker(model_name=model_name, breakpoint_percentile=semantic_percentile)
        self._structure_chunker = StructureAwareChunker()
        self._overlap_chunker = OverlapChunker(chunk_size=overlap_chunk_size, overlap=overlap_amount)
        self._cache = CacheManager()
        self._retriever: Optional[Retriever] = None

    def evaluate(self, document: str, query: str) -> Tuple[Dict[str, Any], str, List[str], float]:
        doc_hash = self._cache.get_doc_hash(document)
        results = {}

        # Run strategies
        for name, chunker in [
            ("semantic", self._semantic_chunker),
            ("structure_aware", self._structure_chunker),
            ("overlap", self._overlap_chunker)
        ]:
            results[name] = self._run_strategy_with_cache(name, chunker, document, query, doc_hash)
            
            # Shared model logic
            if name == "semantic" and self._semantic_chunker._model is not None and self._retriever is None:
                self._retriever = Retriever(model_name=self.model_name, top_k=self.top_k, model=self._semantic_chunker._model)

        best_strategy, best_chunks, final_score = self._select_best(results)
        return results, best_strategy, best_chunks, final_score

    def _run_strategy_with_cache(self, name: str, chunker: Any, document: str, query: str, doc_hash: str) -> Dict[str, Any]:
        logger.info(f"\n[Evaluator] ──── Strategy: {name!r} ────")
        
        cached_data = self._cache.load(doc_hash, name, self.model_name) if self.use_cache else None
        
        if cached_data:
            chunks, embeddings = cached_data
            chunk_time = 0.0
        else:
            t0 = time.perf_counter()
            chunks = chunker.chunk(document)
            chunk_time = time.perf_counter() - t0
            
            if chunks:
                embeddings = self._get_retriever().encode_chunks(chunks)
                if self.use_cache:
                    self._cache.save(doc_hash, name, self.model_name, chunks, embeddings)
            else:
                embeddings = None

        if not chunks:
            return {"top_chunks": [], "similarity_scores": [], "avg_score": 0.0, "num_chunks": 0}

        t1 = time.perf_counter()
        top_chunks, scores, avg_score = self._get_retriever().retrieve(query, chunks, precomputed_embeddings=embeddings)
        retrieval_time = time.perf_counter() - t1

        return {
            "top_chunks": top_chunks,
            "similarity_scores": [round(s, 6) for s in scores],
            "avg_score": round(avg_score, 6),
            "num_chunks": len(chunks),
            "chunk_time_s": round(chunk_time, 3),
            "retrieval_time_s": round(retrieval_time, 3),
            "cached": cached_data is not None
        }

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(model_name=self.model_name, top_k=self.top_k)
        return self._retriever

    def _select_best(self, results: Dict[str, Any]) -> Tuple[str, List[str], float]:
        def sort_key(item):
            name, res = item
            top_score = res["similarity_scores"][0] if res["similarity_scores"] else 0.0
            return (res["avg_score"], top_score)
        best_name, best_result = max(results.items(), key=sort_key)
        return best_name, best_result["top_chunks"], best_result["avg_score"]
