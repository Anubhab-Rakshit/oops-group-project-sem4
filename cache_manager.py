"""
cache_manager.py
----------------
High-performance caching system for document embeddings.

Why this is needed
------------------
For a 200+ page document, calculating embeddings for three different
chunking strategies can take several minutes. If the user asks multiple
questions about the same document, we should not repeat this work.

How it works
------------
1. Generate a SHA-256 hash of the document text.
2. Create a unique cache key based on (doc_hash, strategy_name, model_name).
3. Save/Load the numpy embedding matrices and the chunk list.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Any
import numpy as np
from utils import get_logger

logger = get_logger(__name__)

class CacheManager:
    """
    Handles persistence of embeddings and chunks to avoid redundant computation.
    """
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_base = Path(cache_dir)
        self.cache_base.mkdir(parents=True, exist_ok=True)

    def get_doc_hash(self, text: str) -> str:
        """Create a stable hash for the document content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_paths(self, doc_hash: str, strategy: str, model_name: str) -> Tuple[Path, Path]:
        """Generate unique filenames for a specific document/strategy/model combo."""
        # Sanitise model name for filename
        safe_model = model_name.replace("/", "_").replace("\\", "_")
        prefix = f"{doc_hash[:16]}_{strategy}_{safe_model}"
        
        embed_path = self.cache_base / f"{prefix}_vecs.npy"
        chunk_path = self.cache_base / f"{prefix}_chunks.json"
        return embed_path, chunk_path

    def load(self, doc_hash: str, strategy: str, model_name: str) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Attempt to load cached chunks and embeddings.
        Returns (chunks, embeddings) or None if not found.
        """
        embed_path, chunk_path = self._get_paths(doc_hash, strategy, model_name)

        if embed_path.exists() and chunk_path.exists():
            try:
                with chunk_path.open("r", encoding="utf-8") as f:
                    chunks = json.load(f)
                embeddings = np.load(str(embed_path))
                logger.info(f"[Cache] Successfully loaded {len(chunks)} cached chunks for '{strategy}'")
                return chunks, embeddings
            except Exception as e:
                logger.warning(f"[Cache] Failed to load cache for '{strategy}': {e}")
                return None
        return None

    def save(self, doc_hash: str, strategy: str, model_name: str, chunks: List[str], embeddings: np.ndarray) -> None:
        """Save chunks and embeddings to the cache."""
        embed_path, chunk_path = self._get_paths(doc_hash, strategy, model_name)
        
        try:
            with chunk_path.open("w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False)
            np.save(str(embed_path), embeddings)
            logger.info(f"[Cache] Saved '{strategy}' embeddings to {embed_path.name}")
        except Exception as e:
            logger.warning(f"[Cache] Failed to save cache for '{strategy}': {e}")

    def clear(self):
        """Delete all cached files."""
        for file in self.cache_base.glob("*"):
            file.unlink()
        logger.info("[Cache] All cached embeddings cleared.")
