"""
chunkers/__init__.py
--------------------
Exposes a unified interface for all chunking strategies.

Each chunker follows the same contract:
  chunk(text: str, **kwargs) -> List[str]

This makes the evaluator strategy-agnostic.
"""

from .semantic import SemanticChunker
from .structure_aware import StructureAwareChunker
from .overlap import OverlapChunker

__all__ = [
    "SemanticChunker",
    "StructureAwareChunker",
    "OverlapChunker",
]
