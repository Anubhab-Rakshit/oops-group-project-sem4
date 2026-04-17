"""
chunkers/structure_aware.py
----------------------------
Structure-Aware Chunking — splits a document at explicit structural
boundaries detected by patterns in the text itself.

What counts as a structural boundary?
--------------------------------------
1. **Markdown / Notebook headings**   → `# Title`, `## Section`, `### Sub`
2. **Numbered section titles**        → `1.`, `1.1`, `Chapter 3`, `Part II`
3. **ALL-CAPS section headers**       → `INTRODUCTION`, `METHODOLOGY`
4. **Page markers** (from PDF loader) → `--- Page N ---`
5. **Double blank lines**             → paragraph breaks (last resort)

Why this matters for 200-page notebooks
-----------------------------------------
A Jupyter notebook or a textbook is already structured by its author.
Each chapter is a self-contained topic. Splitting at chapter boundaries
keeps the full context of a section together — meaning a query about
"regularisation" will retrieve the entire regularisation section, not
half of it.

The regex patterns are ordered by specificity: if multiple patterns match
the same line, the most specific one (heading) wins.
"""

from __future__ import annotations

import re
from typing import List
from utils import get_logger, clean_text, count_words

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compiled pattern library
# ---------------------------------------------------------------------------

# Ordered from most specific to least specific.
# Each pattern matches a line that *starts* a new structural block.
_STRUCTURAL_PATTERNS = [
    # PDF page markers injected by our loader
    re.compile(r"^---\s*Page\s+\d+\s*---$", re.MULTILINE),

    # Markdown headings: #, ##, ###, ####
    re.compile(r"^#{1,4}\s+\S", re.MULTILINE),

    # Numbered sections: "1.", "1.1", "1.1.2", "Chapter 3", "Part II"
    re.compile(
        r"^\s*(?:Chapter|Part|Section|Unit|Module|Appendix)[\s\d]+|"
        r"^\s*\d+(?:\.\d+)*\s+[A-Z]",
        re.MULTILINE | re.IGNORECASE,
    ),

    # ALL-CAPS headers (at least 3 words, not mid-sentence)
    re.compile(
        r"^[A-Z][A-Z\s\-:]{10,}$",
        re.MULTILINE,
    ),
]


class StructureAwareChunker:
    """
    Regex-based structural boundary chunker.

    Parameters
    ----------
    min_chunk_words : int
        Minimum number of words per chunk. Smaller chunks are merged with
        the next chunk. Default: 80.
    max_chunk_words : int
        Maximum number of words per chunk. Oversized chunks are split on the
        nearest double-newline (paragraph boundary). Default: 1500.
    merge_small : bool
        Whether to merge small fragments with subsequent chunks.
        Default: True.
    """

    def __init__(
        self,
        min_chunk_words: int = 80,
        max_chunk_words: int = 1500,
        merge_small: bool = True,
    ):
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.merge_small = merge_small

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[str]:
        """
        Split `text` into structurally coherent chunks.

        Parameters
        ----------
        text : str
            The full document text.

        Returns
        -------
        List[str]
            A list of chunk strings, each representing one structural block.
        """
        text = clean_text(text)

        # Detect all structural boundary positions in the document
        boundary_positions = self._find_boundaries(text)

        if len(boundary_positions) <= 1:
            # No structural boundaries found — fall back to paragraph splits
            logger.info(
                "[StructureAwareChunker] No structural patterns found — "
                "falling back to paragraph-level splits."
            )
            return self._split_by_paragraphs(text)

        # Split text at every boundary
        chunks = self._split_at_boundaries(text, boundary_positions)

        # Post-process
        if self.merge_small:
            chunks = self._merge_small_chunks(chunks)
        chunks = self._split_large_chunks(chunks)

        logger.info(
            f"[StructureAwareChunker] Produced {len(chunks)} chunks "
            f"from {len(boundary_positions)} structural boundaries."
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_boundaries(self, text: str) -> List[int]:
        """
        Scan the text and collect the character positions of every line
        that matches any structural pattern.

        Returns
        -------
        List[int]
            Sorted character positions. Position 0 is always included so
            the first segment is captured.
        """
        positions = {0}  # always start from the beginning

        for pattern in _STRUCTURAL_PATTERNS:
            for match in pattern.finditer(text):
                # Use the start of the matched line as boundary position
                positions.add(match.start())

        return sorted(positions)

    def _split_at_boundaries(
        self, text: str, boundaries: List[int]
    ) -> List[str]:
        """
        Slice the text into segments using boundary positions.

        Returns
        -------
        List[str]
            Non-empty text segments.
        """
        segments: List[str] = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
        return segments

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Fallback: split on double newlines (paragraph breaks).
        Used when no structural indicators are present (e.g., plain prose).
        """
        raw_paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []
        buffer = ""

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            # Accumulate paragraphs until we reach min_chunk_words
            if buffer:
                candidate = buffer + "\n\n" + para
                if count_words(candidate) < self.min_chunk_words:
                    buffer = candidate
                else:
                    chunks.append(buffer.strip())
                    buffer = para
            else:
                buffer = para

        if buffer:
            chunks.append(buffer.strip())

        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks below min_chunk_words with the following chunk.

        Example: a chapter title "1. Introduction" on its own line has only
        2 words. Without merging, it becomes a useless one-line chunk.
        """
        if not chunks:
            return chunks

        merged: List[str] = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if count_words(buffer) < self.min_chunk_words:
                buffer = buffer + "\n\n" + chunk
            else:
                merged.append(buffer)
                buffer = chunk

        merged.append(buffer)
        return merged

    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """
        Break oversized chunks at paragraph boundaries.

        We prefer paragraph boundaries over midpoints because they preserve
        prose structure better. If no paragraph boundary exists within the
        chunk, we fall back to a word-midpoint split.
        """
        result: List[str] = []
        for chunk in chunks:
            if count_words(chunk) <= self.max_chunk_words:
                result.append(chunk)
            else:
                result.extend(self._recursive_paragraph_split(chunk))
        return result

    def _recursive_paragraph_split(self, chunk: str) -> List[str]:
        """
        Recursively split a large chunk at paragraph boundaries until all
        sub-chunks are within max_chunk_words.
        """
        if count_words(chunk) <= self.max_chunk_words:
            return [chunk]

        # Try splitting at a double-newline
        parts = re.split(r"\n\s*\n", chunk, maxsplit=1)
        if len(parts) == 2 and all(p.strip() for p in parts):
            left = self._recursive_paragraph_split(parts[0].strip())
            right = self._recursive_paragraph_split(parts[1].strip())
            return left + right
        else:
            # No paragraph break — fall back to word midpoint
            words = chunk.split()
            mid = len(words) // 2
            left = " ".join(words[:mid])
            right = " ".join(words[mid:])
            return self._recursive_paragraph_split(left) + \
                   self._recursive_paragraph_split(right)
