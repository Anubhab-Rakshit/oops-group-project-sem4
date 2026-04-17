"""
loader.py
---------
Document loading utilities.

Supports three input formats:
  - Plain text (.txt)
  - PDF (.pdf) via pdfplumber — handles multi-column, tables, headers
  - JSON (.json) — extracts all string values and concatenates them

Design rationale
----------------
We separate loading from chunking so that each chunker receives a single,
clean `str`. This keeps chunkers pure functions that do not know where the
text came from.
"""

import json
import os
from pathlib import Path


def load_document(path: str) -> str:
    """
    Load a document from disk and return its full text content.

    Parameters
    ----------
    path : str
        Absolute or relative path to the document.

    Returns
    -------
    str
        The full text of the document as a single string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()

    if ext == ".txt":
        return _load_txt(path)
    elif ext == ".pdf":
        return _load_pdf(path)
    elif ext == ".json":
        return _load_json(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .txt, .pdf, .json"
        )


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> str:
    """Read a plain-text file using UTF-8 encoding (fallback to latin-1)."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _load_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber.

    pdfplumber is preferred over PyPDF2 because it correctly handles:
      - Multi-column layouts
      - Tables (extracted as tab-separated text)
      - Header / footer stripping
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF loading. "
            "Install it with: pip install pdfplumber"
        )

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Annotate each page boundary so structure-aware chunker can use it
                pages.append(f"\n--- Page {i + 1} ---\n{text}")
    return "\n".join(pages)


def _load_json(path: Path) -> str:
    """
    Recursively walk a JSON document and concatenate all string leaf values.

    Works for:
      - {"content": "...", "sections": [{"body": "..."}, ...]}
      - Flat {"key": "value", ...}
      - Lists of strings or objects

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    str
        All text values joined by newlines.
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    strings = []
    _extract_strings(data, strings)
    return "\n".join(strings)


def _extract_strings(obj, collector: list):
    """Recursively collect all string leaves from a nested JSON object."""
    if isinstance(obj, str):
        collector.append(obj.strip())
    elif isinstance(obj, dict):
        for value in obj.values():
            _extract_strings(value, collector)
    elif isinstance(obj, list):
        for item in obj:
            _extract_strings(item, collector)
