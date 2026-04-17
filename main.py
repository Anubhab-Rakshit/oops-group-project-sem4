"""
main.py
--------
Entry point for the Intelligent Document Chunking & Retrieval System.

Usage (command line)
--------------------
  python main.py --doc path/to/document.pdf --query "What is backpropagation?"

  # With custom chunking settings:
  python main.py --doc notes.txt --query "Explain attention mechanism" \\
      --top-k 5 --overlap-size 400 --overlap-amount 80 --percentile 95

  # Save results to JSON:
  python main.py --doc textbook.pdf --query "Define gradient descent" \\
      --output results.json

Usage (programmatic)
--------------------
  from main import run_pipeline

  results, best_strategy, best_chunks, final_score = run_pipeline(
      doc_path="notes.txt",
      query="What is regularisation?",
  )

The `run_pipeline` function is the primary API for this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Automatic HuggingFace cache redirect
# ---------------------------------------------------------------------------
# On macOS systems where ~/.cache is not writable (e.g., system Python),
# the HF model download will fail with a PermissionError.
# We detect this early and redirect the cache into the project directory.

def _ensure_hf_cache() -> None:
    """
    If the default HuggingFace cache directory is not writable, redirect
    it to a local `.hf_cache` folder inside the project directory.

    This function is called before any imports of sentence_transformers
    or huggingface_hub so that the env var is set in time.
    """
    if "HF_HOME" in os.environ:
        return  # user has already set a custom cache — respect it

    default_cache = Path.home() / ".cache" / "huggingface"
    try:
        default_cache.mkdir(parents=True, exist_ok=True)
        # Quick write test
        test_file = default_cache / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError):
        # Default cache is not writable — use local fallback
        local_cache = Path(__file__).parent / ".hf_cache"
        local_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(local_cache)

_ensure_hf_cache()  # Must run before any HF imports

# Rich is used only for terminal display — it is never required for the core logic
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from loader import load_document
from evaluator import ChunkingEvaluator
from utils import get_logger, clean_text, count_words

logger = get_logger(__name__)
console = Console() if _RICH_AVAILABLE else None


# ---------------------------------------------------------------------------
# Core pipeline function (programmatic API)
# ---------------------------------------------------------------------------

def run_pipeline(
    doc_path: str,
    query: str,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    overlap_chunk_size: int = 400,
    overlap_amount: int = 80,
    semantic_percentile: int = 95,
) -> Tuple[Dict[str, Any], str, List[str], float]:
    """
    Full end-to-end pipeline: load → chunk → retrieve → evaluate.

    Parameters
    ----------
    doc_path : str
        Path to the document file (.txt, .pdf, .json).
    query : str
        The information retrieval query.
    top_k : int
        Number of top chunks to return per strategy.
    model_name : str
        SentenceTransformer model identifier.
    overlap_chunk_size : int
        Number of words per chunk for the overlap strategy.
    overlap_amount : int
        Number of overlapping words between consecutive chunks.
    semantic_percentile : int
        Percentile threshold for semantic chunker's split detection.

    Returns
    -------
    results : Dict[str, Any]
        Per-strategy results with keys "semantic", "structure_aware", "overlap".
        Each value has: top_chunks, similarity_scores, avg_score, num_chunks.
    best_strategy : str
        Name of the best-performing strategy.
    best_chunks : List[str]
        Top chunks from the winning strategy.
    final_score : float
        Average similarity score of the best strategy.

    Example
    -------
    >>> results, best, chunks, score = run_pipeline("doc.txt", "neural networks")
    >>> print(f"Best strategy: {best}, Score: {score:.4f}")
    >>> for i, chunk in enumerate(chunks, 1):
    ...     print(f"Chunk {i}: {chunk[:120]}...")
    """
    # 1. Load document
    _print_header(f"Loading document: {Path(doc_path).name}")
    document = load_document(doc_path)
    document = clean_text(document)
    word_count = count_words(document)
    _info(f"Document loaded — {word_count:,} words ({len(document):,} characters)")

    if word_count < 20:
        raise ValueError(
            "Document is too short to chunk meaningfully. "
            f"Got only {word_count} words."
        )

    # 2. Run evaluation
    _print_header("Running chunking strategies & retrieval")
    evaluator = ChunkingEvaluator(
        model_name=model_name,
        top_k=top_k,
        semantic_percentile=semantic_percentile,
        overlap_chunk_size=overlap_chunk_size,
        overlap_amount=overlap_amount,
    )
    results, best_strategy, best_chunks, final_score = evaluator.evaluate(
        document=document,
        query=query,
    )

    return results, best_strategy, best_chunks, final_score


# ---------------------------------------------------------------------------
# Human-readable terminal display
# ---------------------------------------------------------------------------

def display_results(
    results: Dict[str, Any],
    best_strategy: str,
    best_chunks: List[str],
    final_score: float,
    query: str,
    top_k: int = 5,
    verbose: bool = False,
) -> None:
    """
    Print a formatted summary of all strategy results.

    Parameters
    ----------
    results : Dict[str, Any]
        The full results dict from run_pipeline.
    best_strategy : str
        Name of the winning strategy.
    best_chunks : List[str]
        Top chunks from the winning strategy.
    final_score : float
        Avg score of the winning strategy.
    query : str
        The original query (displayed in the header).
    top_k : int
        How many top chunks to show per strategy.
    verbose : bool
        If True, print all top chunks. If False, only show the best strategy's
        top chunk.
    """
    if _RICH_AVAILABLE:
        _display_rich(results, best_strategy, best_chunks, final_score, query, verbose)
    else:
        _display_plain(results, best_strategy, best_chunks, final_score, query, verbose)


def _display_rich(results, best_strategy, best_chunks, final_score, query, verbose):
    """Rich-formatted terminal output."""
    console.print()
    console.rule(f"[bold cyan]Query: {query!r}[/bold cyan]")
    console.print()

    # Strategy comparison table
    table = Table(
        title="Chunking Strategy Comparison",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Strategy", style="cyan", width=20)
    table.add_column("Chunks", justify="right", width=8)
    table.add_column("Avg Score", justify="right", width=12)
    table.add_column("Top Score", justify="right", width=12)
    table.add_column("Chunk Time", justify="right", width=12)
    table.add_column("Embed Time", justify="right", width=12)

    _STRATEGY_LABELS = {
        "semantic": "Semantic",
        "structure_aware": "Structure-Aware",
        "overlap": "Overlap",
    }

    for name, res in results.items():
        label = _STRATEGY_LABELS.get(name, name)
        is_best = name == best_strategy
        top_score = res["similarity_scores"][0] if res["similarity_scores"] else 0.0
        style = "bold green" if is_best else ""
        winner_badge = " ★" if is_best else ""
        table.add_row(
            f"[{style}]{label}{winner_badge}[/{style}]" if style else f"{label}{winner_badge}",
            str(res.get("num_chunks", "?")),
            f"{res['avg_score']:.4f}",
            f"{top_score:.4f}",
            f"{res.get('chunk_time_s', '?')}s",
            f"{res.get('retrieval_time_s', '?')}s",
        )
    console.print(table)
    console.print()

    # Best strategy details
    console.print(
        Panel(
            f"[bold green]Best Strategy:[/bold green] [cyan]{_STRATEGY_LABELS.get(best_strategy, best_strategy)}[/cyan]\n"
            f"[bold green]Final Score:[/bold green]   [yellow]{final_score:.4f}[/yellow]",
            title="[bold]Winner",
            border_style="green",
        )
    )
    console.print()

    # Top chunks from best strategy
    console.rule("[bold]Top Retrieved Chunks (Best Strategy)[/bold]")
    display_chunks = best_chunks
    for i, chunk in enumerate(display_chunks, 1):
        preview = chunk[:500] + ("..." if len(chunk) > 500 else "")
        score = results[best_strategy]["similarity_scores"][i - 1] if i - 1 < len(results[best_strategy]["similarity_scores"]) else 0.0
        console.print(
            Panel(
                f"[dim]Score: {score:.4f}[/dim]\n\n{preview}",
                title=f"[bold]Chunk {i}[/bold]",
                border_style="blue",
            )
        )
        console.print()

    if verbose:
        for strategy_name, res in results.items():
            if strategy_name == best_strategy:
                continue
            label = _STRATEGY_LABELS.get(strategy_name, strategy_name)
            console.rule(f"[dim]Other Strategy: {label}[/dim]")
            for i, (chunk, score) in enumerate(
                zip(res["top_chunks"], res["similarity_scores"]), 1
            ):
                preview = chunk[:300] + ("..." if len(chunk) > 300 else "")
                console.print(f"  [dim]Chunk {i} (score={score:.4f}):[/dim] {preview}")
            console.print()


def _display_plain(results, best_strategy, best_chunks, final_score, query, verbose):
    """Plain-text fallback when Rich is not installed."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    print(f"\n{'─'*70}")
    print(f"{'STRATEGY':<20} {'CHUNKS':>8} {'AVG SCORE':>12} {'TOP SCORE':>12}")
    print(f"{'─'*70}")
    for name, res in results.items():
        top_score = res["similarity_scores"][0] if res["similarity_scores"] else 0.0
        winner = " ★" if name == best_strategy else ""
        print(
            f"{name:<20} {res.get('num_chunks',0):>8} "
            f"{res['avg_score']:>12.4f} {top_score:>12.4f}{winner}"
        )
    print(f"{'─'*70}")
    print(f"\nBest strategy : {best_strategy}")
    print(f"Final score   : {final_score:.4f}")
    print(f"\nTop chunks from '{best_strategy}':")
    for i, chunk in enumerate(best_chunks, 1):
        preview = chunk[:400] + ("..." if len(chunk) > 400 else "")
        print(f"\n--- Chunk {i} ---")
        print(preview)


# ---------------------------------------------------------------------------
# Utility print helpers
# ---------------------------------------------------------------------------

def _print_header(msg: str) -> None:
    if console:
        console.rule(f"[bold blue]{msg}[/bold blue]")
    else:
        print(f"\n{'─'*60}\n{msg}\n{'─'*60}")


def _info(msg: str) -> None:
    if console:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"  {msg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chunking_retrieval",
        description=(
            "Intelligent Document Chunking & Retrieval System.\n"
            "Compares semantic, structure-aware, and overlap chunking strategies\n"
            "and returns the most effective one for your query."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--doc", "-d",
        required=True,
        metavar="PATH",
        help="Path to the document (.txt, .pdf, .json).",
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The information retrieval query string.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of top chunks to retrieve per strategy (default: 5).",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=400,
        help="Words per chunk for overlap strategy (default: 400).",
    )
    parser.add_argument(
        "--overlap-amount",
        type=int,
        default=80,
        help="Overlap words between consecutive chunks (default: 80).",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=95,
        choices=range(50, 100),
        metavar="[50-99]",
        help="Distance percentile threshold for semantic chunker (default: 95).",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show top chunks from all strategies, not just the best.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        results, best_strategy, best_chunks, final_score = run_pipeline(
            doc_path=args.doc,
            query=args.query,
            top_k=args.top_k,
            model_name=args.model,
            overlap_chunk_size=args.overlap_size,
            overlap_amount=args.overlap_amount,
            semantic_percentile=args.percentile,
        )
    except (FileNotFoundError, ValueError) as e:
        if console:
            console.print(f"[bold red]Error:[/bold red] {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Display in terminal
    display_results(
        results=results,
        best_strategy=best_strategy,
        best_chunks=best_chunks,
        final_score=final_score,
        query=args.query,
        top_k=args.top_k,
        verbose=args.verbose,
    )

    # Save to JSON if requested
    if args.output:
        output_data = {
            "query": args.query,
            "best_strategy": best_strategy,
            "best_chunks": best_chunks,
            "final_score": final_score,
            "results": results,
        }
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(output_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
