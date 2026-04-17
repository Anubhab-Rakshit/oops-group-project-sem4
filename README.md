# Intelligent Document Chunking & Retrieval System

A production-grade implementation of three document chunking strategies with automatic best-strategy selection using cosine similarity over sentence embeddings.

---

## Quick Start

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Run against any document
python main.py --doc my_textbook.pdf --query "What is gradient descent?"

# 3. Save results to JSON
python main.py --doc notes.txt --query "Explain regularisation" --output results.json
```

---

## Project Structure

```
.
├── main.py                 # CLI entry point + programmatic API
├── loader.py               # Document loading (TXT / PDF / JSON)
├── chunkers/
│   ├── __init__.py
│   ├── semantic.py         # Semantic chunker
│   ├── structure_aware.py  # Structure-aware chunker
│   └── overlap.py          # Overlap (sliding window) chunker
├── retriever.py            # SBERT embedding + cosine similarity
├── evaluator.py            # Strategy orchestration + best selection
├── utils.py                # Text cleaning, tokenisation, helpers
└── requirements.txt
```

---

## Understanding the Three Chunking Strategies

### 1. Semantic Chunking (`chunkers/semantic.py`)

The most sophisticated strategy. It uses machine learning to detect where topics change.

**Algorithm:**
1. Tokenise the document into individual sentences.
2. Embed every sentence using a transformer model (`all-MiniLM-L6-v2`).
3. Compute the **cosine distance** between each pair of adjacent sentences.
4. Find the 95th percentile of all distances — this is the "sensitivity threshold".
5. Any gap whose distance exceeds the threshold is a **chunk boundary**.

**Intuition:** Imagine two sentences:
- *"The gradient is computed by backpropagation."*
- *"The French Revolution began in 1789."*

Their embedding vectors point in completely different directions in 384-dimensional space. The cosine distance is high → a boundary is placed between them.

**When it excels:** Documents with **natural topic transitions** — research papers, textbooks, novel chapters.

**Configurable parameters:**
| Parameter | Default | Meaning |
|---|---|---|
| `breakpoint_percentile` | 95 | Higher → fewer, larger chunks |
| `min_chunk_size` | 50 words | Merge fragments smaller than this |
| `max_chunk_size` | 1000 words | Force-split chunks larger than this |

---

### 2. Structure-Aware Chunking (`chunkers/structure_aware.py`)

Splits the document using **explicit structural markers** present in the text.

**Detected patterns (in priority order):**
1. PDF page markers: `--- Page N ---`
2. Markdown headings: `# Title`, `## Section`
3. Numbered sections: `1. Introduction`, `Chapter 3`, `Part II`
4. ALL-CAPS headers: `METHODOLOGY`, `DATA ANALYSIS`
5. Double blank lines (paragraph breaks — last resort)

**Intuition:** A 200-page Jupyter notebook is already organised by its author. Chapter 5 is about regularisation; Chapter 6 is about optimisation. These boundaries are explicitly marked.

**When it excels:** Academic notebooks, textbooks, structured PDFs, any document where the author used headings.

**Configurable parameters:**
| Parameter | Default | Meaning |
|---|---|---|
| `min_chunk_words` | 80 | Merge small fragments (e.g., standalone headers) |
| `max_chunk_words` | 1500 | Split oversized sections at paragraph boundaries |

---

### 3. Overlap-Based Chunking (`chunkers/overlap.py`)

The **sliding window** approach. Simple, robust, and guaranteed to capture any span of text.

**Algorithm:**
```
Document words: [w0, w1, w2, ..., wN]

chunk_size = 400 words
overlap    = 80 words
stride     = chunk_size - overlap = 320 words

Window 1: words[0:400]
Window 2: words[320:720]   ← 80 words shared with window 1
Window 3: words[640:1040]  ← 80 words shared with window 2
```

**Intuition:** Without overlap, a key sentence split across two chunks is only partially captured by each. With 20% overlap, every sentence is fully contained within at least one chunk.

**Trade-off:** Produces ~3x more chunks than semantic chunking, which means more embeddings to compute. But it's a robust baseline.

**When it excels:** Densely written prose with no structure and no clear topic transitions. Also a great fallback when the other two methods under-perform for a specific query.

---

## Similarity Metric

We use **cosine similarity over SBERT (Sentence-BERT) embeddings**.

### Why not BM25?
BM25 counts keyword frequencies. Given the query *"What is parameter optimisation?"*, it would fail to match a chunk discussing *"updating weights via gradient descent"* — even though they mean the same thing.

### Why not ROUGE / BLEU?
These metrics measure n-gram overlap between a **reference** text and a **hypothesis**. They are designed for summarisation evaluation, not retrieval.

### Why cosine similarity over SBERT?
| Property | Value |
|---|---|
| Handles paraphrase? | ✓ Yes |
| Handles synonyms? | ✓ Yes |
| Requires labelled data? | ✗ No |
| Interpretable? | ✓ 0 = unrelated, 1 = identical |
| Fast enough for 200 pages? | ✓ ~5-15s on CPU |

**Formula:**
```
cosine_similarity(query, chunk) = (query · chunk) / (||query|| × ||chunk||)
```

Since we L2-normalise all embeddings, this simplifies to a dot product, making batch computation a single matrix multiplication: `O(n × d)` where `n` = number of chunks, `d` = embedding dim (384 for MiniLM).

---

## Output Format

```python
results, best_strategy, best_chunks, final_score = run_pipeline(
    doc_path="textbook.pdf",
    query="What is backpropagation?",
)

# results is a dict:
{
    "semantic": {
        "top_chunks": ["chunk text...", "another chunk...", ...],  # top-5 strings
        "similarity_scores": [0.821, 0.793, 0.776, 0.754, 0.741], # descending
        "avg_score": 0.777,          # primary evaluation metric
        "num_chunks": 84,            # total chunks produced
        "chunk_time_s": 12.4,        # time to chunk (seconds)
        "retrieval_time_s": 3.2,     # time to embed + score (seconds)
    },
    "structure_aware": { ... },
    "overlap":         { ... },
}

# top-level results:
best_strategy  # → "semantic"
best_chunks    # → ["Most relevant chunk...", ...]
final_score    # → 0.777
```

---

## CLI Reference

```
usage: python main.py --doc PATH --query TEXT [options]

Required:
  --doc, -d PATH         Path to document (.txt, .pdf, .json)
  --query, -q TEXT       Information retrieval query

Optional:
  --top-k, -k INT        Top chunks to return per strategy (default: 5)
  --overlap-size INT     Words per chunk, overlap strategy (default: 400)
  --overlap-amount INT   Overlap words between chunks (default: 80)
  --percentile [50-99]   Semantic breakpoint sensitivity (default: 95)
  --model TEXT           SentenceTransformer model (default: all-MiniLM-L6-v2)
  --output, -o PATH      Save full results as JSON
  --verbose, -v          Show top chunks from ALL strategies
```

---

## Programmatic API

```python
from main import run_pipeline

results, best_strategy, best_chunks, final_score = run_pipeline(
    doc_path="my_document.pdf",
    query="Explain the vanishing gradient problem",
    top_k=5,
    overlap_chunk_size=400,
    overlap_amount=80,
    semantic_percentile=95,
)

print(f"Best: {best_strategy} (score: {final_score:.4f})")
for i, chunk in enumerate(best_chunks, 1):
    print(f"\nChunk {i}:\n{chunk[:300]}")
```

---

## Performance Notes for 200+ Page Documents

| Document size | Approx. words | Embedding time (CPU) | Notes |
|---|---|---|---|
| 50 pages | ~25,000 | ~30s | Fast |
| 100 pages | ~50,000 | ~60s | Comfortable |
| 200 pages | ~100,000 | ~2-3 min | Acceptable |
| 500 pages | ~250,000 | ~5-7 min | Increase batch_size |

The bottleneck is embedding chunks, not chunking itself. To speed up:
- Use a GPU → install `torch` with CUDA support before `sentence-transformers`
- Reduce `--overlap-size` to produce fewer overlap chunks
- Use `--model "paraphrase-MiniLM-L3-v2"` for a 4x smaller model (slightly less accurate)

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | SBERT embeddings for semantic chunking and retrieval |
| `pdfplumber` | High-quality PDF text extraction |
| `numpy` | Vectorised cosine similarity computation |
| `nltk` | Punkt sentence tokeniser |
| `rich` | Beautiful terminal output (optional) |
| `tqdm` | Progress bars for large encoding batches |
