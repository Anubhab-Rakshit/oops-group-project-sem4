"""
api.py
------
FastAPI web server that exposes the chunking pipeline as a REST API.

This wraps the existing pipeline into HTTP endpoints so a web frontend
or any other client can use the system without touching the CLI.

Endpoints
---------
POST /analyze  — Upload a document + send a query, get full results back
GET  /health   — Health check (used by deployment platforms to verify the app is alive)
GET  /         — Serves the frontend (index.html from /static)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Redirect HF cache before any imports
_local_cache = Path(__file__).parent / ".hf_cache"
_local_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_local_cache))

from main import run_pipeline  # noqa: E402

app = FastAPI(
    title="ChunkIQ — Intelligent Document Retrieval",
    description="Compares Semantic, Structure-Aware, and Overlap chunking strategies for optimal RAG retrieval.",
    version="2.0.0",
)

# Allow requests from the Vercel frontend or any localhost dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — returns 200 if the server is alive."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Document file (.txt, .pdf, .json)"),
    query: str = Form(..., description="Your search query"),
    top_k: int = Form(5, description="Number of top chunks to retrieve"),
):
    """
    Analyze a document against a query using all three chunking strategies.

    Returns a structured JSON with per-strategy results and the best strategy.
    """
    # Validate file extension
    allowed = {".txt", ".pdf", ".json"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {allowed}",
        )

    # Save uploaded file to a temporary location on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        results, best_strategy, best_chunks, final_score = run_pipeline(
            doc_path=tmp_path,
            query=query,
            top_k=top_k,
        )

        return JSONResponse(
            content={
                "query": query,
                "best_strategy": best_strategy,
                "best_chunks": best_chunks,
                "final_score": round(final_score, 6),
                "results": results,
            }
        )

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # Always clean up the temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Serve the frontend (must be last)
# ─────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
