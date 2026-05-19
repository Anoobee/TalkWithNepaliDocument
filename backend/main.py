"""
backend/main.py
---------------
FastAPI RAG backend.

Endpoints:
  POST /query          → full pipeline, returns JSON answer + sources
  POST /query/stream   → streaming SSE answer
  GET  /health         → liveness check
  GET  /docs           → Swagger UI (auto)
"""

from __future__ import annotations
import sys
import os

# Allow importing from project root (ai/ package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai.embedder import encode_query, get_model
from ai.retriever import retrieve
from ai.llm import generate_answer, stream_answer


# App setup


app = FastAPI(
    title="Nepali Document RAG API",
    description="BGE-M3 hybrid retrieval (Vespa) + Gemini answer generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Schemas



class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=1000, description="User query (Nepali or English)"
    )
    top_k_retrieval: int = Field(
        default=20, ge=1, le=100, description="Docs fetched from Vespa"
    )
    top_k_context: int = Field(default=5, ge=1, le=20, description="Docs passed to LLM")


class SourceDoc(BaseModel):
    id: str
    text: str
    relevance: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDoc]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float



# Startup: warm up the model



@app.on_event("startup")
async def startup_event():
    """Load BGE-M3 into memory at startup so first query isn't slow."""
    get_model()
    print("BGE-M3 model loaded and ready.")



# Endpoints



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Full RAG pipeline:
    1. Embed query with BGE-M3 (dense + sparse + ColBERT)
    2. Retrieve top-K from Vespa (HNSW + BM25, two-phase ColBERT rerank)
    3. Pass top context docs to Gemini
    4. Return answer + sources
    """
    t0 = time.perf_counter()

    # Step 1: Embed
    try:
        embeddings = encode_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Step 2: Retrieve
    t1 = time.perf_counter()
    try:
        hits = retrieve(req.query, embeddings, top_k=req.top_k_retrieval)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vespa retrieval failed: {e}")
    retrieval_ms = (time.perf_counter() - t1) * 1000

    # Trim to top_k_context for LLM (Vespa already ranked them)
    context_docs = hits[: req.top_k_context]

    if not context_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # Step 3: Generate
    t2 = time.perf_counter()
    try:
        answer = generate_answer(req.query, context_docs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini generation failed: {e}")
    generation_ms = (time.perf_counter() - t2) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=[SourceDoc(**d) for d in context_docs],
        retrieval_time_ms=round(retrieval_ms, 1),
        generation_time_ms=round(generation_ms, 1),
        total_time_ms=round(total_ms, 1),
    )


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """
    Streaming version — returns SSE text/event-stream.
    Sources are sent as a final JSON line prefixed with 'data: [SOURCES]'.
    """
    # Embed + retrieve synchronously first
    try:
        embeddings = encode_query(req.query)
        hits = retrieve(req.query, embeddings, top_k=req.top_k_retrieval)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    context_docs = hits[: req.top_k_context]

    if not context_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    import json

    def event_generator():
        try:
            # Stream the answer tokens
            for chunk in stream_answer(req.query, context_docs):
                if chunk:
                    # Escape newlines in chunk for proper SSE format
                    escaped_chunk = chunk.replace("\n", "\\n")
                    yield f"data: {escaped_chunk}\n\n"

            # Send sources at the end
            sources_payload = json.dumps(
                [
                    {
                        "id": d["id"],
                        "text": d["text"][:300] + "...",
                        "relevance": d["relevance"],
                    }
                    for d in context_docs
                ]
            )
            yield f"data: [SOURCES]{sources_payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
