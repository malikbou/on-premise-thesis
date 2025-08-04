"""FastAPI wrapper exposing the RAG pipeline for performance testing.

Run locally:
    $ source .venv/bin/activate
    $ uvicorn load_test.rag_server:app --reload --host 0.0.0.0 --port 8000

The server exposes two endpoints:
    POST /query     – body {"question": "..."}
    GET  /metrics   – Prometheus scrape endpoint

Environment variables:
    EMBEDDING_MODEL – Ollama embedding model slug (default: all-minilm)
    CHAT_MODEL      – Ollama chat model slug (default: mistral:7b)
    INDEX_DIR       – Path to FAISS index directory (defaults to .rag_cache/<slug>/faiss_index)
"""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Histogram, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# LangChain + Ollama imports
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# Ollama connection (container -> host)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# ---------------------------------------------------
# Configuration helpers
# ---------------------------------------------------

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral:7b")

# Retrieval parameters (keep in sync with src.build_embeddings)
CHUNK_TOP_K = 3

app = FastAPI(title="RAG Performance Test Server")

# ---------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------

REQUEST_COUNT = Counter(
    "rag_requests_total", "Total number of /query requests received"
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "Latency of /query requests (wall time)"
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds", "Time spent in vector store retrieval"
)
GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds", "Time spent in LLM generation"
)
IN_FLIGHT = Gauge(
    "rag_in_flight", "Number of in-flight /query requests"
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    retrieved_docs: List[str]
    total_time_ms: float
    retrieval_ms: float
    generation_ms: float


# ---------------------------------------------------
# Lazy loaders (so Uvicorn workers each load their own copy)
# ---------------------------------------------------

@lru_cache(maxsize=1)
def _load_retriever():
    """Load FAISS index & return a retriever object."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", DEFAULT_EMBEDDING_MODEL.lower())
    index_dir = os.getenv("INDEX_DIR", os.path.join(".rag_cache", slug, "faiss_index"))
    index_file = os.path.join(index_dir, "index.faiss")

    if not os.path.exists(index_file):
        raise RuntimeError(
            f"FAISS index not found at {index_file}. Build embeddings first."
        )

    embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": CHUNK_TOP_K})


@lru_cache(maxsize=1)
def _load_chat_model():
    return ChatOllama(model=DEFAULT_CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1, timeout=120)


# ---------------------------------------------------
# Middleware to track in-flight requests
# ---------------------------------------------------

class InFlightMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/query":
            IN_FLIGHT.inc()
        try:
            response = await call_next(request)
            return response
        finally:
            if request.url.path == "/query":
                IN_FLIGHT.dec()


app.add_middleware(InFlightMiddleware)


# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.post("/query", response_model=AnswerResponse)
async def query_rag(req: QuestionRequest):
    REQUEST_COUNT.inc()
    start_wall = time.perf_counter()

    # Load heavy objects lazily
    try:
        retriever = _load_retriever()
        llm = _load_chat_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build RAG chain on the fly (cheap)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    # ----- Retrieval timing -----
    t0 = time.perf_counter()
    docs = retriever.get_relevant_documents(req.question)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    RETRIEVAL_LATENCY.observe(retrieval_ms / 1000)

    # ----- Generation timing -----
    t1 = time.perf_counter()
    answer = llm.invoke(req.question).content
    generation_ms = (time.perf_counter() - t1) * 1000
    GENERATION_LATENCY.observe(generation_ms / 1000)

    total_ms = (time.perf_counter() - start_wall) * 1000
    REQUEST_LATENCY.observe(total_ms / 1000)

    return AnswerResponse(
        answer=answer,
        retrieved_docs=[d.page_content for d in docs],
        total_time_ms=total_ms,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
