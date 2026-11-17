"""FastAPI app: one endpoint, POST /ask, plus /health.

The index is built lazily on first use (and at startup when the
processed chunks file exists) so importing the app in tests is cheap.
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import embeddings
from .answer import answer as build_answer
from .config import CHUNKS_FILE, DEFAULT_TOP_K
from .ingest import load_sources
from .retrieval import Retriever

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        chunks = []
        if CHUNKS_FILE.exists():
            chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
        _retriever = Retriever(chunks)
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()
    yield


app = FastAPI(title="Production RAG Eval Harness (Phase 1)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


@app.get("/health")
def health():
    r = get_retriever()
    docs = {c["doc"] for c in r.chunks}
    return {"status": "ok", "docs": len(docs), "chunks": len(r.chunks),
            "embedder": r.embedder}


@app.post("/ask")
def ask(req: AskRequest):
    r = get_retriever()
    if not r.chunks:
        return {"question": req.question, "found": False,
                "answer": "Not found in the corpus. The index is empty; run scripts.build_index first.",
                "score": 0.0, "embedder": r.embedder, "citations": []}
    retrieved = r.search(req.question, top_k=req.top_k)
    result = build_answer(req.question, retrieved)
    result["embedder"] = r.embedder
    # Attach source URLs to citations.
    urls = {s["slug"]: s["source_url"] for s in load_sources()}
    for cit in result["citations"]:
        cit["source_url"] = urls.get(cit["doc"], "")
    return result
