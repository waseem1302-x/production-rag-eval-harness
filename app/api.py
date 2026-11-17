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
from .eligibility import resolve as resolve_eligibility
from .ingest import load_sources
from .retrieval import Retriever

_retriever: Retriever | None = None
_chunks: list[dict] | None = None


def get_chunks() -> list[dict]:
    global _chunks
    if _chunks is None:
        _chunks = []
        if CHUNKS_FILE.exists():
            _chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    return _chunks


class Profile(BaseModel):
    """Optional asker profile for eligibility-first answering.
    Both fields are free text: nationality is a country name
    ("Pakistan"), degree_level is a level ("masters", "PhD",
    "bachelor's"). Unknown or missing fields are treated as
    "cannot be decided", never guessed."""
    nationality: str | None = None
    degree_level: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)
    profile: Profile | None = None
    # Phase 2, Step 1: opt-in generative answer (local model +
    # verification, falls back to extractive). Default False:
    # the extractive default is unchanged.
    generate: bool = False


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


app = FastAPI(title="Production RAG Eval Harness (Phase 2)", lifespan=lifespan)


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
                "score": 0.0, "embedder": r.embedder, "citations": [],
                "abstention_reason": "empty_index", "retrieval_trace": None}
    retrieved = r.search(req.question, top_k=req.top_k)
    result = build_answer(req.question, retrieved)
    result["answer_mode"] = "extractive"
    elig_replaced = False
    if req.profile is not None:
        # Eligibility first: a not-eligible or closed verdict, or an
        # eligibility question, replaces the substantive answer.
        # Otherwise the substantive answer stands and the
        # eligibility verdicts are attached as information.
        profile = req.profile.model_dump()
        elig_answer, eligibility = resolve_eligibility(
            req.question, profile, retrieved, get_chunks(), result)
        result["eligibility"] = eligibility
        if elig_answer is not None:
            result = elig_answer
            elig_replaced = True
    if req.generate:
        if elig_replaced or not result.get("found"):
            # Generative path (opt-in only). Eligibility answers are
            # template + verbatim evidence, never generated; an
            # abstention stays an abstention. `generate_answer`
            # re-runs the extractive gates first and falls back
            # to them when verification fails.
            result["answer_mode"] = result.get("answer_mode", "extractive")
            result["generation"] = {
                "model": None,
                "reason": ("skipped_eligibility_answer" if elig_replaced
                           else "skipped_abstention")}
        else:
            from .generative import generate_answer

            gen = generate_answer(req.question, retrieved)
            if "eligibility" in result:
                gen["eligibility"] = result["eligibility"]
            result = gen
    result["embedder"] = r.embedder
    # Attach source URLs to citations.
    urls = {s["slug"]: s["source_url"] for s in load_sources()}
    for cit in result["citations"]:
        cit["source_url"] = urls.get(cit["doc"], "")
    # Auditable retrieval trace: every candidate the retriever
    # considered, in reranked order, with all three scores — plus
    # which chunks were actually injected into the answer. A wrong
    # answer can be traced back to the exact chunk and score that
    # produced it; an abstention can be traced to the gate that fired.
    injected_ids = {c["chunk_id"] for c in result["citations"]}
    result["retrieval_trace"] = {
        "embedder": r.embedder,
        "candidates": [
            {"chunk_id": c["chunk_id"], "doc": c["doc"],
             "dense_score": c["dense_score"], "bm25_score": c["bm25_score"],
             "combined_score": c["score"], "rerank_score": c["rerank_score"],
             "term_coverage": c.get("term_coverage"),
             "injected": c["chunk_id"] in injected_ids}
            for c in retrieved
        ],
    }
    return result
