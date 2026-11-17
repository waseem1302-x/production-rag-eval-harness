"""Edge-case tests. Every case here was a real failure mode found
while building Phase 1, or a boundary the next change must not break.

Run: python -m pytest tests/test_edge_cases.py -v
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.answer import answer as build_answer
from app.chunking import chunk_document, chunk_text
from app.config import CHUNKS_FILE
from app.retrieval import Retriever


# --- Chunking edge cases -------------------------------------------------

def test_chunking_empty_document():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []
    assert chunk_document("empty-doc", "") == []


def test_chunking_single_long_sentence_is_hard_split():
    # One 5,000-char sentence with no sentence boundary must still
    # be split, and no chunk may exceed the target + overlap.
    text = "word " * 1000  # 5,000 chars, no full stop at all
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 2048 + 200


def test_chunking_short_text_is_not_split():
    assert chunk_text("One short sentence.") == ["One short sentence."]


def test_chunking_duplicate_chunks_keep_stable_ids():
    # The same text twice must produce distinct, stable chunk ids;
    # downstream code keys citations by chunk_id, so a collision
    # would silently merge two different documents.
    chunks = chunk_document("doc-a", "Same text. " * 50) + \
        chunk_document("doc-b", "Same text. " * 50)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunking_overlap_does_not_duplicate_whole_chunk():
    text = " ".join(f"Sentence {i} about funding and study abroad." for i in range(300))
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert chunks[0] != chunks[1]


# --- Retrieval edge cases ------------------------------------------------

def test_retrieval_empty_index_returns_nothing():
    retriever = Retriever([])
    assert retriever.search("What does Chevening fund?", top_k=5) == []


def test_retrieval_duplicate_chunks_do_not_crash():
    chunks = ([{"chunk_id": "a-0", "doc": "a", "text": "Chevening funds one-year masters in the UK."}] * 2 +
              [{"chunk_id": "b-0", "doc": "b", "text": "The SI scholarship funds masters in Sweden."}])
    retriever = Retriever(chunks)
    results = retriever.search("What does Chevening fund?", top_k=5)
    assert results
    assert all("chunk_id" in r and "score" in r for r in results)


def test_retrieval_weak_query_returns_low_scores():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    results = retriever.search("flibberty gibbet wozzeck", top_k=5)
    # A nonsense query must not return a confident top hit: every
    # returned score must be below the top score of a real question.
    real = retriever.search("What does the Chevening Scholarship fund?", top_k=5)
    assert results[0]["rerank_score"] < real[0]["rerank_score"]


# --- Answering / abstention edge cases -----------------------------------

def test_answer_empty_retrieval_abstains():
    result = build_answer("What does Chevening fund?", [])
    assert result["found"] is False
    assert result["answer"] == "Not found in the corpus."
    assert result["citations"] == []


def test_answer_missing_context_abstains():
    # The question names a scholarship whose page is not in this
    # tiny index at all; the system must not answer from another
    # scholarship's page.
    chunks = [{"chunk_id": "chevening-0", "doc": "chevening",
               "text": "Chevening Scholarships enable outstanding emerging leaders "
                       "to pursue one-year master's degrees in the UK."}]
    retriever = Retriever(chunks)
    retrieved = retriever.search("What is the monthly stipend for the MEXT scholarship in yen?", top_k=5)
    result = build_answer("What is the monthly stipend for the MEXT scholarship in yen?", retrieved)
    assert result["found"] is False


def test_answer_weak_retrieval_abstains():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    retrieved = retriever.search("How do I bake sourdough bread at home?", top_k=5)
    result = build_answer("How do I bake sourdough bread at home?", retrieved)
    assert result["found"] is False
    assert result["citations"] == []


def test_answer_gates_off_answers_everything():
    # The naive baseline (gates off) answers even unanswerable
    # questions. This test pins that behaviour so the benchmark's
    # baseline numbers stay honest: if this ever starts abstaining,
    # the baseline is no longer a baseline.
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    retrieved = retriever.search("What is the GPA requirement for the DAAD scholarship?", top_k=5)
    result = build_answer("What is the GPA requirement for the DAAD scholarship?",
                          retrieved, gates=False, scored_picking=False)
    assert result["found"] is True


# --- API edge cases ------------------------------------------------------

def test_api_empty_index_returns_not_found_message():
    api_module._retriever = Retriever([])
    try:
        client = TestClient(api_module.app)
        resp = client.post("/ask", json={"question": "What does Chevening fund?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is False
        assert "index is empty" in body["answer"]
    finally:
        api_module._retriever = None


def test_api_health_with_empty_index():
    api_module._retriever = Retriever([])
    try:
        client = TestClient(api_module.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["chunks"] == 0
    finally:
        api_module._retriever = None
