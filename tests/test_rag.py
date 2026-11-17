"""Tests for chunking, retrieval, and abstention. Run: python -m pytest tests/ -v"""
import json

import pytest

from app.answer import answer as build_answer
from app.chunking import chunk_document, chunk_text
from app.config import CHUNKS_FILE
from app.retrieval import Retriever


def test_chunking_respects_target_size():
    text = " ".join(f"Sentence number {i} about scholarships and funding." for i in range(200))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 2048 + 200, f"chunk too long: {len(chunk)}"


def test_chunking_keeps_short_text_whole():
    assert chunk_text("Short text.") == ["Short text."]
    assert chunk_text("") == []


def test_chunking_overlap_shares_text():
    text = " ".join(f"Word{i} sentence about funding and study abroad." for i in range(300))
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # The tail of chunk 0 must appear at the start of chunk 1 (overlap).
    assert chunks[1].startswith(chunks[0][-200:].split(" ", 1)[-1][:20]) or \
        any(word in chunks[1] for word in chunks[0].split()[-10:])


def test_chunk_document_ids_are_stable():
    chunks = chunk_document("chevening", "First paragraph. " * 100)
    assert chunks[0]["chunk_id"] == "chevening-0"
    assert all(c["doc"] == "chevening" for c in chunks)


@pytest.fixture(scope="module")
def retriever():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    assert chunks, "run scripts.build_index before tests"
    return Retriever(chunks)


def test_retrieval_finds_right_document(retriever):
    results = retriever.search("How much is the Vanier scholarship worth per year?", top_k=5)
    docs = [r["doc"] for r in results]
    assert "vanier" in docs, f"vanier not in top 5: {docs}"


def test_retrieval_returns_citations_with_scores(retriever):
    results = retriever.search("What does the Chevening Scholarship fund?", top_k=3)
    assert len(results) == 3
    for r in results:
        assert r["text"] and r["chunk_id"] and r["score"] >= 0


def test_abstention_on_out_of_corpus_question(retriever):
    # "GPA requirement" is not in the DAAD page. The system must abstain
    # rather than return DAAD sentences that never mention a GPA.
    for question in [
        "What is the GPA requirement for the DAAD scholarship?",
        "What is the deadline for the Chevening Scholarship 2027?",
        "What is the acceptance rate for the Erasmus Mundus programme?",
    ]:
        results = retriever.search(question, top_k=5)
        result = build_answer(question, results)
        assert result["found"] is False, f"did not abstain on: {question} -> {result['answer']}"
        assert result["answer"] == "Not found in the corpus."
        assert result["citations"] == []


def test_abstention_on_unrelated_question(retriever):
    results = retriever.search("How do I bake sourdough bread at home?", top_k=5)
    result = build_answer("How do I bake sourdough bread at home?", results)
    assert result["found"] is False
    assert result["answer"] == "Not found in the corpus."
    assert result["citations"] == []
