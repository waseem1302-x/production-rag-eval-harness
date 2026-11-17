"""Tests for auditable retrieval: traces, abstention reasons, and
the text normaliser. Run: python -m pytest tests/test_audit.py -v
"""
import json

from fastapi.testclient import TestClient

from app import api as api_module
from app.answer import answer as build_answer
from app.config import CHUNKS_FILE
from app.retrieval import Retriever
from app.textnorm import normalize_document, segment_sentences


def test_normalize_unwraps_mid_sentence_line_breaks():
    text = "Masters duration\n1 to 2 academic years (60, 90 or 120\nECTS credits\n)."
    sentences = normalize_document(text).splitlines()
    assert sentences == ["Masters duration 1 to 2 academic years (60, 90 or 120 ECTS credits )."]


def test_normalize_does_not_merge_headers_into_content():
    text = ("Home Page - Vanier Canada Graduate Scholarships\n"
            "We are no longer accepting applications for the Vanier Canada Graduate Scholarships.")
    sentences = normalize_document(text).splitlines()
    assert sentences[0] == "Home Page - Vanier Canada Graduate Scholarships."
    assert sentences[1].startswith("We are no longer accepting")


def test_normalize_adds_terminal_punctuation_to_fragments():
    assert normalize_document("147 Scholarship options for your selection") == \
        "147 Scholarship options for your selection."


def test_segment_sentences_splits_on_punctuation_and_lines():
    text = "First sentence. Second sentence.\nThird line without period"
    assert segment_sentences(text) == [
        "First sentence.", "Second sentence.", "Third line without period"]


def test_abstention_reason_is_recorded():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    cases = [
        ("What is the deadline for the Chevening Scholarship 2027?",
         "missing_specific_term"),
        ("What is the GPA requirement for the DAAD scholarship?",
         "missing_specific_term"),
        ("How do I bake sourdough bread at home?", "weak_overlap"),
    ]
    for question, reason in cases:
        retrieved = retriever.search(question, top_k=5)
        result = build_answer(question, retrieved)
        assert result["found"] is False, question
        assert result["abstention_reason"] == reason, (question, result)


def test_api_response_contains_retrieval_trace():
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={"question": "What does the Chevening Scholarship fund?"})
    assert resp.status_code == 200
    body = resp.json()
    trace = body["retrieval_trace"]
    assert trace["embedder"] == "all-MiniLM-L6-v2"
    assert len(trace["candidates"]) == 5
    candidate = trace["candidates"][0]
    for key in ("chunk_id", "doc", "dense_score", "bm25_score",
                "combined_score", "rerank_score", "term_coverage", "injected"):
        assert key in candidate, key
    injected = [c for c in trace["candidates"] if c["injected"]]
    assert injected, "no chunk marked injected"
    assert {c["chunk_id"] for c in injected} == \
        {c["chunk_id"] for c in body["citations"]}


def test_api_abstention_response_contains_reason():
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={"question": "What is the deadline for the Chevening Scholarship 2027?"})
    body = resp.json()
    assert body["found"] is False
    assert body["abstention_reason"] == "missing_specific_term"
    assert body["retrieval_trace"] is not None
