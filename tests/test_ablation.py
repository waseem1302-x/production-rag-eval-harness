"""Tests for retrieval modes and the ablation configuration.
Run: python -m pytest tests/test_ablation.py -v
"""
import json

import pytest

from app.config import CHUNKS_FILE
from app.retrieval import Retriever
from benchmarks.run_benchmark import CONFIGS

CHUNKS = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
QUESTION = "What does the Chevening Scholarship fund?"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        Retriever(CHUNKS, mode="tfidf")


def test_bm25_mode_never_loads_the_model():
    r = Retriever(CHUNKS, mode="bm25")
    assert r.embedder == "bm25-only"
    assert r.vectors is None
    results = r.search(QUESTION, top_k=5)
    assert len(results) == 5
    assert all(c["dense_score"] == 0.0 for c in results)
    assert results[0]["doc"] == "chevening"


def test_dense_mode_uses_only_dense_scores():
    r = Retriever(CHUNKS, mode="dense")
    assert r.embedder == "all-MiniLM-L6-v2"
    results = r.search(QUESTION, top_k=5)
    assert len(results) == 5
    assert all(c["bm25_score"] == 0.0 for c in results)
    assert results[0]["dense_score"] > 0.0
    assert "chevening" in [c["doc"] for c in results]


def test_hybrid_mode_combines_both_signals():
    r = Retriever(CHUNKS, mode="hybrid")
    results = r.search(QUESTION, top_k=5)
    assert len(results) == 5
    assert any(c["dense_score"] > 0.0 for c in results)
    assert any(c["bm25_score"] > 0.0 for c in results)
    assert "chevening" in [c["doc"] for c in results]


def test_empty_chunks_in_any_mode():
    for mode in ("bm25", "dense", "hybrid"):
        assert Retriever([], mode=mode).search(QUESTION) == []


def test_ablation_configs_cover_each_component():
    names = [c["name"] for c in CONFIGS]
    assert len(CONFIGS) >= 5
    assert len(names) == len(set(names)), "configuration names must be unique"
    for c in CONFIGS:
        assert c["mode"] in ("bm25", "dense", "hybrid")
    # Each ablation pair must exist.
    for required in ("baseline-naive", "hybrid-plain", "improved-bm25",
                     "dense-only", "hybrid-no-rerank", "hybrid-no-gates",
                     "improved-hybrid"):
        assert required in names, f"missing ablation configuration: {required}"
