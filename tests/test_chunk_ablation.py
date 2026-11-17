"""Chunk-size ablation tests (Phase 2, Step 5).

Covers: chunks stay within the token budget for each ablated size
(256 / 512 / 1,024 tokens), overlap is preserved at each size, the
index builder accepts a chunk-size parameter, and the eval runs
against a non-default chunk size without crashing.
"""
import json

import pytest

from app.chunking import chunk_document, chunk_text
from app.config import CHUNK_OVERLAP_CHARS, chunk_chars_for_tokens
from app.eval import load_golden, run_eval
from app.ingest import load_documents
from app.retrieval import Retriever
from scripts.build_index import build_index

LONG_TEXT = " ".join(
    f"Sentence number {i} about scholarships, funding, and study abroad."
    for i in range(400)
)


@pytest.mark.parametrize("tokens", [256, 512, 1024])
def test_chunk_chars_for_tokens(tokens):
    assert chunk_chars_for_tokens(tokens) == tokens * 4


@pytest.mark.parametrize("tokens", [256, 512, 1024])
def test_chunking_within_budget_for_each_size(tokens):
    budget = chunk_chars_for_tokens(tokens)
    chunks = chunk_text(LONG_TEXT, chunk_chars=budget)
    assert len(chunks) > 1
    for chunk in chunks:
        # Overlap may push a chunk up to budget + overlap past target.
        assert len(chunk) <= budget + CHUNK_OVERLAP_CHARS, (
            f"{tokens}-token chunk too long: {len(chunk)} > {budget + CHUNK_OVERLAP_CHARS}")


@pytest.mark.parametrize("tokens", [256, 512, 1024])
def test_overlap_preserved_for_each_size(tokens):
    budget = chunk_chars_for_tokens(tokens)
    chunks = chunk_text(LONG_TEXT, chunk_chars=budget)
    assert len(chunks) >= 2
    for prev, cur in zip(chunks, chunks[1:]):
        # Some word from the tail of the previous chunk must reappear
        # at the start of the next one (the overlap seed).
        tail_words = prev.split()[-10:]
        assert any(word in cur for word in tail_words), (
            f"{tokens}-token chunks share no overlap text")


def test_chunk_document_accepts_size_parameter():
    small = chunk_document("doc", LONG_TEXT, chunk_chars=1024)
    default = chunk_document("doc", LONG_TEXT)
    assert len(small) > len(default)
    assert small[0]["chunk_id"] == "doc-0"
    assert all(c["doc"] == "doc" for c in small)


def test_build_index_rejects_unknown_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="unsupported chunk size"):
        build_index(chunk_tokens=128, out_dir=tmp_path)


def test_build_index_accepts_chunk_size_parameter(tmp_path):
    stats = build_index(chunk_tokens=256, out_dir=tmp_path)
    assert stats["chunk_tokens"] == 256
    assert stats["chunk_chars"] == 1024
    assert stats["overlap_chars"] == 200
    assert stats["n_chunks"] == 60  # same corpus, 256-token chunks
    assert stats["n_vectors"] == 60
    chunks_file = tmp_path / "chunks.jsonl"
    assert chunks_file.exists()
    assert (tmp_path / "faiss.index").exists()
    chunks = [json.loads(line) for line in chunks_file.read_text().splitlines() if line.strip()]
    assert len(chunks) == 60
    assert max(len(c["text"]) for c in chunks) <= 1024 + CHUNK_OVERLAP_CHARS


def test_eval_runs_against_non_default_chunk_size():
    """Full eval path on 256-token chunks: must not crash, even though
    eligibility rules pin 512-token chunk ids (some rules drop to
    unknown at this size — that is a measured ablation finding, not
    a crash). Uses a small golden subset to keep the test fast."""
    docs = load_documents()
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc["doc"], doc["text"], chunk_chars=1024))
    assert len(chunks) == 60
    retriever = Retriever(chunks)
    golden = [q for q in load_golden() if q["id"] in ("q01", "q02", "e01", "o01")]
    result = run_eval(retriever, golden)
    assert result["n_in_corpus"] == 3
    assert result["n_out_of_corpus"] == 1
    assert result["n_eligibility"] == 1
    assert 0.0 <= result["recall@5"] <= 1.0
    assert result["eligibility_accuracy"] in (0.0, 1.0)
    assert len(result["per_question"]) == 3
