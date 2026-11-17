"""Tests for Phase 2 Step 1: generative answers + verification + judge.

The real 0.5B model is NEVER loaded here. All model text comes
through `app.generative.set_completion_override` with a fake
`prompt -> str` callable, so these tests run in seconds and are
deterministic. The real model is exercised only by the generative
eval script (benchmarks/results/generative_eval.json).
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import generative
from app.api import app
from app.config import CHUNKS_FILE
from app.eval import run_eval
from app.retrieval import Retriever
from app.verification import verify_answer, verify_sentence

client = TestClient(app)

CHUNKS = [
    {"doc": "chevening", "chunk_id": "chevening-0", "score": 0.9,
     "rerank_score": 0.9,
     "text": "Chevening Scholarships enable outstanding emerging leaders "
             "from all over the world to pursue one-year master's degrees "
             "in the UK. Chevening Scholarships are fully-funded."},
    {"doc": "vanier", "chunk_id": "vanier-0", "score": 0.8,
     "rerank_score": 0.8,
     "text": "The Vanier Canada Graduate Scholarships program supports "
             "doctoral students with $50,000 per year for three years."},
]

QUESTION = "What does the Chevening Scholarship fund?"


@pytest.fixture(autouse=True)
def _no_model():
    generative.set_completion_override(None)
    yield
    generative.set_completion_override(None)


# --- verification step ---

def test_verify_supported_sentence():
    check = verify_sentence(
        "Chevening Scholarships fund one-year master's degrees in the UK.",
        CHUNKS)
    assert check["supported"] is True
    assert check["chunk_id"] == "chevening-0"
    assert check["coverage"] >= 0.5


def test_verify_removes_unsupported_sentence():
    # No chunk mentions Mars, basket weaving, or underwater anything.
    result = verify_answer(
        "Chevening Scholarships fund one-year master's degrees in the UK. "
        "The scholarship also covers underwater basket weaving on Mars.",
        CHUNKS)
    assert result["kept"] == [
        "Chevening Scholarships fund one-year master's degrees in the UK."]
    assert len(result["dropped"]) == 1
    assert result["dropped"][0]["reason"] == "low_overlap"
    assert "Mars" in result["dropped"][0]["sentence"]


def test_verify_rejects_invented_number():
    # $99,999 appears in no chunk, even though the words overlap.
    check = verify_sentence(
        "The Vanier scholarship pays $99,999 per year for doctoral students.",
        CHUNKS)
    assert check["supported"] is False
    assert check["reason"] == "unsupported_number"


def test_verify_accepts_number_present_in_chunk():
    check = verify_sentence(
        "The Vanier scholarship pays $50,000 per year for doctoral students.",
        CHUNKS)
    assert check["supported"] is True
    assert check["chunk_id"] == "vanier-0"


def test_verify_measures_novel_stems_not_question_words():
    # A sentence that restates the question and adds an ungrounded
    # claim must fail even though the question words overlap the
    # chunk: coverage is measured on the novel stems only.
    check = verify_sentence(
        "Manaaki New Zealand Scholarships cover Governance, Public Health, "
        "Entrepreneurship and Innovation.",
        [{"doc": "manaaki", "chunk_id": "manaaki-0", "score": 0.9,
          "text": "Manaaki New Zealand Scholarships cover all levels of "
                  "study, including undergraduate and postgraduate study "
                  "in New Zealand."}],
        question_stems={"manaaki", "new", "zealand", "scholarship", "cover",
                        "level", "studi"})
    assert check["supported"] is False
    assert check["reason"] == "low_overlap"


def test_verify_empty_sentence_has_no_content():
    check = verify_sentence("... ", CHUNKS)
    assert check["supported"] is False
    assert check["reason"] == "no_content_words"


# --- generative path (fake model) ---

def test_generative_returns_verified_answer():
    generative.set_completion_override(
        lambda prompt: "Chevening Scholarships fund one-year master's degrees in the UK.")
    result = generative.generate_answer(QUESTION, CHUNKS)
    assert result["found"] is True
    assert result["answer_mode"] == "generative"
    assert result["answer"] == "Chevening Scholarships fund one-year master's degrees in the UK."
    assert result["generation"]["model"] == generative.MODEL_ID
    assert result["verification"]["n_dropped"] == 0
    assert result["citations"][0]["chunk_id"] == "chevening-0"


def test_generative_drops_unsupported_sentence_and_keeps_supported():
    generative.set_completion_override(
        lambda prompt: "Chevening Scholarships fund one-year master's degrees in the UK. "
                       "The scholarship also covers underwater basket weaving on Mars.")
    result = generative.generate_answer(QUESTION, CHUNKS)
    assert result["answer_mode"] == "generative"
    assert "Mars" not in result["answer"]
    assert result["verification"]["n_dropped"] == 1


def test_generative_falls_back_when_all_sentences_unsupported():
    generative.set_completion_override(
        lambda prompt: "The scholarship covers underwater basket weaving on Mars.")
    result = generative.generate_answer(QUESTION, CHUNKS)
    assert result["answer_mode"] == "generative_fallback_extractive"
    assert result["found"] is True  # extractive fallback still answers
    assert "Mars" not in result["answer"]
    assert result["generation"]["reason"] == "verification_failed_all_dropped"


def test_generative_falls_back_when_model_unavailable():
    def _boom(prompt):
        raise RuntimeError("model could not load")

    generative.set_completion_override(_boom)
    result = generative.generate_answer(QUESTION, CHUNKS)
    assert result["answer_mode"] == "generative_fallback_extractive"
    assert result["generation"]["reason"] == "model_unavailable"


def test_generative_never_runs_when_extractive_abstains():
    calls = []

    def _fake(prompt):
        calls.append(prompt)
        return "Chevening Scholarships fund one-year master's degrees in the UK."

    generative.set_completion_override(_fake)
    result = generative.generate_answer(
        "How do I bake sourdough bread at home?", CHUNKS)
    assert result["found"] is False
    assert result["answer_mode"] == "abstained"
    assert calls == [], "generator must not run past the abstention gates"


def test_generative_model_abstention_falls_back():
    generative.set_completion_override(lambda prompt: "Not found in the corpus.")
    result = generative.generate_answer(QUESTION, CHUNKS)
    assert result["answer_mode"] == "generative_fallback_extractive"
    assert result["generation"]["reason"] == "model_abstained"


def test_generative_filters_chunks_to_named_scholarship():
    # Regression test for the q17 contamination: the question names
    # Manaaki, the retrieved top-3 mixes Manaaki + SI chunks, and
    # the model copies the SI field list into a Manaaki answer.
    # The named-doc filter restricts grounding to Manaaki chunks,
    # so the SI-grounded sentence is dropped and the answer falls
    # back to extractive instead of misattributing SI's fields.
    retrieved = [
        {"doc": "manaaki", "chunk_id": "manaaki-0", "score": 0.9,
         "rerank_score": 0.9,
         "text": "Manaaki New Zealand Scholarships cover all levels of "
                 "study, including undergraduate and postgraduate study "
                 "in New Zealand."},
        {"doc": "sweden-si", "chunk_id": "sweden-si-0", "score": 0.8,
         "rerank_score": 0.8,
         "text": "The SI Scholarship supports master's studies within "
                 "Governance, Public Health, Entrepreneurship and Innovation."},
    ]
    generative.set_completion_override(
        lambda prompt: "Manaaki New Zealand Scholarships cover Governance, "
                       "Public Health, Entrepreneurship and Innovation.")
    result = generative.generate_answer(
        "What levels of study do Manaaki New Zealand Scholarships cover?",
        retrieved)
    assert result["answer_mode"] == "generative_fallback_extractive"
    assert "Governance" not in result["answer"]


# --- LLM-as-judge ---

def test_judge_returns_scores_in_range():
    verdict = generative.judge_answer(
        QUESTION, "Chevening funds one-year master's degrees.",
        CHUNKS, judge_fn=lambda prompt: "faithfulness: 1\nrelevance: 1")
    assert verdict["faithfulness"] == 1
    assert verdict["relevance"] == 1
    assert verdict["parsed"] is True


def test_judge_zero_scores_parse():
    verdict = generative.judge_answer(
        QUESTION, "Chevening funds underwater basket weaving.",
        CHUNKS, judge_fn=lambda prompt: "faithfulness: 0\nrelevance: 0")
    assert verdict["faithfulness"] == 0
    assert verdict["relevance"] == 0


def test_judge_unparseable_returns_none():
    verdict = generative.judge_answer(
        QUESTION, "Something.", CHUNKS,
        judge_fn=lambda prompt: "I am not sure, maybe.")
    assert verdict["parsed"] is False
    assert verdict["faithfulness"] is None
    assert verdict["relevance"] is None


def test_eval_judge_runs_with_fake_judge():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    retriever = Retriever(chunks, mode="bm25")
    golden = [
        {"id": "q01", "question": "What does the Chevening Scholarship fund?",
         "expected_doc": "chevening", "expected_answer": "one-year master"},
        {"id": "q02", "question": "What is the SI Scholarship for Global Professionals?",
         "expected_doc": "sweden-si", "expected_answer": "SI Scholarship"},
    ]
    result = run_eval(retriever, golden, judge=True,
                      judge_fn=lambda prompt: "faithfulness: 1\nrelevance: 1")
    assert result["judge_faithfulness"] == 1.0
    assert result["judge_relevance"] == 1.0
    assert result["judge_n"] == 2
    assert result["judge_unparsed"] == 0
    # Proxy metrics are still computed alongside the judge.
    assert "answer_only_relevance" in result
    assert "faithfulness" in result


def test_judge_parses_reversed_label_order():
    # The 0.5B judge sometimes answers the lines in the opposite
    # order. Label-aware parsing must not swap the two scores:
    # this output means relevance=1, faithfulness=0.
    verdict = generative.judge_answer(
        QUESTION, "Chevening funds one-year master's degrees.",
        CHUNKS, judge_fn=lambda prompt: "Relevance: 1\nFaithfulness: 0")
    assert verdict["faithfulness"] == 0
    assert verdict["relevance"] == 1
    assert verdict["parsed"] is True


def test_judge_falls_back_to_positional_digits_without_labels():
    verdict = generative.judge_answer(
        QUESTION, "Chevening funds one-year master's degrees.",
        CHUNKS, judge_fn=lambda prompt: "1 0")
    assert verdict["faithfulness"] == 1
    assert verdict["relevance"] == 0


def test_eval_generative_marks_eligibility_answers():
    generative.set_completion_override(
        lambda prompt: "Chevening Scholarships fund one-year master's degrees in the UK.")
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    retriever = Retriever(chunks, mode="bm25")
    golden = [q for q in json.loads(open("eval/golden.json").read())
              if q["id"] == "e01"]
    result = run_eval(retriever, golden, generative=True, judge=True,
                      judge_fn=lambda prompt: "faithfulness: 1\nrelevance: 1")
    assert result["generative_modes"].get("eligibility") == 1
    assert result["per_question"][0]["answer_mode"] == "eligibility"
    # Eligibility statements are not judge-scored.
    assert result["judge_n"] == 0
    assert "judge" not in result["per_question"][0]


def test_eval_generative_mode_with_fake_model():
    generative.set_completion_override(
        lambda prompt: "Chevening Scholarships fund one-year master's degrees in the UK.")
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    retriever = Retriever(chunks, mode="bm25")
    golden = [
        {"id": "q01", "question": "What does the Chevening Scholarship fund?",
         "expected_doc": "chevening", "expected_answer": "one-year master"},
    ]
    result = run_eval(retriever, golden, generative=True)
    assert result["generative_modes"].get("generative") == 1
    assert result["per_question"][0]["answer_mode"] == "generative"


# --- API: generate=true opt-in ---

def test_api_generate_true_returns_verified_answer():
    generative.set_completion_override(
        lambda prompt: "Chevening Scholarships fund one-year master's degrees in the UK.")
    resp = client.post("/ask", json={"question": QUESTION, "generate": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["answer_mode"] == "generative"
    assert body["answer"] == "Chevening Scholarships fund one-year master's degrees in the UK."
    assert body["generation"]["model"] == generative.MODEL_ID
    assert body["citations"], "generative answer must still cite chunks"


def test_api_generate_true_falls_back_on_verification_failure():
    generative.set_completion_override(
        lambda prompt: "The scholarship covers underwater basket weaving on Mars.")
    resp = client.post("/ask", json={"question": QUESTION, "generate": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["answer_mode"] == "generative_fallback_extractive"
    assert "Mars" not in body["answer"]


def test_api_default_stays_extractive():
    resp = client.post("/ask", json={"question": QUESTION})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["answer_mode"] == "extractive"
    assert "generation" not in body


def test_api_generate_true_still_abstains():
    generative.set_completion_override(
        lambda prompt: "Chevening Scholarships fund one-year master's degrees in the UK.")
    resp = client.post("/ask", json={
        "question": "How do I bake sourdough bread at home?", "generate": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["answer"] == "Not found in the corpus."
