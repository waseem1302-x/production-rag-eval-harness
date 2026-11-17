"""Tests for Phase 2 Step 2: dense sentence scoring + answer verification.

Dense scoring uses the real local MiniLM embedder (already loaded by
the module-scoped retriever fixture), so these tests take ~1 minute
with the model load. The verification unit tests inject dense sims
and need no model. The generative o04 test uses a fake model, like
tests/test_generative.py — the real 0.5B model is never loaded here.
"""
import json

import pytest

from app import generative
from app.answer import (
    answer as build_answer,
    dense_sentence_scores,
    verify_answer_relevance,
)
from app.config import CHUNKS_FILE
from app.eval import load_golden, run_eval
from app.retrieval import Retriever

Q03 = ("How long are the master's studies supported by the SI "
       "Scholarship for Global Professionals?")
Q10 = "How long does an Erasmus Mundus Joint Master last?"
Q17 = "What levels of study do Manaaki New Zealand Scholarships cover?"
Q01 = "What does the Chevening Scholarship fund?"
O04 = "Which university in Canada is ranked number one for Vanier scholars?"

# Step 1 baseline (benchmarks/results/generative_eval.json, extractive
# column): the 30 in-corpus questions whose answer contained the
# expected phrase. No Step 2 change may lose one of these.
BASELINE_CORRECT = [
    "e01", "e02", "e03", "e04", "e05", "e06", "e07", "e08", "e09",
    "e10", "q01", "q02", "q04", "q05", "q06", "q07", "q08", "q09",
    "q11", "q12", "q13", "q14", "q15", "q16", "q18", "q19", "q20",
    "q21", "q22", "q23",
]


@pytest.fixture(scope="module")
def retriever():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    assert chunks, "run scripts.build_index before tests"
    return Retriever(chunks)


# --- dense sentence scoring ----------------------------------------------

def test_dense_sim_ranks_duration_sentence_above_boilerplate():
    # Measured on MiniLM (all-MiniLM-L6-v2): the sentence that
    # answers "how long" scores 0.771, the lexically overlapping
    # boilerplate ("world-class teaching...") only 0.426. The gap
    # is large, so this is a stable property of the embedder, not
    # a knife-edge ranking.
    question = "How long does the course last?"
    duration = "The course duration is two years."
    boilerplate = ("The course offers world-class teaching and "
                   "excellent facilities.")
    sims = dense_sentence_scores(question, [duration, boilerplate])
    assert sims is not None
    assert sims[duration] > sims[boilerplate]
    assert sims[duration] > 0.7
    assert sims[boilerplate] < 0.5


def test_dense_sim_ranks_levels_sentence_above_greeting():
    # The q17 shape in miniature: "undergraduate and postgraduate
    # study" (0.658) beats the greeting sentence (0.521) on dense
    # similarity even though the greeting shares the scholarship
    # name. (On the real corpus the full answer still misses q17 —
    # the correct sentence ranks 14th by dense sim among all
    # candidates — so this unit test pins the mechanism, not a
    # golden-set win; see test_no_baseline_correct_answer_lost.)
    question = "What levels of study do the scholarships cover?"
    levels = "Scholarships for undergraduate and postgraduate study."
    greeting = "Welcome to the home of the scholarships."
    sims = dense_sentence_scores(question, [levels, greeting])
    assert sims is not None
    assert sims[levels] > sims[greeting]


def test_dense_scores_return_none_without_sentences():
    assert dense_sentence_scores("What does Chevening fund?", []) is None


def test_dense_scoring_does_not_break_q01(retriever):
    retrieved = retriever.search(Q01, top_k=5)
    result = build_answer(Q01, retrieved)
    assert result["found"] is True
    assert "one-year master" in result["answer"].lower()


# --- answer verification: unit tests (injected sims, no model) ----------

def test_verification_abstains_on_ranking_without_rank_term():
    check = verify_answer_relevance(
        O04,
        ["We are no longer accepting applications for the Vanier "
         "Canada Graduate Scholarships.",
         "Vanier, the first francophone Governor General of Canada, "
         "the Vanier Canada Graduate Scholarships (Vanier CGS) program helps."],
        dense_sims={"x": 0.667})
    assert check["answers"] is False
    assert check["reason"] == "unanswered_ranking"


def test_verification_passes_ranking_with_rank_term():
    check = verify_answer_relevance(
        "Which university is ranked number one in Canada?",
        ["The University of Toronto is ranked number one in Canada."],
        dense_sims={"x": 0.8})
    assert check["answers"] is True
    assert check["reason"] is None


def test_verification_abstains_on_low_dense_similarity():
    check = verify_answer_relevance(
        "What does the Chevening Scholarship fund?",
        ["The weather in London is rainy today."],
        dense_sims={"The weather in London is rainy today.": 0.2})
    assert check["answers"] is False
    assert check["reason"] == "low_answer_similarity"


def test_verification_passes_answerable_question():
    check = verify_answer_relevance(
        Q01,
        ["Chevening Scholarships are fully-funded, leaving you free "
         "to focus on achieving your professional goals."],
        dense_sims={"x": 0.75})
    assert check["answers"] is True


def test_verification_skips_dense_floor_without_sims():
    # No sims and no embedder path in this call (dense_sims={} is
    # falsy, so best_sim is None): the ranking check still runs,
    # and a non-ranking question passes.
    check = verify_answer_relevance(Q01, ["Some sentence."], dense_sims={})
    assert check["answers"] is True
    assert check["best_sim"] is None


# --- answer verification: live (real retriever) --------------------------

def test_verification_abstains_on_o04(retriever):
    retrieved = retriever.search(O04, top_k=5)
    result = build_answer(O04, retrieved)
    assert result["found"] is False
    assert result["answer"] == "Not found in the corpus."
    assert result["abstention_reason"] == "unanswered_ranking"
    assert result["citations"] == []


def test_verification_abstains_on_unseen_ranking_question(retriever):
    # Held-out generality check: the rule is about ranking
    # questions, not about o04. The Chevening page contains no
    # ranking term at all ("ranked" appears only in the Erasmus
    # page), so this must abstain with the same reason. The
    # question is phrased so it passes the earlier lexical gates
    # ("one" appears in the picked Chevening sentences), leaving
    # the verification step as the gate that refuses it.
    question = ("Which university is ranked number one for Chevening "
                "scholars in the UK?")
    retrieved = retriever.search(question, top_k=5)
    result = build_answer(question, retrieved)
    assert result["found"] is False
    assert result["abstention_reason"] == "unanswered_ranking"


def test_verification_does_not_abstain_on_answerable(retriever):
    for question in (Q01, Q03):
        retrieved = retriever.search(question, top_k=5)
        result = build_answer(question, retrieved)
        assert result["found"] is True, question
    # q03's answer-only status is asserted by the regression test
    # below (it is a known remaining miss for the extractive path);
    # this test pins only that verification does not refuse it.


def test_other_out_of_corpus_reasons_unchanged(retriever):
    cases = [
        ("What is the deadline for the Chevening Scholarship 2027?",
         "missing_specific_term"),
        ("What is the monthly stipend for the MEXT scholarship in yen?",
         "missing_specific_term"),
        ("What is the acceptance rate for the Erasmus Mundus programme?",
         "missing_specific_term"),
        ("What is the GPA requirement for the DAAD scholarship?",
         "missing_specific_term"),
    ]
    for question, reason in cases:
        retrieved = retriever.search(question, top_k=5)
        result = build_answer(question, retrieved)
        assert result["found"] is False, question
        assert result["abstention_reason"] == reason, (question, result)


# --- generative path also abstains on o04 ---------------------------------

def test_generative_abstains_on_o04(retriever):
    calls = []

    def _fake(prompt):
        calls.append(prompt)
        return ("The University of Toronto is ranked number one for "
                "Vanier scholars.")

    generative.set_completion_override(_fake)
    try:
        retrieved = retriever.search(O04, top_k=5)
        result = generative.generate_answer(O04, retrieved)
    finally:
        generative.set_completion_override(None)
    assert result["found"] is False
    assert result["answer_mode"] == "abstained"
    assert result["abstention_reason"] == "unanswered_ranking"
    assert calls == [], "generator must not run past the verification"


def test_generative_falls_back_when_kept_sentences_fail_relevance():
    # The extractive answer passes verification (its sentence names
    # a ranking), but the model's kept sentence is a grounded
    # nomination sentence with no ranking term in it: the
    # generative path must fall back to extractive instead of
    # returning the irrelevant kept sentence.
    chunks = [
        {"doc": "vanier", "chunk_id": "vanier-0", "score": 0.9,
         "rerank_score": 0.9,
         "text": "The University of Toronto is ranked number one in "
                 "Canada for Vanier scholars. Candidates must be "
                 "nominated by a Canadian institution with a quota."},
    ]
    generative.set_completion_override(
        lambda prompt: "Candidates must be nominated by a Canadian "
                       "institution with a quota.")
    try:
        result = generative.generate_answer(O04, chunks)
    finally:
        generative.set_completion_override(None)
    assert result["answer_mode"] == "generative_fallback_extractive"
    assert result["generation"]["reason"] == "relevance_verification_failed"
    assert result["found"] is True  # extractive fallback still answers
    assert "ranked number one" in result["answer"]


# --- no-regression test (full eval, as Phase 1 did) -----------------------

def test_no_baseline_correct_answer_lost(retriever):
    """Step 2 must not lose any question Step 1 got right, and must
    fix the o04 abstention. Dense scoring is measured to change no
    golden answer at the shipped weight (2.0), so the correct set
    is exactly the Step 1 set; q03, q10 and q17 remain extractive
    misses (documented remaining failures, not regressions)."""
    result = run_eval(retriever, load_golden())
    assert result["answer_only_relevance"] == 0.909
    assert result["answer_relevance_proxy"] == 1.0
    assert result["faithfulness"] == 1.0
    assert result["abstention_accuracy"] == 1.0
    assert result["eligibility_accuracy"] == 1.0
    correct = {r["id"] for r in result["per_question"]
               if r["answer_only_relevant"]}
    assert set(BASELINE_CORRECT) <= correct, (
        "lost baseline-correct questions: "
        f"{sorted(set(BASELINE_CORRECT) - correct)}")
    assert correct == set(BASELINE_CORRECT), (
        f"unexpected changes vs baseline: added={sorted(correct - set(BASELINE_CORRECT))} "
        f"lost={sorted(set(BASELINE_CORRECT) - correct)}")
    for qid in ("q03", "q10", "q17"):
        row = next(r for r in result["per_question"] if r["id"] == qid)
        assert row["answer_only_relevant"] is False, qid
        assert row["found"] is True, qid
