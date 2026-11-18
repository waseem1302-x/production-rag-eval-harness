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
    _named_scholarship_stems,
    _specific_terms,
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
# Phase 2, Step 4 adds golden e11 (Vanier doctoral applicant, verdict
# "closed"), which is answer-only correct, so the pinned set is 31
# and answer-only relevance is 0.912 (31/34) — the 30 originals are
# still all required, below.
# Phase 2, Step 6 adds golden q24-q32 (one per new document) and
# o06-o07. Eight of the nine new factual questions are answer-only
# correct (q24-q31); q32 is a documented miss (bullet-list merging,
# see the regression test below), so the pinned set is 39 and
# answer-only relevance is 0.907 (39/43). Abstention first fell
# 1.0 -> 0.714 (5/7) on corpus growth — o06 and o07 were answered
# instead of abstained, because the grown corpus gave the lexical
# gates other scholarships' deadline and rate-adjacent sentences to
# quote — and Step 6 then fixed both (see
# tests/test_corpus_growth.py): "scholar" joined the generic terms
# (o07) and a scholarship-subject focus check was added to the
# answer verification (o06), so abstention is 1.0 (7/7) again.
BASELINE_CORRECT = [
    "e01", "e02", "e03", "e04", "e05", "e06", "e07", "e08", "e09",
    "e10", "e11", "q01", "q02", "q04", "q05", "q06", "q07", "q08", "q09",
    "q11", "q12", "q13", "q14", "q15", "q16", "q18", "q19", "q20",
    "q21", "q22", "q23",
]
STEP6_CORRECT = ["q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31"]


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


# --- answer verification: scholarship-subject focus (Phase 2, Step 6) ---

O06 = "What is the deadline for the Australia Awards Scholarship 2027?"
O07 = ("What is the acceptance rate for the Knight-Hennessy Scholars "
       "programme?")


def test_scholar_is_a_generic_term():
    # "scholar(s)" is as generic as "scholarship" in this corpus,
    # so it must not count as a specific term: o07's ask
    # ("acceptance", "rate") is then the whole specific set, and no
    # KHS sentence contains either, so the specific-term gate fires.
    assert "scholar" not in _specific_terms(O07)
    assert _specific_terms(O07) == {"acceptanc", "rat"}


def test_named_scholarship_stems_walk_back_from_scholarship():
    assert _named_scholarship_stems(O06) == {"australia", "award"}
    # A known document's name drops out: its doc-alias stems are
    # excluded, and the doc-name gate owns it anyway.
    assert _named_scholarship_stems(
        "What is the deadline for the Chevening Scholarship 2027?") == set()
    assert _named_scholarship_stems(
        "How do I bake sourdough bread at home?") == set()


def test_verification_abstains_on_unknown_scholarship_subject():
    # The o06 shape in miniature: deadline + year sentences from
    # other scholarships, none naming Australia Awards.
    check = verify_answer_relevance(
        O06,
        ["Each recipient will receive an official scholarship offer "
         "by email by the end of April 2027.",
         "The scholarship must then be accepted or declined within "
         "the deadline stated in the offer."],
        dense_sims={"x": 0.64})
    assert check["answers"] is False
    assert check["reason"] == "unanswered_scholarship_subject"


def test_verification_passes_when_subject_sentence_names_it():
    check = verify_answer_relevance(
        O06,
        ["The Australia Awards Scholarship deadline is 30 April 2027."],
        dense_sims={"x": 0.64})
    assert check["answers"] is True
    assert check["reason"] is None


def test_verification_subject_check_skipped_for_known_document():
    # A known name is owned by the doc-name gate: Chevening
    # sentences that never repeat the word "Chevening" must still
    # pass verification (the gate already restricted candidates to
    # the Chevening page).
    check = verify_answer_relevance(
        "What is the deadline for the Chevening Scholarship 2027?",
        ["The deadline stated in the offer is 30 April 2027."],
        dense_sims={"x": 0.64})
    assert check["answers"] is True


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


def test_verification_abstains_on_o06_and_o07(retriever):
    # Phase 2, Step 6: corpus growth first broke both (abstention
    # 1.0 -> 0.714), then the fix restored them. o06 names a
    # scholarship the corpus does not hold ("Australia Awards"
    # appears nowhere in it) and is refused by the
    # scholarship-subject focus check; o07 asks for a fact the
    # fetched KHS page does not state (no "acceptance" anywhere in
    # the corpus) and is refused by the specific-term gate, which
    # no longer counts "scholar" as specific.
    cases = [
        (O06, "unanswered_scholarship_subject"),
        (O07, "missing_specific_term"),
    ]
    for question, reason in cases:
        retrieved = retriever.search(question, top_k=5)
        result = build_answer(question, retrieved)
        assert result["found"] is False, question
        assert result["answer"] == "Not found in the corpus."
        assert result["abstention_reason"] == reason, (question, result)
        assert result["citations"] == []


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
    misses (documented remaining failures, not regressions).
    Phase 2, Step 4: the pinned set grows by exactly e11 (new
    golden question, answer-only correct) and answer-only relevance
    moves 0.909 -> 0.912 (31/34); e02 and e05 stay in the set with
    their verdicts changed to "closed".
    Phase 2, Step 6: the pinned set grows by exactly q24-q31 (new
    golden questions, answer-only correct) and answer-only relevance
    moves 0.912 -> 0.907 (39/43); q32 is a documented miss (the SEK
    12,000 sentence is merged into one long bullet-list sentence by
    the splitter, so it loses to the page intro on shared stems).
    Abstention is 1.0 (7/7): o06 and o07 abstain again after the
    Step 6 fix (see test_corpus_growth.py) — the 0.714 dip measured
    mid-step, when corpus growth first broke both, is recorded in
    that file's history, not hidden."""
    result = run_eval(retriever, load_golden())
    assert result["answer_only_relevance"] == 0.907
    assert result["answer_relevance_proxy"] == 1.0
    assert result["faithfulness"] == 1.0
    assert result["abstention_accuracy"] == 1.0
    assert result["eligibility_accuracy"] == 1.0
    correct = {r["id"] for r in result["per_question"]
               if r["answer_only_relevant"]}
    expected = set(BASELINE_CORRECT) | set(STEP6_CORRECT)
    assert expected <= correct, (
        "lost baseline-correct questions: "
        f"{sorted(expected - correct)}")
    assert correct == expected, (
        f"unexpected changes vs baseline: added={sorted(correct - expected)} "
        f"lost={sorted(expected - correct)}")
    for qid in ("q03", "q10", "q17", "q32"):
        row = next(r for r in result["per_question"] if r["id"] == qid)
        assert row["answer_only_relevant"] is False, qid
        assert row["found"] is True, qid
