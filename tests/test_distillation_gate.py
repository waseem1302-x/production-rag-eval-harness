"""Tests for the distillation verification gate (pure helpers).

No teacher call, no model load, no network. These tests prove the
gate that decides what enters training data: sentence counts,
expected-phrase checks, new-number checks, judge parsing, and the
golden-question similarity check.
"""
from scripts.build_distillation_data import (
    contains_phrase,
    count_sentences,
    faithfulness_proxy,
    is_verbatim_golden,
    jaccard_similarity,
    key_fact_covered,
    no_new_numbers,
    parse_judge_lines,
    strip_code_fences,
)

CHUNKS = [
    {"chunk_id": "chevening-0", "doc": "chevening",
     "text": "Chevening Scholarships enable outstanding emerging "
             "leaders to pursue one-year master's degrees in the UK. "
             "Chevening Scholarships are fully-funded."},
    {"chunk_id": "sweden-si-0", "doc": "sweden-si",
     "text": "The SI Scholarship covers a monthly allowance of "
             "SEK 12,000 and a travel grant of SEK 15,000."},
]
GOLDEN = [{"id": "q01",
           "question": "What does the Chevening Scholarship fund?"},
          {"id": "q10",
           "question": "How long does an Erasmus Mundus Joint Master last?"}]


def test_count_sentences():
    assert count_sentences("One sentence.") == 1
    assert count_sentences("First sentence. Second sentence.") == 2
    assert count_sentences("One. Two. Three.") == 3
    assert count_sentences("") == 0


def test_contains_phrase_is_case_insensitive():
    assert contains_phrase("The SI Scholarship covers SEK 12,000.",
                           "sek 12,000")
    assert not contains_phrase("The SI Scholarship covers SEK 12,000.",
                               "SEK 11,000")


def test_no_new_numbers():
    assert no_new_numbers("The allowance is SEK 12,000.", CHUNKS)
    assert not no_new_numbers("The allowance is SEK 99,000.", CHUNKS)
    assert no_new_numbers("No numbers here at all.", CHUNKS)


def test_parse_judge_lines():
    assert parse_judge_lines("faithfulness: 1\nrelevance: 0") == (1, 0)
    assert parse_judge_lines("Relevance: 1\nFaithfulness: 0") == (0, 1)
    assert parse_judge_lines("faithfulness: 1") is None
    assert parse_judge_lines("looks good to me") is None
    assert parse_judge_lines("faithfulness: 2\nrelevance: 1") is None


def test_jaccard_similarity():
    q = "What does the Chevening Scholarship fund?"
    assert jaccard_similarity(q, q) == 1.0
    assert jaccard_similarity(q, "How long does an Erasmus Mundus "
                              "Joint Master last?") < 0.3
    assert jaccard_similarity(
        "What does Chevening fund?", q) >= 0.5


def test_is_verbatim_golden():
    assert is_verbatim_golden("What does the Chevening Scholarship fund?",
                              GOLDEN)
    assert not is_verbatim_golden(
        "What does the Chevening Scholarship pay for?", GOLDEN)


def test_strip_code_fences():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('plain text') == 'plain text'


def test_faithfulness_proxy():
    assert faithfulness_proxy(
        "Chevening Scholarships are fully-funded.", CHUNKS) == 1
    assert faithfulness_proxy(
        "Chevening Scholarships pay SEK 99,000 a month.", CHUNKS) == 0


def test_key_fact_covered_allows_rephrasing():
    # Real teacher outputs from the smoke run: the answer is right,
    # the wording differs ("September/October" vs "September or
    # October", "8-24" vs "8 to 24"). A verbatim check drops them.
    assert key_fact_covered(
        "The scholarship studies in the UK begin in September or "
        "October 2027.", "September/October 2027")
    assert key_fact_covered(
        "The application portal for SI Scholarships is open from 8 "
        "to 24 February 2027.", "8-24 February 2027")
    assert key_fact_covered(
        "In accordance with the UK Bribery Act 2010, any applicant "
        "who is convicted of bribery faces a ban from reapplying "
        "for a Commonwealth Scholarship or Fellowship for a duration "
        "of up to five years.",
        "banned from reapplying for a Commonwealth Scholarship or "
        "Fellowship for a period of up to five years")


def test_key_fact_covered_rejects_wrong_numbers():
    assert not key_fact_covered(
        "The allowance is SEK 11,000 per month.", "SEK 12,000")
    assert not key_fact_covered(
        "The scholarship funds doctoral study.", "SEK 12,000")


def test_key_fact_covered_accepts_synonyms():
    # Real smoke-run case: the answer is correct and grounded, but
    # "at least" / "long" / "consist" became "minimum" /
    # "duration" / "comprise", which the stemmer cannot bridge.
    # Coverage is 0.625, so the bar is 0.6, not 0.7.
    assert key_fact_covered(
        "Eligible industry internships must be a minimum of 3 months "
        "in duration and comprise 60 full-time equivalent days of "
        "engagement.",
        "at least 3 months long, consist of 60 full-time equivalent "
        "days of engagement")
