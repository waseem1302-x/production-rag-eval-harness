"""Tests for Phase 2 Step 6: corpus growth (10 -> 19 documents,
31 -> 101 chunks at 512 tokens, 39 -> 50 golden questions).

The new documents are official scholarship pages only, stored as
extracted .txt (no raw HTML). The new golden questions are measured
here: q24-q31 pass, q32 is a documented miss (bullet-list merging),
and o06/o07 abstain: corpus growth first broke both (abstention
1.0 -> 0.714, measured mid-step), and Step 6 fixed them — "scholar"
joined the generic terms (o07) and a scholarship-subject focus
check was added to the answer verification (o06). Pinning the fix
with its reasons means a future regression must update these tests
deliberately, not silently.
"""
import json
import re
from pathlib import Path

import pytest

from app.answer import answer as build_answer
from app.config import CHUNKS_FILE
from app.eligibility import RULES
from app.eval import load_golden, run_eval
from app.ingest import load_sources
from app.retrieval import Retriever
from scripts.check_data_quality import check

RAW_DIR = Path("data/raw")

NEW_DOCS = {
    # slug: (source_url, expected chunks at 512 tokens)
    "gates-cambridge": ("https://www.gatescambridge.org/programme/the-scholarship/", 2),
    "rhodes": ("https://www.rhodeshouse.ox.ac.uk/scholarships/faqs/", 29),
    "knight-hennessy": ("https://knight-hennessy.stanford.edu/program-overview/funding", 2),
    "schwarzman": ("https://www.schwarzmanscholars.org/frequently-asked-questions/", 12),
    "commonwealth-shared": ("https://cscuk.fcdo.gov.uk/scholarships/commonwealth-shared-scholarships-applications/", 12),
    "rtp": ("https://www.education.gov.au/research-block-grants/research-training-program", 4),
    "marshall": ("https://marshallscholarship.org/about/", 2),
    "erasmus-mundus-org": ("https://erasmus-plus.ec.europa.eu/opportunities/organisations/cooperation-among-organisations-and-institutions/erasmus-mundus-joint-masters", 1),
    "si-pioneering-women": ("https://si.se/en/apply/scholarships/pioneeringwomeninstem/", 6),
}

OLD_DOC_CHUNKS = {
    "chevening": 1, "sweden-si": 12, "fulbright": 3, "daad": 1,
    "erasmus-mundus": 2, "eiffel": 3, "vanier": 1, "manaaki": 1,
    "hkpfs": 2, "mext": 5,
}

# Same patterns as the pre-commit secret scan. Only counts are ever
# reported — a match must never be printed or stored.
SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_-]{10,}"),
    re.compile(r"ghp_[0-9A-Za-z]{20,}"),
    re.compile(r"github_pat_[0-9A-Za-z_]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def _chunks():
    return [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
            if line.strip()]


def _golden_by_id():
    return {q["id"]: q for q in load_golden()}


@pytest.fixture(scope="module")
def retriever():
    chunks = _chunks()
    assert chunks, "run scripts.build_index before tests"
    return Retriever(chunks)


# --- corpus size and sources ----------------------------------------------

def test_corpus_size_as_claimed():
    chunks = _chunks()
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c["doc"]] = by_doc.get(c["doc"], 0) + 1
    assert len(by_doc) == 19
    assert len(chunks) == 101
    for slug, expected in {**OLD_DOC_CHUNKS, **{s: n for s, (_, n) in NEW_DOCS.items()}}.items():
        assert by_doc[slug] == expected, (slug, by_doc.get(slug))


def test_new_sources_recorded_with_url_and_fetch_date():
    sources = {s["slug"]: s for s in load_sources()}
    assert len(sources) == 19
    for slug, (url, _n) in NEW_DOCS.items():
        assert sources[slug]["source_url"] == url, slug
        assert sources[slug]["fetched_at"] == "2026-11-18", slug


def test_no_raw_html_in_corpus():
    assert list(RAW_DIR.glob("*.html")) == []
    for slug in NEW_DOCS:
        assert (RAW_DIR / f"{slug}.txt").is_file()
        assert (RAW_DIR / f"{slug}.json").is_file()


def test_no_secret_pattern_in_new_text_files():
    """Grep-count scan only: 0 matches required, values never printed."""
    counts = {}
    for slug in NEW_DOCS:
        text = (RAW_DIR / f"{slug}.txt").read_text()
        counts[slug] = sum(len(p.findall(text)) for p in SECRET_PATTERNS)
    assert counts == {slug: 0 for slug in NEW_DOCS}, counts


# --- new documents are retrievable ----------------------------------------

def test_new_documents_retrievable(retriever):
    golden = _golden_by_id()
    for qid in ("q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31", "q32"):
        q = golden[qid]
        retrieved = retriever.search(q["question"], top_k=5)
        docs = [c["doc"] for c in retrieved]
        assert q["expected_doc"] in docs, (qid, docs)


def test_new_documents_have_no_eligibility_rules_yet():
    """Step 6 adds corpus, not rules: profile checks for the new
    docs have no authored rule, so the verdict is 'unknown' by
    construction. Authoring rules is a separate step."""
    for slug in NEW_DOCS:
        assert RULES.get(slug) is None, slug


# --- new golden questions ---------------------------------------------------

def test_new_golden_questions_pass_except_documented_q32(retriever):
    result = run_eval(retriever, load_golden())
    rows = {r["id"]: r for r in result["per_question"]}
    for qid in ("q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31"):
        assert rows[qid]["answer_only_relevant"] is True, qid
    # q32 is a documented miss, not a regression to hide: the
    # "SEK 12,000" sentence is in the top retrieved chunk
    # (si-pioneering-women-1), but the sentence splitter merges the
    # whole dash-led bullet list into one long sentence, so it shares
    # fewer query stems than the page intro and is never picked.
    q32 = rows["q32"]
    assert q32["answer_only_relevant"] is False
    assert q32["found"] is True
    assert "sek 12,000" not in q32["answer"].lower()
    retrieved = retriever.search(
        _golden_by_id()["q32"]["question"], top_k=5)
    assert retrieved[0]["chunk_id"] == "si-pioneering-women-1"
    assert "SEK 12,000" in retrieved[0]["text"]


def test_new_out_of_corpus_questions_abstain_with_reasons(retriever):
    """o06 (Australia Awards deadline — the scholarship is not in
    the corpus: the phrase "Australia Awards" appears nowhere in
    it) and o07 (KHS acceptance rate — the fact is not on the
    fetched page: "acceptance" appears nowhere in the corpus) both
    ABSTAIN. Mid-step, corpus growth broke both (abstention 1.0 ->
    0.714, 5/7): the grown corpus gave the lexical gates other
    scholarships' deadline and rate-adjacent sentences to quote,
    and the doc-name gate only fires when a known document is
    named. Step 6 fixed them two ways: o06 is refused by the
    scholarship-subject focus check in `verify_answer_relevance`
    (the picked SI + Commonwealth sentences never name Australia
    Awards), and o07 by the specific-term gate, which no longer
    counts "scholar" as a specific term (every KHS sentence
    contains "scholars", so the gate used to pass on that word
    alone; its only specific terms are now "acceptance" and "rate",
    neither in any picked sentence).
    """
    golden = _golden_by_id()
    expected_reasons = {
        "o06": "unanswered_scholarship_subject",
        "o07": "missing_specific_term",
    }
    for qid, reason in expected_reasons.items():
        retrieved = retriever.search(golden[qid]["question"], top_k=5)
        result = build_answer(golden[qid]["question"], retrieved)
        assert result["found"] is False, qid
        assert result["answer"] == "Not found in the corpus.", qid
        assert result["abstention_reason"] == reason, (qid, result)
        assert result["citations"] == [], qid
    result = run_eval(retriever, load_golden())
    assert result["abstention_accuracy"] == 1.0
    assert result["n_out_of_corpus"] == 7


# --- data-quality checker limitation ---------------------------------------

def test_stipend_flag_currency_gap_is_fixed():
    """Phase 2, Step 7: the Step 6 limitation is FIXED, not just
    pinned. The checker used to flag Rhodes, Commonwealth Shared
    and SI Pioneering 'missing stipend' even though all three pages
    state an amount — its MONEY_RE matched €/$ only, not £/SEK —
    and Gates Cambridge was the same gap in the other direction:
    NOT flagged, because the page's "US$210m" founding donation
    matched the pattern, while its real stipend (£23,152) is in
    pounds. RTP was flagged although it states the rates as
    "($) 35,550" (sign detached from digits), and Sweden SI and
    MEXT were flagged although they state SEK 12,000/month and
    ¥117,000–¥242,000/month. The pattern now matches £/SEK/¥ and
    detached "($)", so none of the six is flagged — and Gates is
    still not flagged, now also because its £23,152 stipend
    matches. This test replaces the Step 6 pin of the limitation;
    the history is in tests/test_reverification.py.
    """
    report = check(_chunks(), load_sources())
    missing = report["missing_fields"]
    assert "stipend" not in missing.get("rhodes", [])
    assert "stipend" not in missing.get("commonwealth-shared", [])
    assert "stipend" not in missing.get("si-pioneering-women", [])
    assert "stipend" not in missing.get("gates-cambridge", [])
    assert "stipend" not in missing.get("rtp", [])
    assert "stipend" not in missing.get("sweden-si", [])
    assert "stipend" not in missing.get("mext", [])
    assert "£21,060" in (RAW_DIR / "rhodes.txt").read_text()
    assert "£1,712" in (RAW_DIR / "commonwealth-shared.txt").read_text()
    assert "SEK 12,000" in (RAW_DIR / "si-pioneering-women.txt").read_text()
    assert "£23,152" in (RAW_DIR / "gates-cambridge.txt").read_text()
    assert "US$210m" in (RAW_DIR / "gates-cambridge.txt").read_text()
    assert "35,550" in (RAW_DIR / "rtp.txt").read_text()
