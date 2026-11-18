"""Tests for Phase 2 Step 7: re-verification of stale pages.

On 18 Nov 2026 every one of the 19 source pages was re-fetched
(direct HTTP with the repo's own extractor, plus a browser text
fetch for the 11 pages originally saved that way and for RTP,
whose direct fetch timed out) and compared with the stored .txt.
Result: all 19 live; 18 substantively unchanged, Eiffel changed
only in a sidebar recommended-article teaser. No data/raw/*.txt
was rewritten, so the index and the eval must be unchanged — the
eval test below pins that against the saved before-run file.
The same step fixed the data-quality checker's currency gap
(£/SEK/¥ and detached "($) 35,550"), so the stipend flags it
used to raise for Rhodes, Commonwealth Shared, SI Pioneering
Women, Sweden SI, MEXT and RTP are gone — and the pages that
state two amounts in different fields now carry resolutions.
"""
import json
import re
from pathlib import Path

from app.config import CHUNKS_FILE
from app.ingest import load_sources
from scripts.check_data_quality import MONEY_RE, check

RESULTS_DIR = Path("benchmarks/results")
RAW_DIR = Path("data/raw")
REVERIFICATION = RESULTS_DIR / "reverification_2026-11-18.json"

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


# --- re-verification is recorded per source -------------------------------

def test_every_source_records_a_reverification_date():
    """Step 7: each of the 19 sources carries reverified_at, and the
    machine-readable record covers all 19 pages, all live."""
    sources = load_sources()
    assert len(sources) == 19
    for s in sources:
        assert s["reverified_at"] == "2026-11-18", s["slug"]
    record = json.loads(REVERIFICATION.read_text())
    assert record["reverified_at"] == "2026-11-18"
    by_slug = {p["slug"]: p for p in record["pages"]}
    assert set(by_slug) == {s["slug"] for s in sources}
    assert all(p["status"] == "live" for p in record["pages"])
    # 18 unchanged; the only change is Eiffel's sidebar teaser.
    changed = [p["slug"] for p in record["pages"] if p["content"] != "unchanged"]
    assert changed == ["eiffel"]
    assert "MOPGA" in by_slug["eiffel"]["deadline_stipend"]
    # The re-verification record names its method and its no-rewrite rule.
    assert "No raw HTML" in record["method"]


def test_no_secret_pattern_in_stored_text():
    """Grep-count scan only: 0 matches required, values never printed."""
    counts = {}
    for txt in sorted(RAW_DIR.glob("*.txt")):
        counts[txt.stem] = sum(len(p.findall(txt.read_text()))
                               for p in SECRET_PATTERNS)
    assert counts == {k: 0 for k in counts}, counts


# --- checker currency gap: fixed ------------------------------------------

def test_checker_matches_pound_sek_yen_and_detached_dollar():
    """Step 7 fix. Unit-level: the pattern matches the amounts the
    old €/$-only pattern missed; integration-level: the checker no
    longer flags those pages as missing a stipend."""
    assert MONEY_RE.search("£21,060 for 2026-27")
    assert MONEY_RE.search("£1,712 per month")
    assert MONEY_RE.search("SEK 12,000")
    assert MONEY_RE.search("¥117,000 per month")
    assert MONEY_RE.search("Full-time base RTP stipend rate ($) 35,550")
    # A trailing comma in running text ("up to £4,000, dependent")
    # is not part of the amount.
    assert MONEY_RE.search("up to £4,000, dependent").group(0) == "£4,000"
    report = check(_chunks(), load_sources())
    missing = report["missing_fields"]
    for slug in ("rhodes", "commonwealth-shared", "rtp",
                 "si-pioneering-women", "sweden-si", "mext"):
        assert "stipend" not in missing.get(slug, []), slug
    # Gates was never flagged, but for the wrong reason (its US$210m
    # donation matched); it is now also matched on its real stipend.
    assert "stipend" not in missing.get("gates-cambridge", [])
    assert "£23,152" in (RAW_DIR / "gates-cambridge.txt").read_text()
    # Pages that state no amount are still flagged — the fix did not
    # make the checker indiscriminate.
    for slug in ("chevening", "daad", "erasmus-mundus", "fulbright",
                 "knight-hennessy", "manaaki", "marshall", "schwarzman"):
        assert "stipend" in missing[slug], slug


def test_new_multiple_amounts_findings_are_resolved_with_fields():
    """The £/SEK/¥ fix surfaces a second amount on seven more pages;
    each is resolved by naming the field each amount belongs to —
    the checker must not trade six false 'missing stipend' flags
    for seven unexplained 'multiple amounts' flags."""
    report = check(_chunks(), load_sources())
    multi = {f["doc"]: f for f in report["findings"]
             if f["kind"] == "multiple_amounts"}
    for slug in ("gates-cambridge", "rhodes", "commonwealth-shared",
                 "rtp", "sweden-si", "si-pioneering-women", "mext"):
        assert multi[slug]["status"] == "resolved", slug
        assert multi[slug]["resolution"], slug
    assert "travel grant" in multi["sweden-si"]["resolution"]
    assert "travel grant" in multi["si-pioneering-women"]["resolution"]
    assert "33%" in multi["rhodes"]["resolution"]
    assert "base" in multi["rtp"]["resolution"]
    assert "donation" in multi["gates-cambridge"]["resolution"]
    assert "no single" in multi["mext"]["resolution"]
    # Step 7 correction: the HKPFS resolution now also names the
    # travel allowance, not only the stipend in two currencies.
    assert "travel allowance" in multi["hkpfs"]["resolution"]


# --- eval still passes after the re-fetch updates --------------------------

def test_eval_unchanged_after_reverification():
    """No .txt changed in Step 7, so the re-run eval matched the saved
    before-run file question by question — no golden question was
    silently changed, dropped or lost. The Phase 2 improvement then
    changed exactly two rows on purpose (q03, fixed by the
    bullet-list split; q32, fixed by IDF sentence scoring), so this
    test now pins that: every other row is identical to the Step 7
    before-run file, and answer-only relevance is 0.953 (41/43),
    not 0.907 (39/43)."""
    before = json.loads(
        (RESULTS_DIR / "extractive_eval_before_reverification.json").read_text())
    after = json.loads((RESULTS_DIR / "extractive_eval.json").read_text())
    assert after["n_in_corpus"] == before["n_in_corpus"] == 43
    assert after["n_out_of_corpus"] == before["n_out_of_corpus"] == 7
    assert after["n_eligibility"] == before["n_eligibility"] == 11
    for metric in ("recall@3", "recall@5", "answer_relevance_proxy",
                   "faithfulness",
                   "abstention_accuracy", "eligibility_accuracy"):
        assert after[metric] == before[metric], metric
    assert before["answer_only_relevance"] == 0.907
    assert after["answer_only_relevance"] == 0.953
    before_rows = {r["id"]: r for r in before["per_question"]}
    after_rows = {r["id"]: r for r in after["per_question"]}
    assert set(after_rows) == set(before_rows)
    for qid, row in before_rows.items():
        if qid in ("q03", "q32"):
            # Fixed by the Phase 2 improvement (bullet-list split
            # for q03, IDF sentence scoring for q32).
            assert row["answer_only_relevant"] is False
            assert after_rows[qid]["answer_only_relevant"] is True
            assert row["found"] is True
            assert after_rows[qid]["found"] is True
            continue
        assert after_rows[qid]["answer_only_relevant"] == row["answer_only_relevant"], qid
        assert after_rows[qid]["found"] == row["found"], qid
