"""Tests for the before/after report. Run: python -m pytest tests/test_before_after.py -v

The report is generated from the measured results files, so these
tests check that the generated text cannot drift from the data:
the delta table must carry the benchmark's numbers, and the failure
tables must match the per-question results.
"""
import json
import re
from pathlib import Path

from benchmarks.before_after import main as generate

RESULTS_DIR = Path("benchmarks/results")


def _report() -> str:
    generate()
    return (RESULTS_DIR / "before_after.md").read_text()


def test_delta_table_matches_benchmark_results():
    text = _report()
    baseline = json.loads((RESULTS_DIR / "baseline-naive.json").read_text())
    improved = json.loads((RESULTS_DIR / "improved-hybrid.json").read_text())
    assert f"| Answer-only relevance | {baseline['quality']['answer_only_relevance']} | " \
           f"{improved['quality']['answer_only_relevance']} |" in text
    assert f"| Abstention accuracy | {baseline['quality']['abstention_accuracy']} | " \
           f"{improved['quality']['abstention_accuracy']} |" in text
    assert f"| recall@5 | {baseline['quality']['recall@5']} | " \
           f"{improved['quality']['recall@5']} |" in text


def test_failure_table_counts():
    text = _report()
    in_corpus_rows = [l for l in text.splitlines() if re.match(r"\| q\d+ \|", l)]
    out_rows = [l for l in text.splitlines() if re.match(r"\| o\d+ \|", l)]
    # Phase 2, Step 6: 32 factual in-corpus (q01-q32) + 7 out (o01-o07).
    assert len(in_corpus_rows) == 32
    assert len(out_rows) == 7
    # Improved column: 28 of 32 in-corpus correct (misses q03, q10,
    # q17, q32), 7 of 7 abstained.
    assert sum(1 for l in in_corpus_rows if l.split("|")[4].strip() == "✓") == 28
    assert sum(1 for l in out_rows if "✓ abstained" in l) == 7


def test_still_failing_questions_are_paraphrase_gaps():
    text = _report()
    # q10 and q17: both pipelines miss them, paraphrase gap.
    for qid in ("q10", "q17"):
        row = next(l for l in text.splitlines() if l.startswith(f"| {qid} |"))
        assert "✗ | ✗" in row, qid
        assert "paraphrase gap" in row, qid
    # q03: the paraphrase-gap miss the naive baseline now happens
    # to get on the grown corpus (its BM25 top chunk changed, so
    # first-sentence picking lands on the right sentence), while the
    # improved pipeline still misses it — same miss as Step 5, not
    # a new loss. The generator labels it REGRESSION; see
    # test_no_regressions_in_failure_table for why that label is a
    # baseline-side change.
    q03 = next(l for l in text.splitlines() if l.startswith("| q03 |"))
    assert "✓ | ✗" in q03
    assert "REGRESSION" in q03
    # o04: fixed in Step 2, still abstained (the audit was stale at
    # Step 5 and said "answered"; it was regenerated in Step 6).
    o04 = next(l for l in text.splitlines() if l.startswith("| o04 |"))
    assert "✓ abstained" in o04
    assert "unanswered_ranking" in o04


def test_no_regressions_in_failure_table():
    """No question the improved pipeline got right may be lost — a
    regression in the improved column would be a red flag, not a
    footnote. The only (baseline ✓, improved ✗) rows are exactly
    q03 and q32, and both are pre-existing improved-pipeline
    misses (q03 paraphrase gap, q32 bullet-list merging),
    unchanged since Step 5 in the improved column: what changed on
    corpus growth is the BASELINE column — with 101 chunks the
    baseline's BM25 top chunk changed, so its first-sentence
    picking now lands on the right sentence for those two. A new
    (✓, ✗) row, or a REGRESSION label anywhere else, fails."""
    text = _report()
    regressions = []
    for line in text.splitlines():
        if re.match(r"\| q\d+ \|", line):
            cells = [c.strip() for c in line.split("|")]
            if cells[3] == "✓" and cells[4] == "✗":
                regressions.append(cells[1])
    assert regressions == ["q03", "q32"], regressions
