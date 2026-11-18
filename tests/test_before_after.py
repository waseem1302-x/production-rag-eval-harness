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
    # Improved column: 30 of 32 in-corpus correct (misses q10,
    # q17 only), 7 of 7 abstained.
    assert sum(1 for l in in_corpus_rows if l.split("|")[4].strip() == "✓") == 30
    assert sum(1 for l in out_rows if "✓ abstained" in l) == 7


def test_still_failing_questions_are_paraphrase_gaps():
    text = _report()
    # q10 and q17: both pipelines miss them, paraphrase gap.
    for qid in ("q10", "q17"):
        row = next(l for l in text.splitlines() if l.startswith(f"| {qid} |"))
        assert "✗ | ✗" in row, qid
        assert "paraphrase gap" in row, qid
    # q03: fixed by the Phase 2 improvement (bullet-list
    # splitting) — it now passes in both columns, so it is no
    # longer a REGRESSION row at all.
    q03 = next(l for l in text.splitlines() if l.startswith("| q03 |"))
    assert "✓ | ✓" in q03
    assert "REGRESSION" not in q03
    # q32: fixed by the Phase 2 improvement (IDF sentence
    # scoring) — it now passes in the improved column, so it is
    # no longer a REGRESSION row at all.
    q32 = next(l for l in text.splitlines() if l.startswith("| q32 |"))
    assert "✓ | ✓" in q32
    assert "REGRESSION" not in q32
    # o04: fixed in Step 2, still abstained (the audit was stale at
    # Step 5 and said "answered"; it was regenerated in Step 6).
    o04 = next(l for l in text.splitlines() if l.startswith("| o04 |"))
    assert "✓ abstained" in o04
    assert "unanswered_ranking" in o04


def test_no_regressions_in_failure_table():
    """No question the improved pipeline got right may be lost — a
    regression in the improved column would be a red flag, not a
    footnote. There is no (baseline ✓, improved ✗) row left: q32
    was the only one, a pre-existing improved-pipeline miss
    (bullet-list scoring), and the Phase 2 improvement (IDF
    sentence scoring) fixed it, so the regressions list is empty.
    q03 was the second pinned row until the Phase 2 improvement
    (bullet-list splitting) fixed it."""
    text = _report()
    regressions = []
    for line in text.splitlines():
        if re.match(r"\| q\d+ \|", line):
            cells = [c.strip() for c in line.split("|")]
            if cells[3] == "✓" and cells[4] == "✗":
                regressions.append(cells[1])
    assert regressions == [], regressions
