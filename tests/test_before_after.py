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
    assert len(in_corpus_rows) == 23
    assert len(out_rows) == 5
    # Improved column: 20 of 23 in-corpus correct, 4 of 5 abstained.
    assert sum(1 for l in in_corpus_rows if l.split("|")[4].strip() == "✓") == 20
    assert sum(1 for l in out_rows if "✓ abstained" in l) == 4


def test_still_failing_questions_are_paraphrase_gaps():
    text = _report()
    for qid in ("q03", "q10", "q17"):
        row = next(l for l in text.splitlines() if l.startswith(f"| {qid} |"))
        assert "✗ | ✗" in row, qid
        assert "paraphrase gap" in row, qid
    o04 = next(l for l in text.splitlines() if l.startswith("| o04 |"))
    assert "✗ answered" in o04


def test_no_regressions_in_failure_table():
    """No question the baseline got right may be lost by the improved
    pipeline — a regression would be a red flag, not a footnote."""
    text = _report()
    for line in text.splitlines():
        if re.match(r"\| q\d+ \|", line):
            cells = [c.strip() for c in line.split("|")]
            assert not (cells[3] == "✓" and cells[4] == "✗"), line
            assert "REGRESSION" not in line, line
