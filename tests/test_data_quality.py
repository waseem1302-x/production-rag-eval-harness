"""Tests for the data-quality checker. Run: python -m pytest tests/test_data_quality.py -v

The checker must find the known messy items in the real corpus and
must not flag the false positive that was caught during
development (the SI page pointing at another site's database).
"""
import json
from pathlib import Path

from app.config import CHUNKS_FILE
from app.ingest import load_sources
from scripts.check_data_quality import check


def _report():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    return check(chunks, load_sources())


def _findings(report, kind, doc=None):
    return [f for f in report["findings"]
            if f["kind"] == kind and (doc is None or f["doc"] == doc)]


def test_vanier_closure_found_with_evidence():
    report = _report()
    found = _findings(report, "closure", "vanier")
    assert len(found) == 2
    assert all(f["severity"] == "high" for f in found)
    assert any("no longer accepting applications" in f["sentence"] for f in found)
    assert any("new harmonized program" in f["sentence"] for f in found)
    assert all(f["status"] == "resolved" for f in found)
    assert all(f["chunk_id"] == "vanier-0" for f in found)


def test_daad_database_page_found():
    report = _report()
    found = _findings(report, "database_page", "daad")
    assert len(found) == 4
    assert any("147 Scholarship options" in f["sentence"] for f in found)
    assert all(f["status"] == "resolved" for f in found)


def test_no_false_positive_for_si_database_mention():
    """The SI page mentions 'Study in Sweden's scholarship database'
    as somewhere else to look. That is not the SI page being a
    database; the checker must not flag it."""
    report = _report()
    assert _findings(report, "database_page", "sweden-si") == []


def test_eiffel_two_amounts_flagged_and_resolved():
    report = _report()
    found = _findings(report, "multiple_amounts", "eiffel")
    assert len(found) == 1
    assert "€1,200" in found[0]["note"] and "€2,000" in found[0]["note"]
    assert found[0]["status"] == "resolved"
    assert "Master's level" in found[0]["resolution"]


def test_hkpfs_amounts_are_one_figure_two_currencies():
    report = _report()
    found = _findings(report, "multiple_amounts", "hkpfs")
    assert len(found) == 1
    assert found[0]["status"] == "resolved"
    assert "two currencies" in found[0]["resolution"]


def test_approximate_and_year_specific_figures():
    report = _report()
    fulbright = _findings(report, "approximate_figure", "fulbright")
    assert any("Approximately 4,000" in f["sentence"] for f in fulbright)
    hkpfs_year = _findings(report, "year_specific_figure", "hkpfs")
    assert any("2026/27" in f["sentence"] for f in hkpfs_year)
    assert any("2027/28" in f["sentence"] for f in hkpfs_year)


def test_stale_page_markers():
    report = _report()
    stale = _findings(report, "stale_page_marker")
    by_doc: dict[str, list[str]] = {}
    for f in stale:
        by_doc.setdefault(f["doc"], []).append(f["sentence"])
    assert any("26 June 2025" in s for s in by_doc["erasmus-mundus"])
    assert any("October 2022" in s for s in by_doc["eiffel"])
    assert any("December 2025" in s for s in by_doc["eiffel"])


def test_missing_fields_matrix():
    report = _report()
    missing = report["missing_fields"]
    assert missing["chevening"] == ["deadline", "stipend", "citizenship_rule"]
    # Phase 2, Step 7: Sweden SI no longer appears at all — the
    # £/SEK fix finds its SEK 12,000/month allowance, so it states
    # a stipend, and it already stated a deadline.
    assert "sweden-si" not in missing
    # Phase 2 improvement, Step 5: Vanier no longer appears at all
    # — it is closed, so a missing-deadline flag is noise (the
    # closure finding already records that nobody can apply), and
    # it states a stipend, so no field is missing.
    assert "vanier" not in missing
    assert "deadline" in missing["erasmus-mundus"]
    # Phase 2, Step 7: MEXT states yen amounts (¥117,000–¥242,000
    # per month, per category), so "stipend" is no longer missing —
    # only the deadline and the (unauthored) citizenship rule are.
    assert missing["mext"] == ["deadline", "citizenship_rule"]
    assert "degree_levels" in missing["fulbright"]
    assert "degree_levels" in missing["daad"]
    # Phase 2, Step 6: the 9 new documents. Two systematic reasons,
    # not nine missing facts: (a) no eligibility RULES are authored
    # for the new docs yet, so "degree_levels" and "citizenship_rule"
    # are flagged for all nine by construction — several pages do
    # state them (e.g. si-pioneering-women lists its 10 eligible
    # countries); (b) before Step 7 the checker's MONEY_RE matched
    # €/$ only, not £/SEK, so "stipend" was flagged even where the
    # page stated an amount in pounds or kronor. Step 7 fixed (b):
    # Rhodes, Commonwealth Shared, RTP and SI Pioneering no longer
    # flag "stipend" (see
    # tests/test_reverification.py::test_checker_matches_pound_sek_yen_and_detached_dollar).
    # Gates was never flagged — before Step 7 because its "US$210m"
    # donation matched, now also because its £23,152 stipend does.
    # Phase 2 improvement, Step 4 fixed (a) for 8 of the 9 docs by
    # authoring rules from the corpus (Schwarzman states no rule, so
    # it still flags both): Gates, RTP and Erasmus Mundus org now
    # flag only "deadline"; Rhodes only "deadline" and
    # "citizenship_rule" (constituency-based, no country list in the
    # corpus); Knight-Hennessy "deadline", "stipend" and
    # "citizenship_rule" (no citizenship rule stated); Marshall only
    # "stipend"; Commonwealth Shared and SI Pioneering flag nothing.
    assert missing["gates-cambridge"] == ["deadline"]
    assert missing["rhodes"] == ["deadline", "citizenship_rule"]
    assert missing["knight-hennessy"] == ["deadline", "stipend", "citizenship_rule"]
    # Phase 2 improvement, Step 5: DATE_RE now matches US-format
    # dates ("May 20"), so Schwarzman's China-track application
    # deadline is found and "deadline" is no longer flagged.
    assert missing["schwarzman"] == ["stipend", "degree_levels", "citizenship_rule"]
    # Commonwealth Shared and SI Pioneering flag nothing, so the
    # checker omits them from the missing-fields map entirely.
    assert missing.get("commonwealth-shared", []) == []
    assert missing["rtp"] == ["deadline"]
    assert missing["marshall"] == ["stipend"]
    assert missing["erasmus-mundus-org"] == ["deadline"]
    assert missing.get("si-pioneering-women", []) == []


def test_erasmus_org_two_amounts_are_different_fields():
    """Phase 2, Step 6: the EMJM organisations page lists €5 million
    (max grant per project) and €1,400/month (scholarship per
    individual) — different fields, not a contradiction."""
    report = _report()
    found = _findings(report, "multiple_amounts", "erasmus-mundus-org")
    assert len(found) == 1
    assert found[0]["status"] == "resolved"
    assert "different fields" in found[0]["resolution"]
    assert "€1,400 per month" in found[0]["resolution_evidence"]


def test_report_files_are_written_and_valid():
    results = Path("benchmarks/results")
    data = json.loads((results / "data_quality.json").read_text())
    # Phase 2, Step 7: 72 -> 73 findings (16 resolved, 57 open).
    # Six false "missing stipend" flags are gone (Rhodes, Commonwealth
    # Shared, RTP, SI Pioneering, Sweden SI, MEXT now match £/SEK/¥
    # or detached "($)"), and seven new multiple-amounts findings
    # appear, all resolved with their field named.
    # Phase 2 improvement, Step 4: 73 -> 58 findings (16 resolved,
    # 42 open). Authoring eligibility rules for 8 of the 9 Step 6
    # docs resolves 14 "missing_field" findings (the degree-level
    # and citizenship-rule flags for those docs), and the Step 1
    # bullet-list split removes 1 false "approximate_figure" finding
    # for Schwarzman (the merged sentence had attached
    # "approximately" to "25 minutes"); the 2 multiple-amounts
    # findings for Commonwealth Shared and SI Pioneering are
    # re-cited with split sentences, same status and note.
    # Phase 2 improvement, Step 5: 58 -> 56 findings (16 resolved,
    # 40 open). DATE_RE now matches US-format dates, so
    # Schwarzman's "May 20" deadline is found; and closed
    # programmes skip the deadline check, so Vanier no longer gets
    # a missing-deadline flag.
    assert len(data["findings"]) == 56
    assert sum(1 for f in data["findings"] if f["status"] == "resolved") == 16
    assert data["missing_fields"]["chevening"] == ["deadline", "stipend", "citizenship_rule"]
    assert (results / "data_quality.md").read_text().startswith("# Data quality report")
