"""Tests for eligibility-first answering. Run: python -m pytest tests/test_eligibility.py -v

Every verdict here is grounded in a verbatim corpus sentence; the
checker itself validates each rule's evidence against the chunk
text at load time and drops any rule whose evidence is missing.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.config import CHUNKS_FILE
from app.eligibility import EligibilityChecker, normalize_level

CHUNKS = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def checker():
    return EligibilityChecker(CHUNKS)


def test_all_rules_have_verbatim_evidence(checker):
    """No rule survives load unless its evidence sentence is found
    verbatim in the chunk it cites. A dropped rule becomes unknown,
    never a silent wrong verdict."""
    for doc, rule in checker.rules.items():
        for field, ev in rule["evidence"].items():
            chunk = checker.chunks[ev["chunk_id"]]
            assert chunk["doc"] == doc
            from app.textnorm import norm_text
            assert norm_text(ev["sentence"]) in norm_text(chunk["text"]), (doc, field)


def test_normalize_level():
    assert normalize_level("PhD") == "phd"
    assert normalize_level("doctoral") == "phd"
    assert normalize_level("Master's") == "masters"
    assert normalize_level("bachelor's") == "undergraduate"
    assert normalize_level("postgraduate") == "postgraduate"
    assert normalize_level(None) is None


# (profile, doc, expected verdict)
CASES = [
    # Degree level
    ({"degree_level": "phd"}, "chevening", "not_eligible"),
    ({"degree_level": "masters"}, "chevening", "eligible"),
    ({"degree_level": "masters"}, "vanier", "closed"),
    ({"degree_level": "phd"}, "vanier", "closed"),
    ({"degree_level": "masters"}, "hkpfs", "not_eligible"),
    ({"degree_level": "phd"}, "hkpfs", "eligible"),
    ({"degree_level": "undergraduate"}, "manaaki", "eligible"),
    ({"degree_level": "masters"}, "manaaki", "eligible"),
    ({"degree_level": "phd"}, "eiffel", "eligible"),
    ({"degree_level": "masters"}, "eiffel", "eligible"),
    ({"degree_level": "undergraduate"}, "sweden-si", "not_eligible"),
    ({"degree_level": "masters"}, "erasmus-mundus", "eligible"),
    ({"degree_level": "phd"}, "erasmus-mundus", "not_eligible"),
    # Citizenship — the SI list is in the corpus; Pakistan is not on it.
    ({"nationality": "Pakistan", "degree_level": "masters"}, "sweden-si", "not_eligible"),
    ({"nationality": "Bangladesh", "degree_level": "masters"}, "sweden-si", "eligible"),
    ({"nationality": "Malaysia", "degree_level": "masters"}, "sweden-si", "eligible"),
    ({"nationality": "Canada", "degree_level": "phd"}, "vanier", "closed"),
    ({"nationality": "Pakistan", "degree_level": "phd"}, "vanier", "closed"),
    ({"nationality": "Pakistan"}, "hkpfs", "eligible"),
    ({"nationality": "France", "degree_level": "phd"}, "eiffel", "not_eligible"),
    ({"nationality": "Pakistan", "degree_level": "phd"}, "eiffel", "eligible"),
    # Unknown is the safe default when the corpus states no rule.
    ({"degree_level": "masters"}, "fulbright", "unknown"),
    ({"nationality": "Pakistan"}, "fulbright", "unknown"),
    ({"nationality": "Pakistan"}, "mext", "unknown"),
    ({"degree_level": "masters"}, "daad", "unknown"),
    ({"nationality": "Pakistan"}, "daad", "eligible"),
    ({"nationality": "Germany"}, "daad", "not_eligible"),
]


@pytest.mark.parametrize("profile,doc,expected", CASES)
def test_verdicts(checker, profile, doc, expected):
    verdict = checker.check(profile, doc)
    assert verdict["verdict"] == expected, (profile, doc, verdict)


def test_verdict_carries_reason_and_evidence(checker):
    verdict = checker.check({"nationality": "Pakistan", "degree_level": "masters"}, "sweden-si")
    assert verdict["verdict"] == "not_eligible"
    assert any("34 eligible countries" in r for r in verdict["reasons"])
    assert verdict["evidence"], "a grounded verdict must cite its evidence"
    assert {ev["chunk_id"] for ev in verdict["evidence"]} == {"sweden-si-0", "sweden-si-2"}


def test_unknown_verdict_has_no_evidence(checker):
    verdict = checker.check({"degree_level": "masters"}, "fulbright")
    assert verdict["verdict"] == "unknown"
    assert verdict["evidence"] == []
    assert "does not state" in " ".join(verdict["reasons"])


# --- Programme closure (Phase 2, Step 4) -------------------------------

def test_all_closures_have_verbatim_evidence(checker):
    """Every closure entry survives load only if each evidence
    sentence is found verbatim in the chunk it cites — the same
    rule as the eligibility rules."""
    from app.textnorm import norm_text
    for doc, closure in checker.closures.items():
        assert closure["replacement"], f"{doc}: closure must name its replacement"
        for ev in closure["evidence"]:
            chunk = checker.chunks[ev["chunk_id"]]
            assert chunk["doc"] == doc
            assert norm_text(ev["sentence"]) in norm_text(chunk["text"]), (doc, ev)


def test_closed_verdict_for_doctoral_applicant(checker):
    """The known gap: a doctoral applicant who meets every Vanier
    rule must be told the programme is closed, not 'doctoral
    only' / eligible."""
    verdict = checker.check({"nationality": "Canada", "degree_level": "phd"}, "vanier")
    assert verdict["verdict"] == "closed"
    assert any("closed" in r.lower() for r in verdict["reasons"])
    assert any("Canada Graduate Research Scholarship - Doctoral program" in r
               for r in verdict["reasons"]), "answer must name what replaced it"
    assert {ev["chunk_id"] for ev in verdict["evidence"]} == {"vanier-0"}
    assert any("no longer accepting applications" in ev["sentence"]
               for ev in verdict["evidence"])


def test_closed_verdict_for_other_degree_levels(checker):
    """Closure does not depend on the degree level: a master's or
    undergraduate applicant gets the same closed verdict, not a
    degree-level rejection."""
    for level in ("masters", "undergraduate", "bachelor"):
        verdict = checker.check({"degree_level": level}, "vanier")
        assert verdict["verdict"] == "closed", (level, verdict)
        assert "doctoral only" not in " ".join(verdict["reasons"]).lower()


def test_closed_verdict_with_nationality_only(checker):
    """Closure is a property of the programme, not of the profile:
    it fires even when the profile carries no degree level."""
    verdict = checker.check({"nationality": "Pakistan"}, "vanier")
    assert verdict["verdict"] == "closed"


def test_open_programmes_unaffected(checker):
    """No other programme is flagged closed: the open verdicts are
    exactly what they were before this step."""
    assert checker.check({"degree_level": "phd"}, "hkpfs")["verdict"] == "eligible"
    assert checker.check({"degree_level": "masters"}, "chevening")["verdict"] == "eligible"
    assert checker.check({"degree_level": "masters"}, "fulbright")["verdict"] == "unknown"
    assert set(checker.closures) == {"vanier"}


def test_closure_entry_dropped_when_evidence_missing():
    """Verbatim validation: if a re-ingest removed the closure
    sentence from the chunk, the entry is dropped (with a warning)
    and the programme falls back to its rules instead of citing a
    sentence that is no longer there."""
    mutated = []
    for chunk in CHUNKS:
        if chunk["chunk_id"] == "vanier-0":
            chunk = dict(chunk, text=chunk["text"].replace(
                "We are no longer accepting applications for the Vanier "
                "Canada Graduate Scholarships.", "").replace(
                "Please refer to the Canada Graduate Research Scholarship - "
                "Doctoral program web page for information regarding the new "
                "harmonized program.", ""))
        mutated.append(chunk)
    with pytest.warns(UserWarning, match="closure"):
        checker = EligibilityChecker(mutated)
    assert "vanier" not in checker.closures
    # Falls back to the pre-closure behaviour: doctoral is eligible.
    assert checker.check({"degree_level": "phd"}, "vanier")["verdict"] == "eligible"


def test_closure_table_matches_data_quality_checker(checker):
    """The flag file and the data-quality checker's closure finding
    must not drift apart: every closure evidence sentence is part
    of the checker's recorded resolution evidence for that doc."""
    from scripts.check_data_quality import RESOLUTIONS
    for doc, closure in checker.closures.items():
        resolution = RESOLUTIONS.get((doc, "closure"))
        assert resolution is not None, f"{doc}: no closure resolution in the checker"
        for ev in closure["evidence"]:
            assert ev["sentence"] in resolution["evidence"], (doc, ev)


def test_api_closed_verdict_with_citation():
    """A Vanier doctoral applicant gets the closed answer through
    the API, with the closure cited — not the $50,000 fact answer."""
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={
        "question": "I am a Canadian citizen starting a PhD. Can I apply for the Vanier Scholarship?",
        "profile": {"nationality": "Canada", "degree_level": "phd"}})
    body = resp.json()
    assert resp.status_code == 200
    assert body["found"] is True
    assert "Closed" in body["answer"]
    assert "no longer accepting applications" in body["answer"]
    assert "Canada Graduate Research Scholarship - Doctoral program" in body["answer"]
    vanier = next(v for v in body["eligibility"] if v["doc"] == "vanier")
    assert vanier["verdict"] == "closed"
    assert vanier["evidence"], "closed verdict must cite its evidence"
    assert body["citations"], "closed answer must carry citations"
    assert any(c["chunk_id"] == "vanier-0" for c in body["citations"])


def test_api_substantive_vanier_question_with_profile_is_closed():
    """Eligibility first, closure version: a PhD applicant asking
    what Vanier is worth is told it is closed, not given the
    $50,000 stipend as if they could still apply for it."""
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={
        "question": "How much is the Vanier Canada Graduate Scholarship worth per year?",
        "profile": {"nationality": "Pakistan", "degree_level": "phd"}})
    body = resp.json()
    assert "Closed" in body["answer"]
    assert "no longer accepting applications" in body["answer"]


def test_api_profile_returns_eligibility_block():
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={
        "question": "I am from Pakistan. Can I apply for the SI Scholarship for Global Professionals?",
        "profile": {"nationality": "Pakistan", "degree_level": "masters"}})
    body = resp.json()
    assert resp.status_code == 200
    assert body["found"] is True
    assert "Not eligible" in body["answer"]
    assert "34 eligible countries" in body["answer"]
    si = next(v for v in body["eligibility"] if v["doc"] == "sweden-si")
    assert si["verdict"] == "not_eligible"
    assert si["evidence"], "eligibility answer must cite evidence"
    assert body["citations"], "eligibility answer must carry citations"


def test_api_substantive_question_with_incompatible_profile():
    """Eligibility first: a PhD applicant asking what Chevening
    funds is told they are not eligible, not given the fact answer."""
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={
        "question": "What does the Chevening Scholarship fund?",
        "profile": {"nationality": "Pakistan", "degree_level": "phd"}})
    body = resp.json()
    assert "Not eligible" in body["answer"]
    assert "one-year master's degrees" in body["answer"]


def test_api_substantive_question_with_compatible_profile():
    """An eligible profile gets the normal answer plus the
    eligibility block as supplementary information."""
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={
        "question": "What does the Chevening Scholarship fund?",
        "profile": {"nationality": "Pakistan", "degree_level": "masters"}})
    body = resp.json()
    assert "fully-funded" in body["answer"]
    chev = next(v for v in body["eligibility"] if v["doc"] == "chevening")
    assert chev["verdict"] == "eligible"


def test_api_without_profile_has_no_eligibility_block():
    client = TestClient(api_module.app)
    resp = client.post("/ask", json={"question": "What does the Chevening Scholarship fund?"})
    assert "eligibility" not in resp.json()
