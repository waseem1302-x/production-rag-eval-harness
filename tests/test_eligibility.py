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
    ({"degree_level": "masters"}, "vanier", "not_eligible"),
    ({"degree_level": "phd"}, "vanier", "eligible"),
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
    ({"nationality": "Canada", "degree_level": "phd"}, "vanier", "eligible"),
    ({"nationality": "Pakistan", "degree_level": "phd"}, "vanier", "eligible"),
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
