"""IDF-weighted sentence scoring (Phase 2 improvement, Step 2).

The q32 question shares only two stems ("monthly", "allowance")
with its correct sentence, while the competing eligibility sentence
shares seven, mostly stopwords and doc-name stems. Plain stem
counts therefore rank the wrong sentence first. Weighting each
shared stem by its inverse document frequency over the candidate
sentences makes the rare content stems outweigh the ubiquitous
ones. These tests use no model (dense=False) so they run fast and
deterministically.
"""
import json
from pathlib import Path

from app.answer import (
    _idf_weights,
    _pick_sentences,
    _specific_terms,
    _stems,
)
from app.config import CHUNKS_FILE

REPO = Path(__file__).resolve().parent.parent
Q32 = "What is the monthly allowance for a Pioneering Women in STEM scholar?"
SEK_SENTENCE = (
    "A monthly allowance of SEK 12,000 to cover your living expenses "
    "throughout the study period."
)
APPLY_SENTENCE = (
    "Apply to a STEM-related master's programme eligible for the "
    "Pioneering Women in STEM scholarship."
)


def _q32_chunks():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    wanted = {"si-pioneering-women-1", "si-pioneering-women-0",
              "si-pioneering-women-3"}
    return [c for c in chunks if c["chunk_id"] in wanted]


def test_idf_ranks_sek_sentence_first_for_q32():
    """On the real top-3 chunks for q32, the SEK 12,000 sentence is
    picked first. Under plain stem counts the eligibility sentence
    wins 11 to 8; with IDF weights the SEK sentence wins."""
    picked = _pick_sentences(
        _q32_chunks(), _stems(Q32), Q32, _specific_terms(Q32),
        dense=False)
    assert picked[0] == SEK_SENTENCE
    assert APPLY_SENTENCE in picked


def test_idf_weights_are_population_relative():
    """A stem in every candidate sentence weighs 1.0; a stem in
    exactly one weighs more. The weight is ln((N+1)/(df+1)) + 1."""
    import math

    sentences = [
        "The monthly allowance is paid.",
        "The monthly report is here.",
        "The monthly update is here.",
    ]
    weights = _idf_weights(sentences)
    # "the" and "monthly" are in all 3 sentences: ln(4/4) + 1 = 1.0.
    assert weights["the"] == 1.0
    assert weights["monthli"] == 1.0
    # "allowance" is in 1 of 3: ln(4/2) + 1.
    assert abs(weights["allowanc"] - (math.log(4 / 2) + 1)) < 1e-9
    assert weights["allowanc"] > weights["monthli"]
