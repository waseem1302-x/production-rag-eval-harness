"""Query expansion (Phase 2 improvement, query expansion).

q10 and q17 are the last two extractive misses, both paraphrase
gaps: q10's correct sentence shares only "master" with its
question, q17's only "scholarship"/"study". The corpus states
durations as "duration"/"years" and study levels as
"undergraduate"/"postgraduate"/"doctoral"/"masters", never in the
question's own words, so a small explicit synonym map expands the
query stems before sentence picking. These tests use no model
(dense=False) so they run fast and deterministically.
"""
import json
from pathlib import Path

from app.answer import (
    _pick_sentences,
    _query_terms,
    _specific_terms,
    _stem,
    _stems,
    expand_query_stems,
    expansion_stems,
)
from app.config import CHUNKS_FILE

REPO = Path(__file__).resolve().parent.parent

Q10 = "How long does an Erasmus Mundus Joint Master last?"
Q10_SENTENCE = (
    "Masters duration 1 to 2 academic years (60, 90 or 120 ECTS credits )."
)
Q17 = "What levels of study do Manaaki New Zealand Scholarships cover?"
Q17_SENTENCE = "Scholarships for undergraduate and postgraduate study."


def _chunks_for(doc: str):
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    return [c for c in chunks if c["doc"] == doc]


def test_expand_long_adds_duration_stems():
    """'long' expands to duration/years/months, stemmed."""
    expanded = expand_query_stems({_stem("long")})
    assert _stem("duration") in expanded
    assert _stem("years") in expanded
    assert _stem("months") in expanded
    assert _stem("long") in expanded


def test_expand_last_adds_duration_stems():
    """'last' (as in 'how long does X last') expands to duration/years."""
    expanded = expand_query_stems({_stem("last")})
    assert _stem("duration") in expanded
    assert _stem("years") in expanded
    assert _stem("months") not in expanded


def test_expand_level_adds_level_names():
    """'level' expands to the four level names the corpus uses."""
    expanded = expand_query_stems({_stem("level")})
    for word in ("undergraduate", "postgraduate", "doctoral", "masters"):
        assert _stem(word) in expanded


def test_expand_unknown_stems_unchanged():
    """Stems with no trigger come back unchanged."""
    stems = {_stem("deadline"), _stem("chevening")}
    assert expand_query_stems(stems) == stems


def test_expand_does_not_mutate_input():
    """expand_query_stems returns a new set."""
    stems = {_stem("long")}
    expanded = expand_query_stems(stems)
    assert stems == {_stem("long")}
    assert expanded != stems


def test_expansion_stems_returns_only_added_stems():
    """expansion_stems is the expanded set minus the question's
    own stems — the bonus applies to the added stems only."""
    base = {_stem("long"), _stem("master")}
    added = expansion_stems(base)
    assert _stem("duration") in added
    assert _stem("long") not in added
    assert _stem("master") not in added


def test_expansion_bonus_counts_double():
    """An expansion stem counts double, like a specific stem:
    expansion alone does not fix q10 (the duration sentence
    scores 6.95 and ranks 7th), with the bonus it scores 17.66
    and ranks 1st on the real top-3 chunks (dense=False)."""
    chunks = _chunks_for("erasmus-mundus")
    base = {_stem(t) for t in _query_terms(Q10)}
    picked = _pick_sentences(
        chunks, expand_query_stems(base), Q10, _specific_terms(Q10),
        expansion_stems(base), dense=False)
    assert picked[0] == Q10_SENTENCE


def test_expansion_picks_q10_duration_sentence():
    """On the real erasmus-mundus chunks, the '1 to 2 academic
    years' sentence is picked once the query is expanded. Without
    expansion it shares only 'master' and loses to intro sentences
    that share four doc-name stems."""
    chunks = _chunks_for("erasmus-mundus")
    base = {_stem(t) for t in _query_terms(Q10)}
    picked_without = _pick_sentences(
        chunks, base, Q10, _specific_terms(Q10), dense=False)
    assert Q10_SENTENCE not in picked_without
    picked = _pick_sentences(
        chunks, expand_query_stems(base), Q10, _specific_terms(Q10),
        expansion_stems(base), dense=False)
    assert Q10_SENTENCE in picked


def test_expansion_picks_q17_levels_sentence():
    """On the real manaaki chunks, the 'undergraduate and
    postgraduate study' sentence is picked once 'level' expands to
    the level names. Without expansion it shares only
    'scholarship'/'study' and loses to sentences sharing the
    four doc-name stems."""
    chunks = _chunks_for("manaaki")
    base = {_stem(t) for t in _query_terms(Q17)}
    picked_without = _pick_sentences(
        chunks, base, Q17, _specific_terms(Q17), dense=False)
    assert Q17_SENTENCE not in picked_without
    picked = _pick_sentences(
        chunks, expand_query_stems(base), Q17, _specific_terms(Q17),
        expansion_stems(base), dense=False)
    assert Q17_SENTENCE in picked
