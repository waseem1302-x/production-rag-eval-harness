"""Bullet-list sentence splitting (Phase 2 improvement, Step 1).

The SI Pioneering Women page lists what the scholarship includes as
dash-led bullet items on lines that the normaliser joins into one
line, so the sentence splitter glued three items into one long
"sentence":

  "The fee is paid directly by SI to your Swedish university each
  semester. - A monthly allowance of SEK 12,000 to cover your living
  expenses throughout the study period. - A travel grant of SEK 15,000."

That glued sentence shares only "monthly" and "allowance" with the
q32 question, so it ties with the page intro on lexical score and
loses on document order. Splitting dash-led items into their own
sentences lets the SEK 12,000 sentence be picked on its own.
Each split piece is still a verbatim substring of the chunk text,
so faithfulness stays measurable.
"""
import json

from app.config import CHUNKS_FILE
from app.textnorm import norm_text, segment_sentences

GLUED = (
    "The fee is paid directly by SI to your Swedish university each "
    "semester. - A monthly allowance of SEK 12,000 to cover your living "
    "expenses throughout the study period. - A travel grant of SEK 15,000."
)


def test_bullets_split_after_terminal_punctuation():
    parts = segment_sentences(GLUED)
    assert parts == [
        "The fee is paid directly by SI to your Swedish university each semester.",
        "A monthly allowance of SEK 12,000 to cover your living expenses throughout the study period.",
        "A travel grant of SEK 15,000.",
    ]


def test_bullets_split_after_colon_intro():
    parts = segment_sentences(
        "The SI scholarship includes: - Full tuition fee coverage. "
        "- A monthly allowance of SEK 12,000.")
    assert parts == [
        "The SI scholarship includes:",
        "Full tuition fee coverage.",
        "A monthly allowance of SEK 12,000.",
    ]


def test_non_bullet_dashes_untouched():
    text = "Apply between 8-24 February 2027. The portal closes at 14:59 CET."
    assert segment_sentences(text) == [
        "Apply between 8-24 February 2027.",
        "The portal closes at 14:59 CET.",
    ]


def test_q32_sentence_is_its_own_pickable_sentence():
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines()
              if line.strip()]
    chunk = next(c for c in chunks if c["chunk_id"] == "si-pioneering-women-1")
    sentences = segment_sentences(chunk["text"])
    sek = [s for s in sentences if "SEK 12,000" in s]
    assert len(sek) == 1
    assert sek[0] == ("A monthly allowance of SEK 12,000 to cover your living "
                      "expenses throughout the study period.")
    # Still verbatim: the split sentence is a substring of the chunk.
    assert norm_text(sek[0]) in norm_text(chunk["text"])
