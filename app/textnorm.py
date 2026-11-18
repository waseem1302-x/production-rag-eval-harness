"""Text normalisation for messy official pages.

Official scholarship pages are written for browsers, not for RAG.
Their extracted text has three recurring defects, all found in this
corpus:

1. Hard line breaks mid-sentence. "Masters duration" on one line and
   "1 to 2 academic years (60, 90 or 120 ECTS credits)." on the next
   are one sentence, but a naive splitter sees two fragments.
2. Sentences with no terminal punctuation. "147 Scholarship options
   for your selection" is the last line of the DAAD page and has no
   full stop, so a splitter that only accepts punctuated sentences
   can never pick it — even though it is the answer to "How many
   scholarship options does the DAAD database list?".
3. Header and nav fragments ("Home Page - Vanier Canada Graduate
   Scholarships", "Welcome to the home of Manaaki New Zealand
   scholarships.") mixed in with real content.

`normalize_document` fixes (1) and (2) by unwrapping lines, with one
strict rule: a line is merged into the previous sentence ONLY when
it is clearly a continuation — it starts lowercase, with a digit or
an opening bracket, the previous line ends with a comma/colon or an
unfinished word (a conjunction or preposition), or a bracket is still
open. A line starting with a capital letter starts a new sentence.
That distinction matters: the first version of this function merged
every line until a full stop, which glued page headers onto the
following sentence ("Home Page - Vanier Canada Graduate Scholarships
We are no longer accepting applications...") and those long
header-sentences then outscored the real answers. Headers stay
separate sentences now; they can still be picked, but only if they
genuinely share the most query terms.
The result is one sentence per line. Defect (3) is not removed —
deleting text is a data decision, not a formatting one — but the
sentence scorer in `app.answer` prefers sentences that share
specific query terms, so header fragments lose to real content.

Normalisation happens in `app.chunking.chunk_text`, not in
ingestion: the raw files in `data/raw/` stay verbatim (provenance),
and the processed chunks in `data/processed/chunks.jsonl` are the
normalised form. Every sentence the API returns is a substring of
a chunk, so faithfulness stays measurable.
"""
import re

_TERMINAL_RE = re.compile(r'[.!?]["\')\]]?\s*$')


def _continues(buffer: str, line: str) -> bool:
    """Is `line` a continuation of the sentence in `buffer`?

    True when the line starts lowercase, with a digit or an opening
    bracket; when the buffer ends with a comma, colon or an
    unfinished word (conjunction/preposition/article); or when a
    bracket opened in the buffer is still open. A line starting
    with a capital letter is a new sentence, not a continuation —
    that is what keeps page headers out of content sentences.
    """
    stripped = buffer.rstrip()
    if stripped.count("(") > stripped.count(")"):
        return True
    if stripped.count("[") > stripped.count("]"):
        return True
    if stripped and stripped[-1] in ",:;-—(":
        return True
    last_word = stripped.rsplit(None, 1)[-1].lower().strip(",;:") if stripped else ""
    if last_word in {"and", "or", "the", "a", "an", "of", "to", "in",
                     "for", "with", "by", "at", "on", "as"}:
        return True
    first = line[0] if line else ""
    return first.islower() or first.isdigit() or first in "\"'([{"


def norm_text(text: str) -> str:
    """Collapse all whitespace (including newlines) and normalise
    curly apostrophes, for verbatim comparisons. Official pages put
    line breaks inside sentences, so a raw substring check would
    miss a sentence that is split across lines in the chunk."""
    return " ".join(text.replace("’", "'").split())


def normalize_document(text: str) -> str:
    """Unwrap hard line breaks into one sentence per line."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    sentences: list[str] = []
    buf: list[str] = []
    for line in lines:
        # A finished sentence never absorbs the next line, even if
        # that line starts lowercase: "He left. he went home" is
        # two sentences, not one.
        if buf and _TERMINAL_RE.search(" ".join(buf)):
            sentences.append(" ".join(buf))
            buf = [line]
        elif buf and _continues(" ".join(buf), line):
            buf.append(line)
        else:
            if buf:
                sentences.append(_finish(" ".join(buf)))
            buf = [line]
    if buf:
        sentences.append(_finish(" ".join(buf)))
    return "\n".join(sentences)


def _finish(sentence: str) -> str:
    """Ensure a sentence ends with terminal punctuation."""
    if not _TERMINAL_RE.search(sentence):
        sentence += "."
    return sentence


_BULLET_RE = re.compile(r"(?<=[.!?:])\s+-\s+(?=[A-Z0-9])")


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences. Each line is first treated as a
    sentence unit (normalised documents are one sentence per line);
    a line holding several sentences is split on sentence-ending
    punctuation. The result is what `app.answer` picks from and
    what the eval's faithfulness check verifies against.

    Dash-led bullet items are split off as their own sentences
    (Phase 2 improvement, Step 1): the normaliser joins a bullet
    list into one line ("...includes: - Full tuition fee coverage.
    - A monthly allowance of SEK 12,000 ..."), and without this
    split the whole list is one long "sentence" that shares few
    query stems and loses to the page intro (golden q32). A bullet
    marker is only split after terminal punctuation or a colon,
    so inline dashes ("8-24 February", "well-known") are untouched.
    Each split piece is still a verbatim substring of the chunk
    text, so faithfulness stays measurable."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])",
                         line)
        for part in parts:
            out.extend(p.strip() for p in _BULLET_RE.split(part) if p.strip())
    if not out and text.strip():
        out = [text.strip()]
    return out
