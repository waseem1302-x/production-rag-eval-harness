"""Recursive, sentence-aware chunking.

Target: ~512 tokens (~2,048 chars), ~200 chars overlap.
No LangChain: the behaviour must be fully visible and testable.

Algorithm:
1. Split the document into paragraphs (blank lines).
2. Pack paragraphs into a chunk until adding the next one would
   exceed CHUNK_CHARS.
3. If a single paragraph is too long, split it into sentences
   and pack sentences the same way.
4. If a single sentence is still too long, hard-split it on
   word boundaries.
5. Overlap: each new chunk starts with the last OVERLAP chars
   of the previous chunk, so a fact on a boundary is not lost.
   Overlap is only applied between chunks of the same document.
"""
import re

from .config import CHUNK_CHARS, CHUNK_OVERLAP_CHARS

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_RE = re.compile(r"\S+\s*")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Split a too-long sentence on word boundaries, no word broken."""
    words = _WORD_RE.findall(text)
    out, cur = [], ""
    for w in words:
        if len(cur) + len(w) > size and cur:
            out.append(cur.strip())
            cur = w
        else:
            cur += w
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


def _pack(units: list[str], size: int) -> list[str]:
    """Pack units (paragraphs or sentences) into chunks of at most `size` chars."""
    chunks, cur = [], ""
    for unit in units:
        if len(unit) > size:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.extend(_hard_split(unit, size))
            continue
        if cur and len(cur) + 1 + len(unit) > size:
            chunks.append(cur.strip())
            cur = unit
        else:
            cur = f"{cur} {unit}".strip() if cur else unit
    if cur:
        chunks.append(cur.strip())
    return chunks


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS,
               overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks of at most `chunk_chars` chars."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # First pass: pack paragraphs; long paragraphs are split into sentences.
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_chars:
            units.append(para)
        else:
            units.extend(_split_sentences(para))
    chunks = _pack(units, chunk_chars)

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap_chars:]
        # Start the next chunk at a word boundary inside the tail.
        cut = tail.find(" ")
        seed = tail[cut + 1:] if cut != -1 else tail
        merged = f"{seed} {cur}".strip()
        # Merging must not push the chunk far past the target; if it does,
        # keep the original chunk (overlap is a bonus, not a requirement).
        overlapped.append(merged if len(merged) <= chunk_chars + overlap_chars else cur)
    return overlapped


def chunk_document(doc_id: str, text: str) -> list[dict]:
    """Chunk one document, returning chunk dicts with stable ids."""
    return [
        {"chunk_id": f"{doc_id}-{i}", "doc": doc_id, "text": chunk}
        for i, chunk in enumerate(chunk_text(text))
    ]
