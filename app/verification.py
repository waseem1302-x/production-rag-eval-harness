"""Verification of generated answers against the cited chunks.

Phase 2, Step 1. A generative answer is only trusted sentence by
sentence: every sentence of the generated draft must be *grounded*
in the retrieved chunks, or it is dropped. If nothing survives, the
caller falls back to the extractive answer (or abstains).

This check is deliberately deterministic and lexical — no model
call — so it cannot itself hallucinate, it is fast, and it is
unit-testable without loading a model:

- A sentence is SUPPORTED by a chunk when it shares at least 2
  content stems with that chunk and those shared stems cover at
  least half of the sentence's content stems (a one-content-stem
  sentence needs that single stem to match). Content stems are
  stemmed tokens minus stopwords, using the same stemmer and
  stopword list as the extractive answerer, so "funding" in the
  answer matches "funded" in the chunk.
- Coverage is measured on the sentence's *novel* stems: content
  stems that do not already appear in the question. A sentence
  that only restates the question ("The Manaaki Scholarships
  cover ...") earns no grounding from the question words — the
  new claims are what must be in the chunk. Without this, a
  sentence that copies another scholarship's facts but keeps the
  question's scholarship name scores ~0.5 on question-word
  overlap alone (measured on q17: the SI field list attributed
  to Manaaki). A sentence with no novel stems carries no
  verifiable claim and is unsupported.
- A sentence is UNSUPPORTED when it contains a number (a digit
  sequence, e.g. "50,000", "$1,200", "2027") that appears in no
  retrieved chunk. Numbers are where a wrong scholarship answer
  does the most damage (a stipend, a deadline year), so they get
  a strict rule: an invented number fails the sentence even if
  the words overlap. The number is normalised (commas and
  currency symbols stripped) before comparison.

Known limit, stated honestly: grounding is not relevance. A
sentence can be grounded in a chunk yet not answer the question
(the o04 failure: Vanier sentences that never name a ranked
university are all "supported" by the Vanier chunk). That gap is
what the LLM-as-judge relevance score measures, and what the
eval reports — verification stops invented facts, it does not
prove the answer is the right one.
"""
import re

from .answer import _stem
from .retrieval import _query_terms, tokenize

_NUMBER_RE = re.compile(r"\$?\d[\d,]*(\.\d+)?")

_STOPWORDS = {
    "what", "which", "where", "when", "who", "how", "does", "do",
    "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
    "and", "or", "by", "with", "from", "it", "its", "this", "that",
    "be", "been", "being", "was", "were", "will", "would", "can",
    "could", "should", "may", "might", "must", "shall", "as", "at",
    "are", "is", "if", "than", "then", "there", "here", "not", "no",
    "yes", "but", "so", "such", "into", "over", "under", "between",
    "through", "after", "before", "during", "each", "other", "some",
    "any", "all", "both", "also", "only", "more", "most", "many",
    "much", "very", "just", "per", "via", "etc",
}


def _content_stems(text: str) -> set[str]:
    """Stemmed content words of a sentence: tokens minus stopwords."""
    return {_stem(t) for t in tokenize(text)
            if t not in _STOPWORDS and len(t) > 2}


def _numbers(text: str) -> set[str]:
    """Digit sequences in a sentence, normalised (commas/$ stripped)."""
    return {m.group(0).replace(",", "").lstrip("$")
            for m in _NUMBER_RE.finditer(text)}


def _chunk_numbers(chunk_text: str) -> set[str]:
    return _numbers(chunk_text)


def verify_sentence(sentence: str, chunks: list[dict],
                    question_stems: set[str] | None = None) -> dict:
    """Check one generated sentence against the retrieved chunks.

    Returns {"supported": bool, "chunk_id": str|None,
    "coverage": float, "reason": str}. `coverage` is the best
    shared-stem fraction over the chunks (0.0 when the sentence
    has no content stems). Coverage is measured on the sentence's
    novel stems (content stems not in the question); see the
    module docstring for why.
    """
    content = _content_stems(sentence)
    if not content:
        return {"supported": False, "chunk_id": None, "coverage": 0.0,
                "reason": "no_content_words"}
    novel = content - (question_stems or set())
    if not novel:
        return {"supported": False, "chunk_id": None, "coverage": 0.0,
                "reason": "no_novel_content"}
    sent_numbers = _numbers(sentence)
    if sent_numbers:
        all_chunk_numbers: set[str] = set()
        for chunk in chunks:
            all_chunk_numbers |= _chunk_numbers(chunk["text"])
        missing = sent_numbers - all_chunk_numbers
        if missing:
            return {"supported": False, "chunk_id": None, "coverage": 0.0,
                    "reason": "unsupported_number",
                    "unsupported_numbers": sorted(missing)}
    best: dict | None = None
    for chunk in chunks:
        chunk_stems = _content_stems(chunk["text"])
        shared = novel & chunk_stems
        coverage = len(shared) / len(novel)
        if best is None or coverage > best["coverage"]:
            best = {"chunk_id": chunk["chunk_id"], "coverage": coverage,
                    "shared": len(shared)}
    assert best is not None
    if len(novel) == 1:
        supported = best["shared"] == 1
    else:
        supported = best["shared"] >= 2 and best["coverage"] >= 0.5
    return {
        "supported": supported,
        "chunk_id": best["chunk_id"] if supported else None,
        "coverage": round(best["coverage"], 3),
        "reason": "supported" if supported else "low_overlap",
    }


def verify_answer(answer_text: str, chunks: list[dict],
                  question: str | None = None) -> dict:
    """Split a generated answer into sentences and verify each one.

    Returns {"kept": [sentences], "dropped": [{"sentence": ...,
    "reason": ...}], "answer": " ".join(kept)}. Sentences are
    split with the same sentence segmenter the extractive
    answerer uses, so a kept sentence is a sentence the corpus
    can be checked against, not a re-joined fragment.
    """
    from .textnorm import segment_sentences

    question_stems = _content_stems(question) if question else set()
    kept: list[str] = []
    dropped: list[dict] = []
    for sent in segment_sentences(answer_text):
        sent = sent.strip()
        if not sent:
            continue
        check = verify_sentence(sent, chunks, question_stems)
        if check["supported"]:
            kept.append(sent)
        else:
            dropped.append({"sentence": sent, "reason": check["reason"],
                            "coverage": check["coverage"]})
    return {"kept": kept, "dropped": dropped, "answer": " ".join(kept)}
