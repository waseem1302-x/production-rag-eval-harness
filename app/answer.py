"""Extractive answering with citations and explicit abstention.

Phase 1 has no generative LLM. The answer is sentences taken
verbatim from the cited chunks, so faithfulness is measurable.

Abstention rules (all lexical, all explainable — see README):
1. No retrieved chunks, or the best score is below threshold.
2. Doc-name consistency: if the question names a scholarship
   (e.g. "MEXT", "Vanier"), only chunks from that scholarship's
   page may answer. Answering a MEXT question from the Eiffel
   page is a wrong answer, not a helpful one.
3. Specific-term gate: the picked answer sentences must contain
   at least one *specific* query term (stemmed) — a term that is
   not a generic word ("scholarship", "programme") and not the
   document name. "What is the deadline for Chevening?" names
   Chevening, but no sentence mentions a deadline, so it abstains.
   "What is the GPA requirement for DAAD?" abstains the same way.

Known limit (found by the eval, documented honestly): a question
can pass all three rules and still not be answered in substance,
e.g. "Which university is ranked number one for Vanier scholars?"
returns Vanier sentences that never name a ranked university.
Lexical checks cannot catch that; Phase 2 adds a verification
step (does this chunk actually answer the question?).
"""
import re

from .config import ABSTAIN_THRESHOLD
from .retrieval import _query_terms, tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

GENERIC_TERMS = {
    "scholarship", "scholarships", "program", "programs",
    "programme", "programmes", "study", "studies", "student",
    "students", "university", "universities", "application",
    "applications", "eligible", "eligibility", "funded", "funding",
    "degree", "degrees", "master", "masters", "level", "levels",
}

# Stemmed doc-name token -> doc slug. "si" covers "SI Scholarship".
DOC_ALIASES = {
    "chevening": "chevening", "mext": "mext", "vanier": "vanier",
    "daad": "daad", "eiffel": "eiffel", "erasmus": "erasmus-mundus",
    "mundus": "erasmus-mundus", "fulbright": "fulbright",
    "hkpfs": "hkpfs", "manaaki": "manaaki", "zealand": "manaaki",
    "sweden": "sweden-si", "si": "sweden-si", "hong": "hkpfs",
    "kong": "hkpfs",
}


def _stem(word: str) -> str:
    """Light stemmer: funded/fund, submitted/submits, studies/study.

    Rules are applied in order and must be consistent: the same
    root must stem the same way whether it arrives as "nominate",
    "nominated", "studies" or "study".
    """
    w = word.lower().strip("'")
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    # Undoubled consonant: submitted -> submit (not submitt).
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("y"):
        w = w[:-1] + "i"
    return w


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text)}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _named_docs(question: str) -> set[str]:
    # Alias keys must be stemmed the same way as the question tokens:
    # "chevening" stems to "cheven", "erasmus" to "erasmu".
    stems = _stems(question)
    aliases = {_stem(k): v for k, v in DOC_ALIASES.items()}
    return {aliases[s] for s in stems if s in aliases}


def _specific_terms(question: str) -> set[str]:
    terms = {_stem(t) for t in _query_terms(question)}
    generic = {_stem(g) for g in GENERIC_TERMS}
    doc_tokens = {_stem(k) for k in DOC_ALIASES}
    return {t for t in terms if t not in generic and t not in doc_tokens}


def _pick_sentences(chunks: list[dict], query_stems: set[str],
                    specific_stems: set[str] | None = None,
                    max_sentences: int = 3, scored: bool = True) -> list[str]:
    """Pick the best sentences, scored by query-term overlap.

    With scored=True (default) each sentence is scored (1 point per
    distinct query stem, +2 per distinct specific stem) and the top
    scorers win. Ties keep document order, so the answer still reads
    in order. With scored=False the first sentences in chunk order
    are taken; the benchmark uses that mode for the naive baseline
    so the scoring step's contribution is measured, not assumed.
    Only sentences ending in . ! ? are pickable: header and list
    fragments without terminal punctuation join badly — two fragments
    joined with a space re-split as one "sentence" that appears in
    no chunk, which breaks the faithfulness measurement (found on
    the q07 DAAD question).
    """
    specific_stems = specific_stems or set()
    if not scored:
        picked: list[str] = []
        for chunk in chunks:
            for sent in _sentences(chunk["text"]):
                if not re.search(r"[.!?]$", sent):
                    continue
                if query_stems & _stems(sent) and sent not in picked:
                    picked.append(sent)
                if len(picked) >= max_sentences:
                    return picked
        if not picked and chunks:
            picked = _sentences(chunks[0]["text"])[:1]
        return picked
    scored_list: list[tuple[int, int, str]] = []
    for order, chunk in enumerate(chunks):
        for sent in _sentences(chunk["text"]):
            if not re.search(r"[.!?]$", sent):
                continue
            stems = _stems(sent)
            shared = query_stems & stems
            if not shared:
                continue
            score = len(shared) + 2 * len(specific_stems & stems)
            scored_list.append((score, -order, sent))
    scored_list.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [sent for _, _, sent in scored_list[:max_sentences]]
    if not picked and chunks:
        picked = _sentences(chunks[0]["text"])[:1]
    return picked


def answer(question: str, retrieved: list[dict], threshold: float = ABSTAIN_THRESHOLD,
           gates: bool = True, scored_picking: bool = True) -> dict:
    """Build the API answer dict from retrieved chunks.

    `gates=False` skips every abstention rule and always answers
    from the top chunks. That is the naive baseline behaviour the
    benchmark compares against: a tutorial chatbot answers
    everything, including questions the corpus cannot answer.
    `scored_picking=False` takes the first sentences in chunk order
    instead of the best-scored sentences.
    """
    # Content stems only: _stems(question) would include stopwords
    # ("how", "do", "at", "i"), and a sentence sharing only "at" and
    # "i" with the question is not evidence of anything.
    query_stems = {_stem(t) for t in _query_terms(question)}
    best = retrieved[0]["rerank_score"] if retrieved else 0.0
    if not retrieved:
        return _abstain(question, best)
    if gates and best < threshold:
        return _abstain(question, best)

    # Rule 2: doc-name consistency.
    named = _named_docs(question)
    candidates = retrieved
    if gates and named:
        candidates = [c for c in retrieved if c["doc"] in named]
        if not candidates:
            return _abstain(question, best)

    sentences = _pick_sentences(candidates[:3], query_stems,
                                _specific_terms(question), scored=scored_picking)
    if not sentences:
        return _abstain(question, best)

    if gates:
        # Shared-stem gate: the picked sentences must share at least two
        # distinct query stems with the question. One shared word is not
        # evidence — "How do I bake sourdough bread at home?" shares only
        # "home" with the Manaaki page ("Welcome to the home of Manaaki
        # New Zealand scholarships") and must abstain, not answer.
        shared_stems = query_stems & _stems(" ".join(sentences))
        if len(shared_stems) < 2:
            return _abstain(question, best)

        # Rule 3: specific-term gate.
        specific = _specific_terms(question)
        if specific:
            present = set()
            for sent in sentences:
                present |= specific & _stems(sent)
            if not present:
                return _abstain(question, best)
        elif not (query_stems & _stems(" ".join(sentences))):
            return _abstain(question, best)

    citations = [
        {"doc": c["doc"], "chunk_id": c["chunk_id"], "score": c["score"],
         "text": c["text"]}
        for c in candidates[:3]
    ]
    return {
        "question": question,
        "found": True,
        "answer": " ".join(sentences),
        "score": round(float(best), 4),
        "citations": citations,
    }


def _abstain(question: str, best: float) -> dict:
    return {
        "question": question,
        "found": False,
        "answer": "Not found in the corpus.",
        "score": round(float(best), 4),
        "citations": [],
    }
