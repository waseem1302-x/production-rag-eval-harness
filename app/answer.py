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
Lexical checks cannot catch that. Phase 2, Step 2 adds two things
(see `verify_answer_relevance`): a dense-similarity floor on the
picked sentences, and a question-focus check — a question that asks
about a ranking must be answered by a sentence that mentions a
ranking. Dense similarity alone was measured and cannot separate
that question: o04's picked sentences score 0.667, above several
answerable questions (q19: 0.562, q14: 0.591, q12: 0.615), so the
focus check does the work there and the dense floor only catches
unrelated questions. The same verification runs on the generative
path's kept sentences (see app/generative.py).

Phase 2, Step 6 (corpus growth) found the same shape twice more
and added a second focus check, same pattern: a question that names
a scholarship the corpus does not hold must be answered by a
sentence that mentions that scholarship's name, or the answer is
refused ("unanswered_scholarship_subject"). Growing the corpus from
10 to 19 documents gave the lexical gates other scholarships'
sentences to quote: o06 ("deadline for the Australia Awards
Scholarship 2027") was answered with SI + Commonwealth dates, and
o07 ("acceptance rate for the Knight-Hennessy Scholars programme")
was answered with KHS funding sentences. The o07 half was the
specific-term gate passing on the word "scholar" alone — every KHS
sentence contains "scholars" — so "scholar" joined the generic
terms (in a scholarship corpus it is as generic as "scholarship"
or "student"); after that, o07's only specific terms are
"acceptance" and "rate", neither of which any KHS sentence
contains, so it abstains with "missing_specific_term". The check is
narrow, like the ranking check: it looks for the named
scholarship's *name terms* in the picked sentences, not for the
scholarship as an entity, and it only fires when no known document
is named (a known name is already owned by the doc-name gate).
"""
import math
import re

from . import embeddings
from .config import (
    ABSTAIN_THRESHOLD,
    DENSE_SENTENCE_WEIGHT,
    DENSE_VERIFY_MIN_SIM,
)
from .retrieval import _query_terms, tokenize
from .textnorm import segment_sentences as _sentences

GENERIC_TERMS = {
    "scholarship", "scholarships", "program", "programs",
    "programme", "programmes", "study", "studies", "student",
    "students", "university", "universities", "application",
    "applications", "eligible", "eligibility", "funded", "funding",
    "degree", "degrees", "master", "masters", "level", "levels",
    # Phase 2, Step 6: "scholar(s)" is generic in a scholarship
    # corpus — every document is about scholars. Without this, the
    # specific-term gate passes on "scholar" alone, so a question
    # whose real ask ("acceptance", "rate") appears nowhere is
    # answered with funding sentences (golden o07).
    "scholar", "scholars",
}

# Stemmed doc-name token -> doc slug. "si" covers "SI Scholarship".
DOC_ALIASES = {
    "chevening": "chevening", "mext": "mext", "vanier": "vanier",
    "daad": "daad", "eiffel": "eiffel", "erasmus": "erasmus-mundus",
    "mundus": "erasmus-mundus", "fulbright": "fulbright",
    "hkpfs": "hkpfs", "manaaki": "manaaki", "zealand": "manaaki",
    "sweden": "sweden-si", "si": "sweden-si", "hong": "hkpfs",
    "kong": "hkpfs",
    # Phase 2, Step 6 (corpus growth): aliases for the 9 new
    # documents. "emjm" names the organisations-side Erasmus page
    # (erasmus-mundus-org), whose distinct fact is the EU
    # scholarship amount; "erasmus"/"mundus" still name the
    # student-side page. "pioneering" names the SI Pioneering
    # Women in STEM page; "si" still names Global Professionals.
    "gates": "gates-cambridge", "cambridge": "gates-cambridge",
    "rhodes": "rhodes", "oxford": "rhodes",
    "knight": "knight-hennessy", "hennessy": "knight-hennessy",
    "stanford": "knight-hennessy",
    "schwarzman": "schwarzman", "tsinghua": "schwarzman",
    "commonwealth": "commonwealth-shared",
    "rtp": "rtp", "marshall": "marshall",
    "emjm": "erasmus-mundus-org",
    "pioneering": "si-pioneering-women",
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


# Query expansion (Phase 2 improvement, query expansion): a small,
# explicit, corpus-grounded synonym map for the two question shapes
# the corpus never states in the question's own words.
#
# - "How long ...?" / "... last?": the corpus states durations as
#   "duration" and in years/months ("Masters duration 1 to 2
#   academic years"), never as "long". q10's correct sentence shares
#   only "master" with its question, so it loses to intro sentences
#   that share four doc-name stems. Expanding "long"/"last" to
#   "duration"/"years" gives it two more shared stems, and
#   "duration" is rare among the candidates, so its IDF weight
#   lifts the sentence into the picked set.
# - "What levels of study ...?": the corpus states levels as
#   "undergraduate", "postgraduate", "doctoral", "masters" ("At
#   which degree levels ..." for Eiffel, "Scholarships for
#   undergraduate and postgraduate study" for Manaaki). q17's
#   correct sentence shares only "scholarship"/"study" with its
#   question; expanding "level" to the four level names gives it
#   the rare stems that identify it.
#
# Deliberately small and explicit, not embedding-based and not an
# LLM call: every expansion term is a word the corpus itself uses
# for the concept, everything stays local, CPU-only, $0. Limits,
# stated honestly: the map only covers these two shapes — a
# question paraphrased any other way ("how many years", "what
# kind of degree") gets no expansion, and a trigger word in an
# unrelated sense ("last year of bachelor studies" is a sentence,
# not a question word here) would expand anyway, so the map must
# stay short and the full eval must prove zero regressions.
# Keys and values are stemmed at use time with _stem, so the map
# is written in plain words and can never drift from the stemmer.
QUERY_EXPANSIONS = {
    "long": ["duration", "years", "months"],
    "last": ["duration", "years"],
    "level": ["undergraduate", "postgraduate", "doctoral", "masters"],
}


def expand_query_stems(stems: set[str]) -> set[str]:
    """Add the expansion stems for any trigger stem present.

    Returns a new set; the input is not mutated. A stem with no
    trigger in the map comes back unchanged.
    """
    expanded = set(stems)
    triggers = {_stem(k): v for k, v in QUERY_EXPANSIONS.items()}
    for stem in stems:
        for word in triggers.get(stem, ()):
            expanded.add(_stem(word))
    return expanded


def expansion_stems(base_stems: set[str]) -> set[str]:
    """The stems expansion added, i.e. the expanded set minus the
    question's own stems. These are the question's ask in the
    corpus's words, and _pick_sentences counts them double — the
    same factor as the specific-stem bonus, for the same reason:
    they identify what the question asks, not background vocabulary.
    Without the bonus, expansion alone does not fix q10: the
    duration sentence shares "master"/"duration"/"year" for 6.95
    and ranks 7th, behind intro sentences sharing four doc-name
    stems. With the bonus it scores 17.66 and ranks 1st."""
    return expand_query_stems(base_stems) - set(base_stems)


def dense_sentence_scores(question: str, sentences: list[str]) -> dict[str, float] | None:
    """Cosine similarity of each sentence to the question, using the
    existing local MiniLM embedder ($0, no API). Embeddings are
    L2-normalised, so the dot product is the cosine. Returns None
    when the embedder is unavailable (the caller falls back to
    lexical-only scoring) or when there are no sentences — a missing
    model must degrade the answerer, never crash it.
    """
    if not sentences:
        return None
    try:
        vectors = embeddings.embed([question] + list(sentences))
    except Exception:  # noqa: BLE001 - embedder load can fail many ways
        return None
    query_vec = vectors[0]
    return {sent: round(float(vec @ query_vec), 4)
            for sent, vec in zip(sentences, vectors[1:])}


def _ranking_question(question: str) -> bool:
    """Does the question ask about a ranking? Matches "rank",
    "ranked", "ranking", "number one" and "number 1" as whole
    words/phrases. Deliberately narrow: "top" and "best" are not
    included — "top reasons to apply" is not a ranking question.
    """
    return bool(re.search(
        r"\b(rank|ranked|ranking|number one|number 1)\b",
        question or "", re.IGNORECASE))


def _mentions_ranking(sentences: list[str]) -> bool:
    """Does any sentence mention a ranking? The stem "rank" covers
    rank/ranked/ranking; "number one"/"number 1" are matched as
    phrases. Note the limit, stated honestly: this checks for a
    ranking *term*, not a ranked *entity* — a sentence about
    "best-ranked students" passes even though it names no
    university. It catches the o04 shape (no ranking mentioned at
    all), not every unanswerable ranking question.
    """
    joined = " ".join(sentences).lower()
    return ("rank" in _stems(joined)
            or "number one" in joined or "number 1" in joined)


def _named_scholarship_stems(question: str) -> set[str]:
    """Stemmed name terms of the scholarship a question names, if
    any: the capitalised words immediately before "Scholarship(s)".
    Walks backwards from that word while words are capitalised, so
    "the Australia Awards Scholarship" yields {australia, award}
    and "the Chevening Scholarship" yields {chevening}; a lowercase
    word ("the", "for") stops the walk, and a lowercase-spelled
    name yields nothing. Generic stems and doc-alias stems are
    dropped: a known document's name is owned by the doc-name gate,
    not by this check.
    """
    words = re.findall(r"[A-Za-z][A-Za-z\-]*", question or "")
    idx = next((i for i, w in enumerate(words)
                if w.lower() in ("scholarship", "scholarships")), None)
    if idx is None:
        return set()
    generic = {_stem(g) for g in GENERIC_TERMS}
    doc_tokens = {_stem(k) for k in DOC_ALIASES}
    stems = set()
    for j in range(idx - 1, -1, -1):
        if not words[j][0].isupper():
            break
        stem = _stem(words[j])
        if stem not in generic and stem not in doc_tokens:
            stems.add(stem)
    return stems


def verify_answer_relevance(question: str, sentences: list[str],
                            dense_sims: dict[str, float] | None = None) -> dict:
    """Answer-verification step (Phase 2, Step 2): do the chosen
    sentences actually answer the question?

    Three checks, all deterministic and explainable:
    1. Dense floor: the best picked sentence must reach
       DENSE_VERIFY_MIN_SIM cosine similarity to the question, or
       the answer is refused ("low_answer_similarity"). This is a
       floor for unrelated questions, not a relevance proof.
    2. Ranking focus: a question that asks about a ranking must be
       answered by a sentence that mentions a ranking, or the
       answer is refused ("unanswered_ranking"). This is the check
       that catches o04: the Vanier sentences score 0.667 on dense
       similarity (above several answerable questions), so no dense
       threshold can separate them — but none of them mentions a
       ranking at all.
    3. Scholarship-subject focus (Phase 2, Step 6): a question
       that names a scholarship no known document covers must be
       answered by a sentence that mentions that scholarship's
       name, or the answer is refused
       ("unanswered_scholarship_subject"). This is the check that
       catches o06: "Australia Awards" is not in the corpus, the
       doc-name gate only fires for known documents, and the
       picked SI + Commonwealth sentences share the question's
       "deadline" and "2027" but never name Australia Awards.
       The check is skipped when a known document is named — the
       doc-name gate already restricts the candidates to it — and
       it looks for name *terms*, not the scholarship as an entity:
       a sentence mentioning only "Awards" would pass it.

    `dense_sims`, when given, is a sentence -> cosine map (tests
    inject it so they need no model); otherwise the sims are
    computed here with the local embedder, and the floor is skipped
    when the embedder is unavailable. Returns {"answers": bool,
    "reason": str|None, "best_sim": float|None}.
    """
    if not sentences:
        return {"answers": False, "reason": "no_answer_sentences",
                "best_sim": None}
    if dense_sims is None:
        dense_sims = dense_sentence_scores(question, sentences)
    best_sim = max(dense_sims.values()) if dense_sims else None
    if best_sim is not None and best_sim < DENSE_VERIFY_MIN_SIM:
        return {"answers": False, "reason": "low_answer_similarity",
                "best_sim": best_sim}
    if _ranking_question(question) and not _mentions_ranking(sentences):
        return {"answers": False, "reason": "unanswered_ranking",
                "best_sim": best_sim}
    if not _named_docs(question):
        name_stems = _named_scholarship_stems(question)
        if name_stems and not (name_stems & _stems(" ".join(sentences))):
            return {"answers": False,
                    "reason": "unanswered_scholarship_subject",
                    "best_sim": best_sim}
    return {"answers": True, "reason": None, "best_sim": best_sim}


def _idf_weights(sentences: list[str]) -> dict[str, float]:
    """Inverse document frequency of each stem over the given
    sentences: ln((N+1)/(df+1)) + 1, where N is the sentence count
    and df the number of sentences containing the stem. A stem in
    every sentence weighs 1.0; a stem in exactly one sentence weighs
    ln((N+1)/2) + 1. The population is the candidate sentences of
    one _pick_sentences call, so the weights describe the sentences
    actually competing for the answer, not the whole corpus.
    """
    n = len(sentences)
    df: dict[str, int] = {}
    for stems in (_stems(s) for s in sentences):
        for stem in stems:
            df[stem] = df.get(stem, 0) + 1
    return {stem: math.log((n + 1) / (count + 1)) + 1.0
            for stem, count in df.items()}


def _pick_sentences(chunks: list[dict], query_stems: set[str],
                    question: str | None = None,
                    specific_stems: set[str] | None = None,
                    expansion_stems: set[str] | None = None,
                    max_sentences: int = 3, scored: bool = True,
                    dense: bool = True) -> list[str]:
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

    Dense sentence scoring (Phase 2, Step 2): with dense=True (and
    scored=True) each sentence's lexical score gains
    DENSE_SENTENCE_WEIGHT * its cosine similarity to the question
    (local MiniLM embedder, $0). Measured honestly on the golden
    set, this does NOT improve answer-only relevance on this
    corpus: at weights 0–2 no golden answer changes at all, and
    at weights 3–4 it trades one question for another (q03 fixed,
    q16 lost — q16's correct sentence scores only 0.394 dense
    similarity, below the boilerplate that replaces it), net
    0.909 either way. The weight is therefore 2.0: the largest
    weight with zero regressions, kept in the path so the mechanism
    exists and is measured, not because it earns a gain here.
    The paraphrase gap is real but dense similarity is too weak a
    signal at this corpus size to close it: q03's correct sentence
    ranks 4th by dense sim (0.630), q10's 8th (0.438) and q17's
    14th (0.527). A sentence still needs at least one shared query
    stem to be pickable (unchanged), and when the embedder is
    unavailable the scores are lexical-only.

    IDF weighting (Phase 2 improvement, Step 2): the lexical part
    of the score is no longer a stem count. Each shared stem counts
    its IDF weight over the candidate sentences (see _idf_weights),
    so a stem that appears in nearly every competing sentence
    ("the", "pioneering", "women") counts about 1 while a stem in
    one sentence ("monthly", "allowance") counts about ln((N+1)/2)
    + 1. Measured on the 50-question golden set: answer-only
    0.930 (40/43) to 0.953 (41/43), q32 fixed, zero regressions.

    Query expansion (Phase 2 improvement, query expansion):
    `expansion_stems`, when given, count double — the same factor
    as the specific-stem bonus. They are the question's ask in the
    corpus's words ("duration"/"years" for "how long", the level
    names for "levels of study"), so they identify the answer the
    way specific stems do. Expansion without the bonus does not
    fix q10 (see `expansion_stems`).
    """
    specific_stems = specific_stems or set()
    expansion_stems = expansion_stems or set()
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
    # Collect the pickable sentences first so they are embedded in
    # one batch (one model call per question, not one per sentence).
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()
    for order, chunk in enumerate(chunks):
        for sent in _sentences(chunk["text"]):
            if not re.search(r"[.!?]$", sent):
                continue
            if sent in seen:
                continue
            seen.add(sent)
            stems = _stems(sent)
            if not (query_stems & stems):
                continue
            candidates.append((order, sent))
    sims: dict[str, float] = {}
    if dense and question is not None:
        sims = dense_sentence_scores(
            question, [sent for _, sent in candidates]) or {}
    # IDF weighting (Phase 2 improvement, Step 2): each shared stem
    # counts by its rarity among the competing sentences, not 1.
    # Plain counts let stopwords ("the", "for", "in") and doc-name
    # stems ("pioneering", "women", "stem") outvote the rare content
    # stems that actually identify the answer ("monthly",
    # "allowance" for q32). The specific-stem bonus still doubles
    # the weight, it doubles the IDF weight instead of the count.
    weights = _idf_weights([sent for _, sent in candidates])
    scored_list: list[tuple[float, int, str]] = []
    for order, sent in candidates:
        stems = _stems(sent)
        shared = query_stems & stems
        score = sum(weights.get(t, 1.0) for t in shared)
        score += 2 * sum(weights.get(t, 1.0)
                         for t in specific_stems & stems)
        # Query-expansion bonus: expansion stems count double, like
        # specific stems (see the _pick_sentences docstring).
        score += 2 * sum(weights.get(t, 1.0)
                         for t in expansion_stems & stems)
        # Short fragments are usually headers ("Scholarships |
        # Chevening.", "Home Page - Vanier..."). They share the
        # document name and little else, so they outscore real
        # content on raw stem count. Demote them by one point.
        if len(sent) < 50:
            score -= 1
        score += DENSE_SENTENCE_WEIGHT * sims.get(sent, 0.0)
        scored_list.append((score, -order, sent))
    scored_list.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [sent for _, _, sent in scored_list[:max_sentences]]
    if not picked and chunks:
        picked = _sentences(chunks[0]["text"])[:1]
    return picked


def answer(question: str, retrieved: list[dict], threshold: float = ABSTAIN_THRESHOLD,
           gates: bool = True, scored_picking: bool = True,
           dense: bool = True) -> dict:
    """Build the API answer dict from retrieved chunks.

    `gates=False` skips every abstention rule and always answers
    from the top chunks. That is the naive baseline behaviour the
    benchmark compares against: a tutorial chatbot answers
    everything, including questions the corpus cannot answer.
    `scored_picking=False` takes the first sentences in chunk order
    instead of the best-scored sentences. `dense=False` turns off
    dense sentence scoring (lexical scores only); the eval uses it
    to measure the dense contribution separately.
    """
    # Content stems only: _stems(question) would include stopwords
    # ("how", "do", "at", "i"), and a sentence sharing only "at" and
    # "i" with the question is not evidence of anything.
    base_stems = {_stem(t) for t in _query_terms(question)}
    query_stems = base_stems
    # Query expansion (Phase 2 improvement): only for scored
    # picking. The naive baseline (scored_picking=False) must keep
    # its original query stems, or it stops being a baseline — the
    # before/after comparison would measure the expansion twice.
    expansion_only: set[str] = set()
    if scored_picking:
        query_stems = expand_query_stems(base_stems)
        expansion_only = expansion_stems(base_stems)
    best = retrieved[0]["rerank_score"] if retrieved else 0.0
    if not retrieved:
        return _abstain(question, best, "empty_index")
    if gates and best < threshold:
        return _abstain(question, best, "weak_score")

    # Rule 2: doc-name consistency.
    named = _named_docs(question)
    candidates = retrieved
    if gates and named:
        candidates = [c for c in retrieved if c["doc"] in named]
        if not candidates:
            return _abstain(question, best, "doc_name_mismatch")

    sentences = _pick_sentences(candidates[:3], query_stems,
                                question,
                                _specific_terms(question),
                                expansion_only,
                                max_sentences=5, scored=scored_picking,
                                dense=dense)
    if not sentences:
        return _abstain(question, best, "no_matching_sentences")

    if gates:
        # Shared-stem gate: the picked sentences must share at least two
        # distinct query stems with the question. One shared word is not
        # evidence — "How do I bake sourdough bread at home?" shares only
        # "home" with the Manaaki page ("Welcome to the home of Manaaki
        # New Zealand scholarships") and must abstain, not answer.
        shared_stems = query_stems & _stems(" ".join(sentences))
        if len(shared_stems) < 2:
            return _abstain(question, best, "weak_overlap")

        # Rule 3: specific-term gate.
        specific = _specific_terms(question)
        if specific:
            present = set()
            for sent in sentences:
                present |= specific & _stems(sent)
            if not present:
                return _abstain(question, best, "missing_specific_term")
        elif not (query_stems & _stems(" ".join(sentences))):
            return _abstain(question, best, "weak_overlap")

        # Phase 2, Step 2 — answer verification: do the chosen
        # sentences actually answer the question? (Dense floor +
        # ranking-focus check; see `verify_answer_relevance`.)
        # Runs after the lexical gates, and only when gates are on:
        # the naive baseline (gates=False) must keep answering
        # everything, or it stops being a baseline.
        relevance = verify_answer_relevance(question, sentences)
        if not relevance["answers"]:
            result = _abstain(question, best, relevance["reason"])
            result["relevance_verification"] = relevance
            return result
    else:
        relevance = None

    citations = [
        {"doc": c["doc"], "chunk_id": c["chunk_id"], "score": c["score"],
         "text": c["text"]}
        for c in candidates[:3]
    ]
    result = {
        "question": question,
        "found": True,
        "answer": " ".join(sentences),
        "score": round(float(best), 4),
        "citations": citations,
        "answer_sentences": sentences,
    }
    if relevance is not None:
        result["relevance_verification"] = relevance
    return result


def _abstain(question: str, best: float, reason: str = "abstained") -> dict:
    """Every abstention names its reason, so the eval's failure table
    can say WHY the system refused — "weak_overlap" and
    "missing_specific_term" are different bugs with different fixes."""
    return {
        "question": question,
        "found": False,
        "answer": "Not found in the corpus.",
        "score": round(float(best), 4),
        "citations": [],
        "abstention_reason": reason,
    }
