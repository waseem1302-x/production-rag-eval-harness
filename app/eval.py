"""Eval harness: golden set + recall@k, faithfulness, answer relevance.

Definitions (all string-based proxies in Phase 1 — no LLM judge):
- recall@k: fraction of in-corpus questions where the expected
  document appears in the top-k retrieved documents.
- faithfulness: fraction of answer sentences that appear verbatim
  (as a substring) in any cited chunk. Extractive answers should
  score near 1.0; anything lower means the answer builder is broken.
- answer relevance (proxy): fraction of in-corpus questions where
  the expected answer phrase (lowercased) appears in the answer
  text or in any cited chunk. This measures whether the evidence
  retrieved actually contains the answer, not whether a generator
  wrote it well. Phase 2 replaces this with an LLM-as-judge.
- answer-only relevance: fraction where the expected phrase appears
  in the answer text alone, not just in the citations. This is the
  metric that separates the naive baseline from the improved
  pipeline: the baseline returns the first sentences of the top
  chunk (often nav/header text), so the answer itself rarely
  contains the answer even when the citations do.
- abstention accuracy: fraction of out-of-corpus questions answered
  with found=False.
- eligibility accuracy: fraction of questions that carry a profile
  where the verdict for the expected document matches the expected
  verdict ("eligible" / "not_eligible" / "unknown"). Eligibility
  questions are also counted in the answer metrics above, because
  their expected answer phrase is part of the eligibility statement.
"""
import json

from .answer import _sentences
from .config import CHUNKS_FILE, GOLDEN_FILE
from .retrieval import Retriever
from .textnorm import norm_text


def _answer_question(question: str, profile: dict | None, retrieved: list[dict],
                     chunks: list[dict], answer_kwargs: dict) -> tuple[dict, list[dict]]:
    """One question through the same path the API uses: substantive
    answer, with eligibility-first routing when a profile is given."""
    from .answer import answer as build_answer
    from .eligibility import resolve as resolve_eligibility

    substantive = build_answer(question, retrieved, **answer_kwargs)
    if not profile:
        return substantive, []
    answer, eligibility = resolve_eligibility(
        question, profile, retrieved, chunks, substantive)
    return (answer if answer is not None else substantive), eligibility


def load_golden(path=GOLDEN_FILE) -> list[dict]:
    return json.loads(path.read_text())


def run_eval(retriever: Retriever, golden: list[dict] | None = None,
             search_kwargs: dict | None = None,
             answer_kwargs: dict | None = None) -> dict:
    """Run the golden set. `search_kwargs` / `answer_kwargs` let the
    benchmark run other pipeline configurations (e.g. the naive
    baseline: no rerank, no abstention gates) on the same questions.
    Defaults are the full Phase 1 pipeline."""
    from .answer import answer as build_answer

    golden = golden if golden is not None else load_golden()
    search_kwargs = search_kwargs or {}
    answer_kwargs = answer_kwargs or {}
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    in_corpus = [q for q in golden if not q.get("out_of_corpus")]
    out_corpus = [q for q in golden if q.get("out_of_corpus")]

    recall3 = recall5 = 0
    relevant = 0
    answer_only_relevant = 0
    faithful_sents = total_sents = 0
    elig_correct = elig_total = 0
    per_question = []

    for q in in_corpus:
        retrieved = retriever.search(q["question"], top_k=5, **search_kwargs)
        docs = [c["doc"] for c in retrieved]
        hit3 = q["expected_doc"] in docs[:3]
        hit5 = q["expected_doc"] in docs
        recall3 += hit3
        recall5 += hit5
        result, eligibility = _answer_question(
            q["question"], q.get("profile"), retrieved, chunks, answer_kwargs)
        if q.get("profile"):
            elig_total += 1
            verdict = next((v["verdict"] for v in eligibility
                            if v["doc"] == q["expected_doc"]), "missing")
            if verdict == q.get("expected_verdict"):
                elig_correct += 1
        expected = q["expected_answer"].lower()
        haystack = (result["answer"] + " " + " ".join(c["text"] for c in result["citations"])).lower()
        is_relevant = expected in haystack
        relevant += is_relevant
        if expected in result["answer"].lower():
            answer_only_relevant += 1
        is_answer_only = expected in result["answer"].lower()
        # Faithfulness is only meaningful for answers that were given.
        # An abstention ("Not found in the corpus.") is not a claim
        # drawn from the chunks, so it is neither faithful nor unfaithful.
        # Check the stored answer sentences directly: re-splitting the
        # joined answer string cannot recover sentences that were
        # joined without punctuation.
        if result["found"]:
            # When the answer carries its own sentence list, use it —
            # even when it is empty. An eligibility answer with no
            # evidence (verdict "unknown") makes no extractive claim
            # about the corpus, so it contributes no sentences to
            # the faithfulness denominator, like an abstention.
            # Only answers without a sentence list are re-split
            # from the answer text.
            sentences = (result["answer_sentences"] if "answer_sentences" in result
                         else _sentences(result["answer"]))
            for sent in sentences:
                total_sents += 1
                # Compare normalised text: chunk text carries the
                # page's line breaks, so a raw substring check would
                # miss a sentence split across lines (e.g. the Vanier
                # citizenship list). Same normalisation as the
                # eligibility evidence validation in app/textnorm.py.
                cited = norm_text(" ".join(c["text"] for c in result["citations"]))
                if norm_text(sent) in cited:
                    faithful_sents += 1
        per_question.append({
            "id": q["id"], "question": q["question"],
            "expected_doc": q["expected_doc"], "found_doc_top5": docs,
            "recall@3": hit3, "recall@5": hit5,
            "found": result["found"], "answer_relevant": is_relevant,
            "answer_only_relevant": is_answer_only,
            "eligibility_verdict": next(
                (v["verdict"] for v in eligibility if v["doc"] == q["expected_doc"]),
                None) if q.get("profile") else None,
        })

    abstained = 0
    for q in out_corpus:
        retrieved = retriever.search(q["question"], top_k=5, **search_kwargs)
        result = build_answer(q["question"], retrieved, **answer_kwargs)
        if not result["found"]:
            abstained += 1

    n = max(1, len(in_corpus))
    return {
        "n_in_corpus": len(in_corpus),
        "n_out_of_corpus": len(out_corpus),
        "n_eligibility": elig_total,
        "recall@3": round(recall3 / n, 3),
        "recall@5": round(recall5 / n, 3),
        "answer_relevance_proxy": round(relevant / n, 3),
        "answer_only_relevance": round(answer_only_relevant / n, 3),
        "faithfulness": round(faithful_sents / max(1, total_sents), 3),
        "abstention_accuracy": round(abstained / max(1, len(out_corpus)), 3),
        "eligibility_accuracy": round(elig_correct / elig_total, 3) if elig_total else None,
        "per_question": per_question,
    }
