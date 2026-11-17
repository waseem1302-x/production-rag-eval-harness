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
"""
import json

from .answer import _sentences
from .config import GOLDEN_FILE
from .retrieval import Retriever


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
    in_corpus = [q for q in golden if not q.get("out_of_corpus")]
    out_corpus = [q for q in golden if q.get("out_of_corpus")]

    recall3 = recall5 = 0
    relevant = 0
    answer_only_relevant = 0
    faithful_sents = total_sents = 0
    per_question = []

    for q in in_corpus:
        retrieved = retriever.search(q["question"], top_k=5, **search_kwargs)
        docs = [c["doc"] for c in retrieved]
        hit3 = q["expected_doc"] in docs[:3]
        hit5 = q["expected_doc"] in docs
        recall3 += hit3
        recall5 += hit5
        result = build_answer(q["question"], retrieved, **answer_kwargs)
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
        if result["found"]:
            for sent in _sentences(result["answer"]):
                total_sents += 1
                cited = " ".join(c["text"] for c in result["citations"])
                if sent in cited:
                    faithful_sents += 1
        per_question.append({
            "id": q["id"], "question": q["question"],
            "expected_doc": q["expected_doc"], "found_doc_top5": docs,
            "recall@3": hit3, "recall@5": hit5,
            "found": result["found"], "answer_relevant": is_relevant,
            "answer_only_relevant": is_answer_only,
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
        "recall@3": round(recall3 / n, 3),
        "recall@5": round(recall5 / n, 3),
        "answer_relevance_proxy": round(relevant / n, 3),
        "answer_only_relevance": round(answer_only_relevant / n, 3),
        "faithfulness": round(faithful_sents / max(1, total_sents), 3),
        "abstention_accuracy": round(abstained / max(1, len(out_corpus)), 3),
        "per_question": per_question,
    }
