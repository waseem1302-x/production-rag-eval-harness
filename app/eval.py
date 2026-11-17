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
  wrote it well. Phase 2 adds an LLM-as-judge alongside it
  (`judge=True` in `run_eval`): the proxy stays so before/after
  is comparable, and the judge scores faithfulness and relevance
  semantically — the proxy cannot judge a paraphrasing generator,
  whose sentences are grounded but not verbatim.
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
  verdict ("eligible" / "not_eligible" / "unknown" / "closed"). Eligibility
  questions are also counted in the answer metrics above, because
  their expected answer phrase is part of the eligibility statement.
"""
import json

from .answer import _sentences
from .config import CHUNKS_FILE, GOLDEN_FILE
from .retrieval import Retriever
from .textnorm import norm_text


def _answer_question(question: str, profile: dict | None, retrieved: list[dict],
                     chunks: list[dict], answer_kwargs: dict,
                     answer_fn=None) -> tuple[dict, list[dict]]:
    """One question through the same path the API uses: substantive
    answer, with eligibility-first routing when a profile is given.

    `answer_fn`, when given, builds the substantive answer instead
    of the extractive answerer (the generative path passes
    `generate_answer` here); eligibility routing is identical.
    """
    from .answer import answer as build_answer
    from .eligibility import resolve as resolve_eligibility

    build = answer_fn or build_answer
    substantive = build(question, retrieved, **answer_kwargs) \
        if answer_fn is None else build(question, retrieved)
    if not profile:
        return substantive, []
    answer, eligibility = resolve_eligibility(
        question, profile, retrieved, chunks, substantive)
    if answer is None:
        return substantive, eligibility
    # Tag the replacement so the eval can tell an eligibility
    # statement (template + verbatim evidence, never generated)
    # apart from a substantive answer: the judge scores the
    # substantive answers only, and the generative-mode counts
    # report these as "eligibility", not "unknown".
    answer["answer_mode"] = "eligibility"
    return answer, eligibility


def load_golden(path=GOLDEN_FILE) -> list[dict]:
    return json.loads(path.read_text())


def run_eval(retriever: Retriever, golden: list[dict] | None = None,
             search_kwargs: dict | None = None,
             answer_kwargs: dict | None = None,
             generative: bool = False, judge: bool = False,
             judge_fn=None, progress=None) -> dict:
    """Run the golden set. `search_kwargs` / `answer_kwargs` let the
    benchmark run other pipeline configurations (e.g. the naive
    baseline: no rerank, no abstention gates) on the same questions.
    Defaults are the full Phase 1 pipeline.
    `progress`, when given, is called as progress(i, n, question_id)
    after each in-corpus question (the generative run takes
    minutes; the caller uses it for a progress log).

    Phase 2: `generative=True` answers with the local generative
    model + verification step (see app/generative.py) instead of
    the extractive answerer; `judge=True` additionally scores each
    given answer with the same local model as judge (one short
    completion per question, two 0/1 scores). The string-proxy
    metrics are ALWAYS computed, so generative and extractive runs
    stay comparable — but note the proxy `faithfulness` measures
    verbatim substring overlap, which a paraphrasing generator
    fails by design; the judge's faithfulness is the meaningful
    number for generative answers. `judge_fn`, when given, replaces
    the model for judging (tests inject a fake).
    """
    from .answer import answer as build_answer
    from .generative import generate_answer, judge_answer

    golden = golden if golden is not None else load_golden()
    search_kwargs = search_kwargs or {}
    answer_kwargs = answer_kwargs or {}
    # Eligibility is resolved against the chunks the retriever
    # actually searched, not the default chunks file on disk:
    # an eval against a non-default chunk size (Phase 2, Step 5
    # ablation) must use that size's chunks, whose ids and text
    # differ from the 512-token file. Fall back to the file only
    # when the retriever carries no chunks (e.g. an empty index).
    chunks = (list(retriever.chunks) if getattr(retriever, "chunks", None)
              else [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()])
    in_corpus = [q for q in golden if not q.get("out_of_corpus")]
    out_corpus = [q for q in golden if q.get("out_of_corpus")]

    recall3 = recall5 = 0
    relevant = 0
    answer_only_relevant = 0
    faithful_sents = total_sents = 0
    elig_correct = elig_total = 0
    judge_faith = judge_rel = judge_n = judge_unparsed = 0
    generative_modes: dict[str, int] = {}
    per_question = []

    for qi, q in enumerate(in_corpus, start=1):
        retrieved = retriever.search(q["question"], top_k=5, **search_kwargs)
        docs = [c["doc"] for c in retrieved]
        hit3 = q["expected_doc"] in docs[:3]
        hit5 = q["expected_doc"] in docs
        recall3 += hit3
        recall5 += hit5
        if generative:
            result, eligibility = _answer_question(
                q["question"], q.get("profile"), retrieved, chunks,
                answer_kwargs, answer_fn=generate_answer)
            mode = result.get("answer_mode", "unknown")
            generative_modes[mode] = generative_modes.get(mode, 0) + 1
        else:
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
        row = {
            "id": q["id"], "question": q["question"],
            "expected_doc": q["expected_doc"], "found_doc_top5": docs,
            "recall@3": hit3, "recall@5": hit5,
            "found": result["found"], "answer": result["answer"],
            "answer_relevant": is_relevant,
            "answer_only_relevant": is_answer_only,
            "eligibility_verdict": next(
                (v["verdict"] for v in eligibility if v["doc"] == q["expected_doc"]),
                None) if q.get("profile") else None,
        }
        if generative:
            row["answer_mode"] = result.get("answer_mode")
            if result.get("generation", {}).get("latency_s") is not None:
                row["generation_latency_s"] = result["generation"]["latency_s"]
        # The judge scores substantive answers only: an eligibility
        # statement is a template + verbatim evidence, not a
        # generated answer, and the 0.5B judge misreads its format.
        if judge and result["found"] and result.get("answer_mode") != "eligibility":
            judge_chunks = result["citations"] or retrieved[:3]
            verdict = judge_answer(q["question"], result["answer"],
                                   judge_chunks, judge_fn=judge_fn)
            row["judge"] = verdict
            if verdict["parsed"]:
                judge_n += 1
                judge_faith += verdict["faithfulness"]
                judge_rel += verdict["relevance"]
            else:
                judge_unparsed += 1
        per_question.append(row)
        if progress is not None:
            progress(qi, len(in_corpus), q["id"])

    abstained = 0
    for q in out_corpus:
        retrieved = retriever.search(q["question"], top_k=5, **search_kwargs)
        if generative:
            result = generate_answer(q["question"], retrieved)
        else:
            result = build_answer(q["question"], retrieved, **answer_kwargs)
        if not result["found"]:
            abstained += 1

    n = max(1, len(in_corpus))
    out = {
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
    if generative:
        out["generative_modes"] = generative_modes
    if judge:
        out["judge_faithfulness"] = round(judge_faith / judge_n, 3) if judge_n else None
        out["judge_relevance"] = round(judge_rel / judge_n, 3) if judge_n else None
        out["judge_n"] = judge_n
        out["judge_unparsed"] = judge_unparsed
    return out
