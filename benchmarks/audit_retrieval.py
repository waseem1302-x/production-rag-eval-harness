"""Retrieval audit: exact chunks retrieved, reranked and injected.

Run: python -m benchmarks.audit_retrieval

For every golden question, record the full retrieval trace: each
candidate chunk with its dense score, BM25 score, combined score,
rerank score and term coverage, in reranked order, and mark which
chunks were actually injected into the answer (the cited ones).
Also record the abstention reason for refused questions.

Output: benchmarks/results/retrieval_audit.json (machine-readable)
and benchmarks/results/retrieval_audit.md (human-readable).

This is the artefact that makes retrieval auditable: when an answer
is wrong, this file shows whether retrieval failed (right chunk not
in the candidates), reranking failed (right chunk retrieved but
ranked below the cut), or answering failed (right chunk injected
but the wrong sentences picked).
"""
import json
from pathlib import Path

from app.config import CHUNKS_FILE
from app.eval import _answer_question, load_golden
from app.retrieval import Retriever

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def audit() -> list[dict]:
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    golden = load_golden()
    rows = []
    for q in golden:
        retrieved = retriever.search(q["question"], top_k=5)
        result, eligibility = _answer_question(
            q["question"], q.get("profile"), retrieved, chunks, {})
        injected = {c["chunk_id"] for c in result["citations"]}
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_doc": q.get("expected_doc") or None,
            "out_of_corpus": bool(q.get("out_of_corpus")),
            "found": result["found"],
            "abstention_reason": result.get("abstention_reason"),
            "answer": result["answer"],
            "candidates": [
                {"rank": i + 1, "chunk_id": c["chunk_id"], "doc": c["doc"],
                 "dense_score": c["dense_score"], "bm25_score": c["bm25_score"],
                 "combined_score": c["score"], "rerank_score": c["rerank_score"],
                 "term_coverage": c.get("term_coverage"),
                 "injected": c["chunk_id"] in injected,
                 "text": c["text"][:300]}
                for i, c in enumerate(retrieved)
            ],
        })
    return rows


def main() -> None:
    rows = audit()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_audit.json").write_text(json.dumps(rows, indent=2))
    lines = ["# Retrieval audit (Phase 1)\n",
             "For each golden question: the candidates in reranked order, "
             "with dense / BM25 / combined / rerank scores, and whether "
             "each chunk was injected into the answer.\n"]
    for row in rows:
        lines.append(f"## {row['id']}: {row['question']}")
        lines.append(f"Expected doc: {row['expected_doc'] or '— (out of corpus)'} | "
                     f"found: {row['found']} | "
                     f"abstention: {row['abstention_reason'] or '—'}")
        lines.append(f"Answer: {row['answer'][:200]}")
        for c in row["candidates"]:
            mark = "INJECTED" if c["injected"] else "—"
            lines.append(f"- #{c['rank']} {c['chunk_id']} [{mark}] "
                         f"dense={c['dense_score']} bm25={c['bm25_score']} "
                         f"combined={c['combined_score']} rerank={c['rerank_score']} "
                         f"coverage={c['term_coverage']} :: {c['text'][:120]}")
        lines.append("")
    (RESULTS_DIR / "retrieval_audit.md").write_text("\n".join(lines))
    injected = sum(1 for r in rows for c in r["candidates"] if c["injected"])
    print(f"audited {len(rows)} questions, {injected} injected chunks; "
          f"wrote {RESULTS_DIR}/retrieval_audit.json")


if __name__ == "__main__":
    main()
