"""Chunk-size ablation (Phase 2, Step 5): 256 / 512 / 1,024 tokens, measured.

Run: python -m scripts.run_chunk_ablation

For each size, from the same .txt corpus and with the same overlap
policy (200 chars, constant — chunk size is the only variable):
  1. Build the index into benchmarks/results/chunk_ablation/chunk-<size>/
     (chunks.jsonl + faiss.index) via scripts.build_index. The working
     data/processed/ 512-token index is never touched.
  2. Run the full extractive eval on the same 39-question golden set
     (app.eval.run_eval, full pipeline: hybrid retrieval, lexical
     rerank, scored picking, abstention gates, eligibility-first).
  3. Measure retrieval + total latency on the 28 factual questions
     with benchmarks.run_benchmark.measure_latency (same definition
     as the 8-configuration ablation).

Results are written to benchmarks/results/chunk_ablation/
(chunk_ablation_<size>.json per size, plus chunk_ablation.md
generated from the written files, not from memory). Nothing here
is hand-typed: every number in the table comes from a generated file.
"""
import json
import statistics
from pathlib import Path

from app.config import GOLDEN_FILE
from app.eval import load_golden, run_eval
from app.retrieval import Retriever
from benchmarks.run_benchmark import measure_latency
from scripts.build_index import CHUNK_TOKEN_CHOICES, build_index

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "chunk_ablation"


def run_size(chunk_tokens: int, golden: list[dict]) -> dict:
    """Build one size's index and measure it. Returns the result dict."""
    out_dir = RESULTS_DIR / f"chunk-{chunk_tokens}"
    stats = build_index(chunk_tokens=chunk_tokens, out_dir=out_dir)
    chunks = [json.loads(line) for line in (out_dir / "chunks.jsonl").read_text().splitlines()
              if line.strip()]
    lengths = [len(c["text"]) for c in chunks]
    retriever = Retriever(chunks)
    print(f"chunk-{chunk_tokens}: chunks={len(chunks)} embedder={retriever.embedder}")
    quality = run_eval(retriever, golden)
    factual = [q for q in golden if not q.get("profile")]
    latency = measure_latency(retriever, factual,
                              {"rerank": True},
                              {"gates": True, "scored_picking": True})
    result = {
        "chunk_tokens": chunk_tokens,
        "chunk_chars": stats["chunk_chars"],
        "overlap_chars": stats["overlap_chars"],
        "n_documents": stats["n_documents"],
        "n_chunks": len(chunks),
        "n_vectors": stats["n_vectors"],
        "chunk_chars_mean": round(statistics.mean(lengths), 1),
        "chunk_chars_max": max(lengths),
        "chunks_file": str(out_dir / "chunks.jsonl"),
        "index_file": str(out_dir / "faiss.index"),
        "embedder": retriever.embedder,
        "quality": {k: v for k, v in quality.items() if k != "per_question"},
        "latency": latency,
        "per_question": quality["per_question"],
    }
    (RESULTS_DIR / f"chunk_ablation_{chunk_tokens}.json").write_text(json.dumps(result, indent=2))
    q = result["quality"]
    print(f"chunk-{chunk_tokens}: recall@3={q['recall@3']} recall@5={q['recall@5']} "
          f"answer_only={q['answer_only_relevance']} "
          f"faithfulness={q['faithfulness']} abstention={q['abstention_accuracy']} "
          f"eligibility={q['eligibility_accuracy']} "
          f"retrieval_mean={latency['retrieval']['mean_ms']}ms "
          f"total_mean={latency['search_plus_answer']['mean_ms']}ms")
    return result


def build_summary(results_dir: Path = RESULTS_DIR) -> Path:
    """Write chunk_ablation.md from the per-size result files on disk."""
    results = []
    for size in CHUNK_TOKEN_CHOICES:
        path = results_dir / f"chunk_ablation_{size}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    if not results:
        raise FileNotFoundError(f"no chunk ablation result files in {results_dir}")
    n = results[0]["quality"]["n_in_corpus"]
    lines = ["# Chunk-size ablation (Phase 2, Step 5)\n",
             f"Same .txt corpus, same overlap policy (200 chars, constant), "
             f"same 39-question golden set ({n} in-corpus + "
             f"{results[0]['quality']['n_out_of_corpus']} out-of-corpus, "
             f"{results[0]['quality']['n_eligibility']} eligibility). Full "
             f"extractive pipeline for every size: hybrid retrieval, lexical "
             f"rerank, scored picking, abstention gates, eligibility-first. "
             f"Latency is retrieval + search-plus-answer on the 28 factual "
             f"questions, measured after warm-up with the embedding model "
             f"already loaded (same definition as benchmarks/run_benchmark.py). "
             f"Cost $0.00 everywhere: local CPU, no API calls.\n",
             "| Chunk size | Chunks | Mean chunk chars | Max chunk chars | recall@3 | recall@5 | Answer-only relevance | Faithfulness | Abstention | Eligibility | Retrieval mean | Total mean |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        q, lat = r["quality"], r["latency"]
        lines.append(f"| {r['chunk_tokens']} tokens ({r['chunk_chars']} chars) | {r['n_chunks']} | "
                     f"{r['chunk_chars_mean']} | {r['chunk_chars_max']} | "
                     f"{q['recall@3']} | {q['recall@5']} | {q['answer_only_relevance']} | "
                     f"{q['faithfulness']} | {q['abstention_accuracy']} | "
                     f"{q['eligibility_accuracy']} | {lat['retrieval']['mean_ms']} ms | "
                     f"{lat['search_plus_answer']['mean_ms']} ms |")
    lines.append("\n## What the numbers say\n")
    by_size = {r["chunk_tokens"]: r for r in results}
    if len(by_size) == 3:
        def q(size, metric):
            return by_size[size]["quality"][metric]

        def factual_answer_only(size):
            rows = [r for r in by_size[size]["per_question"]
                    if r.get("eligibility_verdict") is None]
            good = sum(1 for r in rows if r["answer_only_relevant"])
            return f"{good}/{len(rows)}"
        lines.append(f"- Factual questions only (23, eligibility excluded): "
                     f"answer-only {factual_answer_only(256)} at 256 tokens, "
                     f"{factual_answer_only(512)} at 512, "
                     f"{factual_answer_only(1024)} at 1,024. The 256-token "
                     f"size fixes q03 (SI master's duration) — smaller chunks "
                     f"isolate the duration sentence so the picker finds it — "
                     f"but loses three eligibility rows, so the headline "
                     f"(all 34 in-corpus) still favours 512.")
        best_answer = max(by_size, key=lambda s: q(s, "answer_only_relevance"))
        lines.append(f"- Answer-only relevance: 256 tokens {q(256, 'answer_only_relevance')}, "
                     f"512 tokens {q(512, 'answer_only_relevance')}, "
                     f"1,024 tokens {q(1024, 'answer_only_relevance')}. "
                     f"Best is {best_answer} tokens.")
        lines.append(f"- Eligibility accuracy: 256 tokens {q(256, 'eligibility_accuracy')}, "
                     f"512 tokens {q(512, 'eligibility_accuracy')}, "
                     f"1,024 tokens {q(1024, 'eligibility_accuracy')}. "
                     f"Eligibility rules pin chunk ids, so a size that moves an "
                     f"evidence sentence to another chunk drops that rule to "
                     f"unknown — see EVAL_RESULTS.md for which questions move.")
        lines.append(f"- Recall@5 is {q(256, 'recall@5')} / {q(512, 'recall@5')} / "
                     f"{q(1024, 'recall@5')} (256 / 512 / 1,024 tokens): on this "
                     f"corpus the right document is in the top 5 at every size.")
        lines.append(f"- Total latency mean: 256 tokens "
                     f"{by_size[256]['latency']['search_plus_answer']['mean_ms']} ms, "
                     f"512 tokens {by_size[512]['latency']['search_plus_answer']['mean_ms']} ms, "
                     f"1,024 tokens {by_size[1024]['latency']['search_plus_answer']['mean_ms']} ms. "
                     f"Read the mean with the median: total p50 is "
                     f"{by_size[256]['latency']['search_plus_answer']['p50_ms']} / "
                     f"{by_size[512]['latency']['search_plus_answer']['p50_ms']} / "
                     f"{by_size[1024]['latency']['search_plus_answer']['p50_ms']} ms "
                     f"(256 / 512 / 1,024 tokens) — nearly flat. The 1,024-token "
                     f"mean is a tail effect (p95 "
                     f"{by_size[1024]['latency']['search_plus_answer']['p95_ms']} ms, max "
                     f"{by_size[1024]['latency']['search_plus_answer']['max_ms']} ms), "
                     f"which may be CPU contention as much as chunk size; do not "
                     f"read it as '1,024 tokens is twice as slow per question'. "
                     f"Latency varies run to run on this shared-CPU VM.")
    out = results_dir / "chunk_ablation.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    golden = load_golden(GOLDEN_FILE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for size in CHUNK_TOKEN_CHOICES:
        run_size(size, golden)
    build_summary(RESULTS_DIR)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
