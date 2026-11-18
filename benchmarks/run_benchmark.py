"""Benchmark and ablation: every pipeline component measured, not assumed.

Run: python -m benchmarks.run_benchmark

All configurations run on the same golden set (the factual
questions from eval/golden.json; the 10 eligibility questions are
excluded here and measured separately by the eval harness and the
CI gate, because they need a profile). Latency is measured after
warm-up with the embedding model already loaded. Cost per query is
$0.00 for every configuration: everything runs locally on CPU, no
API calls.

Ablation design — each configuration changes ONE thing from its
nearest neighbour, so each component's contribution is isolated:
- baseline-naive vs hybrid-plain: the retrieval signal only
  (BM25 vs hybrid), same naive answering.
- improved-bm25 vs dense-only vs improved-hybrid: the retrieval
  signal at full answering discipline (BM25 vs dense vs hybrid).
- hybrid-no-rerank vs improved-hybrid: the lexical reranker only.
- hybrid-no-gates vs improved-hybrid: the abstention gates only
  (scored picking and rerank held constant).
- improved-hybrid vs hybrid-cross-encoder: the reranker only
  (lexical coverage bonus vs cross-encoder pair scoring).
- baseline-naive vs improved-bm25: the full answering discipline
  (rerank + scored picking + gates) at fixed BM25 retrieval.

Configurations:
- baseline-naive: BM25, no rerank, no abstention gates,
  first-sentence picking, always answers. The tutorial chatbot.
- hybrid-plain: hybrid retrieval, otherwise naive answering.
- improved-bm25: BM25 + rerank + scored picking + gates.
- dense-only: dense retrieval + rerank + scored picking + gates.
- hybrid-no-rerank: hybrid, no rerank, scored picking + gates.
- hybrid-no-gates: hybrid + rerank + scored picking, no gates.
- improved-hybrid: hybrid + rerank + scored picking + gates.
  The full Phase 1 pipeline.
- hybrid-cross-encoder: hybrid + cross-encoder rerank + scored
  picking + gates. The Phase 2, Step 3 ablation: everything
  identical to improved-hybrid except the reranker, so any delta
  is the cross-encoder's.

Results are written to benchmarks/results/<config>.json and a
human-readable ablation summary to benchmarks/results/summary.md.
The summary is generated from the written result files
(`build_summary`), not from the in-memory run, so the table cannot
drift from the files. Nothing here is invented: every number comes
from this run.
"""
import json
import statistics
import time
from pathlib import Path

from app.answer import answer as build_answer
from app.config import CHUNKS_FILE, GOLDEN_FILE
from app.eval import load_golden, run_eval
from app.retrieval import Retriever

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = [
    {
        "name": "baseline-naive",
        "description": "BM25, no rerank, no abstention gates, first-sentence picking, always answers.",
        "mode": "bm25",
        "search_kwargs": {"rerank": False},
        "answer_kwargs": {"gates": False, "scored_picking": False},
    },
    {
        "name": "hybrid-plain",
        "description": "Hybrid retrieval, otherwise naive answering (isolates the retrieval signal).",
        "mode": "hybrid",
        "search_kwargs": {"rerank": False},
        "answer_kwargs": {"gates": False, "scored_picking": False},
    },
    {
        "name": "improved-bm25",
        "description": "BM25 + rerank + scored picking + abstention gates.",
        "mode": "bm25",
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
    {
        "name": "dense-only",
        "description": "Dense retrieval + rerank + scored picking + abstention gates.",
        "mode": "dense",
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
    {
        "name": "hybrid-no-rerank",
        "description": "Hybrid, no rerank, scored picking + abstention gates (isolates the reranker).",
        "mode": "hybrid",
        "search_kwargs": {"rerank": False},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
    {
        "name": "hybrid-no-gates",
        "description": "Hybrid + rerank + scored picking, no abstention gates (isolates the gates).",
        "mode": "hybrid",
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": False, "scored_picking": True},
    },
    {
        "name": "improved-hybrid",
        "description": "Hybrid + rerank + scored picking + abstention gates (full Phase 1).",
        "mode": "hybrid",
        "reranker": "lexical",
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
    {
        "name": "hybrid-cross-encoder",
        "description": "Hybrid + cross-encoder rerank + scored picking + abstention gates (Phase 2, Step 3).",
        "mode": "hybrid",
        "reranker": "cross-encoder",
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
]


def _percentiles(times_ms: list[float]) -> dict:
    times_ms = sorted(times_ms)
    n = len(times_ms)
    return {
        "mean_ms": round(statistics.mean(times_ms), 1),
        "p50_ms": round(times_ms[n // 2], 1),
        "p95_ms": round(times_ms[int(n * 0.95) - 1], 1),
        "max_ms": round(times_ms[-1], 1),
    }


def measure_latency(retriever: Retriever, golden: list[dict],
                    search_kwargs: dict, answer_kwargs: dict) -> dict:
    """Time retrieval alone and search + answer per question
    (warm-up first, model loaded). Retrieval-only latency is what
    separates the signals: dense embedding dominates it, BM25 is
    near-free."""
    questions = [q["question"] for q in golden]
    # Warm-up: first call includes one-off costs, exclude it.
    retriever.search(questions[0], top_k=5, **search_kwargs)
    retrieval_ms = []
    total_ms = []
    for question in questions:
        start = time.perf_counter()
        retrieved = retriever.search(question, top_k=5, **search_kwargs)
        retrieval_ms.append((time.perf_counter() - start) * 1000)
        build_answer(question, retrieved, **answer_kwargs)
        total_ms.append((time.perf_counter() - start) * 1000)
    return {
        "n_queries": len(questions),
        "retrieval": _percentiles(retrieval_ms),
        "search_plus_answer": _percentiles(total_ms),
    }


def run_config(chunks: list[dict], golden: list[dict], config: dict) -> dict:
    retriever = Retriever(chunks, mode=config["mode"],
                          reranker=config.get("reranker", "lexical"))
    quality = run_eval(retriever, golden,
                       search_kwargs=config["search_kwargs"],
                       answer_kwargs=config["answer_kwargs"])
    latency = measure_latency(retriever, golden,
                              config["search_kwargs"], config["answer_kwargs"])
    return {
        "config": config["name"],
        "description": config["description"],
        "retrieval_mode": config["mode"],
        "reranker": retriever.reranker,
        "embedder": retriever.embedder,
        "n_chunks": len(chunks),
        "quality": {k: v for k, v in quality.items() if k != "per_question"},
        "latency": latency,
        # Local CPU, no API calls: cost is genuinely zero, not estimated.
        "cost_per_query_usd": 0.0,
        "per_question": quality["per_question"],
    }


def _ablation_lines(results: list[dict]) -> list[str]:
    """Computed comparisons between neighbouring configurations.
    Every delta is derived from the measured numbers above, so the
    text cannot drift from the table. A comparison is only written
    when both configurations have result files — a partial results
    directory still produces a usable summary instead of crashing.
    """
    by_name = {r["config"]: r for r in results}

    def q(name, metric):
        return by_name[name]["quality"][metric]

    def lat(name):
        return by_name[name]["latency"]["retrieval"]["mean_ms"]

    def have(*names):
        return all(n in by_name for n in names)

    lines = ["\n## What the ablation shows\n"]
    if have("baseline-naive", "hybrid-plain"):
        lines.append(
            f"- Retrieval signal, naive answering: hybrid-plain vs baseline-naive. "
            f"Recall@5 {q('baseline-naive', 'recall@5')} → {q('hybrid-plain', 'recall@5')}, "
            f"answer-only relevance {q('baseline-naive', 'answer_only_relevance')} → "
            f"{q('hybrid-plain', 'answer_only_relevance')}. If these are equal, the "
            f"dense signal buys nothing on this corpus at naive answering.")
    if have("improved-bm25", "dense-only", "improved-hybrid"):
        lines.append(
            f"- Retrieval signal, full discipline: BM25 {q('improved-bm25', 'recall@5')} / "
            f"dense {q('dense-only', 'recall@5')} / hybrid {q('improved-hybrid', 'recall@5')} "
            f"recall@5; answer-only {q('improved-bm25', 'answer_only_relevance')} / "
            f"{q('dense-only', 'answer_only_relevance')} / "
            f"{q('improved-hybrid', 'answer_only_relevance')}. The signal that wins "
            f"here is the one worth paying the latency for.")
    if have("hybrid-no-rerank", "improved-hybrid"):
        lines.append(
            f"- Reranker: hybrid-no-rerank vs improved-hybrid. Answer-only relevance "
            f"{q('hybrid-no-rerank', 'answer_only_relevance')} → "
            f"{q('improved-hybrid', 'answer_only_relevance')}, recall@5 "
            f"{q('hybrid-no-rerank', 'recall@5')} → {q('improved-hybrid', 'recall@5')}.")
    if have("improved-hybrid", "hybrid-cross-encoder"):
        lines.append(
            f"- Cross-encoder reranker: improved-hybrid (lexical) vs "
            f"hybrid-cross-encoder. Answer-only relevance "
            f"{q('improved-hybrid', 'answer_only_relevance')} → "
            f"{q('hybrid-cross-encoder', 'answer_only_relevance')}, recall@5 "
            f"{q('improved-hybrid', 'recall@5')} → "
            f"{q('hybrid-cross-encoder', 'recall@5')}, abstention "
            f"{q('improved-hybrid', 'abstention_accuracy')} → "
            f"{q('hybrid-cross-encoder', 'abstention_accuracy')}, retrieval mean "
            f"{lat('improved-hybrid')} ms → {lat('hybrid-cross-encoder')} ms. "
            f"Same retrieval, same answering — any delta here is the "
            f"cross-encoder's, and the latency is what it costs.")
    if have("hybrid-no-gates", "improved-hybrid"):
        lines.append(
            f"- Abstention gates: hybrid-no-gates vs improved-hybrid. Abstention accuracy "
            f"{q('hybrid-no-gates', 'abstention_accuracy')} → "
            f"{q('improved-hybrid', 'abstention_accuracy')}, answer-only relevance "
            f"{q('hybrid-no-gates', 'answer_only_relevance')} → "
            f"{q('improved-hybrid', 'answer_only_relevance')}. The gates are what stop "
            f"the system answering questions the corpus cannot answer.")
    if have("improved-bm25", "dense-only", "improved-hybrid"):
        lines.append(
            f"- Latency (retrieval mean): BM25 {lat('improved-bm25')} ms, dense "
            f"{lat('dense-only')} ms, hybrid {lat('improved-hybrid')} ms. Dense "
            f"embedding is the cost; BM25 is near-free. Any recall gain from dense "
            f"must be weighed against this.")
    if have("baseline-naive", "improved-hybrid"):
        lines.append(
            f"- Full pipeline vs naive baseline: answer-only relevance "
            f"{q('baseline-naive', 'answer_only_relevance')} → "
            f"{q('improved-hybrid', 'answer_only_relevance')}, abstention "
            f"{q('baseline-naive', 'abstention_accuracy')} → "
            f"{q('improved-hybrid', 'abstention_accuracy')}, faithfulness "
            f"{q('baseline-naive', 'faithfulness')} → {q('improved-hybrid', 'faithfulness')}.")
    return lines


def build_summary(results_dir: Path = RESULTS_DIR) -> Path:
    """Write summary.md from the per-configuration result files.

    The summary is generated from the files on disk, not from the
    in-memory run: re-running it on the same directory reproduces
    the same table, and a stale or edited file shows up in the
    summary instead of being silently overwritten by memory.
    Configurations are listed in CONFIGS order; a result file for
    a configuration not in CONFIGS is ignored.
    """
    results = []
    for config in CONFIGS:
        path = results_dir / f"{config['name']}.json"
        if not path.exists():
            continue
        results.append(json.loads(path.read_text()))
    if not results:
        raise FileNotFoundError(f"no benchmark result files in {results_dir}")
    n_in = results[0]["quality"]["n_in_corpus"]
    n_out = results[0]["quality"]["n_out_of_corpus"]
    lines = ["# Benchmark results (Phase 2 improvement — re-run with current code)\n",
             f"Same golden set for every configuration: {n_in} in-corpus + "
             f"{n_out} out-of-corpus questions. Latency is measured after "
             "warm-up with the embedding model already loaded. Cost is $0.00: "
             "everything runs locally on CPU, no API calls. The Step 3 "
             "numbers in the docs were measured on the 31-chunk corpus "
             "and the 39-question golden set, so this table is a fresh "
             "re-run of all configurations with the current code on the "
             "grown corpus.\n",
             "Latency varies run to run by tens of milliseconds on this VM "
             "(shared CPU, 1.5 cores). The order of magnitude — BM25 ~1 ms, "
             "dense ~50–150 ms, cross-encoder rerank ~seconds per query — "
             "is stable across runs; the exact figures are one measured "
             "run, not a guarantee.\n",
             "| Configuration | Mode | Reranker | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        q = r["quality"]
        ret, tot = r["latency"]["retrieval"], r["latency"]["search_plus_answer"]
        lines.append(f"| {r['config']} | {r['retrieval_mode']} | {r.get('reranker', 'lexical')} | "
                     f"{q['recall@3']} | {q['recall@5']} | "
                     f"{q['answer_relevance_proxy']} | {q['answer_only_relevance']} | "
                     f"{q['faithfulness']} | {q['abstention_accuracy']} | "
                     f"{ret['mean_ms']} ms | {tot['mean_ms']} ms | "
                     f"${r['cost_per_query_usd']:.2f} |")
    lines.extend(_ablation_lines(results))
    out = results_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    # Benchmarks compare the retrieval/answering pipeline on the
    # factual questions. The 10 eligibility questions carry a profile
    # and are measured separately by the eval harness (eligibility
    # accuracy) and the CI gate — mixing them in would compare the
    # baseline's substantive answers against eligibility verdicts.
    golden = [q for q in load_golden(GOLDEN_FILE) if not q.get("profile")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        result = run_config(chunks, golden, config)
        (RESULTS_DIR / f"{config['name']}.json").write_text(json.dumps(result, indent=2))
        q = result["quality"]
        print(f"{config['name']}: recall@3={q['recall@3']} recall@5={q['recall@5']} "
              f"relevance={q['answer_relevance_proxy']} "
              f"answer_only={q['answer_only_relevance']} "
              f"faithfulness={q['faithfulness']} "
              f"abstention={q['abstention_accuracy']} "
              f"retrieval_mean={result['latency']['retrieval']['mean_ms']}ms "
              f"total_mean={result['latency']['search_plus_answer']['mean_ms']}ms "
              f"cost=${result['cost_per_query_usd']:.2f}/query")

    build_summary(RESULTS_DIR)
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
