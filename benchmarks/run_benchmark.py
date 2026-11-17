"""Benchmark: baseline vs improved pipeline on the same golden set.

Run: python -m benchmarks.run_benchmark

Configurations (all on eval/golden.json, 20 in-corpus + 5 out-of-corpus):
- baseline-naive: BM25-only, no rerank, no abstention gates,
  first-sentence picking, always answers. This is the tutorial
  chatbot behaviour: retrieve by keyword, return the first
  sentences of the top chunk, never say "I don't know".
- improved-bm25: BM25-only + lexical rerank + scored sentence
  picking + abstention gates. Same retrieval signal as the
  baseline; only the answering discipline changes.
- improved-hybrid: dense (all-MiniLM-L6-v2, FAISS) + BM25 hybrid
  + rerank + scored picking + abstention gates. The full Phase 1
  pipeline.

Metrics per configuration: recall@3, recall@5, answer relevance
(proxy), faithfulness, abstention accuracy, and query latency
(mean / p50 / p95, milliseconds, search + answer after warm-up,
embedding model already loaded). Cost per query is $0.00 for every
configuration: everything runs locally on CPU, no API calls.

Results are written to benchmarks/results/<config>.json and a
human-readable summary to benchmarks/results/summary.md. Nothing
here is invented: every number comes from this run.
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
        "description": "BM25-only, no rerank, no abstention gates, first-sentence picking, always answers.",
        "dense": False,
        "search_kwargs": {"rerank": False},
        "answer_kwargs": {"gates": False, "scored_picking": False},
    },
    {
        "name": "improved-bm25",
        "description": "BM25-only + lexical rerank + scored sentence picking + abstention gates.",
        "dense": False,
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
    {
        "name": "improved-hybrid",
        "description": "Dense + BM25 hybrid + rerank + scored picking + abstention gates (full Phase 1).",
        "dense": True,
        "search_kwargs": {"rerank": True},
        "answer_kwargs": {"gates": True, "scored_picking": True},
    },
]


def measure_latency(retriever: Retriever, golden: list[dict],
                    search_kwargs: dict, answer_kwargs: dict) -> dict:
    """Time search + answer per question (warm-up first, model loaded)."""
    questions = [q["question"] for q in golden]
    # Warm-up: first call includes one-off costs, exclude it.
    retriever.search(questions[0], top_k=5, **search_kwargs)
    times_ms = []
    for question in questions:
        start = time.perf_counter()
        retrieved = retriever.search(question, top_k=5, **search_kwargs)
        build_answer(question, retrieved, **answer_kwargs)
        times_ms.append((time.perf_counter() - start) * 1000)
    times_ms.sort()
    n = len(times_ms)
    return {
        "n_queries": n,
        "mean_ms": round(statistics.mean(times_ms), 1),
        "p50_ms": round(times_ms[n // 2], 1),
        "p95_ms": round(times_ms[int(n * 0.95) - 1], 1),
        "max_ms": round(times_ms[-1], 1),
    }


def run_config(chunks: list[dict], golden: list[dict], config: dict) -> dict:
    retriever = Retriever(chunks, dense=config["dense"])
    quality = run_eval(retriever, golden,
                       search_kwargs=config["search_kwargs"],
                       answer_kwargs=config["answer_kwargs"])
    latency = measure_latency(retriever, golden,
                              config["search_kwargs"], config["answer_kwargs"])
    return {
        "config": config["name"],
        "description": config["description"],
        "embedder": retriever.embedder,
        "n_chunks": len(chunks),
        "quality": {k: v for k, v in quality.items() if k != "per_question"},
        "latency": latency,
        # Local CPU, no API calls: cost is genuinely zero, not estimated.
        "cost_per_query_usd": 0.0,
        "per_question": quality["per_question"],
    }


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    # Benchmarks compare the retrieval/answering pipeline on the 25
    # factual questions. The 10 eligibility questions carry a profile
    # and are measured separately by the eval harness (eligibility
    # accuracy) and the CI gate — mixing them in would compare the
    # baseline's substantive answers against eligibility verdicts.
    golden = [q for q in load_golden(GOLDEN_FILE) if not q.get("profile")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for config in CONFIGS:
        result = run_config(chunks, golden, config)
        results.append(result)
        (RESULTS_DIR / f"{config['name']}.json").write_text(json.dumps(result, indent=2))
        q = result["quality"]
        print(f"{config['name']}: recall@3={q['recall@3']} recall@5={q['recall@5']} "
              f"relevance={q['answer_relevance_proxy']} "
              f"answer_only={q['answer_only_relevance']} "
              f"faithfulness={q['faithfulness']} "
              f"abstention={q['abstention_accuracy']} "
              f"latency_mean={result['latency']['mean_ms']}ms "
              f"cost=${result['cost_per_query_usd']:.2f}/query")

    lines = ["# Benchmark results (Phase 1)\n",
             "Same golden set for every configuration: 20 in-corpus + "
             "5 out-of-corpus questions. Latency is search + answer after "
             "warm-up, embedding model already loaded. Cost is $0.00: "
             "everything runs locally on CPU, no API calls.\n",
             "| Configuration | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Latency mean | Cost / query |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        q, lat = r["quality"], r["latency"]
        lines.append(f"| {r['config']} | {q['recall@3']} | {q['recall@5']} | "
                     f"{q['answer_relevance_proxy']} | {q['answer_only_relevance']} | "
                     f"{q['faithfulness']} | "
                     f"{q['abstention_accuracy']} | {lat['mean_ms']} ms | "
                     f"${r['cost_per_query_usd']:.2f} |")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nResults written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
