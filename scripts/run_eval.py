"""Run the eval harness and print real numbers. Usage: python -m scripts.run_eval

Also writes benchmarks/results/extractive_eval.json (full result,
per-question rows included) so the numbers are recorded by the
script, not hand-typed into EVAL_RESULTS.md.
"""
import json
from pathlib import Path

from app.config import CHUNKS_FILE
from app.eval import load_golden, run_eval
from app.retrieval import Retriever

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

if __name__ == "__main__":
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    print(f"chunks: {len(chunks)}  embedder: {retriever.embedder}")
    result = run_eval(retriever, load_golden())
    for key in ["n_in_corpus", "n_out_of_corpus", "recall@3", "recall@5",
                "answer_relevance_proxy", "answer_only_relevance",
                "faithfulness", "abstention_accuracy"]:
        print(f"{key}: {result[key]}")
    print()
    for row in result["per_question"]:
        print(f"{row['id']}: recall@3={row['recall@3']} recall@5={row['recall@5']} "
              f"found={row['found']} relevant={row['answer_relevant']} "
              f"top5={row['found_doc_top5']}")
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "extractive_eval.json").write_text(json.dumps(result, indent=2))
    print("wrote benchmarks/results/extractive_eval.json")
