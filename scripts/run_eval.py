"""Run the eval harness and print real numbers. Usage: python -m scripts.run_eval"""
import json

from app.config import CHUNKS_FILE
from app.eval import load_golden, run_eval
from app.retrieval import Retriever

if __name__ == "__main__":
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    print(f"chunks: {len(chunks)}  embedder: {retriever.embedder}")
    result = run_eval(retriever, load_golden())
    for key in ["n_in_corpus", "n_out_of_corpus", "recall@3", "recall@5",
                "answer_relevance_proxy", "faithfulness", "abstention_accuracy"]:
        print(f"{key}: {result[key]}")
    print()
    for row in result["per_question"]:
        print(f"{row['id']}: recall@3={row['recall@3']} recall@5={row['recall@5']} "
              f"found={row['found']} relevant={row['answer_relevant']} "
              f"top5={row['found_doc_top5']}")
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=2))
