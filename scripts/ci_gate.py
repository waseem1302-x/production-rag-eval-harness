"""CI gate: run the eval harness and fail if quality drops below threshold.

Run: python -m scripts.ci_gate
Run strict (all thresholds at 1.0, to prove the gate can fail):
  CI_GATE_STRICT=1 python -m scripts.ci_gate

Thresholds are set below the numbers measured on 18 Nov 2026 so
normal run-to-run jitter (embedding latency, tie-breaking) does not
fail the build, but a real regression does:
  recall@5            >= 0.90  (measured 1.0)
  answer_only_relevance >= 0.60  (measured 0.85)
  faithfulness        >= 0.95  (measured 1.0)
  abstention_accuracy >= 0.60  (measured 0.8)

If a threshold is ever lowered, the reason must be written in the
commit message — never silently weaken the gate to make a red build
green.
"""
import os
import sys

from app.config import CHUNKS_FILE
from app.eval import load_golden, run_eval
from app.retrieval import Retriever
import json

THRESHOLDS = {
    "recall@5": 0.90,
    "answer_only_relevance": 0.60,
    "faithfulness": 0.95,
    "abstention_accuracy": 0.60,
    "eligibility_accuracy": 0.80,
}


def main() -> int:
    strict = os.environ.get("CI_GATE_STRICT") == "1"
    thresholds = {k: 1.0 for k in THRESHOLDS} if strict else THRESHOLDS
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    retriever = Retriever(chunks)
    result = run_eval(retriever, load_golden())
    failed = []
    for metric, minimum in thresholds.items():
        value = result[metric]
        if value is None:
            print(f"SKIP: {metric} not measured (no eligibility questions in set)")
            continue
        status = "PASS" if value >= minimum else "FAIL"
        print(f"{status}: {metric} = {value} (threshold {minimum})")
        if value < minimum:
            failed.append(metric)
    if failed:
        print(f"CI GATE FAILED: {', '.join(failed)} below threshold.")
        return 1
    print("CI GATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
