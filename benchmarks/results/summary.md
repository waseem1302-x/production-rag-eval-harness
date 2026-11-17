# Benchmark results (Phase 1)

Same golden set for every configuration: 20 in-corpus + 5 out-of-corpus questions. Latency is search + answer after warm-up, embedding model already loaded. Cost is $0.00: everything runs locally on CPU, no API calls.

| Configuration | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Latency mean | Cost / query |
|---|---|---|---|---|---|---|---|---|
| baseline-naive | 1.0 | 1.0 | 1.0 | 0.45 | 1.0 | 0.0 | 0.4 ms | $0.00 |
| improved-bm25 | 1.0 | 1.0 | 1.0 | 0.7 | 1.0 | 0.8 | 2.2 ms | $0.00 |
| improved-hybrid | 1.0 | 1.0 | 1.0 | 0.7 | 1.0 | 0.8 | 175.1 ms | $0.00 |
