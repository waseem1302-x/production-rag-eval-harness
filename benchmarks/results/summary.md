# Benchmark results (Phase 1)

Same golden set for every configuration: 23 in-corpus + 5 out-of-corpus questions. Latency is measured after warm-up with the embedding model already loaded. Cost is $0.00: everything runs locally on CPU, no API calls.

Latency varies run to run by tens of milliseconds on this VM (shared CPU, 1.5 cores). The order of magnitude — BM25 ~1 ms, dense ~50–70 ms — is stable across runs; the exact figures are one measured run, not a guarantee.

| Configuration | Mode | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | 1.0 | 1.0 | 1.0 | 0.522 | 1.0 | 0.0 | 0.2 ms | 0.5 ms | $0.00 |
| hybrid-plain | hybrid | 1.0 | 1.0 | 1.0 | 0.565 | 1.0 | 0.0 | 39.9 ms | 42.7 ms | $0.00 |
| improved-bm25 | bm25 | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 0.8 | 1.3 ms | 2.3 ms | $0.00 |
| dense-only | dense | 1.0 | 1.0 | 1.0 | 0.826 | 1.0 | 0.8 | 108.8 ms | 110.2 ms | $0.00 |
| hybrid-no-rerank | hybrid | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 0.8 | 99.9 ms | 103.8 ms | $0.00 |
| hybrid-no-gates | hybrid | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 0.0 | 69.9 ms | 75.6 ms | $0.00 |
| improved-hybrid | hybrid | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 0.8 | 66.4 ms | 70.2 ms | $0.00 |

## What the ablation shows

- Retrieval signal, naive answering: hybrid-plain vs baseline-naive. Recall@5 1.0 → 1.0, answer-only relevance 0.522 → 0.565. If these are equal, the dense signal buys nothing on this corpus at naive answering.
- Retrieval signal, full discipline: BM25 1.0 / dense 1.0 / hybrid 1.0 recall@5; answer-only 0.87 / 0.826 / 0.87. The signal that wins here is the one worth paying the latency for.
- Reranker: hybrid-no-rerank vs improved-hybrid. Answer-only relevance 0.87 → 0.87, recall@5 1.0 → 1.0.
- Abstention gates: hybrid-no-gates vs improved-hybrid. Abstention accuracy 0.0 → 0.8, answer-only relevance 0.87 → 0.87. The gates are what stop the system answering questions the corpus cannot answer.
- Latency (retrieval mean): BM25 1.3 ms, dense 108.8 ms, hybrid 66.4 ms. Dense embedding is the cost; BM25 is near-free. Any recall gain from dense must be weighed against this.
- Full pipeline vs naive baseline: answer-only relevance 0.522 → 0.87, abstention 0.0 → 0.8, faithfulness 1.0 → 1.0.
