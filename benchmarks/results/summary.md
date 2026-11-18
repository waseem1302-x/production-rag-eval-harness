# Benchmark results (Phase 2, Step 6 — re-run with current code)

Same golden set for every configuration: 32 in-corpus + 7 out-of-corpus questions. Latency is measured after warm-up with the embedding model already loaded. Cost is $0.00: everything runs locally on CPU, no API calls. The Step 3 numbers in the docs were measured on the 31-chunk corpus and the 39-question golden set, so this table is a fresh re-run of all configurations with the current code on the grown corpus.

Latency varies run to run by tens of milliseconds on this VM (shared CPU, 1.5 cores). The order of magnitude — BM25 ~1 ms, dense ~50–150 ms, cross-encoder rerank ~seconds per query — is stable across runs; the exact figures are one measured run, not a guarantee.

| Configuration | Mode | Reranker | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.625 | 1.0 | 0.0 | 0.3 ms | 0.7 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.656 | 1.0 | 0.0 | 84.1 ms | 84.6 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 2.0 ms | 1179.2 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 1.0 | 1.0 | 0.844 | 1.0 | 1.0 | 77.7 ms | 1443.7 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 60.2 ms | 1407.1 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.844 | 1.0 | 0.0 | 145.3 ms | 2038.0 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 185.9 ms | 2566.1 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 3987.1 ms | 5374.9 ms | $0.00 |

## What the ablation shows

- Retrieval signal, naive answering: hybrid-plain vs baseline-naive. Recall@5 1.0 → 1.0, answer-only relevance 0.625 → 0.656. If these are equal, the dense signal buys nothing on this corpus at naive answering.
- Retrieval signal, full discipline: BM25 1.0 / dense 1.0 / hybrid 1.0 recall@5; answer-only 0.875 / 0.844 / 0.875. The signal that wins here is the one worth paying the latency for.
- Reranker: hybrid-no-rerank vs improved-hybrid. Answer-only relevance 0.875 → 0.875, recall@5 1.0 → 1.0.
- Cross-encoder reranker: improved-hybrid (lexical) vs hybrid-cross-encoder. Answer-only relevance 0.875 → 0.875, recall@5 1.0 → 1.0, abstention 1.0 → 1.0, retrieval mean 185.9 ms → 3987.1 ms. Same retrieval, same answering — any delta here is the cross-encoder's, and the latency is what it costs.
- Abstention gates: hybrid-no-gates vs improved-hybrid. Abstention accuracy 0.0 → 1.0, answer-only relevance 0.844 → 0.875. The gates are what stop the system answering questions the corpus cannot answer.
- Latency (retrieval mean): BM25 2.0 ms, dense 77.7 ms, hybrid 185.9 ms. Dense embedding is the cost; BM25 is near-free. Any recall gain from dense must be weighed against this.
- Full pipeline vs naive baseline: answer-only relevance 0.625 → 0.875, abstention 0.0 → 1.0, faithfulness 1.0 → 1.0.
