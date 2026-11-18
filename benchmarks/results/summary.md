# Benchmark results (Phase 2 improvement — re-run with current code)

Same golden set for every configuration: 32 in-corpus + 7 out-of-corpus questions. Latency is measured after warm-up with the embedding model already loaded. Cost is $0.00: everything runs locally on CPU, no API calls. The Step 3 numbers in the docs were measured on the 31-chunk corpus and the 39-question golden set, so this table is a fresh re-run of all configurations with the current code on the grown corpus.

Latency varies run to run by hundreds of milliseconds on this VM (shared CPU, 1.5 cores; the dense configs embed every candidate sentence per question, so their totals move the most). The order of magnitude — BM25 ~1 ms, dense ~50–150 ms, cross-encoder rerank ~seconds per query — is stable across runs; the exact figures are one measured run, not a guarantee.

| Configuration | Mode | Reranker | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.625 | 1.0 | 0.0 | 0.3 ms | 0.7 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.656 | 1.0 | 0.0 | 63.3 ms | 63.9 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.938 | 1.0 | 0.857 | 5.0 ms | 887.1 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 1.0 | 1.0 | 0.906 | 1.0 | 0.857 | 44.2 ms | 833.5 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.938 | 1.0 | 1.0 | 58.5 ms | 984.6 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.906 | 1.0 | 0.0 | 46.9 ms | 1041.3 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.938 | 1.0 | 1.0 | 35.5 ms | 890.0 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 1.0 | 1.0 | 0.938 | 1.0 | 1.0 | 3307.2 ms | 4290.0 ms | $0.00 |

## What the ablation shows

- Retrieval signal, naive answering: hybrid-plain vs baseline-naive. Recall@5 1.0 → 1.0, answer-only relevance 0.625 → 0.656. If these are equal, the dense signal buys nothing on this corpus at naive answering.
- Retrieval signal, full discipline: BM25 1.0 / dense 1.0 / hybrid 1.0 recall@5; answer-only 0.938 / 0.906 / 0.938. The signal that wins here is the one worth paying the latency for.
- Reranker: hybrid-no-rerank vs improved-hybrid. Answer-only relevance 0.938 → 0.938, recall@5 1.0 → 1.0.
- Cross-encoder reranker: improved-hybrid (lexical) vs hybrid-cross-encoder. Answer-only relevance 0.938 → 0.938, recall@5 1.0 → 1.0, abstention 1.0 → 1.0, retrieval mean 35.5 ms → 3307.2 ms. Same retrieval, same answering — any delta here is the cross-encoder's, and the latency is what it costs.
- Abstention gates: hybrid-no-gates vs improved-hybrid. Abstention accuracy 0.0 → 1.0, answer-only relevance 0.906 → 0.938. The gates are what stop the system answering questions the corpus cannot answer.
- Latency (retrieval mean): BM25 5.0 ms, dense 44.2 ms, hybrid 35.5 ms. Dense embedding is the cost; BM25 is near-free. Any recall gain from dense must be weighed against this.
- Full pipeline vs naive baseline: answer-only relevance 0.625 → 0.938, abstention 0.0 → 1.0, faithfulness 1.0 → 1.0.
