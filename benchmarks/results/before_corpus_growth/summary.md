# Benchmark results (Phase 2, Step 3 — re-run with current code)

Same golden set for every configuration: 23 in-corpus + 5 out-of-corpus questions. Latency is measured after warm-up with the embedding model already loaded. Cost is $0.00: everything runs locally on CPU, no API calls. The Step 1 numbers in the docs were stale after Step 2 changed the answer() defaults, so this table is a fresh re-run of all configurations with the current code.

Latency varies run to run by tens of milliseconds on this VM (shared CPU, 1.5 cores). The order of magnitude — BM25 ~1 ms, dense ~50–150 ms, cross-encoder rerank ~seconds per query — is stable across runs; the exact figures are one measured run, not a guarantee.

| Configuration | Mode | Reranker | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.522 | 1.0 | 0.0 | 0.2 ms | 0.5 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.565 | 1.0 | 0.0 | 103.5 ms | 106.5 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 6.2 ms | 1070.7 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 1.0 | 1.0 | 0.826 | 1.0 | 1.0 | 57.6 ms | 961.3 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 77.4 ms | 1307.1 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.826 | 1.0 | 0.0 | 74.9 ms | 1342.9 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 123.7 ms | 1490.6 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 4238.0 ms | 5429.1 ms | $0.00 |

## What the ablation shows

- Retrieval signal, naive answering: hybrid-plain vs baseline-naive. Recall@5 1.0 → 1.0, answer-only relevance 0.522 → 0.565. If these are equal, the dense signal buys nothing on this corpus at naive answering.
- Retrieval signal, full discipline: BM25 1.0 / dense 1.0 / hybrid 1.0 recall@5; answer-only 0.87 / 0.826 / 0.87. The signal that wins here is the one worth paying the latency for.
- Reranker: hybrid-no-rerank vs improved-hybrid. Answer-only relevance 0.87 → 0.87, recall@5 1.0 → 1.0.
- Cross-encoder reranker: improved-hybrid (lexical) vs hybrid-cross-encoder. Answer-only relevance 0.87 → 0.87, recall@5 1.0 → 1.0, abstention 1.0 → 1.0, retrieval mean 123.7 ms → 4238.0 ms. Same retrieval, same answering — any delta here is the cross-encoder's, and the latency is what it costs.
- Abstention gates: hybrid-no-gates vs improved-hybrid. Abstention accuracy 0.0 → 1.0, answer-only relevance 0.826 → 0.87. The gates are what stop the system answering questions the corpus cannot answer.
- Latency (retrieval mean): BM25 6.2 ms, dense 57.6 ms, hybrid 123.7 ms. Dense embedding is the cost; BM25 is near-free. Any recall gain from dense must be weighed against this.
- Full pipeline vs naive baseline: answer-only relevance 0.522 → 0.87, abstention 0.0 → 1.0, faithfulness 1.0 → 1.0.
