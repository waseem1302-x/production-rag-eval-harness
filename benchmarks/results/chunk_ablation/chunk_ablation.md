# Chunk-size ablation (Phase 2, Step 5)

Same .txt corpus, same overlap policy (200 chars, constant), same 39-question golden set (34 in-corpus + 5 out-of-corpus, 11 eligibility). Full extractive pipeline for every size: hybrid retrieval, lexical rerank, scored picking, abstention gates, eligibility-first. Latency is retrieval + search-plus-answer on the 28 factual questions, measured after warm-up with the embedding model already loaded (same definition as benchmarks/run_benchmark.py). Cost $0.00 everywhere: local CPU, no API calls.

| Chunk size | Chunks | Mean chunk chars | Max chunk chars | recall@3 | recall@5 | Answer-only relevance | Faithfulness | Abstention | Eligibility | Retrieval mean | Total mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 tokens (1024 chars) | 60 | 1023.2 | 1224 | 1.0 | 1.0 | 0.853 | 1.0 | 1.0 | 0.818 | 93.0 ms | 1011.5 ms |
| 512 tokens (2048 chars) | 31 | 1797.8 | 2248 | 1.0 | 1.0 | 0.912 | 1.0 | 1.0 | 1.0 | 97.6 ms | 1264.0 ms |
| 1024 tokens (4096 chars) | 19 | 2808.6 | 4283 | 1.0 | 1.0 | 0.853 | 1.0 | 1.0 | 0.909 | 256.3 ms | 2463.6 ms |

## What the numbers say

- Factual questions only (23, eligibility excluded): answer-only 21/23 at 256 tokens, 20/23 at 512, 20/23 at 1,024. The 256-token size fixes q03 (SI master's duration) — smaller chunks isolate the duration sentence so the picker finds it — but loses three eligibility rows, so the headline (all 34 in-corpus) still favours 512.
- Answer-only relevance: 256 tokens 0.853, 512 tokens 0.912, 1,024 tokens 0.853. Best is 512 tokens.
- Eligibility accuracy: 256 tokens 0.818, 512 tokens 1.0, 1,024 tokens 0.909. Eligibility rules pin chunk ids, so a size that moves an evidence sentence to another chunk drops that rule to unknown — see EVAL_RESULTS.md for which questions move.
- Recall@5 is 1.0 / 1.0 / 1.0 (256 / 512 / 1,024 tokens): on this corpus the right document is in the top 5 at every size.
- Total latency mean: 256 tokens 1011.5 ms, 512 tokens 1264.0 ms, 1,024 tokens 2463.6 ms. Read the mean with the median: total p50 is 879.0 / 1120.7 / 1190.4 ms (256 / 512 / 1,024 tokens) — nearly flat. The 1,024-token mean is a tail effect (p95 5900.8 ms, max 10920.9 ms), which may be CPU contention as much as chunk size; do not read it as '1,024 tokens is twice as slow per question'. Latency varies run to run on this shared-CPU VM.
