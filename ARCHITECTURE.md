# Architecture — Production RAG Eval Harness (Phase 1)

## What this system is

A small, honest, production-shaped RAG system over 10 public official
scholarship pages. It answers questions **only from the fetched
corpus**, cites the exact chunks it used, checks eligibility before
answering when a profile is given, and says **"Not found in the
corpus"** when the evidence is weak — instead of hallucinating.

Phase 1 is **retrieval + extractive answers + a measured eval
harness**. There is no generative LLM. That is deliberate: if
retrieval is wrong, a generator only hides it. Phase 2 adds a
generator and an LLM-as-judge.

## System overview

```
                        build time (scripts/)
official pages ──► ingest.py ──► data/raw/*.txt + *.json (verbatim + metadata)
                              ──► chunking.py ──► data/processed/chunks.jsonl (31 chunks)
                                              ──► embeddings.py ──► FAISS IndexFlatIP (384-dim)
                                                              └─► BM25Okapi (rank-bm25)

                        query time (app/)
POST /ask ──► retrieval.py ──► top-10 candidates (hybrid 0.5 dense + 0.5 BM25)
          ──► lexical rerank ──► top-5
          ──► eligibility.py (if profile) ──► verdict + reasons + evidence
          ──► answer.py ──► extractive answer + citations, or abstention
          ──► response: answer, score, citations, eligibility, retrieval_trace

                        quality (eval/, benchmarks/, scripts/)
eval/golden.json (38 questions) ──► eval.py ──► metrics
benchmarks/run_benchmark.py ──► 7 configurations, same golden set
benchmarks/audit_retrieval.py ──► per-question retrieval trace
scripts/check_data_quality.py ──► corpus mess report
scripts/ci_gate.py + .github/workflows/eval.yml ──► quality gate
```

## Component map

| File | Purpose | Why it exists |
|---|---|---|
| `scripts/ingest.py` | Fetch the 10 official pages, extract text, save raw text + metadata (URL, fetch method, HTTP status, date) | Provenance. Every answer can be traced to a fetched page and a fetch date. Raw text is never edited after fetching. |
| `app/textnorm.py` | Normalise document text before chunking: unwrap hard line breaks inside sentences, keep headers separate, add terminal punctuation to fragments | Official pages put line breaks in the middle of sentences and mix headers with content. Without this, chunks contain broken sentences and answers quote fragments. Normalisation happens at chunking time, so `data/raw/` stays verbatim. |
| `app/chunking.py` | Recursive, sentence-aware chunking: ~512 tokens (~2,048 chars), 200-char overlap | One chunk should hold one complete idea. Overlap keeps facts that straddle a boundary. See "Chunking" below. |
| `app/embeddings.py` | `sentence-transformers` `all-MiniLM-L6-v2`, 384-dim, L2-normalised | The dense signal. Falls back to BM25-only and reports `"embedder": "bm25-only"` if the model cannot load — never silently pretends dense ran. |
| `app/retrieval.py` | FAISS `IndexFlatIP` (cosine) + BM25, min-max normalised, combined 0.5/0.5, then a lexical reranker (query-term coverage bonus +0.1). `mode` selects `bm25` / `dense` / `hybrid` | Hybrid retrieval + an auditable rerank step. The `mode` parameter exists for the ablation: each signal's contribution is measured, not assumed. |
| `app/answer.py` | Extractive answers: up to 5 scored sentences from the top 3 chunks, with abstention gates | The answer is verbatim sentences from cited chunks, so faithfulness is measurable. The gates (score threshold, doc-name consistency, shared-stem, specific-term) are what stop the system answering questions the corpus cannot answer. Every refusal records its reason. |
| `app/eligibility.py` | Degree-level and citizenship checks against corpus-grounded rules, validated at load time against the chunk text | A scholarship answer is useless if the reader cannot apply for it. Rules are written as a table, but each rule's evidence sentence must be found verbatim in the named chunk or the rule drops to `unknown`. |
| `app/api.py` | FastAPI: `GET /health`, `POST /ask` (optional `profile`) | One endpoint. The response carries the answer, citations with source URLs, the eligibility verdicts, and a full retrieval trace. |
| `app/eval.py` | Golden-set harness: recall@k, answer relevance (proxy + answer-only), faithfulness, abstention accuracy, eligibility accuracy | The measurement instrument. Every claim in the README is a number this file produces. |
| `benchmarks/` | 7-configuration ablation (`run_benchmark.py`), per-question retrieval audit (`audit_retrieval.py`), generated before/after report (`before_after.py`) | The proof that each component earns its place — or does not. See `benchmarks/results/summary.md` and `before_after.md`. |
| `scripts/check_data_quality.py` | Finds the corpus's messy data (closures, two amounts, database pages, approximate/year-specific figures, missing fields, stale markers) | Real pages are messy. A system that hides the mess will invent answers for it. |
| `scripts/ci_gate.py` | Fails the build if any metric drops below its threshold | The eval harness is a gate, not a report. Thresholds are set below the measured numbers so jitter does not fail the build; lowering a threshold requires a reason in the commit message. |

## Request path (query time)

1. `POST /ask` with `{"question", "top_k", "profile?"}`.
2. `Retriever.search` embeds the question (dense modes), scores all
   31 chunks with BM25, takes the union of the top 10 of each signal,
   normalises both to 0–1, combines them, and reranks by query-term
   coverage. Returns the top 5 with all four scores
   (`dense_score`, `bm25_score`, `combined_score`, `rerank_score`).
3. If a profile is given, `eligibility.resolve` checks the named
   scholarship(s) first. A `not_eligible` verdict — or an eligibility
   question — replaces the substantive answer with the verdict, its
   reasons, and the verbatim evidence sentences. Otherwise the
   substantive answer stands and the verdicts are attached.
4. `answer.answer` picks up to 5 scored sentences from the top 3
   chunks (short fragments penalised, duplicates removed) and applies
   the gates. If any gate fires, the answer is "Not found in the
   corpus." with the reason (`weak_score`, `doc_name_mismatch`,
   `no_matching_sentences`, `weak_overlap`, `missing_specific_term`).
5. The response includes the retrieval trace: every candidate with
   all scores, term coverage, and whether it was injected into the
   answer. A wrong answer can be traced to the exact chunk and score
   that produced it.

## Decisions, trade-offs, and rejected alternatives

### 1. Hand-rolled pipeline vs LangChain

**Chosen: hand-rolled, no framework.** Every step (chunking,
retrieval, reranking, answering) is ~50–150 lines of plain Python
in `app/`.

**Why:** the point of this project is to learn how RAG works and to
be able to explain every part in an interview. A framework hides the
decisions that matter (how chunks are split, how scores combine,
when to abstain) behind defaults. Hand-rolled code is also fully
testable: each function has unit tests, and the ablation can switch
individual components off.

**Cost:** more code to maintain; no built-in integrations (vector
stores, loaders, agent loops). For a 10-document corpus that cost is
trivial. For a production system with dozens of sources, a framework
or a proper orchestration layer would be the right call — that is a
Phase 2+ decision, and it would be made with measurements, not
fashion.

**Rejected:** LangChain / LlamaIndex. They would have made the first
version faster to write and much harder to explain, debug, and
ablate.

### 2. FAISS vs Chroma / pgvector

**Chosen: FAISS `IndexFlatIP` in memory, built at startup from
`chunks.jsonl`.**

**Why:** 31 chunks. An exact brute-force index is instant, has no
server, no persistence format to manage, and no dependency beyond
`faiss-cpu`. The index is rebuilt from the committed chunks file on
every start, so there is no separate index artefact that can go stale.

**Cost:** no persistence, no metadata filtering, no scaling story
beyond a few thousand chunks. `IndexFlatIP` is O(n) per query —
fine at 31 chunks, wrong at 31 million.

**Rejected:** Chroma (a server and a persistence layer for 31
chunks is overhead, not engineering), pgvector (needs Postgres;
the corpus is a file, not a database). When the corpus grows past a
few thousand chunks, the right move is an approximate index
(FAISS `IndexIVFFlat`/`HNSW`) or a managed vector store — measured
then, not now.

### 3. Hybrid vs dense-only vs BM25-only

**Chosen: hybrid 0.5 dense + 0.5 BM25, with a `mode` switch so the
choice is measured.**

**Measured (28 factual questions, same golden set):**

| Signal (full discipline) | recall@5 | Answer-only relevance | Retrieval latency (mean) |
|---|---|---|---|
| BM25-only | 1.0 | 0.87 | 1.3 ms |
| Dense-only | 1.0 | 0.826 | 108.8 ms |
| Hybrid | 1.0 | 0.87 | 66.4 ms |

**Why hybrid is the default anyway:** recall is saturated at 1.0
for every signal on this corpus — the right document is always in
the top 5 by keyword alone. Dense buys no measured recall here and
slightly *hurts* answer-only relevance (0.826 vs 0.87), at ~50–80x
the latency. On this corpus BM25-only is the better engineering
choice. Hybrid is kept as the default because dense is expected to
matter as the corpus grows (paraphrases, synonyms, larger
candidate sets), and the ablation will tell us when — the claim is
"measured on this corpus", not "dense is better".

**Rejected:** dense-only (worse answer-only, far slower, no
measured benefit here), BM25-only as the *only* option (no dense
signal to measure against, and it would miss paraphrases on a
larger corpus).

### 4. Extractive vs generative answers

**Chosen: extractive only in Phase 1.** The answer is verbatim
sentences from cited chunks. No LLM writes any word of the answer.

**Why:** faithfulness becomes measurable instead of claimed
(measured 1.0: every answer sentence appears verbatim in a cited
chunk). A generative answer can be fluent, confident, and wrong
about a deadline or a stipend — the worst failure mode for a
scholarship assistant. Extractive answers also make every failure
diagnosable: the audit shows exactly which sentences were picked.

**Cost:** answers read like quoted fragments, not prose. Questions
whose answer requires combining or rephrasing information (the
paraphrase-gap failures q03, q10, q17) cannot be answered well.
The eligibility answers are the one exception: the verdict and
reasons are generated from templates, but every factual claim in
them is a verbatim evidence sentence appended to the answer.

**Rejected (for Phase 1):** a generative LLM answer. It is the
Phase 2 step, and it will be added *with* a verification step
(the generated answer must be checked against the cited chunks)
and an LLM-as-judge — not as an unmeasured replacement.

### 5. Lexical reranker vs cross-encoder

**Chosen: a transparent lexical reranker** — query-term coverage
bonus (+0.1 × coverage) on top of the combined score.

**Why:** it is explainable (the trace shows the coverage number),
costs nothing (no model call), and cannot hallucinate.

**Measured:** it adds nothing on this corpus — answer-only
relevance is 0.87 with and without it, recall identical. It is kept
because the mechanism is sound and the corpus is small; on a larger
corpus it must be re-measured before it is claimed to help.

**Rejected (for Phase 1):** a cross-encoder (e.g. `ms-marco-MiniLM`).
It would cost a model call per candidate and its benefit is
unproven here. Phase 2 adds it as an ablation configuration and
measures it against the lexical reranker on the same golden set.

### 6. Local embeddings vs API embeddings

**Chosen: local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim),
CPU, no API calls. Cost per query is $0.00.**

**Why:** the corpus is tiny, the model is small (90 MB), inference
is ~50 ms on CPU, and there is no data leaving the machine, no API
key, no per-query cost, and no vendor dependency. The system works
offline once the model is downloaded.

**Cost:** model download on first run (~90 MB), ~30–60 s load time
in this VM, and a small general-purpose model instead of a
state-of-the-art embedding API. For a larger corpus or a multilingual
one, a hosted embedding model (or a larger local one) would be worth
measuring.

**Rejected:** OpenAI / Cohere / Voyage embeddings. They would add
cost, latency, a secret to manage, and a network dependency — for no
measured benefit on 31 chunks.

### 7. Chunking: 512 tokens, 200-char overlap

**Chosen:** target ~512 tokens (~2,048 chars), 200-char overlap
(~50 tokens), recursive and sentence-aware (paragraphs, then
sentences, then words only for a single over-long sentence).

**Why:** 512 tokens holds one complete idea (a paragraph about
eligibility, a funding bullet list) while keeping the embedding
specific. Overlap exists so a fact split across a boundary is not
lost. Sentence-awareness matters because the answerer quotes whole
sentences — a chunk that cuts a sentence in half produces a broken
quotation.

**Measured effect:** the text normaliser (which unwraps hard line
breaks before chunking) moved answer-only relevance from 0.70 to
0.85; the 5-sentence cap and short-fragment penalty moved it to
0.87. Chunk size itself was not ablated in Phase 1 — that is a
documented gap (see "Known limitations").

### 8. Abstention threshold 0.15, top_k 5, 10 rerank candidates

**Chosen:** `ABSTAIN_THRESHOLD = 0.15`, `DEFAULT_TOP_K = 5`,
`RERANK_CANDIDATES = 10`, all in `app/config.py`.

**Why:** the threshold is on the reranked score of the best
candidate. It is deliberately low: the gates (shared-stem,
specific-term, doc-name) do most of the refusing, and the threshold
is a backstop for genuinely weak retrieval. `top_k = 5` is the
number of candidates the answerer sees; 10 rerank candidates give
the reranker room to reorder without embedding the whole corpus per
query.

**Cost:** these are tuned by inspection on 28 questions, not by a
grid search. The CI gate pins the resulting metrics, so a future
change to any of them that regresses quality fails the build.

## Invariants (rules the system never breaks)

1. **Verbatim corpus.** `data/raw/` is never edited after fetching.
   Normalisation happens at chunking time; chunks are the normalised
   form, raw is the record.
2. **No invented rules.** Eligibility verdicts come only from
   sentences in the corpus. Where the corpus is silent, the verdict
   is `unknown`. Evidence sentences are validated verbatim at load.
3. **No silent fallback.** If the embedding model cannot load, the
   response says `"embedder": "bm25-only"`. If the index is empty,
   `/ask` says so.
4. **Every answer is traceable.** Citations carry `chunk_id` and
   `source_url`; the retrieval trace carries all four scores per
   candidate and whether it was injected.
5. **Every refusal names its reason.** Abstention reasons are part
   of the response and part of the eval's failure table.
6. **No invented numbers.** Every number in the README, `EVAL_RESULTS.md`,
   and `benchmarks/results/` comes from a real local run. If a number
   cannot be measured yet, it is left blank.

## Failure modes

| Failure | Detection | Behaviour |
|---|---|---|
| Embedding model unavailable | `embedder` field in every response | BM25-only retrieval, reported honestly |
| Empty index | `/health` shows 0 chunks; `/ask` returns `empty_index` | "Not found in the corpus. The index is empty…" |
| Question names a scholarship not in the corpus | Doc-name gate | Abstains (`doc_name_mismatch`) |
| Corpus page closed / stale | `scripts/check_data_quality.py` | Documented in `DATA_QUALITY.md`; closure is a golden question (q21) |
| Metric regression | `scripts/ci_gate.py` + GitHub workflow | Build fails; thresholds are pinned |
| Corpus re-ingest changes the facts | Re-run eval + data-quality checker | Numbers must be re-measured before any claim |

## Known limitations (Phase 2)

- **Paraphrase gap (q03, q10, q17):** lexical sentence picking cannot
  find sentences whose key terms are absent from the question. Fix:
  dense sentence scoring, or a generative answer with verification.
- **No answer verification (o04):** the gates check word overlap, not
  whether the picked sentences actually answer the question. Fix: a
  verification step on the picked sentences.
- **Eligibility ignores closure:** the Vanier rules say "doctoral
  only" but the programme is closed; the verdict does not factor
  that in. Fix: the eligibility module should read the checker's
  closure flag.
- **Chunk size not ablated:** 512 tokens was chosen by reasoning,
  not by measurement. Fix: ablate 256 / 512 / 1,024 tokens.
- **Reranker unproven:** adds nothing on this corpus. Fix: add a
  cross-encoder configuration and measure it.
- **Corpus is 10 pages, 31 chunks:** every "dense buys nothing"
  finding is a finding *about this corpus size*. Re-measure as it grows.
