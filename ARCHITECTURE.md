# Architecture — Production RAG Eval Harness (Phase 1)

## What this system is

A small, honest, production-shaped RAG system over 10 public official
scholarship pages. It answers questions **only from the fetched
corpus**, cites the exact chunks it used, checks eligibility before
answering when a profile is given, and says **"Not found in the
corpus"** when the evidence is weak — instead of hallucinating.

Phase 1 is **retrieval + extractive answers + a measured eval
harness**. There is no generative LLM. That is deliberate: if
retrieval is wrong, a generator only hides it. Phase 2, Step 1
adds a generator and an LLM-as-judge — as an **opt-in** path
(`POST /ask` with `"generate": true`), with a deterministic
verification step and fallback to the extractive answer, measured
on the same golden set. The generator is worse than quoting the
source on this corpus (answer-only 0.697 vs 0.909); that is a
measured result, not a failure of the step — see `EVAL_RESULTS.md`.

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
benchmarks/run_benchmark.py ──► 8 configurations, same golden set
benchmarks/audit_retrieval.py ──► per-question retrieval trace
scripts/check_data_quality.py ──► corpus mess report
scripts/ci_gate.py + .github/workflows/eval.yml ──► quality gate
```

## Component map

| File | Purpose | Why it exists |
|---|---|---|
| `scripts/ingest.py` | Fetch the 10 official pages, extract text, save raw text + metadata (URL, fetch method, HTTP status, date) | Provenance. Every answer can be traced to a fetched page and a fetch date. Raw text is never edited after fetching. |
| `app/textnorm.py` | Normalise document text before chunking: unwrap hard line breaks inside sentences, keep headers separate, add terminal punctuation to fragments | Official pages put line breaks in the middle of sentences and mix headers with content. Without this, chunks contain broken sentences and answers quote fragments. Normalisation happens at chunking time, so `data/raw/` stays verbatim. |
| `app/chunking.py` | Recursive, sentence-aware chunking: ~512 tokens (~2,048 chars), 200-char overlap | One chunk should hold one complete idea. Overlap keeps facts that straddle a boundary. See "Chunking" below. Phase 2, Step 5: `chunk_document()` takes `chunk_chars` / `overlap_chars` parameters, and the size was ablated (256/512/1,024 tokens, measured — 512 kept); `app/config.py:chunk_chars_for_tokens` maps tokens to chars (1 token ~ 4 chars). |
| `app/embeddings.py` | `sentence-transformers` `all-MiniLM-L6-v2`, 384-dim, L2-normalised | The dense signal. Falls back to BM25-only and reports `"embedder": "bm25-only"` if the model cannot load — never silently pretends dense ran. |
| `app/retrieval.py` | FAISS `IndexFlatIP` (cosine) + BM25, min-max normalised, combined 0.5/0.5, then a reranker. `mode` selects `bm25` / `dense` / `hybrid`; `reranker` selects `lexical` (query-term coverage bonus +0.1, the default) or `cross-encoder` (ms-marco MiniLM-L6, joint pair scoring, sigmoid-mapped to 0-1; falls back to lexical if the model cannot load, and the trace says which ran) | Hybrid retrieval + an auditable rerank step. The `mode` parameter exists for the ablation: each signal's contribution is measured, not assumed. |
| `app/cross_encoder.py` | Cross-encoder reranker (Phase 2, Step 3): `sentence-transformers` `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`, 22.7M params, ~91 MB, CPU-only, $0. Lazy-loaded like the embedder; `score_pairs` returns raw logits or `None` when the model is unavailable | The reranker the lexical bonus is measured against. Same lazy-load / honest-fallback pattern as `app/embeddings.py`: a missing model degrades the reranker, never crashes a search. |
| `app/answer.py` | Extractive answers: up to 5 scored sentences from the top 3 chunks, with abstention gates + answer verification | The answer is verbatim sentences from cited chunks, so faithfulness is measurable. The gates (score threshold, doc-name consistency, shared-stem, specific-term) are what stop the system answering questions the corpus cannot answer. Every refusal records its reason. Phase 2, Step 2: sentence scores add `2.0 ×` dense cosine similarity to the question (local MiniLM, $0), and after the lexical gates `verify_answer_relevance` checks the picked sentences actually answer the question (dense floor 0.45 + ranking-focus check). Measured: dense scoring buys nothing on this corpus (see §4), verification fixes o04. |
| `app/generative.py` | Generative answers (Phase 2, Step 1): local Qwen2.5-0.5B-Instruct, greedy, + LLM-as-judge | The opt-in generator. Loads the 0.5B model lazily (bf16, CPU, 2 torch threads — measured faster than 4 on this VM), prompts it with the top-3 chunks of the *named* scholarship only, and returns a verified answer or falls back to extractive. `judge_answer` scores one answer with the same model (label-aware parsing: the judge sometimes swaps the two lines). All model text goes through `complete()`, which honours a test override so unit tests never load the model. Phase 2, Step 2: the kept sentences also pass `verify_answer_relevance`; on failure the path falls back to extractive (`relevance_verification_failed`). |
| `app/verification.py` | Deterministic verification of generated sentences against the cited chunks | The guard, not the judge. A sentence is kept only if its *novel* stems (content stems not already in the question) are covered ≥50% by one chunk, and every number in it appears in the chunks. Two fixes came from measured failures: the named-doc filter (q17 — the model copied SI's field list into a Manaaki answer) and novel-stem coverage (question-word overlap alone passed an ungrounded sentence). Verification stops invented facts; it cannot tell whether a grounded sentence answers the question (o04) or contradicts a corpus fact the model did not read (q21). |
| `app/eligibility.py` | Degree-level and citizenship checks against corpus-grounded rules, validated at load time against the chunk text | A scholarship answer is useless if the reader cannot apply for it. Rules are written as a table, but each rule's evidence sentence must be found verbatim in the named chunk or the rule drops to `unknown`. Phase 2, Step 4: the checker also reads `data/closures.json` — one entry per programme the corpus says is closed, with verbatim evidence validated the same way — and a closed programme gets the distinct verdict `closed` (checked before degree/citizenship, whatever the profile), with the replacement programme named. An entry whose evidence is not found is dropped with a warning and the programme is treated as open. |
| `app/api.py` | FastAPI: `GET /health`, `POST /ask` (optional `profile`) | One endpoint. The response carries the answer, citations with source URLs, the eligibility verdicts, and a full retrieval trace. |
| `app/eval.py` | Golden-set harness: recall@k, answer relevance (proxy + answer-only), faithfulness, abstention accuracy, eligibility accuracy, LLM-as-judge | The measurement instrument. Every claim in the README is a number this file produces. `run_eval(generative=True, judge=True)` runs the same golden set through the generative path and scores each substantive answer with the local model as judge; the string proxies are always computed too, so generative and extractive stay comparable. Eligibility statements are tagged `answer_mode: "eligibility"` and are not judge-scored (template + verbatim evidence, not a generated answer). Phase 2, Step 5: `run_eval` resolves eligibility against the retriever's own chunks, not the default chunks file — otherwise an eval at a non-default chunk size would silently measure 512-token eligibility. |
| `benchmarks/` | 8-configuration ablation (`run_benchmark.py`), per-question retrieval audit (`audit_retrieval.py`), generated before/after report (`before_after.py`) | The proof that each component earns its place — or does not. See `benchmarks/results/summary.md` and `before_after.md`. |
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
   scholarship(s) first. A `not_eligible` or `closed` verdict — or an eligibility
   question — replaces the substantive answer with the verdict, its
   reasons, and the verbatim evidence sentences. Otherwise the
   substantive answer stands and the verdicts are attached.
4. `answer.answer` picks up to 5 scored sentences from the top 3
   chunks (lexical score + `2.0 ×` dense cosine similarity to the
   question; short fragments penalised, duplicates removed) and
   applies the gates. If any gate fires, the answer is "Not found
   in the corpus." with the reason (`weak_score`,
   `doc_name_mismatch`, `no_matching_sentences`, `weak_overlap`,
   `missing_specific_term`, `low_answer_similarity`,
   `unanswered_ranking`). The last two are the Phase 2, Step 2
   answer-verification step, which runs after the lexical gates:
   the picked sentences must clear a dense-similarity floor and,
   for ranking questions, mention a ranking.
5. With `"generate": true` (opt-in only), `generative.generate_answer`
   instead prompts the local model with the top-3 chunks of the
   named scholarship, verifies every generated sentence against
   those chunks (`verification.verify_answer`), then checks the
   kept sentences actually answer the question
   (`answer.verify_answer_relevance`), and returns the kept
   sentences — or the extractive answer when nothing survives
   verification, the model abstains, the relevance check fails, or
   the model cannot load. Eligibility answers and abstentions are
   never generated.
6. The response includes the retrieval trace: every candidate with
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
| BM25-only | 1.0 | 0.87 | 6.2 ms |
| Dense-only | 1.0 | 0.826 | 57.6 ms |
| Hybrid | 1.0 | 0.87 | 123.7 ms |

(Fresh re-run, Phase 2 Step 3; the Step 1 run measured 1.3 / 108.8 /
66.4 ms. Latency varies run to run on this shared-CPU VM; the order
of magnitude is the stable finding.)

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

**Chosen: extractive by default in Phase 1; generative opt-in in
Phase 2, Step 1, with verification and fallback.** The answer is
verbatim sentences from cited chunks. No LLM writes any word of
the default answer.

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

**Phase 2, Step 1 outcome (measured, same 38 questions):** the
local 0.5B generator (Qwen2.5-0.5B-Instruct, 988.1 MB bf16, CPU,
greedy, $0) is *worse* than extractive at answer-only relevance
(0.697 vs 0.909) — it paraphrases past the expected phrase, drops
the key sentence (q01 in the final run), and sometimes declines
when the evidence is in context (q15). Verification + fallback
(7 of 33 fell back to extractive) kept it safe, and it fixed one
paraphrase-gap failure (q03). The LLM-as-judge scored 0.957 /
0.957 (n=23, 0 unparsed) but overstates quality: it gave q10's
"one academic year" (expected "1 to 2 academic years") 1/1, and
the correct Vanier-closure answer (q21) 0/0. A 0.5B judge grading
its own family is a weak instrument; the proxy metrics and the
failure tables are the numbers to trust. The extractive default
is therefore unchanged — the generator is an opt-in, not a
replacement, until a stronger model or a better verification step
earns the default on the same golden set.

### 5. Lexical reranker vs cross-encoder

**Chosen: a transparent lexical reranker** — query-term coverage
bonus (+0.1 × coverage) on top of the combined score.

**Why:** it is explainable (the trace shows the coverage number),
costs nothing (no model call), and cannot hallucinate.

**Measured:** it adds nothing on this corpus — answer-only
relevance is 0.87 with and without it, recall identical. It is kept
because the mechanism is sound and the corpus is small; on a larger
corpus it must be re-measured before it is claimed to help.

**Phase 2, Step 3 — cross-encoder measured:** `hybrid-cross-encoder`
is now an ablation configuration (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
22.7M params, ~91 MB, CPU-only, $0; re-sorts the top 10 by
sigmoid(logit), term coverage recorded but unused). On the same
28-question golden set it changes nothing either: answer-only
relevance 0.87 → 0.87, recall@5 1.0 → 1.0, abstention 1.0 → 1.0 —
but retrieval mean latency is 4,238.0 ms vs 123.7 ms for the lexical
reranker (~34x; scoring 10 full chunks takes ~2.5–3.4 s per question
on this VM). It does reorder: 17 of 23 in-corpus top-5 lists change
order, but the top chunk and every picked answer sentence stay the
same, so no metric moves. On a synthetic trap — a chunk repeating
every query term without answering, vs the answering chunk — the
cross-encoder does flip the order where the lexical reranker does
not (logits 7.13 vs 5.97), so the mechanism works; the corpus is
just too small and too clean for it to matter. The cross-encoder
also scores like a lexical model more often than its name suggests:
on a second synthetic trap (MEXT deadline) it kept the trap first
(7.51 vs 6.92). Lexical stays the default; the cross-encoder is a
measured option, not a win.

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
0.87.

**Measured in Phase 2, Step 5 — chunk size ablated, 512 kept:**
`python -m scripts.run_chunk_ablation`, same .txt corpus, overlap
held constant at 200 chars, same 39-question golden set, full
extractive pipeline. Answer-only relevance 0.853 / **0.912** / 0.853
and eligibility 0.818 / **1.0** / 0.909 for 256 / 512 / 1,024 tokens;
recall, faithfulness and abstention 1.0 at every size. 256 fixes q03
but drops 7 eligibility rules (evidence sentences move chunk ids),
so a Pakistani SI applicant is wrongly told `eligible`; 1,024 drops
2 rules and fails e03 the same way. Total-latency medians are nearly
flat (879 / 1,121 / 1,190 ms), so latency does not decide it. The
512-token chunking is what the eligibility rules were authored
against — that home-field advantage is stated, not hidden — and a
future size change must re-pin the rules first and re-run the
ablation. Per-size indexes live in
`benchmarks/results/chunk_ablation/chunk-<size>/`; the builder takes
`--chunk-tokens` and `--out-dir` (`scripts/build_index.py`), and
`app/eval.py:run_eval` resolves eligibility against the retriever's
own chunks so a non-default size is really measured.

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
| Corpus page closed / stale | `scripts/check_data_quality.py` + `data/closures.json` | Documented in `DATA_QUALITY.md`; closure is a golden question (q21), and eligibility returns verdict `closed` for a closed programme (Phase 2, Step 4) |
| Metric regression | `scripts/ci_gate.py` + GitHub workflow | Build fails; thresholds are pinned |
| Corpus re-ingest changes the facts | Re-run eval + data-quality checker | Numbers must be re-measured before any claim |

## Known limitations (Phase 2)

- **Paraphrase gap (q03, q10, q17):** lexical sentence picking cannot
  find sentences whose key terms are absent from the question. The
  generative path fixed q03 (answer-only True) but not q10 or q17;
  q17 additionally fell back to extractive after verification
  dropped the model's cross-document field list. Phase 2, Step 2
  added dense sentence scoring (weight 2.0) and measured it: weights
  0–2 change no golden answer, weights 3–4 fix q03 but lose q16
  (its correct sentence scores 0.394 dense similarity), net 0.909
  either way — dense similarity is too weak a signal at this corpus
  size. q03, q10, q17 remain extractive misses. Fix: a stronger
  generator, or a cross-encoder reranker over sentences.
- **No answer verification (o04):** the gates check word overlap, not
  whether the picked sentences actually answer the question. A
  deterministic verification step now exists (Phase 2, Step 1) and
  drops ungrounded sentences, but o04 still answers instead of
  abstaining: its Vanier sentences are grounded, so verification
  cannot catch that they do not answer the question. **Fixed in
  Phase 2, Step 2:** `verify_answer_relevance` (dense floor 0.45 +
  ranking-focus check) abstains on o04 with reason
  `unanswered_ranking`, in both paths; abstention is 1.0 (5/5).
  Dense similarity alone was measured and cannot separate o04
  (0.667, above several answerable questions), so the focus check
  does the work. The check is narrow — it covers ranking questions
  only — and it checks for a ranking *term*, not a ranked entity,
  so a sentence about "best-ranked students" passes it. The
  LLM-as-judge relevance score remains the proxy for other
  question shapes, and at 0.5B it is too weak (it scored o04's
  sibling failure q21's correct answer 0/0).
- **Judge is weak at 0.5B:** same-model grading is lenient (q10
  scored 1/1 for a wrong duration) and noisy (q21 scored 0/0 for a
  correct closure answer). The judge is reported alongside the
  proxies, never instead of them. Fix: a larger judge model, or
  human-graded spot checks on the failure table.
- **Eligibility ignores closure:** the Vanier rules say "doctoral
  only" but the programme is closed; the verdict does not factor
  that in. **Fixed in Phase 2, Step 4:** `data/closures.json` holds
  one entry per closed programme (Vanier, with its two verbatim
  closure sentences from chunk `vanier-0` and its replacement, the
  Canada Graduate Research Scholarship - Doctoral program), read
  and verbatim-validated at load time by `app/eligibility.py`
  exactly like the rules; `check()` returns verdict `closed`
  before any degree or citizenship check, whatever the profile.
  Measured: eligibility 1.0 (11/11, with new golden e11 for a
  foreign doctoral applicant); e02 and e05 flipped to `closed`;
  no other golden row changed. Adding a future closure is one
  JSON entry plus the matching checker resolution — no code change.
- **Chunk size not ablated:** 512 tokens was chosen by reasoning,
  not by measurement. **Measured in Phase 2, Step 5:** 256 / 512 /
  1,024 tokens ablated on the same 39-question golden set — 512
  wins (answer-only 0.912 vs 0.853, eligibility 1.0 vs 0.818/0.909)
  and stays the default; see §7. The 256-token size fixes q03 but
  drops 7 eligibility rules, including the one that makes a
  Pakistani SI applicant not eligible.
- **Reranker unproven:** adds nothing on this corpus. **Measured in
  Phase 2, Step 3:** the cross-encoder configuration
  (`hybrid-cross-encoder`) changes no metric either (0.87 answer-only,
  1.0 abstention, both ways) at ~34x the retrieval latency — see §5.
  Both rerankers are now measured non-wins on this corpus; re-measure
  on a larger corpus.
- **Corpus is 10 pages, 31 chunks:** every "dense buys nothing"
  finding is a finding *about this corpus size*. Re-measure as it grows.
