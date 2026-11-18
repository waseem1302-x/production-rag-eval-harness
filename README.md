# Production RAG with Eval Harness

A small, honest, production-shaped RAG system over public official scholarship pages. It answers questions only from the fetched corpus, cites the exact chunks it used, checks eligibility before answering when a profile is given, and says "Not found in the corpus" when the evidence is weak, instead of hallucinating.

Read first:

- [ARCHITECTURE.md](ARCHITECTURE.md) - what was built, why, and the trade-offs
- [EVAL_RESULTS.md](EVAL_RESULTS.md) - every measured number
- [DATA_QUALITY.md](DATA_QUALITY.md) - the corpus's messy data, documented not hidden
- [benchmarks/results/before_after.md](benchmarks/results/before_after.md) - before/after with failure tables

## Results at a glance

Final state after Phase 2, Step 7, measured 18 Nov 2026 on a local run. No number here is invented.

| Metric | Value |
|---|---|
| Corpus | 19 documents, 101 chunks |
| Golden questions | 50 (32 factual + 11 eligibility + 7 out-of-corpus) |
| recall@3 / recall@5 | 1.0 / 1.0 |
| Answer-only relevance | 0.907 (39/43) |
| Faithfulness | 1.0 |
| Abstention accuracy | 1.0 (7/7) |
| Eligibility accuracy | 1.0 (11/11) |
| Tests | 179 passed, 0 failed |
| Cost per query | $0.00 (local CPU, no API calls) |
| Data-quality findings | 73 (16 resolved, 57 open) |
| CI gate | Pass (1.0, 0.907, 1.0, 1.0, 1.0) |

## Tech stack

What was actually used, and what each piece does.

| Layer | Tool | What it does here |
|---|---|---|
| Language | Python 3.12 | All code: app, scripts, eval, benchmarks |
| API framework | FastAPI | `GET /health` and `POST /ask` |
| Embeddings | sentence-transformers, all-MiniLM-L6-v2 (384-dim) | Dense retrieval signal, local, CPU-only |
| Keyword search | BM25 (rank-bm25) | Keyword retrieval signal |
| Vector index | FAISS, IndexFlatIP (cosine) | In-memory exact search over 101 chunks |
| Reranker (ablation) | cross-encoder/ms-marco-MiniLM-L-6-v2 (22.7M params, ~91 MB) | Measured as an ablation config, not the default |
| Generative model (opt-in) | Hugging Face, Qwen/Qwen2.5-0.5B-Instruct (0.5B params, 988.1 MB, bf16, CPU-only) | Opt-in generative answer with verification and fallback |
| LLM-as-judge | Same local Qwen2.5-0.5B model | Judges faithfulness and relevance alongside the string proxies |
| Testing | pytest | 179 tests |
| CI | GitHub Actions (`.github/workflows/eval.yml`) | Build index, run tests, run the CI gate on every push and PR |
| Containers | Docker (Dockerfile builds the index at image build time) | See the Docker note below; no container can start in this dev VM |
| Data | Plain files: `.txt` + `.json` in `data/raw`, `chunks.jsonl`, FAISS index | No database server; the corpus is a file |

What was not used, and why:

- **PostgreSQL with pgvector: not built.** The docs do not list it as planned. It was considered and rejected for this corpus: pgvector needs a Postgres server, and the corpus is a file of 101 chunks. See `ARCHITECTURE.md`, decision 2. When the corpus grows past a few thousand chunks, the right move is an approximate FAISS index or a managed vector store, measured then, not now.
- **LangChain / LlamaIndex: not used.** The pipeline is hand-rolled in plain Python so every step (chunking, scoring, abstention) is visible, testable, and explainable in an interview. See `ARCHITECTURE.md`, decision 1.
- **Hosted embedding or LLM APIs: not used.** No API key, no per-query cost, no data leaving the machine. Cost per query is $0.00.

## Corpus

19 public official scholarship pages, one page per scholarship. All fetched 18 Nov 2026. See `data/sources.json` and `data/raw/*.json` for the exact URL, fetch method, HTTP status, and fetch date of every document. Raw text is saved verbatim and never edited after fetching. No raw HTML is stored (`data/raw/*.html` is gitignored and was removed from history); only extracted `.txt` and metadata `.json`.

| # | Scholarship | Source |
|---|---|---|
| 1 | Chevening (UK) | https://www.chevening.org/scholarships/ |
| 2 | Sweden SI Global Professionals | https://si.se/en/apply/scholarships/swedish-institute-scholarships-for-global-professionals/ |
| 3 | Fulbright Foreign Student Program (USA) | https://foreign.fulbrightonline.org/about/foreign-student-program |
| 4 | DAAD (Germany) | https://www.daad.de/en/study-and-research-in-germany/scholarships/ |
| 5 | Erasmus Mundus Joint Masters | https://erasmus-plus.ec.europa.eu/opportunities/opportunities-for-individuals/students/erasmus-mundus-joint-masters |
| 6 | France Excellence Eiffel | https://www.campusfrance.org/en/the-france-excellence-eiffel-scholarship-program |
| 7 | Vanier Canada Graduate Scholarships | https://vanier.gc.ca/en/home-accueil.html |
| 8 | Manaaki New Zealand Scholarships | https://www.nzscholarships.govt.nz/ |
| 9 | Hong Kong PhD Fellowship Scheme (HKPFS) | https://cerg1.ugc.edu.hk/hkpfs/index.html |
| 10 | MEXT (Japan) | https://www.studyinjapan.go.jp/en/planning/scholarships/mext-scholarships/ |
| 11 | Gates Cambridge | https://www.gatescambridge.org/programme/the-scholarship/ |
| 12 | Rhodes Scholarship (FAQs) | https://www.rhodeshouse.ox.ac.uk/scholarships/faqs/ |
| 13 | Knight-Hennessy Scholars, Funding (Stanford) | https://knight-hennessy.stanford.edu/program-overview/funding |
| 14 | Schwarzman Scholars (FAQs) | https://www.schwarzmanscholars.org/frequently-asked-questions/ |
| 15 | Commonwealth Shared Scholarships | https://cscuk.fcdo.gov.uk/scholarships/commonwealth-shared-scholarships-applications/ |
| 16 | Research Training Program (RTP, Australia) | https://www.education.gov.au/research-block-grants/research-training-program |
| 17 | Marshall Scholarship | https://marshallscholarship.org/about/ |
| 18 | Erasmus Mundus Joint Masters (organisations page) | https://erasmus-plus.ec.europa.eu/opportunities/organisations/cooperation-among-organisations-and-institutions/erasmus-mundus-joint-masters |
| 19 | SI Scholarship Pioneering Women in STEM | https://si.se/en/apply/scholarships/pioneeringwomeninstem/ |

Chunk counts at 512 tokens, measured by `scripts.build_index`: chevening 1, sweden-si 12, fulbright 3, daad 1, erasmus-mundus 2, eiffel 3, vanier 1, manaaki 1, hkpfs 2, mext 5, gates-cambridge 2, rhodes 29, knight-hennessy 2, schwarzman 12, commonwealth-shared 12, rtp 4, marshall 2, erasmus-mundus-org 1, si-pioneering-women 6. Total: 101 chunks. Rhodes is 29 chunks because its FAQ page is long (50,171 chars of extracted text).

Fetch notes:

- Chevening returned HTTP 403 to a direct fetch and DAAD's direct fetch failed (proxy connection error), so both were saved via a text fetcher from the same official URLs, recorded as `fetch_method: "browser-text-fetch"` in `data/raw/*.json`. The other 8 original pages returned HTTP 200.
- The 9 new pages were also saved via a text fetcher from the same official URLs on 2026-11-18, recorded the same way, with the reason per page (thin or JS-rendered pages for Gates Cambridge, Rhodes, Knight-Hennessy, Schwarzman, and Commonwealth Shared; direct fetch timed out for RTP).
- No Reddit data. No tutorial corpus. No scraped private data.

## How it works

```
official pages -> ingest.py (fetch + extract text, save raw + metadata)
  -> chunking.py (text normalisation + recursive chunking, ~512 tokens, 200-char overlap)
  -> embeddings.py (sentence-transformers all-MiniLM-L6-v2, 384-dim)
  -> retrieval.py (FAISS dense + BM25, 0.5/0.5 hybrid, lexical rerank;
     cross-encoder rerank available as an ablation configuration)
  -> eligibility.py (degree + citizenship checks, corpus-grounded, only if profile given; closed programmes return verdict `closed`)
  -> answer.py (extractive answer + citations + abstention gates)
  -> generative.py (OPT-IN: local LLM answer + verification, falls back to extractive)
  -> api.py (FastAPI, POST /ask)
  -> eval.py (golden set, recall@k, faithfulness, answer relevance, eligibility accuracy,
              + LLM-as-judge faithfulness / relevance with the same local model)
```

### Chunking: why 512 tokens

- Target 512 tokens, about 2,048 characters, with ~200 characters overlap (~50 tokens). One token is roughly 4 characters of English text.
- That size is big enough to hold one complete idea (a paragraph about eligibility, or a funding bullet list) and small enough that the embedding still means something specific.
- Overlap exists so a fact split across a chunk boundary is not lost.
- Chunking is recursive and sentence-aware: split on paragraphs, then sentences, then words only if a single sentence is still too long. It is implemented from scratch in `app/chunking.py` (no LangChain) so the behaviour is fully visible and testable.
- Before chunking, `app/textnorm.py` normalises the text: it unwraps hard line breaks inside sentences, keeps headers separate from content, and adds terminal punctuation to fragments. This step alone moved answer-only relevance from 0.70 to 0.85.
- **512 is a measured choice, not a reasoned one.** Phase 2, Step 5 ablated 256 / 512 / 1,024 tokens on the same 39-question golden set. See the chunk-size table in the Results section below.

### Retrieval: why hybrid

- Dense embeddings alone miss exact names and numbers ("Vanier", "$50,000", "SI Scholarship"). BM25 alone misses paraphrases ("fully funded master's in the UK" vs "one-year master's degrees").
- So retrieval is hybrid: FAISS cosine similarity over all-MiniLM-L6-v2 embeddings, plus BM25 over chunk tokens, each normalised to 0-1 and combined 0.5/0.5. The top 10 are then reranked with a transparent lexical reranker (query-term coverage bonus).
- Honest finding from the ablation: on this 101-chunk corpus, dense retrieval buys no measured recall (recall@5 is 1.0 for every signal) and slightly hurts answer-only relevance (dense-only 0.844 vs BM25-only 0.875) at about 39x the retrieval latency (77.7 ms vs 2.0 ms). The improvement comes from the answering discipline, not the retrieval signal.

### Answering: why extractive, why abstain

- Phase 1 has no generative LLM in the default path, so the "answer" is sentences taken verbatim from the top cited chunks. That makes faithfulness measurable instead of claimed (measured 1.0).
- If the best score is below the abstention threshold, or no retrieved chunk shares a content word with the question, the API returns:

```json
{"found": false, "answer": "Not found in the corpus.", "citations": [],
 "abstention_reason": "missing_specific_term"}
```

- Every refusal names its reason (`weak_score`, `doc_name_mismatch`, `no_matching_sentences`, `weak_overlap`, `missing_specific_term`, `low_answer_similarity`, `unanswered_ranking`, `unanswered_scholarship_subject`, `empty_index`). Abstaining is a feature, not a failure. A scholarship assistant that invents a deadline is worse than one that says it does not know.
- Phase 2, Step 2 added an answer-verification step after the lexical gates (`app/answer.py:verify_answer_relevance`): the picked sentences must clear a dense-similarity floor (best picked sentence at least 0.45 cosine to the question) and, when the question asks about a ranking, must mention a ranking. Otherwise the answer is refused as `low_answer_similarity` or `unanswered_ranking`.
- Phase 2, Step 6 added a second question-focus check: a question naming a scholarship no known document covers must be answered by a sentence that mentions that scholarship's name (`unanswered_scholarship_subject`). It also added "scholar" to the generic terms, because every KHS sentence contains "scholars" and the specific-term gate was passing on that word alone.

### Eligibility-first answering

`POST /ask` accepts an optional `profile`:

```json
{"question": "Can I apply for the SI Scholarship for Global Professionals?",
 "profile": {"nationality": "Pakistan", "degree_level": "masters"}}
```

- Before the substantive answer, the system checks degree level and citizenship against rules grounded only in corpus sentences. Every rule's evidence sentence is validated verbatim against the chunk text at load time; a rule whose evidence is missing drops to `unknown`. Where the corpus states no rule, the verdict is `unknown`, never guessed from general knowledge.
- A real finding: **Pakistan is not among the SI Scholarship's 34 eligible countries** (the list is in the corpus, chunk `sweden-si-2`), so a Pakistani master's applicant is correctly told "not eligible" with the full list cited.
- A closed programme is never reported as eligible: Vanier is closed (chunk `vanier-0`), so any profile check for it returns verdict `closed`, states the programme is closed, names its replacement (the Canada Graduate Research Scholarship - Doctoral program), and cites the closure sentences.
- Measured: eligibility accuracy **1.0 (11/11 golden questions)**.

### Auditable retrieval

- Every `/ask` response carries a `retrieval_trace`: every candidate chunk with its dense score, BM25 score, combined score, rerank score and term coverage, in reranked order, and whether it was injected into the answer. A wrong answer can be traced to the exact chunk and score that produced it.
- `benchmarks/audit_retrieval.py` writes the same trace for all 50 golden questions to `benchmarks/results/retrieval_audit.json` (50 questions, 72 injected chunks).

## Phase 2: the 7 steps

Each step is one branch, merged with `--no-ff`, measured on the golden set before merging.

- **Step 1, generative answer + verification + LLM-as-judge.** Opt-in generative answer from local Qwen2.5-0.5B-Instruct (0.5B params, 988.1 MB, CPU-only, $0), with a deterministic verification step: every generated sentence is checked against the cited chunks, unsupported sentences are dropped, otherwise the answer falls back to extractive. Measured on 38 questions: generative answer-only 0.697 (23/33) vs extractive 0.909 (30/33). The generator is worse than quoting the source, so extractive stays the default. The judge scored 0.957 / 0.957 (n=23) but overstates quality: it scored q10's wrong "one academic year" 1/1 and the correct Vanier-closure answer (q21) 0/0. It fixed one paraphrase-gap failure (q03).
- **Step 2, dense sentence scoring + answer verification.** Sentence picking now adds 2.0 x cosine similarity (same local MiniLM embedder, $0) to the lexical score. Weight 2.0 shipped as the largest with zero regressions: weights 3-4 fix q03 but lose q16. The verification step fixed o04, which now abstains with reason `unanswered_ranking`. Abstention went 0.8 (4/5) to 1.0 (5/5). Dense scoring itself buys nothing measurable on this corpus.
- **Step 3, cross-encoder reranker ablation.** Model `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M params, ~91 MB, CPU-only, $0) as the `hybrid-cross-encoder` configuration. Measured on the 31-chunk corpus: answer-only 0.87 with and without it, abstention 1.0 both ways, at 4,238.0 ms vs 123.7 ms retrieval mean (about 34x). Re-measured on the grown 101-chunk corpus: still 0.875 both ways, at 3,987.1 ms vs 185.9 ms (about 21x). It reorders 17 of 23 top-5 lists but no picked answer changes. Lexical stays the default.
- **Step 4, eligibility reads the closure flag.** Vanier is closed, so a profile check returns verdict `closed`, not `not_eligible`, with the closure evidence and the replacement programme, whatever the profile. New file `data/closures.json` (version 1), evidence validated verbatim at load time. Golden set 38 to 39 questions (new e11, Pakistani doctoral applicant, Vanier). Eligibility 1.0 (11/11). Answer-only 0.909 to 0.912 (31/34).
- **Step 5, chunk-size ablation.** Same `.txt` corpus, overlap constant at 200 chars, three indexes (256 / 512 / 1,024 tokens), same 39-question golden set. 256 gives 60 chunks, answer-only 0.853, eligibility 0.818. 512 gives 31 chunks, answer-only 0.912, eligibility 1.0. 1,024 gives 19 chunks, answer-only 0.853, eligibility 0.909. Recall, faithfulness, and abstention are 1.0 at all sizes. **512 kept, by the numbers.** 256 fixes q03 but drops 7 eligibility rules and tells a Pakistani SI applicant they are `eligible` when they are not. Honest confound: the rules were authored against 512-token chunking, so 512 has home-field advantage.
- **Step 6, corpus growth.** 10 to 19 documents, 31 to 101 chunks, 39 to 50 golden questions. Nine new docs, all fetched 18 Nov 2026 as `.txt` only: Gates Cambridge, Rhodes, Knight-Hennessy, Schwarzman, Commonwealth Shared, RTP, Marshall, Erasmus Mundus organisations page, SI Pioneering Women in STEM. Answer-only 0.912 to 0.907 (8 of 9 new questions pass, q32 is a documented miss). Growing the corpus first broke abstention, 1.0 to 0.714 (5/7): o06 was answered with SI + Commonwealth dates and o07 with KHS funding sentences. Both were fixed in the same step, so abstention is 1.0 (7/7) again, with the dip recorded, not hidden.
- **Step 7, re-verify stale pages.** All 19 source pages re-fetched 18 Nov 2026. All 19 live, 18 substantively unchanged, Eiffel changed only in a sidebar teaser. No `.txt` was rewritten, the rebuilt index is byte-identical (101 chunks), and the re-run eval is byte-identical to the before-run file. Same step fixes the data-quality checker's currency gap: its money pattern now matches £, SEK, and ¥ plus detached "($) 35,550" rates, so stipend-missing flags went 14 of 19 to 8 of 19, and findings went 72 to 73 (9 to 16 resolved, 63 to 57 open).

## Quickstart (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch the corpus (8 pages by HTTP; Chevening + DAAD text already saved)
python -m scripts.ingest

# 2. Chunk + embed + build the FAISS index
python -m scripts.build_index

# 3. Run the eval harness (prints real numbers from this machine)
python -m scripts.run_eval

# 4. Run the API
uvicorn app.api:app --reload --port 8000

# 5. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Chevening Scholarship fund?"}'

# 6. Ask with a profile (eligibility-first)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I apply for the SI Scholarship for Global Professionals?",
       "profile": {"nationality": "Pakistan", "degree_level": "masters"}}'
```

## API

`GET /health` returns `{"status": "ok", "docs": 19, "chunks": 101, "embedder": ...}`.

`POST /ask` takes `{"question": "...", "top_k": 5, "profile": {"nationality": "...", "degree_level": "..."}, "generate": false}`. `profile` is optional; `generate` is optional, default `false`, so the extractive default is unchanged.

With `"generate": true` the answer comes from the local generative model instead, verified sentence-by-sentence against the cited chunks (`app/verification.py`): unsupported sentences are dropped, and if none survives the answer falls back to the extractive one. The response then carries `"answer_mode": "generative" | "generative_fallback_extractive" | "abstained" | "eligibility"`, a `"generation"` object (`model`, `latency_s`, `reason`), and, for verified answers, a `"verification"` object (`kept`, `dropped`, `n_dropped`). Eligibility answers are never generated, and an abstention stays an abstention: `generate=true` cannot override the gates. Model: Qwen/Qwen2.5-0.5B-Instruct, 0.5B params, 988.1 MB bf16 weights, CPU-only, greedy decoding, $0.00 per query; mean generation latency 4.75 s on this VM (1.5 shared CPUs).

Example response:

```json
{
  "question": "What does the Chevening Scholarship fund?",
  "found": true,
  "answer": "Chevening Scholarships enable outstanding emerging leaders ... one-year master's degrees in the UK. Chevening Scholarships are fully-funded ...",
  "score": 0.62,
  "embedder": "all-MiniLM-L6-v2",
  "citations": [
    {"doc": "chevening", "source_url": "https://www.chevening.org/scholarships/",
     "chunk_id": "chevening-0", "score": 0.62, "text": "..."}
  ],
  "retrieval_trace": {"embedder": "all-MiniLM-L6-v2", "candidates": [
    {"chunk_id": "chevening-0", "doc": "chevening", "dense_score": 0.9,
     "bm25_score": 0.7, "combined_score": 0.8, "rerank_score": 0.85,
     "term_coverage": 0.6, "injected": true}]}
}
```

If the sentence-transformers model cannot be downloaded (offline machine), the embedder falls back to BM25-only retrieval and the response reports `"embedder": "bm25-only"`. The API never silently pretends dense retrieval ran when it did not.

## Eval harness

`eval/golden.json` holds 50 hand-written questions (32 in-corpus factual, 7 out-of-corpus, 11 eligibility). Every question was checked against the actual fetched text before it was added: the expected answer phrase must appear in the cited document.

`python -m scripts.run_eval` prints:

- **recall@3 / recall@5**: fraction of questions where the expected document is in the top 3 / top 5 retrieved documents.
- **faithfulness**: fraction of answer sentences that appear verbatim in a cited chunk. For extractive answers this should be high; if it is not, something is broken.
- **answer relevance (proxy)**: fraction of questions where the expected answer phrase appears in the answer or cited chunks.
- **answer-only relevance**: fraction where the expected phrase appears in the answer text alone, not just the citations. This is the metric that separates the naive baseline from the improved pipeline.
- **abstention accuracy**: fraction of out-of-corpus questions correctly answered "Not found in the corpus."
- **eligibility accuracy**: fraction of profile questions where the verdict matches the expected verdict.

All numbers in this README come from a real local run. See [EVAL_RESULTS.md](EVAL_RESULTS.md). No number here is invented.

## Benchmarks and ablation

```bash
python -m benchmarks.run_benchmark   # 8 configurations, same golden set
```

Re-run in Phase 2, Step 6 with the current code on the grown corpus (101 chunks, 50 questions). Latency is measured after warm-up with the embedding model already loaded. Cost is $0.00 everywhere (local CPU, no API calls). Latency varies run to run by tens of milliseconds on this VM (shared CPU, 1.5 cores). The order of magnitude is stable; the exact figures are one measured run.

| Configuration | Mode | Reranker | recall@5 | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 0.625 | 1.0 | 0.0 | 0.3 ms | 0.7 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 0.656 | 1.0 | 0.0 | 84.1 ms | 84.6 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 0.875 | 1.0 | 1.0 | 2.0 ms | 1179.2 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 0.844 | 1.0 | 1.0 | 77.7 ms | 1443.7 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 0.875 | 1.0 | 1.0 | 60.2 ms | 1407.1 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 0.844 | 1.0 | 0.0 | 145.3 ms | 2038.0 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 0.875 | 1.0 | 1.0 | 185.9 ms | 2566.1 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 0.875 | 1.0 | 1.0 | 3987.1 ms | 5374.9 ms | $0.00 |

Each configuration changes one thing from its neighbour, so each component's contribution is isolated. What the ablation shows:

- The improvement is the answering discipline, not the retrieval signal. Baseline to improved-hybrid: answer-only 0.625 to 0.875, abstention 0.0 to 1.0. Hybrid-plain (hybrid retrieval, naive answering) scores only 0.656.
- Dense buys no recall here: recall@5 is 1.0 for every configuration.
- Dense slightly hurts answer-only relevance (dense-only 0.844 vs BM25-only 0.875) at about 39x the retrieval latency of BM25.
- The reranker adds nothing measurable (0.875 with and without it). The cross-encoder adds nothing either, at about 21x the retrieval latency of the lexical reranker on the grown corpus.
- The gates are the entire abstention story (hybrid-no-gates 0.0 to improved-hybrid 1.0).

Full results: `benchmarks/results/<config>.json`. Summary: `benchmarks/results/summary.md`.

## Before / after

```bash
python -m benchmarks.before_after   # generated from the results files
```

Baseline-naive to improved-hybrid, on the grown corpus (39 factual questions):

| Metric | Before (baseline-naive) | After (improved-hybrid) | Change |
|---|---|---|---|
| recall@5 | 1.0 | 1.0 | +0.000 |
| Answer-only relevance | 0.625 | 0.875 | +0.250 |
| Faithfulness | 1.0 | 1.0 | +0.000 |
| Abstention accuracy | 0.0 | 1.0 | +1.000 |
| Retrieval latency (mean) | 0.3 ms | 185.9 ms | +185.6 ms |
| Total latency (mean) | 0.7 ms | 2566.1 ms | +2565.4 ms |
| Cost per query | $0.00 | $0.00 | $0.00 |

Why it changed: the answering discipline (scored picking + gates), not the retrieval signal. The baseline returned the first sentences of the top chunk, usually headers, navigation, or a marketing intro. Full report with failure tables: [benchmarks/results/before_after.md](benchmarks/results/before_after.md).

## Results by phase

### Phase 1 (38 questions, 18 Nov 2026)

| Metric | Value |
|---|---|
| Golden questions | 38 (23 factual + 5 out-of-corpus + 10 eligibility) |
| recall@3 / recall@5 | 1.0 / 1.0 |
| Answer relevance (proxy) | 1.0 |
| Answer-only relevance | 0.909 (30/33) |
| Faithfulness | 1.0 |
| Abstention accuracy | 0.8 (4/5) |
| Eligibility accuracy | 1.0 (10/10) |
| Tests | 87 passed, 0 failed |
| Data-quality findings | 36 (8 resolved, 28 open) |
| Cost per query | $0.00 (local CPU, no API calls) |

### Phase 2, Step 5, chunk-size ablation (39 questions, 18 Nov 2026)

Command: `python -m scripts.run_chunk_ablation` (437.9 s; per-size results in `benchmarks/results/chunk_ablation/`).

| Chunk size | Chunks | recall@5 | Answer-only relevance | Eligibility | Retrieval / total mean |
|---|---|---|---|---|---|
| 256 tokens | 60 | 1.0 | 0.853 (29/34) | 0.818 (9/11) | 93.0 ms / 1,011.5 ms |
| **512 tokens** | **31** | **1.0** | **0.912 (31/34)** | **1.0 (11/11)** | **97.6 ms / 1,264.0 ms** |
| 1,024 tokens | 19 | 1.0 | 0.853 (29/34) | 0.909 (10/11) | 256.3 ms / 2,463.6 ms |

256 fixes q03 but drops 7 eligibility rules, so a Pakistani SI applicant is wrongly told `eligible`. 1,024 drops 2 rules and fails e03 the same way. Recall, faithfulness, and abstention are 1.0 at every size. **512 stays the default, by the numbers.**

### Phase 2, Step 6, corpus growth (50 questions, 18 Nov 2026)

| Metric | Before (10 docs, 31 chunks, 39 q) | After (19 docs, 101 chunks, 50 q) | Change |
|---|---|---|---|
| n_in_corpus / n_out_of_corpus | 34 / 5 | 43 / 7 | +9 factual (q24-q32), +2 out-of-corpus (o06, o07) |
| n_eligibility | 11 | 11 | none (no new rules authored) |
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 | none |
| answer_relevance_proxy | 1.0 | 1.0 | none |
| answer_only_relevance | 0.912 (31/34) | 0.907 (39/43) | +8 correct, +1 miss (q32) |
| faithfulness | 1.0 | 1.0 | none |
| abstention_accuracy | 1.0 (5/5) | **1.0 (7/7)** | dipped to 0.714 (5/7) mid-step, fixed in-step |
| eligibility_accuracy | 1.0 (11/11) | 1.0 (11/11) | none |

### Phase 2, Step 7, re-verification (50 questions, 18 Nov 2026)

| Metric | Before (Step 6) | After (Step 7) | Change |
|---|---|---|---|
| Pages re-verified live / substantively unchanged | - | 19 / 18 (Eiffel: sidebar teaser only) | - |
| Data-quality findings (resolved / open) | 72 (9 / 63) | 73 (16 / 57) | +7 resolved, -6 open |
| Pages flagged "stipend missing" | 14 of 19 | **8 of 19** | -6 false flags (Rhodes, Commonwealth, SI Pioneering, Sweden SI, MEXT, RTP) |
| Pages flagged "deadline missing" | 15 of 19 | 15 of 19 | none |
| `multiple_amounts` findings (resolved) | 3 | 10 | +7, each resolved with its field |
| answer_only_relevance | 0.907 (39/43) | 0.907 (39/43) | none, same 4 misses (q03, q10, q17, q32) |
| recall@5 / faithfulness / abstention / eligibility | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 / 1.0 | none |
| Golden questions changed / dropped | - | 0 / 0 | none, no page fact changed |
| Chunks (rebuilt index) | 101 | 101 (byte-identical) | none |

Read honestly: the re-verification found almost nothing, and that is the result. A launch claim needs the pages checked, not assumed fresh. The checker fix is the real change, and it changed the documentation, not the answers: the corpus always stated those six stipends; the checker could not see them.

### Phase 2, Step 1, generative eval (38 questions, 18 Nov 2026)

Command: `python -m scripts.run_generative_eval` (288.1 s on this VM; model load 3.6 s; full results in `benchmarks/results/generative_eval.json`).

| Metric | Extractive (Phase 1) | Generative (Phase 2 Step 1) |
|---|---|---|
| Answer-only relevance | 0.909 (30/33) | 0.697 (23/33) |
| Faithfulness (verbatim proxy) | 1.0 | 0.754 |
| Judge faithfulness / relevance | - | 0.957 / 0.957 (n=23, 0 unparsed) |
| Abstention / eligibility accuracy | 0.8 / 1.0 | 0.8 / 1.0 |
| Answer modes (n=33) | - | 16 generative, 7 fallback to extractive, 10 eligibility |
| Mean generation / judge latency | - | 4.75 s / 3.82 s |
| Cost per query | $0.00 | $0.00 (local CPU, no API calls) |

Read honestly: the 0.5B generator is worse than quoting the source at putting the expected phrase in the answer (0.697 vs 0.909). It paraphrases, omits the key phrase, and sometimes declines when the evidence is there (q15). Verification + fallback kept it safe (7 fallbacks; every kept sentence passed the grounding and number checks, by construction), and it did fix one Phase 1 paraphrase-gap failure (q03). The judge's 0.957 overstates quality: the same small model grades itself leniently. It scored q10's "one academic year" (expected "1 to 2 academic years") 1/1, and scored the correct Vanier-closure extractive answer (q21) 0/0. The judge is a weak instrument at 0.5B; the proxy metrics and the failure tables are the numbers to trust.

## Known failures

No regressions in the improved pipeline: no question it got right is lost. The failure table below is the honest record.

| Question | Why it still fails |
|---|---|
| q03, q10, q17 | Paraphrase gap. The expected answer's key terms are not in the question, so lexical picking cannot find the sentence. Dense sentence scoring (Step 2, weight 2.0) does not fix them either: q03's correct sentence ranks 4th by dense sim (0.630), q10's 8th (0.438), q17's 14th (0.527). The generative path fixes q03 only. |
| q32 (SI Pioneering Women allowance) | Bullet-list merging. The "SEK 12,000" sentence sits in a dash-led bullet list, the splitter merges the whole list into one long sentence, and it loses to the page intro on shared stems. The evidence is in the top retrieved chunk (`si-pioneering-women-1`); the picker does not surface it. New in Step 6. |
| o04 (Vanier ranked university) | **Fixed in Step 2.** The answer-verification step now abstains with reason `unanswered_ranking` (abstention 1.0, 7/7). Dense similarity alone could not catch it (o04 scored 0.667, above several answerable questions). |
| o06, o07 (abstention lost on corpus growth) | **Fixed in Step 6.** Growing the corpus to 19 documents first broke both: o06 answered with SI + Commonwealth dates, o07 with KHS funding sentences (abstention 1.0 to 0.714 mid-step). Fixed two ways: the verification gained a scholarship-subject focus check (o06, `unanswered_scholarship_subject`) and "scholar" joined the generic terms (o07, `missing_specific_term`). Abstention is 1.0 (7/7) again; the dip is recorded in `tests/test_corpus_growth.py`, not hidden. |
| Eligibility vs Vanier closure | **Fixed in Step 4.** A profile check for Vanier now returns verdict `closed` with the closure evidence and the replacement programme (e02, e05, e11 all pass; eligibility 1.0, 11/11). |

## Data quality (messy real scholarship data)

```bash
python -m scripts.check_data_quality
```

73 findings (16 resolved, 57 open), each citing sentence + chunk + source URL. Headline findings:

- Vanier is **closed** and replaced by a harmonized doctoral programme, and eligibility now returns verdict `closed` for it (Step 4).
- Eiffel EUR 1,200 is the Master's allowance and EUR 2,000 the Doctorate stipend (not a contradiction).
- The EMJM organisations page's EUR 5 million (max grant per project) and EUR 1,400/month (scholarship per individual) are different fields, not a contradiction (resolved in Step 6).
- HKPFS HK$344,400 is about US$44,150, one 2026/27 stipend in two currencies, plus a HK$14,400 travel allowance (resolution corrected in Step 7).
- DAAD is a 147-option database, not one award.
- 15 of 19 pages state no deadline and **8 of 19 state no stipend** (by the checker's rules: a deadline needs a date-bearing sentence with a deadline word, a stipend needs a EUR/$/HK$/US$/£/SEK/¥ amount or a detached "($)" rate).
- Step 7 fixed the known checker gap and re-verified every page 18 Nov 2026 (all 19 live, 18 substantively unchanged, Eiffel changed only in a sidebar teaser). Its money pattern now matches £/SEK/¥, so it no longer flags "stipend missing" for Rhodes (£21,060), Commonwealth (£1,712/month), SI Pioneering (SEK 12,000/month), Sweden SI (SEK 12,000/month), MEXT (¥117,000-¥242,000/month), or RTP ("($) 35,550").
- No eligibility rules are authored for the 9 new documents yet, so the checker flags degree-levels/citizenship as missing for all nine by construction. Authoring rules is a separate step.

Remaining gaps, named in [DATA_QUALITY.md](DATA_QUALITY.md) section 10: a money amount appearing is still not proof it is a stipend (Gates's "US$210m" donation still matches), and MEXT's ¥ amounts cannot be tied to a category from the extracted text.

Full documentation: [DATA_QUALITY.md](DATA_QUALITY.md).

## CI gate

```bash
python -m scripts.ci_gate
```

Thresholds: recall@5 at least 0.90, answer-only relevance at least 0.60, faithfulness at least 0.95, abstention at least 0.60, eligibility at least 0.80. Measured (Phase 2, Step 7, re-run after re-verification): 1.0, 0.907, 1.0, 1.0, 1.0. **Gate passes.** Strict mode (`CI_GATE_STRICT=1`, all thresholds 1.0) correctly fails on answer-only relevance (0.907), proving the gate can fail. GitHub workflow: `.github/workflows/eval.yml` (build index, run tests, run gate on every push/PR).

## Tests

```bash
python -m pytest tests/ -q   # 179 passed
```

Tests cover chunking (size, overlap, sentence boundaries, text normalisation), retrieval (modes, known questions, empty index), answering and abstention (edge cases: empty documents, missing context, weak retrieval, duplicate chunks, single long sentence), auditable retrieval (abstention reasons, retrieval trace), eligibility (verdict cases, evidence grounding, API profile behaviour, closure), data quality (all known findings, no false positives), ablation (retrieval modes, configuration coverage), chunk-size ablation (256/512/1,024 tokens), corpus growth (19 documents, 101 chunks, 50 questions, o06/o07 abstaining), re-verification (19 pages live, £/SEK/¥ matching, eval identical to before-run), and report integrity (generated before/after report matches the results files). Phase 2, Step 1 adds 25 generative tests (`tests/test_generative.py`, fake model only; the real 0.5B model is never loaded in tests).

## Docker

```bash
docker build -t production-rag-eval-harness .
docker run -p 8000:8000 production-rag-eval-harness
```

The Dockerfile builds the index at image build time from the committed `data/` so the container starts without re-fetching the corpus. **Note (verified 18 Nov 2026):** Docker 29.1.3 is installed in this development VM and the daemon runs (vfs storage driver, `--iptables=false --bridge=none`, data-root under the home directory; `python:3.12-slim` pulls fine), but **no container can start here**: the VM's seccomp policy blocks `setns` into any mount or network namespace, so `docker run hello-world` and the `RUN` steps of `docker build` both fail (`operation not permitted`). The image has therefore **not been built or run in Docker here**. Build it on a machine whose Docker can start containers before trusting it. (The Dockerfile itself parses and its base image resolves; only container execution is blocked.)

## What is not done yet

- No query rewriting, no multi-hop retrieval.
- No deployment (local + Docker only; not deployed anywhere).
- No auto-refresh of the corpus (re-run `scripts.ingest` manually, then re-run the eval and the data-quality checker).
- No eligibility rules for the 9 new Step 6 documents (Gates, Rhodes, Knight-Hennessy, Schwarzman, Commonwealth Shared, RTP, Marshall, Erasmus Mundus organisations page, SI Pioneering Women). Profile checks for them return `unknown` by construction.
- No fix yet for the q32 bullet-list merging bug (split list items into separate pickable sentences).
- No stronger generator or judge: the 0.5B model is measured worse than extractive (0.697 vs 0.909), and the 0.5B judge overstates quality.

Done in Phase 2, Step 1-7: opt-in generative answer, cross-encoder reranker as an ablation config, LLM-as-judge, dense sentence scoring, answer-verification step, Vanier closure verdict, chunk-size ablation, corpus growth, and stale-page re-verification.

## What to learn from this (interview)

- **Chunking:** why 512 tokens, why overlap, what breaks when chunks are too small (lost context) or too big (diluted embedding). Why text normalisation mattered more than chunk size here (0.70 to 0.85).
- **Hybrid retrieval:** when dense wins, when BM25 wins, why you combine them, and how you prove it with recall@k instead of vibes. On this corpus: dense buys nothing. Be able to say why, with numbers.
- **Ablation:** how to isolate one component at a time (signal, reranker, gates) and read the table honestly, including the components that add nothing.
- **Evals:** a golden set with a baseline beats a demo. Be able to say "recall@5 is X, and it went up/down when I changed Y".
- **Abstention:** the most important production behaviour in a scholarship assistant is refusing to answer when the evidence is not there, and naming the reason.
- **Eligibility-first:** checking whether the asker can apply, from corpus-grounded rules, before answering the fact question.
- **Messy data:** real pages close, contradict, and omit. Document the mess with evidence; never let the system paper over it.
- **Observability:** every answer carries its citations, its score, and its retrieval trace, so any wrong answer can be traced back to the chunk.
