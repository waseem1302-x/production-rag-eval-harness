# Production RAG with Eval Harness — Phase 2, Step 5

A small, honest, production-shaped RAG system over public official
scholarship pages. It answers questions **only from the fetched corpus**,
cites the exact chunks it used, checks eligibility before answering
when a profile is given, and says **"Not found in the corpus"** when
the evidence is weak — instead of hallucinating.

Phase 1 is **retrieval + extractive answers + eligibility-first
answering + a measured eval harness**. Phase 2, Step 1 (done, on
branch `feat/generative-answer`) adds an **opt-in generative answer**
from a local $0 model (Qwen2.5-0.5B-Instruct, 988.1 MB, CPU) with a
**deterministic verification step** (every generated sentence checked
against the cited chunks; unsupported sentences dropped, otherwise
fall back to the extractive answer) and an **LLM-as-judge** alongside
the string-proxy metrics. Phase 2, Step 2 adds **dense sentence scoring**
to the answer picker and an **answer-verification step** that checks
whether the picked sentences actually answer the question: the
o04 failure (answering a Vanier ranking question the corpus cannot
answer) now abstains, so abstention accuracy is 1.0 (5/5). The
dense scoring itself is measured to buy **nothing** on this corpus
(see the numbers in [`EVAL_RESULTS.md`](EVAL_RESULTS.md)) — and
that is reported, not hidden. Phase 2, Step 4 (this branch,
`feat/eligibility-closure`) makes eligibility read the closure
flag: Vanier is closed, so a profile check for Vanier now returns
verdict `closed` — with the closure evidence and the replacement
programme — whatever the profile, even a doctoral applicant who
meets every other rule. Eligibility accuracy is 1.0 (11/11 golden
questions; one new Vanier doctoral question, e11, added to measure
the fix). Phase 2, Step 5 (this branch,
`feat/chunk-size-ablation`) ablates the chunk size — 256 / 512 /
1,024 tokens, measured on the same 39-question golden set — and
**keeps 512**: answer-only 0.912 (31/34) and eligibility 1.0
(11/11) at 512, vs 0.853 / 0.818 at 256 and 0.853 / 0.909 at 1,024.
256 fixes q03 but tells a Pakistani SI applicant they are
`eligible` when they are not (its citizenship rule drops when the
evidence sentence moves chunk). The extractive default is unchanged,
and the generator is deliberately the weaker path on this corpus.

**Read first:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (what was built,
why, trade-offs) · [`EVAL_RESULTS.md`](EVAL_RESULTS.md) (every measured
number) · [`DATA_QUALITY.md`](DATA_QUALITY.md) (the corpus's messy
data, documented not hidden) · [`benchmarks/results/before_after.md`](benchmarks/results/before_after.md)
(before/after with failure tables).

## Corpus (Phase 1: 10 documents, 31 chunks)

Public official pages for 10 of the 21 family scholarships from
The Next Scholar. One page per scholarship, fetched 18 Nov 2026.
See `data/sources.json` and `data/raw/*.json` for the exact URL, fetch
method, HTTP status and fetch date of every document. Raw text is
saved verbatim and never edited after fetching.

| # | Scholarship | Source |
|---|-------------|--------|
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

Chevening returned HTTP 403 to a direct fetch and DAAD's direct fetch
failed (proxy connection error), so both were saved via a text fetcher
from the same official URLs — recorded as
`fetch_method: "browser-text-fetch"` in `data/raw/*.json`, not hidden.
The other 8 returned HTTP 200. No Reddit data. No tutorial corpus.
No scraped private data.

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

### Chunking choice (why 512 tokens, why overlap)

Target **512 tokens, ~200 characters overlap (~50 tokens)**.
One token is roughly 4 characters of English text, so 512 tokens is
about 2,048 characters. That size is big enough to hold one complete
idea (a paragraph about eligibility, or a funding bullet list) and
small enough that the embedding still means something specific.
Overlap exists so a fact split across a chunk boundary is not lost.
Chunking is recursive and sentence-aware: split on paragraphs, then
sentences, then words only if a single sentence is still too long.
Implemented from scratch in `app/chunking.py` (no LangChain) so the
behaviour is fully visible and testable.

**512 is now a measured choice, not a reasoned one** (Phase 2,
Step 5: `python -m scripts.run_chunk_ablation`, same .txt corpus,
overlap held constant at 200 chars, same 39-question golden set):

| Chunk size | Chunks | recall@5 | Answer-only relevance | Eligibility | Retrieval mean | Total mean |
|---|---|---|---|---|---|---|
| 256 tokens | 60 | 1.0 | 0.853 (29/34) | 0.818 (9/11) | 93.0 ms | 1,011.5 ms |
| **512 tokens** | **31** | **1.0** | **0.912 (31/34)** | **1.0 (11/11)** | **97.6 ms** | **1,264.0 ms** |
| 1,024 tokens | 19 | 1.0 | 0.853 (29/34) | 0.909 (10/11) | 256.3 ms | 2,463.6 ms |

256 fixes q03 (SI master's duration) but drops 7 eligibility rules
— its evidence sentences move to other chunk ids, so a Pakistani SI
applicant is wrongly told `eligible`. 1,024 drops 2 rules and loses
e03 the same way. Recall, faithfulness and abstention are 1.0 at
every size and do not separate them; total-latency medians are
nearly flat (879 / 1,121 / 1,190 ms), so latency does not decide
it either. One honest confound: the eligibility rules were authored
against the 512-token chunking, so 512 has home-field advantage on
eligibility — changing size would mean re-authoring every pinned
chunk id first. Full analysis in [`EVAL_RESULTS.md`](EVAL_RESULTS.md);
per-size indexes in `benchmarks/results/chunk_ablation/chunk-<size>/`.

Before chunking, `app/textnorm.py` normalises the text: it unwraps
hard line breaks inside sentences, keeps headers separate from
content, and adds terminal punctuation to fragments. Official pages
put line breaks in the middle of sentences; without this, chunks
contain broken sentences and answers quote fragments. This step
alone moved answer-only relevance from 0.70 to 0.85.

### Retrieval choice (why hybrid)

Dense embeddings alone miss exact names and numbers ("Vanier",
"$50,000", "SI Scholarship"). BM25 alone misses paraphrases
("fully funded master's in the UK" vs "one-year master's degrees").
So retrieval is hybrid: FAISS cosine similarity over
`all-MiniLM-L6-v2` embeddings, plus BM25 over chunk tokens, each
normalised to 0–1 and combined 0.5/0.5. The top 10 are then reranked
with a transparent lexical reranker (query-term coverage bonus).
Phase 2, Step 3 added a cross-encoder reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22.7M params, ~91 MB,
CPU-only, $0) as an ablation configuration (`hybrid-cross-encoder`)
and measured it against the lexical reranker on the same 38-question
golden set: it reorders 17 of 23 top-5 lists but changes no answer
and no metric — answer-only relevance 0.87 with and without it,
abstention 1.0 both ways — at 4,238 ms retrieval mean vs 123.7 ms
for the lexical reranker (~34x). The lexical reranker stays the
default; the cross-encoder is kept as a measured option, not a win.

Honest finding from the ablation: on this 31-chunk corpus, dense
retrieval buys **no measured recall** (recall@5 is 1.0 for every
signal) and slightly hurts answer-only relevance (dense-only 0.826
vs BM25-only 0.87) at ~50–80x the latency. The improvement comes
from the answering discipline, not the retrieval signal. See
"Benchmarks" below and [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Answering choice (why extractive, why abstain)

Phase 1 has no generative LLM, so the "answer" is sentences taken
verbatim from the top cited chunks. That makes faithfulness
measurable instead of claimed (measured 1.0). If the best score is
below the abstention threshold, or no retrieved chunk shares a
content word with the question, the API returns:

```json
{"found": false, "answer": "Not found in the corpus.", "citations": [],
 "abstention_reason": "missing_specific_term"}
```

Every refusal names its reason (`weak_score`, `doc_name_mismatch`,
`no_matching_sentences`, `weak_overlap`, `missing_specific_term`,
`low_answer_similarity`, `unanswered_ranking`, `empty_index`).
Abstaining is a feature, not a failure. A scholarship
assistant that invents a deadline is worse than one that says it
does not know.

Phase 2, Step 2 adds an **answer-verification step** after the
lexical gates (`app/answer.py:verify_answer_relevance`): the picked
sentences must clear a dense-similarity floor (best picked sentence
≥ 0.45 cosine to the question) and, when the question asks about a
ranking, must mention a ranking — otherwise the answer is refused
as `low_answer_similarity` or `unanswered_ranking`. The check was
added because dense similarity alone was measured and **cannot**
separate the o04 failure: o04's picked sentences score 0.667,
above several answerable questions (q19: 0.562, q14: 0.591,
q12: 0.615). The same check runs on the generative path's kept
sentences, falling back to the extractive answer when they fail.
The local 0.5B model was also probed as a verifier: as judge it
scored o04's answer 0/0 and the answerable probes 1/1, but it is
not used as a guard — it is the same weak, lenient instrument
that scored q10's wrong duration 1/1 in Step 1, and the guard
stays deterministic.

### Dense sentence scoring (Phase 2, Step 2)

Sentence picking is lexical + dense: each candidate sentence scores
1 point per shared query stem, +2 per shared specific stem, plus
`2.0 ×` its cosine similarity to the question (same local MiniLM
embedder as retrieval, $0, one batch embed per question). The
weight is 2.0 because the eval says so: weights 0, 1 and 2 change
no golden answer at all, and weights 3, 3.5 and 4 fix q03 but lose
q16 (its correct sentence scores only 0.394 dense similarity,
below the boilerplate that replaces it) — net answer-only 0.909
either way. So dense sentence scoring, like dense retrieval before
it, buys nothing measurable on this 31-chunk corpus: q03's correct
sentence ranks 4th by dense similarity (0.630), q10's 8th (0.438)
and q17's 14th (0.527). The mechanism stays in the path at the
largest non-regressing weight so it is measured, not assumed — and
the failure table below still lists q03, q10 and q17 as extractive
misses. (A rarity-weighted specific-term bonus was tried first and
reverted: it changed no golden answer on its own.)

### Eligibility-first answering

`POST /ask` accepts an optional `profile`:

```json
{"question": "Can I apply for the SI Scholarship for Global Professionals?",
 "profile": {"nationality": "Pakistan", "degree_level": "masters"}}
```

Before the substantive answer, the system checks degree level and
citizenship against rules grounded **only** in corpus sentences.
Every rule's evidence sentence is validated verbatim against the
chunk text at load time; a rule whose evidence is missing drops to
`unknown`. Where the corpus states no rule, the verdict is `unknown`
— never guessed from general knowledge.

```json
{"found": true,
 "answer": "Not eligible for the SI Scholarship for Global Professionals. ... Pakistan is not among the 34 eligible countries ... Evidence: Citizens of the following countries are eligible to apply for the SI Scholarship for Global Professionals: Armenia, Azerbaijan, Bangladesh, ...",
 "eligibility": [{"doc": "sweden-si", "verdict": "not_eligible",
   "reasons": ["..."], "evidence": [{"sentence": "...", "chunk_id": "sweden-si-2"}]}]}
```

A real finding: **Pakistan is not among the SI Scholarship's 34
eligible countries** (the list is in the corpus, chunk
`sweden-si-2`), so a Pakistani master's applicant is correctly told
"not eligible" with the full list cited. And a closed programme is
never reported as eligible: Vanier is closed (chunk `vanier-0`),
so any profile check for it returns verdict `closed`, states the
programme is closed, names its replacement (the Canada Graduate
Research Scholarship - Doctoral program), and cites the closure
sentences. Measured: eligibility
accuracy **1.0 (11/11 golden questions)**.

### Auditable retrieval

Every `/ask` response carries a `retrieval_trace`: every candidate
chunk with its dense score, BM25 score, combined score, rerank
score and term coverage, in reranked order, and whether it was
injected into the answer. A wrong answer can be traced to the exact
chunk and score that produced it. `benchmarks/audit_retrieval.py`
writes the same trace for all 38 golden questions to
`benchmarks/results/retrieval_audit.json` (38 questions, 52 injected
chunks).

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

`GET /health` — `{"status": "ok", "docs": 10, "chunks": 31, "embedder": ...}`

`POST /ask` — body `{"question": "...", "top_k": 5, "profile": {"nationality": "...", "degree_level": "..."}, "generate": false}`
(`profile` is optional; `generate` is optional, default `false` —
the extractive default is unchanged.)

With `"generate": true` the answer comes from the local generative
model instead, verified sentence-by-sentence against the cited
chunks (`app/verification.py`): unsupported sentences are dropped,
and if none survives the answer falls back to the extractive one.
The response then carries `"answer_mode": "generative" |
"generative_fallback_extractive" | "abstained" | "eligibility"`,
a `"generation"` object (`model`, `latency_s`, `reason`), and, for
verified answers, a `"verification"` object (`kept`, `dropped`,
`n_dropped`). Eligibility answers are never generated, and an
abstention stays an abstention — `generate=true` cannot override
the gates. Model: Qwen/Qwen2.5-0.5B-Instruct, 0.5B params,
988.1 MB bf16 weights, CPU-only, greedy decoding, $0.00 per query;
mean generation latency 4.75 s on this VM (1.5 shared CPUs).

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

If the sentence-transformers model cannot be downloaded (offline
machine), the embedder falls back to BM25-only retrieval and the
response reports `"embedder": "bm25-only"`. The API never silently
pretends dense retrieval ran when it did not.

## Eval harness

`eval/golden.json` holds 39 hand-written questions (23 in-corpus
factual, 5 out-of-corpus, 11 eligibility). Every question was checked
against the actual fetched text before it was added — the expected
answer phrase must appear in the cited document.

`python -m scripts.run_eval` prints:

- **recall@3 / recall@5** — fraction of questions where the expected
  document is in the top 3 / top 5 retrieved documents.
- **faithfulness** — fraction of answer sentences that appear verbatim
  in a cited chunk (extractive answers, so this should be high; if it
  is not, something is broken).
- **answer relevance (proxy)** — fraction of questions where the
  expected answer phrase appears in the answer or cited chunks.
- **answer-only relevance** — fraction where the expected phrase
  appears in the answer text alone, not just the citations. This is
  the metric that separates the naive baseline from the improved
  pipeline.
- **abstention accuracy** — fraction of out-of-corpus questions
  correctly answered "Not found in the corpus."
- **eligibility accuracy** — fraction of profile questions where the
  verdict matches the expected verdict.

All numbers in this README's "Results" section come from a real
local run — see [`EVAL_RESULTS.md`](EVAL_RESULTS.md). No number here
is invented.

## Benchmarks and ablation

```bash
python -m benchmarks.run_benchmark   # 8 configurations, same golden set
```

Re-run in Phase 2, Step 3 with the current code — the Step 1 numbers
were stale after Step 2 changed the `answer()` defaults (dense
sentence scoring on, verification on). Two things changed on the
re-run: abstention is now 1.0, not 0.8 (Step 2's verification
abstains on o04), and total latency is ~1–1.5 s, not ~70 ms (Step 2's
dense sentence scoring embeds the candidate sentences per question;
retrieval itself is unchanged). The cross-encoder buys nothing: it
reorders 17 of 23 top-5 lists but changes no answer and no metric,
at ~34x the retrieval latency.

| Configuration | Mode | Reranker | recall@5 | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 0.522 | 1.0 | 0.0 | 0.2 ms | 0.5 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 0.565 | 1.0 | 0.0 | 103.5 ms | 106.5 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 0.87 | 1.0 | 1.0 | 6.2 ms | 1070.7 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 0.826 | 1.0 | 1.0 | 57.6 ms | 961.3 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 0.87 | 1.0 | 1.0 | 77.4 ms | 1307.1 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 0.826 | 1.0 | 0.0 | 74.9 ms | 1342.9 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 0.87 | 1.0 | 1.0 | 123.7 ms | 1490.6 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 0.87 | 1.0 | 1.0 | 4238.0 ms | 5429.1 ms | $0.00 |

Each configuration changes one thing from its neighbour, so each
component's contribution is isolated. Full results:
`benchmarks/results/<config>.json`, summary:
`benchmarks/results/summary.md`.

## Before / after

```bash
python -m benchmarks.before_after   # generated from the results files
```

| Metric | Before (baseline-naive) | After (improved-hybrid) | Change |
|---|---|---|---|
| recall@5 | 1.0 | 1.0 | +0.000 |
| Answer-only relevance | 0.522 | 0.87 | +0.348 |
| Faithfulness | 1.0 | 1.0 | +0.000 |
| Abstention accuracy | 0.0 | 1.0 | +1.000 |
| Retrieval latency (mean) | 0.2 ms | 123.7 ms | +123.5 ms |
| Cost per query | $0.00 | $0.00 | $0.00 |

Why it changed: the answering discipline (scored picking + gates),
not the retrieval signal. The baseline returned the first sentences
of the top chunk — usually headers, navigation or a marketing intro.
Full report with failure tables:
[`benchmarks/results/before_after.md`](benchmarks/results/before_after.md).

### Failures (known, not hidden)

| Question | Why it still fails |
|---|---|
| q03, q10, q17 | Paraphrase gap — the expected answer's key terms are not in the question, so lexical picking cannot find the sentence; dense sentence scoring (Step 2, weight 2.0) does not fix them either — q03's correct sentence ranks 4th by dense sim, q10's 8th, q17's 14th. The generative path fixes q03 only. |
| o04 (Vanier ranked university) | **Fixed in Step 2:** the answer-verification step now abstains with reason `unanswered_ranking` (abstention 1.0, 5/5). Dense similarity alone could not catch it (o04 scored 0.667, above several answerable questions). |
| Eligibility vs Vanier closure | **Fixed in Step 4:** a profile check for Vanier now returns verdict `closed` with the closure evidence and the replacement programme (e02, e05, e11 all pass; eligibility 1.0, 11/11). |

No regressions: no question the baseline got right is lost by the
improved pipeline (pinned by `tests/test_dense_verification.py::test_no_baseline_correct_answer_lost`).

## Data quality (messy real scholarship data)

```bash
python -m scripts.check_data_quality
```

36 findings (8 resolved, 28 open), each citing sentence + chunk +
source URL. Headline findings: Vanier is **closed** and replaced by
a harmonized doctoral programme — and eligibility now returns
verdict `closed` for it (Step 4); Eiffel €1,200 is the Master's
allowance and €2,000 the Doctorate stipend (not a contradiction);
HKPFS HK$344,400 ≈ US$44,150 is one 2026/27 stipend in two
currencies; DAAD is a 147-option database, not one award; only SI
states a deadline; only Vanier/Eiffel/HKPFS state a stipend.
Full documentation: [`DATA_QUALITY.md`](DATA_QUALITY.md).

## CI gate

```bash
python -m scripts.ci_gate
```

Thresholds: recall@5 ≥ 0.90, answer-only relevance ≥ 0.60,
faithfulness ≥ 0.95, abstention ≥ 0.60, eligibility ≥ 0.80.
Measured (Phase 2, Step 4): 1.0, 0.912, 1.0, 1.0, 1.0 — **gate passes**. Strict mode
(`CI_GATE_STRICT=1`, all thresholds 1.0) correctly fails on answer-only relevance (0.912),
proving the gate can fail. GitHub workflow: `.github/workflows/eval.yml`
(build index → tests → gate on every push/PR).

## Results (Phase 1, real local run, 18 Nov 2026)

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

Full detail: [`EVAL_RESULTS.md`](EVAL_RESULTS.md).

### Results (Phase 2, Step 5 — chunk-size ablation, same 39 questions, 18 Nov 2026)

Command: `python -m scripts.run_chunk_ablation` (437.9 s;
per-size results in `benchmarks/results/chunk_ablation/`).

| Chunk size | Chunks | recall@5 | Answer-only relevance | Eligibility | Retrieval / total mean |
|---|---|---|---|---|---|
| 256 tokens | 60 | 1.0 | 0.853 (29/34) | 0.818 (9/11) | 93.0 ms / 1,011.5 ms |
| **512 tokens** | **31** | **1.0** | **0.912 (31/34)** | **1.0 (11/11)** | **97.6 ms / 1,264.0 ms** |
| 1,024 tokens | 19 | 1.0 | 0.853 (29/34) | 0.909 (10/11) | 256.3 ms / 2,463.6 ms |

Read honestly: **512 stays the default, by the numbers.** 256 fixes
q03 (factual answer-only 21/23 vs 20/23) but drops 7 eligibility
rules and tells a Pakistani SI applicant they are `eligible` when
they are not; 1,024 drops 2 rules and fails e03 the same way.
Recall, faithfulness and abstention are 1.0 at every size. The
512-column is a re-run of the Step 4 baseline (rebuilt 512 chunks
byte-identical to `data/processed/chunks.jsonl`). Full analysis in
[`EVAL_RESULTS.md`](EVAL_RESULTS.md).

### Results (Phase 2, Step 2 — dense scoring + answer verification, same 38 questions, 18 Nov 2026)

Commands: `python -m scripts.run_eval` (writes
`benchmarks/results/extractive_eval.json`) and
`python -m scripts.run_generative_eval` (writes
`benchmarks/results/generative_eval.json`).

| Metric | Step 1 extractive | Step 2 extractive | Step 2 generative |
|---|---|---|---|
| Answer-only relevance | 0.909 (30/33) | 0.909 (30/33) | 0.697 (23/33) |
| Abstention accuracy | 0.8 (4/5) | **1.0 (5/5)** | **1.0 (5/5)** |
| Faithfulness (verbatim proxy) | 1.0 | 1.0 | 0.754 |
| Judge faithfulness / relevance | — | — | 0.957 / 0.957 (n=23, 0 unparsed) |
| Eligibility accuracy | 1.0 (10/10) | 1.0 (10/10) | 1.0 (10/10) |
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| Answer modes (n=33) | — | — | 16 generative, 7 fallback to extractive, 10 eligibility |
| Mean generation / judge latency | — | — | 12.53 s / 7.36 s (shared CPU under contention; Step 1 measured 4.75 s / 3.82 s on a quieter VM) |

Read honestly: Step 2 fixes exactly one thing — o04 now abstains
(`unanswered_ranking`) in both paths, taking abstention from 0.8
to 1.0 with no other per-question change (verified by the full-eval
diff: zero changed rows in either column). Dense sentence scoring
changes no golden answer at the shipped weight 2.0; at weights 3–4
it fixes q03 but loses q16, net zero — so q03, q10 and q17 remain
extractive misses, and the generative column is unchanged from
Step 1 (0.697, same 16/7/10 modes; the new relevance check fired
0 times on the golden set — its generative-side effect is covered
by a synthetic unit test, not by a golden question). Full analysis
in [`EVAL_RESULTS.md`](EVAL_RESULTS.md).

### Results (Phase 2, Step 1 — generative, same 38 questions, 18 Nov 2026)

Command: `python -m scripts.run_generative_eval` (288.1 s on this VM;
model load 3.6 s; full results in `benchmarks/results/generative_eval.json`).

| Metric | Extractive (Phase 1) | Generative (Phase 2 Step 1) |
|---|---|---|
| Answer-only relevance | 0.909 (30/33) | 0.697 (23/33) |
| Faithfulness (verbatim proxy) | 1.0 | 0.754 |
| Judge faithfulness / relevance | — | 0.957 / 0.957 (n=23, 0 unparsed) |
| Abstention / eligibility accuracy | 0.8 / 1.0 | 0.8 / 1.0 |
| Answer modes (n=33) | — | 16 generative, 7 fallback to extractive, 10 eligibility |
| Mean generation / judge latency | — | 4.75 s / 3.82 s |
| Cost per query | $0.00 | $0.00 (local CPU, no API calls) |

Read honestly: the 0.5B generator is **worse** than quoting the
source at putting the expected phrase in the answer (0.697 vs
0.909) — it paraphrases, omits the key phrase, and sometimes
declines when the evidence is there (q15). Verification + fallback
kept it safe (7 fallbacks; every kept sentence passed the grounding
and number checks, by construction),
and it did fix one Phase 1 paraphrase-gap failure (q03). The judge's
0.957 overstates quality: the same small model grades itself
leniently — it scored q10's "one academic year" (expected "1 to 2
academic years") 1/1, and scored the correct Vanier-closure
extractive answer (q21) 0/0. The judge is a weak instrument at
0.5B; the proxy metrics and the failure tables are the numbers to
trust. Full analysis in [`EVAL_RESULTS.md`](EVAL_RESULTS.md).

## Tests

```bash
python -m pytest tests/ -q   # 158 passed
```

Tests cover chunking (size, overlap, sentence boundaries, text
normalisation), retrieval (modes, known questions, empty index),
answering and abstention (edge cases: empty documents, missing
context, weak retrieval, duplicate chunks, single long sentence),
auditable retrieval (abstention reasons, retrieval trace),
eligibility (27 verdict cases, evidence grounding, API profile
behaviour), data quality (all known findings, no false positives),
ablation (retrieval modes, configuration coverage), and report
integrity (generated before/after report matches the results files).
Phase 2, Step 1 adds 25 tests (`tests/test_generative.py`, fake
model only — the real 0.5B model is never loaded in tests):
verification (supported / invented-number / low-overlap /
novel-stem sentences), generative fallback on verification failure
and on model failure, judge parsing (including reversed label
order), and the API `generate=true` opt-in with the extractive
default unchanged. Phase 2, Step 2 adds 16 tests
(`tests/test_dense_verification.py`): dense similarity ranking
(duration / levels sentences above boilerplate), verification unit
tests with injected sims (ranking with/without rank term, low-sim
floor, no-sim degradation), o04 abstaining with reason
`unanswered_ranking` (live + held-out Chevening ranking question),
other out-of-corpus reasons unchanged, generative path abstaining
on o04 without running the model, generative fallback when kept
sentences fail relevance, and the full-eval no-regression test
(the Step 1 correct set of 30 is exactly preserved, abstention
1.0, q03/q10/q17 still misses). Phase 2, Step 5 adds 13 tests
(`tests/test_chunk_ablation.py`): chunks within budget and overlap
preserved at 256/512/1,024 tokens, the index builder accepting a
chunk-size parameter (256 → 60 chunks / 60 vectors, invalid size
rejected), and the eval running against a non-default chunk size
without crashing.

## Docker

```bash
docker build -t production-rag-eval-harness .
docker run -p 8000:8000 production-rag-eval-harness
```

The Dockerfile builds the index at image build time from the
committed `data/` so the container starts without re-fetching the
corpus. **Note (verified 18 Nov 2026):** Docker 29.1.3 is installed
in this development VM and the daemon runs (vfs storage driver,
`--iptables=false --bridge=none`, data-root under the home
directory; `python:3.12-slim` pulls fine), but **no container can
start here**: the VM's seccomp policy blocks `setns` into any mount
or network namespace, so `docker run hello-world` and the `RUN`
steps of `docker build` both fail (`operation not permitted`). The
image has therefore **not been built or run in Docker here** — build
it on a machine whose Docker can start containers before trusting
it. (The Dockerfile itself parses and its base image resolves;
only container execution is blocked.)

## What Phase 1 does NOT do (Phase 2)

- ~~No generative LLM answer (extractive only).~~ Done in Phase 2,
  Step 1: opt-in generative answer, local Qwen2.5-0.5B-Instruct,
  `POST /ask` with `"generate": true`; extractive stays the default.
- ~~No cross-encoder reranker (lexical rerank only).~~ Done in Phase 2,
  Step 3: `cross-encoder/ms-marco-MiniLM-L-6-v2` as the
  `hybrid-cross-encoder` ablation configuration — measured to buy
  nothing on this corpus (0.87 answer-only with and without it,
  abstention 1.0 both ways) at ~34x the retrieval latency. Lexical
  stays the default.
- ~~No LLM-as-judge (string-match proxies only).~~ Done in Phase 2,
  Step 1: same local model judges faithfulness and relevance
  alongside the proxies (`scripts/run_generative_eval.py`).
- No query rewriting, no multi-hop retrieval.
- ~~No dense sentence scoring (the paraphrase-gap fix).~~ Done in
  Phase 2, Step 2 — and measured to buy nothing on this corpus
  (answer-only 0.909 before and after; weights 3–4 trade q03 for
  q16, net zero). The mechanism stays in the path at weight 2.0.
- ~~No answer-verification step (the o04 fix).~~ Done in Phase 2,
  Step 1: deterministic verification in `app/verification.py` —
  every generated sentence grounded in the cited chunks or dropped,
  with fallback to the extractive answer. (o04 itself still answers
  instead of abstaining: its Vanier sentences are grounded, so
  verification cannot catch that they do not answer the question.)
  Done properly in Phase 2, Step 2: `app/answer.py:verify_answer_relevance`
  (dense floor + ranking-focus check) now abstains on o04 with
  reason `unanswered_ranking`, in both the extractive and the
  generative paths; abstention accuracy is 1.0 (5/5).
- No eligibility check for programme closure (the Vanier gap).
- No deployment (local + Docker only; not deployed anywhere).
- No auto-refresh of the corpus (re-run `scripts.ingest` manually,
  then re-run the eval and the data-quality checker).

## What to learn from this (interview)

- **Chunking:** why 512 tokens, why overlap, what breaks when chunks
  are too small (lost context) or too big (diluted embedding). Why
  text normalisation mattered more than chunk size here (0.70 → 0.85).
- **Hybrid retrieval:** when dense wins, when BM25 wins, why you
  combine them, and how you prove it with recall@k instead of vibes.
  On this corpus: dense buys nothing — be able to say why, with numbers.
- **Ablation:** how to isolate one component at a time (signal,
  reranker, gates) and read the table honestly, including the
  components that add nothing.
- **Evals:** a golden set with a baseline beats a demo. Be able to
  say "recall@5 is X, and it went up/down when I changed Y".
- **Abstention:** the most important production behaviour in a
  scholarship assistant is refusing to answer when the evidence
  is not there — and naming the reason.
- **Eligibility-first:** checking whether the asker can apply, from
  corpus-grounded rules, before answering the fact question.
- **Messy data:** real pages close, contradict, and omit. Document
  the mess with evidence; never let the system paper over it.
- **Observability:** every answer carries its citations, its score,
  and its retrieval trace, so any wrong answer can be traced back
  to the chunk.
