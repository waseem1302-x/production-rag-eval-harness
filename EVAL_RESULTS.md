# Eval Results — Phase 2, Step 2 (final)

All numbers below come from real local runs on 18 Nov 2026, on this
VM (Python 3.12.3, 1.5 shared CPUs, 6 GB RAM). Nothing here is
invented. Reproduce them with the commands in "How to reproduce".

## Phase 2, Step 2 — dense sentence scoring + answer verification

Commands: `python -m scripts.run_eval` (writes
`benchmarks/results/extractive_eval.json`) and
`python -m scripts.run_generative_eval` (regenerates
`benchmarks/results/generative_eval.json` + `.md`). Same
38-question golden set as Step 1, both columns in one run.
Step 2 total run: 661.4 s (model load 7.5 s; extractive 66.8 s;
generative phase 551.5 s) — slower than Step 1's 288.1 s because
the shared CPU was under contention from other jobs on the VM;
same model, same greedy decoding. Mean generation latency 12.53 s
(Step 1: 4.75 s) and mean judge latency 7.36 s (Step 1: 3.82 s)
for the same reason; the latency difference is contention, not a
code change.

| Metric | Step 1 extractive | Step 2 extractive | Step 1 generative | Step 2 generative |
|---|---|---|---|---|
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
| answer_relevance_proxy | 1.0 | 1.0 | 1.0 | 1.0 |
| answer_only_relevance | 0.909 (30/33) | 0.909 (30/33) | 0.697 (23/33) | 0.697 (23/33) |
| faithfulness (verbatim proxy) | 1.0 | 1.0 | 0.754 | 0.754 |
| judge faithfulness | — | — | 0.957 (n=23, 0 unparsed) | 0.957 (n=23, 0 unparsed) |
| judge relevance | — | — | 0.957 (n=23, 0 unparsed) | 0.957 (n=23, 0 unparsed) |
| abstention_accuracy | 0.8 (4/5) | **1.0 (5/5)** | 0.8 (4/5) | **1.0 (5/5)** |
| eligibility_accuracy | 1.0 (10/10) | 1.0 (10/10) | 1.0 (10/10) | 1.0 (10/10) |
| Answer modes (n=33) | — | — | 16 generative, 7 fallback, 10 eligibility | 16 generative, 7 fallback, 10 eligibility |

### What Step 2 changed (read honestly)

- **Exactly one thing improved: o04 now abstains, in both paths.**
  The answer-verification step (`app/answer.py:verify_answer_relevance`)
  refuses o04 with reason `unanswered_ranking` — the question asks
  about a ranking and no picked Vanier sentence mentions one. The
  full-eval diff vs Step 1 shows **zero other changed rows** in
  either column: same 30 answer-only correct, same 3 misses
  (q03, q10, q17), same 16/7/10 generative modes.
- **Dense sentence scoring buys nothing on this corpus.** The
  picker now adds `2.0 ×` cosine similarity (local MiniLM, $0) to
  each sentence's lexical score. Weight ablation, full eval per
  weight (same golden set, rarity-weighted bonus reverted — see below):

  | Dense weight | answer_only | Changed vs Step 1 baseline |
  |---|---|---|
  | 0, 1, 2 | 0.909 | none |
  | 3, 3.5, 4 | 0.909 | q03 fixed, q16 lost |

  At weights 3–4 the fix and the loss cancel exactly, and q16 is a
  regression the step refuses to ship ("no question the baseline
  got right may be lost"), so the shipped weight is 2.0 — the
  largest weight with zero regressions. The mechanism stays in the
  path so it is measured, not because it earns a gain here.
- **Why dense cannot fix the paraphrase gap here (measured).**
  Rank of the correct sentence by dense similarity among the
  top-3 chunks' sentences: q03 4th (0.630), q10 8th (0.438), q17
  14th (0.527). q16's correct sentence scores only 0.394 — below
  the boilerplate that replaces it at weight ≥ 3. Two variants
  were tried and dropped: embedding the question with the
  scholarship name stripped made things worse (q03 rank 9, q10
  rank 17, q12 rank 50), and a rarity-weighted specific-term bonus
  (2 × (1 − df/N) instead of flat +2, to stop "Global
  Professionals" rewarding every SI sentence) changed no golden
  answer on its own and was reverted to keep the change surface
  minimal.
- **Dense similarity cannot catch o04 either (measured).** o04's
  picked sentences score max 0.667 cosine to the question — above
  several answerable questions (q19: 0.562, q14: 0.591, q12:
  0.615) — so no dense threshold separates it. That is why the
  verification is a dense *floor* (0.45, below the lowest
  answerable max sim 0.562) plus a question-focus check: a ranking
  question must be answered by a sentence that mentions a ranking.
  The check is narrow and honest about its limit: it looks for a
  ranking *term*, not a ranked *entity* — a sentence about
  "best-ranked students" would pass it.
- **The generative model as verifier was probed, not adopted.**
  As judge, the local 0.5B model scored o04's extractive answer
  0/0 and the q01/q03/q10/q17 answers 1/1 — it does catch o04 on
  this run. It is still not the guard: it is the same weak,
  lenient instrument that scored q10's wrong "one academic year"
  1/1 in Step 1, it would add ~4–12 s per question to the
  extractive path, and the repo's guard stays deterministic. The
  new relevance check on the generative path's kept sentences
  fired **0 times** on the golden set (no `relevance_verification_failed`
  in `generative_eval.json`); its effect is covered by a synthetic
  unit test, not by a golden question.
- **Per-question, Step 2:** q03, q10, q17 still extractive misses
  (generative fixes q03 only, as in Step 1); o04 abstains in both
  modes; all other 30 in-corpus answers and all 10 eligibility
  verdicts unchanged.

## Golden set

`eval/golden.json` — 38 hand-written questions, every expected answer
verified against the actual fetched text:

| Group | Count | What it tests |
|---|---|---|
| In-corpus factual (q01–q23) | 23 | Answer the question from the cited document |
| Out-of-corpus (o01–o05) | 5 | Abstain when the corpus cannot answer |
| Eligibility (e01–e10) | 10 | Degree-level / citizenship verdicts with a profile |
| **Total** | **38** | |

## Eval harness — full Phase 1 pipeline

Command: `python -m scripts.run_eval`

| Metric | Value | Meaning |
|---|---|---|
| n_in_corpus | 33 | 23 factual + 10 eligibility |
| n_out_of_corpus | 5 | |
| n_eligibility | 10 | |
| recall@3 | 1.0 | Expected document in top 3 for all 33 |
| recall@5 | 1.0 | Expected document in top 5 for all 33 |
| answer_relevance_proxy | 1.0 | Expected phrase in answer or citations for all 33 |
| answer_only_relevance | 0.909 | Expected phrase in the answer text alone: 30/33 |
| faithfulness | 1.0 | Every answer sentence verbatim in a cited chunk |
| abstention_accuracy | 0.8 | 4/5 out-of-corpus questions correctly refused |
| eligibility_accuracy | 1.0 | 10/10 verdicts correct |

The 3 answer-only misses are q03, q10, q17 — the paraphrase gap
(see "Failures"). The 1 abstention miss is o04 (see "Failures").

## Generative eval — Phase 2, Step 1 (final)

Command: `python -m scripts.run_generative_eval` (288.1 s on this
VM; model load 3.6 s; extractive baseline 1.5 s; generative phase
263.1 s). Full results: `benchmarks/results/generative_eval.json`
and `benchmarks/results/generative_eval.md` (both generated by the
script, not hand-typed). Same 38-question golden set, same
retriever, both columns in one run.

Model: Qwen/Qwen2.5-0.5B-Instruct (Qwen, Apache-2.0), 0.5B params,
`model.safetensors` 988,097,824 bytes (988.1 MB), bf16, CPU-only,
greedy decoding, 2 torch threads (measured faster than 4 on this
VM: one generation 11.0 s vs 19.5 s). Downloaded 18 Nov 2026 from
Hugging Face (revision `7ae55760`); Phi-3-mini-4k-instruct
(3.8B, ~7.3 GB) was rejected without downloading weights — too
large for this VM's RAM. Cost $0.00 per query (local, no API calls).

| Metric | Extractive (Phase 1) | Generative (Phase 2 Step 1) |
|---|---|---|
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 |
| answer_relevance_proxy | 1.0 | 1.0 |
| answer_only_relevance | 0.909 (30/33) | 0.697 (23/33) |
| faithfulness (verbatim proxy) | 1.0 | 0.754 |
| judge faithfulness | — | 0.957 (22/23 parsed) |
| judge relevance | — | 0.957 (22/23 parsed) |
| judge unparsed | — | 0 (23 judged; 10 eligibility answers not judged) |
| abstention_accuracy | 0.8 (4/5) | 0.8 (4/5) |
| eligibility_accuracy | 1.0 (10/10) | 1.0 (10/10) |
| Answer modes (n=33) | — | 16 generative, 7 fallback to extractive, 10 eligibility |
| Mean generation latency | — | 4.75 s |
| Mean judge latency | — | 3.82 s |
| Cost per query | $0.00 | $0.00 |

### What the generative run shows (read honestly)

- **The generator is worse than quoting the source.** Answer-only
  0.697 vs 0.909: the 0.5B model paraphrases past the expected
  phrase (q01, q05, q07, q11–q14, q18), drops the key sentence
  (q01's final answer omits "one-year master's degrees"), and
  sometimes declines when the evidence is in context (q15: "not
  specified in the given context" — verification dropped it and the
  extractive fallback answered correctly with HK$344,400).
- **Verification + fallback worked as designed.** 7 of 33 fell back
  to extractive (q02, q04, q15, q16, q17, q20, q21); no kept
  generative sentence contains a number absent from the chunks
  (enforced by the number rule). Two measured bugs were found and
  fixed during this step: cross-document contamination (q17 — the
  model copied the SI field list into a Manaaki answer; fixed by
  restricting grounding to the named scholarship's chunks) and
  question-word overlap passing ungrounded sentences (fixed by
  measuring coverage on novel stems only).
- **One paraphrase-gap failure fixed:** q03 (SI master's duration)
  is answer-only True generatively; q10 and q17 remain misses.
- **The judge overstates quality.** 0.957 with 0 unparsed looks
  strong, but the same 0.5B model grades itself leniently: q10's
  "An Erasmus Mundus Joint Master lasts for one academic year"
  (expected "1 to 2 academic years") scored 1/1, while the
  correct Vanier-closure extractive answer (q21, "We are no longer
  accepting applications…") scored 0/0. The judge is a weak
  instrument at 0.5B — reported alongside the proxies, never
  instead of them. (An earlier parser bug that swapped the two
  scores when the judge answered the lines in reverse order was
  found by reading the raw outputs and fixed with label-aware
  parsing; the first run's 0.778 / 0.63 was that bug, not the
  model.)
- **Abstention and eligibility are unchanged** (0.8 and 1.0): the
  generator never overrides the gates — o04 still answers instead
  of abstaining, and eligibility answers are never generated.
- **Latency is the price:** 4.75 s mean per generated answer (vs
  ~70 ms total for the extractive pipeline) plus 3.82 s per judge
  call, on 1.5 shared CPUs. Fine for an opt-in path; not a default.

## Benchmarks — 8-configuration ablation

Command: `python -m benchmarks.run_benchmark`

Same golden set for every configuration, factual questions only
(23 in-corpus + 5 out-of-corpus; eligibility questions are measured
by the eval harness, not the benchmark). Latency after warm-up,
embedding model already loaded. Cost $0.00 everywhere (local CPU,
no API calls). Latency varies run to run by tens of milliseconds on
this VM; the order of magnitude is stable, the exact figures are one
measured run.

Re-run in Phase 2, Step 3 with the current code: the Step 1 numbers
were stale after Step 2 changed the `answer()` defaults (dense
sentence scoring on, answer verification on). Two things changed on
the re-run: abstention is now 1.0, not 0.8 (Step 2's verification
abstains on o04), and total latency is ~1–1.5 s, not ~70 ms (Step 2's
dense sentence scoring embeds the candidate sentences per question;
retrieval itself is unchanged). The new configuration,
`hybrid-cross-encoder`, is identical to `improved-hybrid` except
for the reranker, so any delta is the cross-encoder's.

| Configuration | Mode | Reranker | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.522 | 1.0 | 0.0 | 0.2 ms | 0.5 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.565 | 1.0 | 0.0 | 103.5 ms | 106.5 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 6.2 ms | 1070.7 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 1.0 | 1.0 | 0.826 | 1.0 | 1.0 | 57.6 ms | 961.3 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 77.4 ms | 1307.1 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.826 | 1.0 | 0.0 | 74.9 ms | 1342.9 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 123.7 ms | 1490.6 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 1.0 | 1.0 | 0.87 | 1.0 | 1.0 | 4238.0 ms | 5429.1 ms | $0.00 |

### What the ablation shows

- **The improvement is the answering discipline, not the retrieval
  signal.** Baseline → improved-hybrid: answer-only 0.522 → 0.87,
  abstention 0.0 → 1.0. Hybrid-plain (hybrid retrieval, naive
  answering) scores only 0.565 — the dense signal alone adds
  almost nothing.
- **Dense buys no recall here.** recall@5 is 1.0 for every
  configuration. On a 31-chunk corpus the right document is always
  in the top 5 by keyword alone.
- **Dense slightly hurts answer-only relevance** (dense-only 0.826
  vs BM25-only 0.87) at ~10x the retrieval latency of BM25. On this
  corpus BM25-only is the better engineering choice.
- **The reranker adds nothing measurable** (0.87 with and without
  it). Kept because the mechanism is sound; must be re-measured on
  a larger corpus.
- **The cross-encoder adds nothing either** (Phase 2, Step 3).
  `hybrid-cross-encoder` vs `improved-hybrid`: answer-only 0.87 →
  0.87, recall@5 1.0 → 1.0, abstention 1.0 → 1.0, at 4,238.0 ms vs
  123.7 ms retrieval mean (~34x). It does reorder — 17 of 23
  in-corpus top-5 lists change order — but the top chunk and every
  picked answer sentence stay the same, so no metric moves. On a
  synthetic lexical trap the cross-encoder flips the order where
  the lexical reranker does not (logits 7.13 vs 5.97), so the
  mechanism works; the corpus is too small and too clean for it to
  matter. Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, 22.7M
  params, model.safetensors 90,870,598 bytes, CPU-only, $0. The
  lexical reranker stays the default.
- **The gates are the entire abstention story** (hybrid-no-gates
  0.0 → improved-hybrid 1.0). They also protect answer-only
  relevance: with the gates off, sentence picking is no longer
  restricted to the named scholarship's chunks, and q16 loses its
  expected phrase (0.87 → 0.826).

Full per-configuration results: `benchmarks/results/<config>.json`.
Ablation summary: `benchmarks/results/summary.md`.
Before/after report with failure tables:
`benchmarks/results/before_after.md`.

## Before / after (baseline → full pipeline)

| Metric | Before | After | Change |
|---|---|---|---|
| recall@3 | 1.0 | 1.0 | +0.000 |
| recall@5 | 1.0 | 1.0 | +0.000 |
| Evidence relevance | 1.0 | 1.0 | +0.000 |
| Answer-only relevance | 0.522 | 0.87 | +0.348 |
| Faithfulness | 1.0 | 1.0 | +0.000 |
| Abstention accuracy | 0.0 | 1.0 | +1.000 |
| Retrieval latency (mean) | 0.2 ms | 123.7 ms | +123.5 ms |
| Total latency (mean) | 0.5 ms | 1490.6 ms | +1490.1 ms |
| Cost per query | $0.00 | $0.00 | $0.00 |

## Tests

Command: `python -m pytest tests/ -q`

**128 passed, 0 failed** (18 Nov 2026, 465 s — the shared CPU was
under contention from other jobs on the VM; the Step 1 suite took
70 s on a quieter VM). Phase 1's 87 plus 25 new generative tests
plus 16 new Step 2 tests (`tests/test_dense_verification.py`):
dense similarity ranking (duration / levels sentences above
boilerplate), verification unit tests with injected sims (ranking
with/without rank term, low-sim floor, no-sim degradation), o04
abstaining with reason `unanswered_ranking` (live + held-out
Chevening ranking question), other out-of-corpus reasons
unchanged, generative path abstaining on o04 without running the
model, generative fallback when kept sentences fail relevance, and
the full-eval no-regression test (the Step 1 correct set of 30 is
exactly preserved, abstention 1.0, q03/q10/q17 still misses).
Coverage by area:

| Test file | Tests | What it covers |
|---|---|---|
| `test_rag.py` | chunking, retrieval, answering | Chunk size/overlap, hybrid retrieval, extractive answers |
| `test_api.py` | API | `/health`, `/ask`, citations, abstention responses |
| `test_edge_cases.py` | edge cases | Empty documents, empty index, missing context, weak retrieval, duplicate chunks, single long sentence, abstention |
| `test_audit.py` | auditable retrieval | Text normalisation, abstention reasons, retrieval trace in `/ask` |
| `test_eligibility.py` | eligibility | 26 verdict cases, evidence grounding, API profile behaviour |
| `test_data_quality.py` | messy data | All known findings, no false positives, report files valid |
| `test_ablation.py` | retrieval modes | `bm25` / `dense` / `hybrid` modes, invalid mode, config coverage |
| `test_before_after.py` | report integrity | Generated report matches the measured results files |
| `test_generative.py` | generative + verification + judge | 25 tests, fake model only (see above) |
| `test_dense_verification.py` | dense scoring + answer verification | 16 tests, real embedder for scoring, fake model for the generative o04 test (see above) |

## Data quality

Command: `python -m scripts.check_data_quality`

**36 findings (8 resolved, 28 open)**, each citing sentence + chunk
+ source URL. Full report: `benchmarks/results/data_quality.json`,
`benchmarks/results/data_quality.md`. Curated documentation:
`DATA_QUALITY.md`.

Headline findings: Vanier is closed and replaced by a harmonized
doctoral programme; Eiffel €1,200 is the Master's allowance and
€2,000 the Doctorate stipend (not a contradiction); HKPFS HK$344,400
≈ US$44,150 is one 2026/27 stipend in two currencies; DAAD is a
147-option database, not one award; only SI states a deadline; only
Vanier/Eiffel/HKPFS state a stipend.

## Retrieval audit

Command: `python -m benchmarks.audit_retrieval`

38 questions audited, 52 chunks injected into answers. Per-question
traces (every candidate with dense / BM25 / combined / rerank scores
and term coverage, in reranked order):
`benchmarks/results/retrieval_audit.json` and
`benchmarks/results/retrieval_audit.md`.

## Failures (known, not hidden)

| Question | What happens | Why | Fix (Phase 2) |
|---|---|---|---|
| q03 (SI master's duration) | Generative answer now contains "one- or two-year master's studies" (answer-only True) | Paraphrase gap, fixed by the generative path in Phase 2, Step 1 | q10, q17 remain; dense sentence scoring (Step 2) was measured and does not fix them extractively (correct sentence ranks 4th by dense sim, but weights ≥ 3 trade q03 for q16) |
| q10 (Erasmus duration) | Generative answer says "one academic year" (expected "1 to 2 academic years"); judge scored it 1/1 | Paraphrase gap + weak 0.5B judge | As above; judge needs a larger model. Correct sentence ranks 8th by dense sim (0.438) |
| q17 (Manaaki levels) | Generative draft copied the SI field list into a Manaaki answer; verification dropped it and fell back to extractive | Cross-document contamination, caught by verification | Named-doc filter added (Step 1); answer still misses the expected phrase. Correct sentence ranks 14th by dense sim (0.527) |
| o04 (Vanier ranked university) | **Fixed in Step 2:** system now abstains with reason `unanswered_ranking`, in both the extractive and generative paths (abstention 1.0, 5/5) | Lexical gates + grounding verification check overlap/grounding, not whether the sentences answer the question; dense similarity alone cannot separate it (0.667, above several answerable questions) | Answer-verification step (dense floor + ranking-focus check); the check is ranking-only and looks for a ranking term, not a ranked entity |
| Eligibility vs Vanier closure | Verdict says "doctoral only", not "closed" | Eligibility rules do not read the data-quality closure flag | Eligibility module reads the closure flag |

## CI gate

Command: `python -m scripts.ci_gate`

Thresholds: recall@5 ≥ 0.90, answer-only relevance ≥ 0.60,
faithfulness ≥ 0.95, abstention ≥ 0.60, eligibility ≥ 0.80.
Measured (Phase 2, Step 2): 1.0, 0.909, 1.0, 1.0, 1.0 — **gate passes**.
Strict mode (`CI_GATE_STRICT=1`, all thresholds 1.0) correctly
fails on answer-only relevance (0.909), proving the gate can fail.
Workflow: `.github/workflows/eval.yml`
(build index → tests → gate on every push/PR).

## How to reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.build_index        # chunks + embeddings + FAISS index
python -m scripts.run_eval           # eval harness numbers (writes benchmarks/results/extractive_eval.json)
python -m scripts.run_generative_eval # generative + judge, same 38 questions (~5-11 min depending on CPU contention)
python -m benchmarks.run_benchmark   # 8-configuration ablation
python -m benchmarks.audit_retrieval # per-question retrieval traces
python -m benchmarks.before_after   # before/after report + failure tables
python -m scripts.check_data_quality # corpus mess report
python -m scripts.ci_gate           # quality gate
python -m pytest tests/ -q          # 128 tests
```

Note: the dense model (`all-MiniLM-L6-v2`, ~90 MB) downloads on
first use from Hugging Face and takes ~30–60 s to load in this VM;
eval and benchmark runs take minutes, mostly model loading. The
`.tmp/` directory in the repo holds pip's temporary files (~1.4 GB,
gitignored) from the original install — it is not part of the
project and can be deleted.
