# Eval Results — Phase 2, Step 7

All numbers below come from real local runs on 18 Nov 2026, on this
VM (Python 3.12.3, 1.5 shared CPUs, 6 GB RAM). Nothing here is
invented. Reproduce them with the commands in "How to reproduce".

## Phase 2, Step 7 — re-verify stale pages, fix the checker currency gap

Commands: `python -m scripts.check_data_quality` (writes
`benchmarks/results/data_quality.json` + `.md`),
`python -m scripts.build_index` (rebuilt index),
`python -m scripts.run_eval` (writes
`benchmarks/results/extractive_eval.json`) and
`python -m scripts.ci_gate`. Before-run first: the pre-step eval
saved as `benchmarks/results/extractive_eval_before_reverification.json`
and the pre-step data-quality report saved as
`benchmarks/results/data_quality_before_reverification.json`, so
the before/after below is two generated files, not hand-typed
numbers. Per-page record:
`benchmarks/results/reverification_2026-11-18.json`; per-source
date: `reverified_at` in `data/sources.json`.

### Re-verification: all 19 pages live, 18 unchanged, 1 sidebar teaser

Every source URL was re-fetched 18 Nov 2026 — direct HTTP with
the repo's own extractor for all 19, plus a browser text fetch
for the 11 pages originally saved that way and for RTP, whose
direct fetch timed out. Seven pages are byte-identical to the
stored text (Sweden SI, Fulbright, Erasmus Mundus student page,
Vanier, Manaaki, HKPFS, MEXT). The other twelve differ only by
extraction method (line-wrapping, punctuation spacing, nav
duplication, table flattening, a duplicated FAQ section) or, for
Eiffel, in one sidebar recommended-article teaser — **no
scholarship fact changed on any page**, so no `data/raw/*.txt`
or `*.json` was rewritten. The one real change: Eiffel's
"Recommended items" teaser "MOPGA Research Themes" →
"Co-financing of 12- to 24-month postdoctoral research contracts";
its €1,200 / €2,000 amounts and October 2022 / December 2025
markers are unchanged. The stale markers stand as recorded:
Erasmus Mundus "Last updated: 26 June 2025", Eiffel "Mise à jour.
October 2022". One marker is newly observed, not a verified
change: the Erasmus Mundus organisations page footer, visible
only in the direct-HTTP extraction, reads "Last updated:
15 April 2026" — the 18 Nov fetch also lacked that footer, so it
may have been there all along.

| # | Scholarship | Status | Changed? | Re-verified |
|---|---|---|---|---|
| 1 | Chevening (UK) | live | Unchanged — no deadline, no stipend figure | 2026-11-18 |
| 2 | Sweden SI Global Professionals | live | Unchanged — byte-identical | 2026-11-18 |
| 3 | Fulbright Foreign Student Program (USA) | live | Unchanged — byte-identical | 2026-11-18 |
| 4 | DAAD (Germany) | live | Unchanged — 147 options | 2026-11-18 |
| 5 | Erasmus Mundus Joint Masters | live | Unchanged — byte-identical; marker 26 June 2025 | 2026-11-18 |
| 6 | France Excellence Eiffel | live | Sidebar teaser only (see above) | 2026-11-18 |
| 7 | Vanier Canada Graduate Scholarships | live | Unchanged — byte-identical; still closed | 2026-11-18 |
| 8 | Manaaki New Zealand Scholarships | live | Unchanged — byte-identical | 2026-11-18 |
| 9 | Hong Kong PhD Fellowship Scheme (HKPFS) | live | Unchanged — byte-identical | 2026-11-18 |
| 10 | MEXT (Japan) | live | Unchanged — byte-identical | 2026-11-18 |
| 11 | Gates Cambridge | live | Unchanged — around 70/year, £23,152 | 2026-11-18 |
| 12 | Rhodes Scholarship (FAQs) | live | Unchanged — £21,060 for 2026-27 | 2026-11-18 |
| 13 | Knight-Hennessy Scholars — Funding (Stanford) | live | Unchanged — up to three years | 2026-11-18 |
| 14 | Schwarzman Scholars (FAQs) | live | Unchanged — not yet 29 on 1 Aug | 2026-11-18 |
| 15 | Commonwealth Shared Scholarships | live | Unchanged — closing 8 December 2026, £1,712/month | 2026-11-18 |
| 16 | Research Training Program (RTP, Australia) | live | Unchanged — 2027 base $35,550, 2026 base $34,315 | 2026-11-18 |
| 17 | Marshall Scholarship | live | Unchanged — up to 50/year, closing 29 September 2026 | 2026-11-18 |
| 18 | Erasmus Mundus Joint Masters (organisations page) | live | Unchanged — €1,400/month; footer marker 15 April 2026 newly observed | 2026-11-18 |
| 19 | SI Scholarship Pioneering Women in STEM | live | Unchanged — SEK 12,000/month; 8–24 Feb 2027 | 2026-11-18 |

### No golden question changed — and the eval proves it

No expected answer changed, because no page fact changed (every
golden expected answer was checked against its re-fetched page).
No question was dropped. The rebuilt index is byte-identical to
the pre-step one (101 chunks, same per-document counts), and the
re-run eval file is **byte-identical** to the before-run file —
not just the same headline numbers, the same per-question rows.

| Metric | Before (Step 6, 19 docs, 101 chunks, 50 q) | After (Step 7) | Change |
|---|---|---|---|
| n_in_corpus / n_out_of_corpus | 43 / 7 | 43 / 7 | none |
| n_eligibility | 11 | 11 | none |
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 | none |
| answer_relevance_proxy | 1.0 | 1.0 | none |
| answer_only_relevance | 0.907 (39/43) | 0.907 (39/43) | none — same 4 misses (q03, q10, q17, q32) |
| faithfulness | 1.0 | 1.0 | none |
| abstention_accuracy | 1.0 (7/7) | 1.0 (7/7) | none |
| eligibility_accuracy | 1.0 (11/11) | 1.0 (11/11) | none |

### Checker currency gap: fixed, with the count to prove it

`MONEY_RE` in `scripts/check_data_quality.py` matched €/$ only.
It now matches £, SEK and ¥ as well, plus detached "($) 35,550"
table rates, and a value group that ends on a digit ("£4,000,"
in running text records as £4,000). Measured by re-running the
checker on the same corpus:

| Checker output | Before (Step 6) | After (Step 7) | Change |
|---|---|---|---|
| Findings (resolved / open) | 72 (9 / 63) | 73 (16 / 57) | +1 total, +7 resolved, −6 open |
| `missing_field` findings | 52 | 46 | −6 false "missing stipend" flags |
| Pages flagged "stipend missing" | 14 of 19 | **8 of 19** | Rhodes, Commonwealth Shared, SI Pioneering, Sweden SI, MEXT, RTP no longer flagged |
| Pages flagged "deadline missing" | 15 of 19 | 15 of 19 | none — no page gained a deadline |
| `multiple_amounts` findings (resolved) | 3 | 10 | +7, each resolved with its field named |
| Documents with any missing field | 19 | 18 | Sweden SI now has none |

The six pages the checker used to mis-flag: Rhodes (£21,060),
Commonwealth Shared (£1,712/month), SI Pioneering Women
(SEK 12,000/month), Sweden SI (SEK 12,000/month + SEK 15,000
travel grant), MEXT (¥117,000 / ¥143,000–145,000 / ¥242,000 per
month, per category) and RTP ($35,550 base for 2027). The seven
new `multiple_amounts` resolutions name the field for each amount:
Gates (£23,152 maintenance vs £750–£4,000 development funding vs
£12,793/£18,244 child allowance vs the US$210m donation),
Rhodes (£21,060/year = £1,755/month; £6,950 is the 33% secondary-
award declaration threshold, not a payment), Commonwealth
(£1,712/£2,000 stipend vs £622/£154 child allowances), RTP (base
vs maximum per year), Sweden SI and SI Pioneering (SEK 12,000
monthly allowance vs SEK 15,000 one-time travel grant), MEXT
(three per-category amounts whose category headings are lost in
extraction — no single quotable MEXT stipend, so o02 still
abstains). The HKPFS resolution was corrected in the same step:
the page carries a travel allowance (HK$14,400, ~US$1,840) as
well as the stipend, not only one figure in two currencies.

**Remaining gaps (named, not hidden):** the pattern still only
proves a money amount *appears* — Gates's "US$210m" donation
still matches, documented in its resolution rather than
distinguished by the pattern. MEXT's ¥ amounts cannot be tied to
a category from the stored text. Amounts with no currency sign
or code at all (a bare number under a "$" table header, a "kr"
amount with no SEK prefix) would still be missed; none is in
this corpus.

### What Step 7 changed (read honestly)

- **The re-verification found almost nothing — and that is the
  result.** A launch claim needs the pages checked, not assumed
  fresh. Seven pages are byte-identical after a day, eleven more
  are unchanged once extraction noise is discounted, and the
  twelfth differs in a sidebar teaser. The eval did not move by a
  single row, which is what a no-change re-fetch should produce —
  and the byte-identical eval files are the evidence, not the
  claim.
- **The checker fix is the real change, and it changed the
  documentation, not the answers.** The corpus always stated
  those six stipends; the checker could not see them. No golden
  answer changed, because the answers were already right — the
  system quoted £21,060 and SEK 12,000 from the text while the
  checker claimed the pages stated no stipend. The fix aligns the
  checker with the corpus; the "14 of 19" count in the Step 6
  documentation was the checker's count, not the truth, and it is
  corrected to 8 of 19 wherever it appeared.
- **Generative eval not re-run in this step.**
  `benchmarks/results/generative_eval.json` still holds the Step 2
  run on the 38-question set. Step 7 is extractive-path and
  checker-only; no `.txt` changed, so no generative answer could
  change either.

## Phase 2, Step 6 — corpus growth: 10 → 19 documents, 31 → 101 chunks

Commands: `python -m scripts.run_eval` (writes
`benchmarks/results/extractive_eval.json`) and
`python -m benchmarks.run_benchmark` (writes the 8 configuration
files + `benchmarks/results/summary.md`). Before-run first: the
pre-step eval saved as
`benchmarks/results/extractive_eval_before_corpus_growth.json` and
all pre-step results copied to
`benchmarks/results/before_corpus_growth/`, so the before/after
below is two generated files, not hand-typed numbers.

### New sources (all fetched 18 Nov 2026, text fetcher, no raw HTML)

| # | Scholarship | Source URL | Chunks (512 tokens) | Golden question |
|---|---|---|---|---|
| 11 | Gates Cambridge | https://www.gatescambridge.org/programme/the-scholarship/ | 2 | q24 (~70 full-cost scholarships/year) |
| 12 | Rhodes Scholarship (FAQs) | https://www.rhodeshouse.ox.ac.uk/scholarships/faqs/ | 29 | q25 (£21,060 in 2026-27) |
| 13 | Knight-Hennessy Scholars — Funding (Stanford) | https://knight-hennessy.stanford.edu/program-overview/funding | 2 | q26 (up to three years) |
| 14 | Schwarzman Scholars (FAQs) | https://www.schwarzmanscholars.org/frequently-asked-questions/ | 12 | q27 (not yet 29 on 1 Aug) |
| 15 | Commonwealth Shared Scholarships | https://cscuk.fcdo.gov.uk/scholarships/commonwealth-shared-scholarships-applications/ | 12 | q28 (closing date 8 December 2026) |
| 16 | Research Training Program (RTP, Australia) | https://www.education.gov.au/research-block-grants/research-training-program | 4 | q29 (base $34,315 in 2026) |
| 17 | Marshall Scholarship | https://marshallscholarship.org/about/ | 2 | q30 (up to 50/year) |
| 18 | Erasmus Mundus Joint Masters (organisations page) | https://erasmus-plus.ec.europa.eu/opportunities/organisations/cooperation-among-organisations-and-institutions/erasmus-mundus-joint-masters | 1 | q31 (€1,400/month) |
| 19 | SI Scholarship Pioneering Women in STEM | https://si.se/en/apply/scholarships/pioneeringwomeninstem/ | 6 | q32 (SEK 12,000/month) |

Per-source fetch metadata (HTTP status, fetch method, fetch date,
notes) is in `data/sources.json` and `data/raw/*.json`. Two new
out-of-corpus questions were added with the growth: o06 (Australia
Awards deadline — the scholarship is not in the corpus) and o07
(KHS acceptance rate — the fact is not on the fetched page).

| Metric | Before (10 docs, 31 chunks, 39 q) | After (19 docs, 101 chunks, 50 q) | Change |
|---|---|---|---|
| n_in_corpus / n_out_of_corpus | 34 / 5 | 43 / 7 | +9 factual (q24–q32), +2 out-of-corpus (o06, o07) |
| n_eligibility | 11 | 11 | none — no new rules authored |
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 | none |
| answer_relevance_proxy | 1.0 | 1.0 | none |
| answer_only_relevance | 0.912 (31/34) | 0.907 (39/43) | +8 correct (q24–q31), +1 miss (q32) |
| faithfulness | 1.0 | 1.0 | none |
| abstention_accuracy | 1.0 (5/5) | **1.0 (7/7)** | dipped to 0.714 (5/7) mid-step, fixed in-step |
| eligibility_accuracy | 1.0 (11/11) | 1.0 (11/11) | none |

### What Step 6 changed (read honestly)

- **The abstention dip was real, and it was fixed in the same
  step.** Growing the corpus gave the lexical gates other
  scholarships' sentences to quote: o06 ("deadline for the
  Australia Awards Scholarship 2027") was answered with SI +
  Commonwealth dates — the doc-name gate only fires for known
  documents, and the picked sentences share "deadline" and "2027"
  with the question — and o07 ("acceptance rate for the
  Knight-Hennessy Scholars programme") was answered with KHS
  funding sentences — the specific-term gate passed on the word
  "scholar" alone, which every KHS sentence contains. Both
  questions are genuinely unanswerable from the corpus, verified
  by grep, not by assumption: "Australia Awards" appears nowhere
  in it, and "acceptance" appears nowhere in it either. The fix
  is two small, existing-mechanism changes in `app/answer.py`:
  "scholar" joined the generic terms (o07's only specific terms
  are now "acceptance" and "rate", neither in any picked sentence
  — reason `missing_specific_term`), and the answer verification
  gained a scholarship-subject focus check (o06 — a question
  naming a scholarship no known document covers must be answered
  by a sentence that mentions its name — reason
  `unanswered_scholarship_subject`). Abstention is 1.0 (7/7)
  again; the 0.714 dip is recorded in
  `tests/test_corpus_growth.py` and in the Step 6 test comments,
  not hidden. Neither question was reclassified or deleted.
- **q32 is the one new miss, and it is a picker bug, not a
  retrieval failure.** The "SEK 12,000" sentence is in the top
  retrieved chunk (`si-pioneering-women-1`), but the sentence
  splitter merges the whole dash-led bullet list into one long
  sentence, so it shares fewer query stems than the page intro
  and is never picked. Documented in the failure table and pinned
  by a test; the fix (splitting list items into pickable
  sentences) is not in this step.
- **The baseline got luckier, the improved pipeline did not get
  worse.** The before/after failure table labels q03 and q32
  REGRESSION (baseline ✓, improved ✗). The improved column is
  unchanged since Step 5 — it misses exactly the same questions
  (q03, q10, q17, plus new q32). What changed is the baseline
  column: with 101 chunks the naive baseline's BM25 top chunk
  changed, so its first-sentence picking now lands on the right
  sentence for q03 and q32. The no-regression test pins exactly
  those two rows, so a third one fails the build.
- **The Phase 1 findings were re-tested on the grown corpus and
  did not change.** Dense still buys nothing (dense-only 0.844 vs
  BM25-only 0.875, recall@5 1.0 for every signal), the lexical
  reranker still changes nothing (0.875 with and without it), the
  cross-encoder still changes nothing (0.875, abstention 1.0, at
  3,987.1 ms vs 185.9 ms retrieval mean), recall stays 1.0, and
  q03/q10/q17 still miss. See the benchmark table below.
- **Two Step 5 tests were stale on the corpus size, not on the
  code.** `tests/test_chunk_ablation.py` asserted 60 chunks at 256
  tokens — the 10-document count. On the grown corpus 256 tokens
  gives 212 chunks (1,024 tokens gives 53; 512 gives 101). The
  tests now assert 212, measured by building the 256-token index.
  The Step 5 chunk-ablation *results* (256/512/1,024 on the
  39-question set) are historical measurements on the old corpus
  and were not re-run in this step; they stay labelled as such.
- **The data-quality checker has a currency gap, found by the new
  documents.** Its money pattern matches €/$ only, so it flags
  "stipend missing" for Rhodes (£21,060), Commonwealth Shared
  (£1,712/month) and SI Pioneering (SEK 12,000/month) although the
  pages state amounts — and fails to flag Gates, whose "US$210m"
  founding donation matches while its real stipend (£23,152) is in
  pounds. It also flags RTP missing although the page states the
  rates as "($) 35,550" (sign detached from digits). Pinned by
  `tests/test_corpus_growth.py::test_stipend_flag_is_a_currency_gap`
  (renamed in Step 7 to `test_stipend_flag_currency_gap_is_fixed`
  when the gap was fixed).
  Findings went 36 → 72 (9 resolved, 63 open), including the
  resolved EMJM two-amounts finding (€5 million per project vs
  €1,400/month per individual — different fields).
- **No eligibility rules for the 9 new documents.** Profile checks
  for them return `unknown` by construction; authoring rules is a
  separate step. The checker flags degree-levels/citizenship as
  missing for all nine in the meantime.
- **Generative eval not re-run in this step.**
  `benchmarks/results/generative_eval.json` still holds the Step 2
  run on the 38-question set. Step 6 is extractive-path only; the
  generative numbers below are labelled with the set they were
  measured on.

## Phase 2, Step 5 — chunk-size ablation: 256 / 512 / 1,024 tokens

Command: `python -m scripts.run_chunk_ablation` (437.9 s; writes
`benchmarks/results/chunk_ablation/chunk_ablation_<size>.json` per
size plus `benchmarks/results/chunk_ablation/chunk_ablation.md`,
generated from the written files). Same .txt corpus for every size,
same overlap policy — 200 chars, held constant, so chunk size is
the only variable — same 39-question golden set, full extractive
pipeline (hybrid retrieval, lexical rerank, scored picking,
abstention gates, eligibility-first). Each size's index and chunks
are stored under `benchmarks/results/chunk_ablation/chunk-<size>/`
(`chunks.jsonl` + `faiss.index`); the working
`data/processed/` 512-token index was never touched, and the
rebuilt 512-token chunks file is byte-identical to it
(`diff` clean), so the 512 column below is a re-run of the Step 4
baseline, not a new index.

| Chunk size | Chunks | Mean / max chunk chars | recall@3 / recall@5 | Answer relevance (proxy) | Answer-only relevance | Faithfulness | Abstention | Eligibility | Retrieval mean | Total mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 tokens (1,024 chars) | 60 | 1,023.2 / 1,224 | 1.0 / 1.0 | 0.912 | 0.853 (29/34) | 1.0 | 1.0 (5/5) | 0.818 (9/11) | 93.0 ms | 1,011.5 ms |
| **512 tokens (2,048 chars)** | **31** | **1,797.8 / 2,248** | **1.0 / 1.0** | **1.0** | **0.912 (31/34)** | **1.0** | **1.0 (5/5)** | **1.0 (11/11)** | **97.6 ms** | **1,264.0 ms** |
| 1,024 tokens (4,096 chars) | 19 | 2,808.6 / 4,283 | 1.0 / 1.0 | 0.971 | 0.853 (29/34) | 1.0 | 1.0 (5/5) | 0.909 (10/11) | 256.3 ms | 2,463.6 ms |

### Decision: keep 512 tokens — chosen by the numbers

- **512 wins the headline and the safety metric.** Answer-only
  relevance 0.912 (31/34) vs 0.853 (29/34) at both other sizes, and
  eligibility 1.0 (11/11) vs 0.818 (9/11) at 256 and 0.909 (10/11)
  at 1,024. No default change, no index rebuild, no rule change.
- **The losses at other sizes are systematic, not noise.** The
  eligibility rules pin chunk ids (e.g. the SI citizenship list
  must be in `sweden-si-2`); a size that moves an evidence sentence
  to another chunk drops that rule, with a warning, and the verdict
  degrades. Rules dropped: **7 at 256** (erasmus-mundus levels +
  citizenship, manaaki levels + citizenship, mext levels,
  sweden-si citizenship, vanier citizenship) and **2 at 1,024**
  (erasmus-mundus levels — `erasmus-mundus-1` does not exist at
  1,024 tokens, the whole document is one chunk — and sweden-si
  citizenship); **0 at 512**.
- **The worst failure is a false "eligible".** At both 256 and
  1,024, e03 (Pakistani master's applicant, SI Scholarship) returns
  `eligible` instead of `not_eligible`: the citizenship rule that
  says Pakistan is not among the 34 eligible countries is gone, so
  the system tells an ineligible applicant they can apply. At 256,
  e07 (Manaaki undergraduate) additionally drops to `unknown`
  (both manaaki rules gone), and e04's verdict stays `eligible` but
  its answer loses the "Bangladesh is among the 34 eligible
  countries" phrase at both non-default sizes. A wrong `eligible`
  is the failure this system is built not to produce — it outweighs
  any latency gain.
- **256 does win one thing: q03.** Factual questions only (23,
  eligibility excluded), answer-only is 21/23 at 256 vs 20/23 at
  512 and 20/23 at 1,024 — the 256-token chunks isolate the SI
  duration sentence so the picker finds it, fixing the q03
  paraphrase-gap miss. It is a real +1, and it is outweighed by the
  three eligibility rows 256 loses (e03, e04, e07). q10 and q17
  miss at every size.
- **Recall, faithfulness and abstention do not separate the sizes.**
  recall@3/@5 1.0, faithfulness 1.0, abstention 1.0 (5/5) at all
  three sizes — on this 31-chunk corpus the right document is
  always in the top 5, whatever the chunk size.
- **Latency does not rescue 256 or condemn 1,024.** Retrieval mean
  93.0 / 97.6 / 256.3 ms and total mean 1,011.5 / 1,264.0 / 2,463.6 ms
  (256 / 512 / 1,024). But total p50 is 879.0 / 1,120.7 / 1,190.4 ms
  — nearly flat: the 1,024-token mean is a tail effect (p95
  5,900.8 ms, max 10,920.9 ms), which may be CPU contention as much
  as chunk size. 256's ~250 ms total-mean edge over 512 is not
  worth a false `eligible` verdict.
- **Honest confound: the rules were authored at 512.** The
  eligibility evidence chunk ids were written against the 512-token
  chunking, so 512's eligibility advantage is partly home-field
  advantage, not a pure chunk-size effect. The ablation measures
  the system as built — and as built, changing size would also
  require re-authoring every pinned chunk id, for no net gain. A
  future chunk-size change must re-pin the rules first and re-run
  this ablation; the script (`scripts/run_chunk_ablation.py`) and
  the per-size indexes exist for exactly that.
- **What changed in code to make this measurable (not a metric
  change):** `chunk_document()` now takes `chunk_chars` /
  `overlap_chars` parameters; `scripts/build_index.py` is a function
  `build_index(chunk_tokens, out_dir)` with `--chunk-tokens`
  (256/512/1,024) and `--out-dir` flags (defaults reproduce the old
  behaviour exactly); `app/eval.py:run_eval` now resolves
  eligibility against the retriever's own chunks instead of always
  reading `data/processed/chunks.jsonl` — without that fix, an eval
  "at 256 tokens" would silently have used the 512-token chunks for
  every eligibility verdict.

## Phase 2, Step 4 — eligibility reads the closure flag

Command: `python -m scripts.run_eval` (writes
`benchmarks/results/extractive_eval.json`). Baseline re-run first
on the same command with the pre-step code and the 38-question
golden set, saved as
`benchmarks/results/extractive_eval_before_closure.json`, so the
before/after below is two generated files, not hand-typed numbers.

| Metric | Before (38 q, Step 3 code) | After (39 q) | Change |
|---|---|---|---|
| n_in_corpus / n_out_of_corpus | 33 / 5 | 34 / 5 | +1 eligibility question (e11) |
| n_eligibility | 10 | 11 | +1 |
| recall@3 / recall@5 | 1.0 / 1.0 | 1.0 / 1.0 | none |
| answer_relevance_proxy | 1.0 | 1.0 | none |
| answer_only_relevance | 0.909 (30/33) | 0.912 (31/34) | +1 question, still passing |
| faithfulness | 1.0 | 1.0 | none |
| abstention_accuracy | 1.0 (5/5) | 1.0 (5/5) | none |
| eligibility_accuracy | 1.0 (10/10) | **1.0 (11/11)** | e02, e05 now `closed`; e11 `closed` |

### What Step 4 changed (read honestly)

- **Exactly two verdicts changed, one question added, zero lost.**
  Per-question diff of the two generated eval files: e02
  (`not_eligible` → `closed`, master's applicant) and e05
  (`eligible` → `closed`, Canadian doctoral applicant — the known
  gap: a qualified doctoral applicant was told "doctoral only" /
  eligible for a programme that no longer accepts applications).
  New golden e11 (Pakistani doctoral applicant) returns `closed`.
  Every other row is byte-identical in verdict and metric;
  answer-only misses are still exactly q03, q10, q17. No question
  the baseline got right was lost.
- **Why a distinct `closed` verdict, not `not_eligible`.** Closure
  is a property of the programme, not of the applicant: a doctoral
  applicant who meets every Vanier rule must not be told they "do
  not qualify". The verdict set (`eligible` / `not_eligible` /
  `unknown` / `closed`) supports it cleanly — the eval compares
  verdict strings, `resolve()` replaces the substantive answer for
  `closed` exactly as for `not_eligible`, and the statement leads
  with "Closed: …" plus the replacement programme.
- **The mechanism generalises.** `data/closures.json` is the flag
  file: one entry per closed programme (doc, status, replacement,
  verbatim evidence sentences + chunk ids), driven by the
  data-quality checker's closure findings — a test asserts every
  closure evidence sentence appears in the checker's recorded
  resolution evidence, so the two cannot drift apart. The
  eligibility module validates each sentence verbatim against the
  chunk at load time, exactly like the rules; an entry whose
  evidence is missing is dropped with a warning and the programme
  falls back to its rules (covered by a test that mutates the
  chunk text). Adding a future closure is one JSON entry plus the
  matching checker resolution — no code change.
- **e02/e05 expected answers changed, on purpose.** Their old
  expected phrases ("doctoral studies", "Canadian citizens") were
  the old rules' answers; the correct answer for a closed
  programme is the closure, so both now expect "no longer
  accepting applications", which the closure statement contains.
  Both still pass answer-only, as does e11.
- **Generative eval not re-run in this step.** Eligibility answers
  are template + verbatim evidence and bypass the generator
  (`generate_answer` is never called for them), and the closure
  verdict flows through the same `resolve()` path in both modes —
  but `benchmarks/results/generative_eval.json` still holds the
  Step 2 run on the 38-question set, so its eligibility column
  (1.0, 10/10) is stale by one question, not wrong in mechanism.
  Re-running it is a ~10-minute model run; it was not needed to
  measure this fix, which is extractive-path only.

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

`eval/golden.json` — 50 hand-written questions, every expected answer
verified against the actual fetched text:

| Group | Count | What it tests |
|---|---|---|
| In-corpus factual (q01–q32) | 32 | Answer the question from the cited document |
| Out-of-corpus (o01–o07) | 7 | Abstain when the corpus cannot answer |
| Eligibility (e01–e11) | 11 | Degree-level / citizenship verdicts with a profile, including closed-programme verdicts (e02, e05, e11) |
| **Total** | **50** | |

## Eval harness — full Phase 1 pipeline

Command: `python -m scripts.run_eval`

| Metric | Value | Meaning |
|---|---|---|
| n_in_corpus | 43 | 32 factual + 11 eligibility |
| n_out_of_corpus | 7 | |
| n_eligibility | 11 | |
| recall@3 | 1.0 | Expected document in top 3 for all 43 |
| recall@5 | 1.0 | Expected document in top 5 for all 43 |
| answer_relevance_proxy | 1.0 | Expected phrase in answer or citations for all 43 |
| answer_only_relevance | 0.907 | Expected phrase in the answer text alone: 39/43 |
| faithfulness | 1.0 | Every answer sentence verbatim in a cited chunk |
| abstention_accuracy | 1.0 | 7/7 out-of-corpus questions correctly refused |
| eligibility_accuracy | 1.0 | 11/11 verdicts correct |

The 4 answer-only misses are q03, q10, q17 (paraphrase gap) and
q32 (bullet-list merging) — see "Failures". No abstention misses:
o06 and o07 abstain after the Step 6 fix (they were answered
mid-step, when abstention dipped to 0.714).

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
(32 in-corpus + 7 out-of-corpus; eligibility questions are measured
by the eval harness, not the benchmark). Latency after warm-up,
embedding model already loaded. Cost $0.00 everywhere (local CPU,
no API calls). Latency varies run to run by tens of milliseconds on
this VM; the order of magnitude is stable, the exact figures are one
measured run.

Re-run in Phase 2, Step 6 on the grown corpus (101 chunks, 50
questions): the Step 3 numbers were measured on the 31-chunk corpus
and the 39-question set. Two things changed on the re-run:
abstention is now 1.0 for every gated configuration (Step 6's
verification fix restores o06/o07), and total latency is higher
across the board (~1.2–2.6 s vs ~1.0–1.7 s) — the dense sentence
scoring embeds the candidate sentences per question, and this run
shared the VM with other work. The new configuration,
`hybrid-cross-encoder`, is identical to `improved-hybrid` except
for the reranker, so any delta is the cross-encoder's.

| Configuration | Mode | Reranker | recall@3 | recall@5 | Evidence relevance | Answer-only relevance | Faithfulness | Abstention | Retrieval mean | Total mean | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-naive | bm25 | lexical | 1.0 | 1.0 | 1.0 | 0.625 | 1.0 | 0.0 | 0.3 ms | 0.7 ms | $0.00 |
| hybrid-plain | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.656 | 1.0 | 0.0 | 84.1 ms | 84.6 ms | $0.00 |
| improved-bm25 | bm25 | lexical | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 2.0 ms | 1179.2 ms | $0.00 |
| dense-only | dense | lexical | 1.0 | 1.0 | 1.0 | 0.844 | 1.0 | 1.0 | 77.7 ms | 1443.7 ms | $0.00 |
| hybrid-no-rerank | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 60.2 ms | 1407.1 ms | $0.00 |
| hybrid-no-gates | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.844 | 1.0 | 0.0 | 145.3 ms | 2038.0 ms | $0.00 |
| improved-hybrid | hybrid | lexical | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 185.9 ms | 2566.1 ms | $0.00 |
| hybrid-cross-encoder | hybrid | cross-encoder | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 1.0 | 3987.1 ms | 5374.9 ms | $0.00 |

### What the ablation shows

- **The improvement is the answering discipline, not the retrieval
  signal.** Baseline → improved-hybrid: answer-only 0.625 → 0.875,
  abstention 0.0 → 1.0. Hybrid-plain (hybrid retrieval, naive
  answering) scores only 0.656 — the dense signal alone adds
  almost nothing.
- **Dense buys no recall here.** recall@5 is 1.0 for every
  configuration. On a 101-chunk corpus the right document is always
  in the top 5 by keyword alone.
- **Dense slightly hurts answer-only relevance** (dense-only 0.844
  vs BM25-only 0.875) at ~39x the retrieval latency of BM25. On this
  corpus BM25-only is the better engineering choice.
- **The reranker adds nothing measurable** (0.875 with and without
  it). Kept because the mechanism is sound; must be re-measured on
  a larger corpus.
- **The cross-encoder adds nothing either** (Phase 2, Step 3,
  re-measured Step 6).
  `hybrid-cross-encoder` vs `improved-hybrid`: answer-only 0.875 →
  0.875, recall@5 1.0 → 1.0, abstention 1.0 → 1.0, at 3,987.1 ms vs
  185.9 ms retrieval mean (~21x). It does reorder — at Step 3, 17 of
  23 in-corpus top-5 lists changed order — but the top chunk and every
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
  restricted to the named scholarship's chunks, and q30 (Marshall)
  loses its expected phrase (0.875 → 0.844).

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
| Answer-only relevance | 0.625 | 0.875 | +0.250 |
| Faithfulness | 1.0 | 1.0 | +0.000 |
| Abstention accuracy | 0.0 | 1.0 | +1.000 |
| Retrieval latency (mean) | 0.3 ms | 185.9 ms | +185.6 ms |
| Total latency (mean) | 0.7 ms | 2566.1 ms | +2565.4 ms |
| Cost per query | $0.00 | $0.00 | $0.00 |

## Tests

Command: `python -m pytest tests/ -q`

**179 passed, 0 failed** (18 Nov 2026, 958.09 s). Step 6's 174 plus
5 new Step 7 re-verification tests in `tests/test_reverification.py`:
every source records a `reverified_at` date and the per-page record
covers all 19 pages, all live, with Eiffel the only change (sidebar
teaser); the checker matches £/SEK/¥ and detached "($)" amounts and
no longer flags Rhodes / Commonwealth / SI Pioneering / Sweden SI /
MEXT / RTP as missing a stipend; the seven new two-amounts findings
are resolved with their field named (and the HKPFS resolution now
also names the travel allowance); the re-run eval is identical to
the saved before-run file question by question; and no secret
pattern appears in the stored text. Step 6's stipend-gap pin
(`test_stipend_flag_is_a_currency_gap`) was rewritten as
`test_stipend_flag_currency_gap_is_fixed`, and the data-quality
missing-fields matrix and report-count tests were updated to the
new measured counts (73 findings, 16 resolved). Step 5's 158 plus
9 new Step 6 corpus-growth tests in `tests/test_corpus_growth.py`:
19 documents, 101 chunks, per-document chunk counts, the 9 new
sources in `data/sources.json`, the golden set at 50 questions
(32 in-corpus / 7 out-of-corpus / 11 eligibility in the benchmark
split), both new out-of-corpus questions abstaining with their
reasons, and the stipend flag as a currency gap (Rhodes /
Commonwealth / SI Pioneering flagged, Gates not flagged); plus 6 new
dense-verification tests (generic "scholar", the named-scholarship
walk-back, the subject-focus check refusing o06 and passing when a
sentence names it, the check skipped for a known document, and o06 +
o07 abstaining live); plus 1 new data-quality test for the EMJM
two-amounts resolution. Step 4's 145 plus
13 new Step 5 chunk-ablation tests in `tests/test_chunk_ablation.py`:
chunks within budget and overlap preserved at 256/512/1,024 tokens,
index builder accepts a chunk-size parameter (256 → 60 chunks /
60 vectors, invalid size rejected), and the eval runs against a
non-default chunk size without crashing. Step 3's 136 plus
9 new Step 4 closure tests in `tests/test_eligibility.py`: closed
verdict for a doctoral applicant, for other degree levels, and with
nationality only; open programmes unaffected; verbatim-evidence
validation for every closure entry; entry dropped (with a warning)
when its evidence is missing from the chunk; closure table matches
the data-quality checker's resolution evidence; API returns the
closed verdict with citation for both an eligibility question and
a substantive Vanier question. The no-regression pin in
`tests/test_dense_verification.py` was updated for the new golden
question (pinned correct set 30 → 31, answer-only 0.909 → 0.912) —
no baseline-correct question was lost; the pin now requires e11 too.
Phase 1's 87 plus 25 new generative tests
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
| `test_eligibility.py` | eligibility | 44 tests: 26 verdict cases (4 Vanier now `closed`), 9 closure tests, evidence grounding, API profile behaviour |
| `test_data_quality.py` | messy data | All known findings, no false positives, report files valid |
| `test_ablation.py` | retrieval modes | `bm25` / `dense` / `hybrid` modes, invalid mode, config coverage |
| `test_chunk_ablation.py` | chunk-size ablation | 13 tests: chunks within budget + overlap preserved at 256/512/1,024 tokens, index builder accepts a chunk-size parameter (256 → 212 chunks / 212 vectors, invalid size rejected), eval runs against a non-default chunk size without crashing |
| `test_corpus_growth.py` | corpus growth | 9 tests: 19 documents / 101 chunks, per-document chunk counts, new sources in `data/sources.json`, golden set 50 questions, o06/o07 abstaining with reasons, stipend flag currency gap **fixed in Step 7** (was pinned as a limitation in Step 6) |
| `test_reverification.py` | re-verification (Step 7) | 5 tests: `reverified_at` per source + 19-page record, £/SEK/¥ and detached "($)" matching, seven new two-amounts resolutions, eval identical to the before-run file, no secret pattern in stored text |
| `test_before_after.py` | report integrity | Generated report matches the measured results files |
| `test_generative.py` | generative + verification + judge | 25 tests, fake model only (see above) |
| `test_dense_verification.py` | dense scoring + answer verification | 22 tests, real embedder for scoring, fake model for the generative o04 test (see above) |

## Data quality

Command: `python -m scripts.check_data_quality`

**73 findings (16 resolved, 57 open)** after the Phase 2, Step 7
re-verification and checker fix (was 72 / 9 / 63 in Step 6), each
citing sentence + chunk + source URL. Full report:
`benchmarks/results/data_quality.json`,
`benchmarks/results/data_quality.md`. Curated documentation:
`DATA_QUALITY.md`.

Headline findings: Vanier is closed and replaced by a harmonized
doctoral programme, and eligibility now returns verdict `closed`
for it (Phase 2, Step 4); Eiffel €1,200 is the Master's allowance and
€2,000 the Doctorate stipend (not a contradiction); the EMJM
organisations page's €5 million (max grant per project) and
€1,400/month (scholarship per individual) are different fields, not
a contradiction (resolved in Step 6); HKPFS HK$344,400 ≈ US$44,150
is one 2026/27 stipend in two currencies, plus a HK$14,400 travel
allowance (resolution corrected in Step 7); DAAD is a 147-option
database, not one award; 15 of 19 pages state no deadline and
**8 of 19 state no stipend by the checker's rules** (was 14 of 19
before the Step 7 fix — the checker now matches £, SEK, ¥ and
detached "($)" rates, so Rhodes £21,060, Commonwealth £1,712/month,
SI Pioneering SEK 12,000/month, Sweden SI SEK 12,000/month, MEXT
¥117,000–¥242,000/month and RTP $35,550 are no longer falsely
flagged; Gates is now matched on its real £23,152 stipend as well
as its "US$210m" donation); no eligibility rules are authored for
the 9 new documents yet. All 19 pages were re-fetched and
re-verified 18 Nov 2026 (Step 7): all live, 18 substantively
unchanged, Eiffel changed only in a sidebar teaser — full table in
`DATA_QUALITY.md` §11 and
`benchmarks/results/reverification_2026-11-18.json`.

## Retrieval audit

Command: `python -m benchmarks.audit_retrieval`

50 questions audited, 72 chunks injected into answers. Per-question
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
| o04 (Vanier ranked university) | **Fixed in Step 2:** system now abstains with reason `unanswered_ranking`, in both the extractive and generative paths (abstention 1.0, 7/7) | Lexical gates + grounding verification check overlap/grounding, not whether the sentences answer the question; dense similarity alone cannot separate it (0.667, above several answerable questions) | Answer-verification step (dense floor + ranking-focus check); the check is ranking-only and looks for a ranking term, not a ranked entity |
| q32 (SI Pioneering allowance) | Answer misses the "SEK 12,000" sentence; the evidence is in the top retrieved chunk (`si-pioneering-women-1`) | Bullet-list merging: the sentence splitter merges the whole dash-led bullet list into one long sentence, so it loses to the page intro on shared stems | Split list items into separate pickable sentences (not yet done) |
| o06 (Australia Awards deadline) | **Fixed in Step 6:** system now abstains with reason `unanswered_scholarship_subject` | Corpus growth gave the lexical gates other scholarships' deadline sentences to quote; the doc-name gate only fires for known documents, and "Australia Awards" is not in the corpus at all | Scholarship-subject focus check in the answer verification (Step 6); the check looks for name terms, not the scholarship as an entity |
| o07 (KHS acceptance rate) | **Fixed in Step 6:** system now abstains with reason `missing_specific_term` | The specific-term gate passed on the word "scholar" alone — every KHS sentence contains "scholars" — while the real ask ("acceptance", "rate") appears nowhere in the corpus | "Scholar" joined the generic terms (Step 6) |
| Eligibility vs Vanier closure | **Fixed in Step 4:** verdict is now `closed` with the closure evidence and replacement (e02, e05, e11; eligibility 1.0, 11/11) | Closure is a property of the programme, not the profile | `data/closures.json` + `app/eligibility.py:load_closures` / `check()` |

## CI gate

Command: `python -m scripts.ci_gate`

Thresholds: recall@5 ≥ 0.90, answer-only relevance ≥ 0.60,
faithfulness ≥ 0.95, abstention ≥ 0.60, eligibility ≥ 0.80.
Measured (Phase 2, Step 7, re-run after re-verification): 1.0, 0.907, 1.0, 1.0, 1.0 — **gate passes** (identical to Step 6: no `.txt` changed, so no metric could move).
Strict mode (`CI_GATE_STRICT=1`, all thresholds 1.0) correctly
fails on answer-only relevance (0.907), proving the gate can fail.
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
python -m scripts.run_chunk_ablation # chunk-size ablation, 256/512/1,024 tokens (~7 min)
python -m pytest tests/ -q          # 179 tests
```

Note: the dense model (`all-MiniLM-L6-v2`, ~90 MB) downloads on
first use from Hugging Face and takes ~30–60 s to load in this VM;
eval and benchmark runs take minutes, mostly model loading. The
`.tmp/` directory in the repo holds pip's temporary files (~1.4 GB,
gitignored) from the original install — it is not part of the
project and can be deleted.
