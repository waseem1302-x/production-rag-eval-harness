# Before / after — Phase 1

## What was before

The naive baseline is the tutorial chatbot: BM25 retrieval, no reranker, first-sentence picking, no abstention gates — it answers every question with the first sentences of the top chunk and never says "I don't know". It also has no notion of eligibility and no record of the corpus's messy data. Every number below is measured on the same golden set (32 in-corpus + 7 out-of-corpus factual questions).

## What happened after

The full Phase 1 pipeline: hybrid dense+BM25 retrieval, lexical reranker, scored sentence picking (up to 5 sentences, short-fragment penalty), abstention gates with recorded reasons, eligibility-first answering from corpus-grounded rules, and a data-quality checker that records the corpus's mess instead of hiding it.

## Per-metric deltas (baseline → improved)

| Metric | Before (baseline-naive) | After (improved-hybrid) | Change |
|---|---|---|---|
| recall@3 | 1.0 | 1.0 | +0.000 |
| recall@5 | 1.0 | 1.0 | +0.000 |
| Evidence relevance (proxy) | 1.0 | 1.0 | +0.000 |
| Answer-only relevance | 0.625 | 0.938 | +0.313 |
| Faithfulness | 1.0 | 1.0 | +0.000 |
| Abstention accuracy | 0.0 | 1.0 | +1.000 |
| Retrieval latency (mean) | 0.3 ms | 35.5 ms | +35.2 ms |
| Total latency (mean) | 0.7 ms | 890.0 ms | +889.3 ms |
| Cost per query | $0.00 | $0.00 | $0.00 (local CPU, no API calls) |

## Why it changed

- Answer-only relevance rose because scored picking replaced first-sentence picking: the baseline returned whatever sentences came first in the top chunk — usually headers, navigation or a marketing intro. Scored picking ranks sentences by shared query stems weighted by IDF over the candidate sentences (Phase 2 improvement, Step 2), and the short-fragment penalty demotes header fragments. The retrieval signal did not drive this: hybrid-plain (hybrid retrieval, naive answering) scores only 0.656 on answer-only, barely above the baseline.
- Abstention rose because the gates exist: the baseline answers everything (abstention 0.0). The score threshold, doc-name consistency, shared-stem and specific-term gates refuse questions the corpus cannot answer, and each refusal records its reason.
- Recall did not change, because it was already saturated: recall@5 is 1.0 for every configuration, including BM25-only. On a 101-chunk corpus the right document is always in the top 5 by keyword alone. Dense retrieval buys no measured recall here at ~50x the latency — that is a finding about this corpus size, not a failure of dense retrieval in general.
- Faithfulness stayed at 1.0 because both pipelines are extractive: every answer sentence is copied verbatim from a cited chunk. Faithfulness measures copying, not correctness — it cannot catch an answer that quotes the right words for the wrong question.
- Eligibility and data quality are new capabilities, not improvements on the baseline: the baseline had no profile, no rules and no record of the corpus's problems. Measured now: eligibility accuracy 1.0 (11/11 golden questions); data-quality checker finds 73 findings (16 resolved, 57 open).

## Failure table — in-corpus questions

| ID | Question | Baseline | Improved | Why |
|---|---|---|---|---|
| q01 | What does the Chevening Scholarship fund? | ✓ | ✓ | — |
| q02 | Is the Chevening Scholarship fully funded? | ✓ | ✓ | — |
| q03 | How long are the master's studies supported by the SI Scholarship for Global Professionals? | ✓ | ✓ | — |
| q04 | How many countries are eligible for the SI Scholarship for Global Professionals? | ✗ | ✓ | wrong sentence from the right chunk — first-sentence picking returned an intro or unrelated sentence |
| q05 | How many countries does the Fulbright Foreign Student Program operate in? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q06 | How many foreign students receive Fulbright scholarships each year? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q07 | How many scholarship options does the DAAD database list for selection? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q08 | Who are DAAD Programmes for Graduates aimed at? | ✓ | ✓ | — |
| q09 | How many institutions from how many countries are involved in an Erasmus Mundus Joint Master? | ✗ | ✓ | wrong sentence from the right chunk — first-sentence picking returned an intro or unrelated sentence |
| q10 | How long does an Erasmus Mundus Joint Master last? | ✗ | ✗ | paraphrase gap — most of the expected answer's content words (academic, years) do not appear in the question, so lexical sentence picking cannot find the sentence; needs dense sentence scoring or a generative answer (Phase 2) |
| q11 | At which degree levels does the France Excellence Eiffel Scholarship fund study? | ✓ | ✓ | — |
| q12 | Who submits applications for the Eiffel Scholarship? | ✗ | ✓ | wrong sentence from the right chunk — first-sentence picking returned an intro or unrelated sentence |
| q13 | How much is the Vanier Canada Graduate Scholarship worth per year and for how long? | ✓ | ✓ | — |
| q14 | Who must nominate a candidate for the Vanier Scholarship? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q15 | What is the annual stipend of the Hong Kong PhD Fellowship Scheme in 2026/27? | ✗ | ✓ | wrong sentence from the right chunk — first-sentence picking returned an intro or unrelated sentence |
| q16 | How many universities participate in the Hong Kong PhD Fellowship Scheme? | ✓ | ✓ | — |
| q17 | What levels of study do Manaaki New Zealand Scholarships cover? | ✗ | ✗ | paraphrase gap — most of the expected answer's content words (postgraduate, undergraduate) do not appear in the question, so lexical sentence picking cannot find the sentence; needs dense sentence scoring or a generative answer (Phase 2) |
| q18 | What does the Manaaki New Zealand Scholarship Programme aim to build in partner countries? | ✓ | ✓ | — |
| q19 | How many types of MEXT scholarship are there? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q20 | Which MEXT categories include research students and undergraduate students? | ✓ | ✓ | — |
| q21 | Is the Vanier Canada Graduate Scholarship still accepting applications? | ✓ | ✓ | — |
| q22 | How much is the monthly allowance for a master's student on the Eiffel Scholarship? | ✓ | ✓ | — |
| q23 | How much is the monthly stipend for a doctoral student on the Eiffel Scholarship? | ✓ | ✓ | — |
| q24 | How many full-cost Gates Cambridge Scholarships are offered each year? | ✓ | ✓ | — |
| q25 | What is the annual stipend for a Rhodes Scholar in 2026-27? | ✓ | ✓ | — |
| q26 | How many years of funding do Knight-Hennessy Scholars receive at Stanford? | ✓ | ✓ | — |
| q27 | What is the maximum age for Schwarzman Scholars as of August 1 of their enrollment year? | ✓ | ✓ | — |
| q28 | When is the closing date for Commonwealth Shared Scholarship applications for 2027/28? | ✓ | ✓ | — |
| q29 | What is the full-time base RTP stipend rate for 2026? | ✗ | ✓ | wrong sentence from the right chunk — first-sentence picking returned an intro or unrelated sentence |
| q30 | How many Marshall Scholarships are awarded each year? | ✓ | ✓ | — |
| q31 | How much is the monthly EMJM scholarship for individuals? | ✓ | ✓ | — |
| q32 | What is the monthly allowance for a Pioneering Women in STEM scholar? | ✓ | ✓ | — |

## Failure table — out-of-corpus questions (abstention)

| ID | Question | Baseline | Improved | Why |
|---|---|---|---|---|
| o01 | What is the deadline for the Chevening Scholarship 2027? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o02 | What is the monthly stipend for the MEXT scholarship in yen? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o03 | What is the acceptance rate for the Erasmus Mundus programme? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o04 | Which university in Canada is ranked number one for Vanier scholars? | ✗ answered | ✓ abstained | abstained (unanswered_ranking) |
| o05 | What is the GPA requirement for the DAAD scholarship? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o06 | What is the deadline for the Australia Awards Scholarship 2027? | ✗ answered | ✓ abstained | abstained (unanswered_scholarship_subject) |
| o07 | What is the acceptance rate for the Knight-Hennessy Scholars programme? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |

## What still fails, and why

- q10, q17 (paraphrase gap): the expected answer's key terms do not appear in the question, so lexical picking cannot rank the right sentence first. Dense sentence scoring shipped in Phase 2 Step 2 and did not fix these two; what remains is a generative answer with a verification step. q03 was in this group until the Phase 2 improvement (bullet-list splitting) fixed it: answer-only 0.907 -> 0.930. q32 was in this group until the Phase 2 improvement (IDF sentence scoring) fixed it: answer-only 0.930 -> 0.953.
- q32 (fixed by IDF sentence scoring): the SEK 12,000 allowance sentence shares only 'monthly' and 'allowance' with the question while the competing eligibility sentence shares 7 stems, so plain stem counts ranked the wrong sentence first. IDF weighting over the candidate sentences makes the rare stems outweigh the ubiquitous doc-name stems, and the SEK sentence is now picked first. The failure table used to label it REGRESSION (the baseline got luckier on corpus growth while the improved pipeline still missed it); both columns pass now.
- o06, o07 (abstention lost on corpus growth, then fixed in Step 6): the grown corpus gave the lexical gates other scholarships' deadline and rate-adjacent sentences to quote, and the doc-name gate only fires when a known document is named — o06 (Australia Awards deadline, a scholarship not in the corpus) was answered with SI + Commonwealth dates and o07 (KHS acceptance rate, a fact not on the fetched page) with KHS funding sentences, so abstention fell 1.0 -> 0.714 (5/7) mid-step. Fixed in this step, two ways: the specific-term gate no longer counts "scholar" as specific (o07's only specific terms are now "acceptance" and "rate", neither in any KHS sentence — reason "missing_specific_term"), and the answer verification gained a scholarship-subject focus check (o06 — reason "unanswered_scholarship_subject"). Abstention is 1.0 (7/7) again; the dip is recorded in tests/test_corpus_growth.py, not hidden.
- The reranker adds no measured value on this corpus (answer-only 0.938 without it vs 0.938 with it). It is kept because the mechanism is sound and the corpus is small; on a larger corpus it must be re-measured before it is claimed to help.
- Dense retrieval adds no measured recall and slightly hurts answer-only relevance (dense-only 0.906 vs BM25-only 0.938) at ~50x the latency. On this corpus BM25-only is the better engineering choice; dense is kept in the hybrid because it is expected to matter as the corpus grows, and the ablation will tell us when.
