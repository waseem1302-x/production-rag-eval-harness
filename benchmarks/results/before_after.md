# Before / after — Phase 1

## What was before

The naive baseline is the tutorial chatbot: BM25 retrieval, no reranker, first-sentence picking, no abstention gates — it answers every question with the first sentences of the top chunk and never says "I don't know". It also has no notion of eligibility and no record of the corpus's messy data. Every number below is measured on the same golden set (23 in-corpus + 5 out-of-corpus factual questions).

## What happened after

The full Phase 1 pipeline: hybrid dense+BM25 retrieval, lexical reranker, scored sentence picking (up to 5 sentences, short-fragment penalty), abstention gates with recorded reasons, eligibility-first answering from corpus-grounded rules, and a data-quality checker that records the corpus's mess instead of hiding it.

## Per-metric deltas (baseline → improved)

| Metric | Before (baseline-naive) | After (improved-hybrid) | Change |
|---|---|---|---|
| recall@3 | 1.0 | 1.0 | +0.000 |
| recall@5 | 1.0 | 1.0 | +0.000 |
| Evidence relevance (proxy) | 1.0 | 1.0 | +0.000 |
| Answer-only relevance | 0.522 | 0.87 | +0.348 |
| Faithfulness | 1.0 | 1.0 | +0.000 |
| Abstention accuracy | 0.0 | 0.8 | +0.800 |
| Retrieval latency (mean) | 0.2 ms | 66.4 ms | +66.2 ms |
| Total latency (mean) | 0.5 ms | 70.2 ms | +69.7 ms |
| Cost per query | $0.00 | $0.00 | $0.00 (local CPU, no API calls) |

## Why it changed

- Answer-only relevance rose because scored picking replaced first-sentence picking: the baseline returned whatever sentences came first in the top chunk — usually headers, navigation or a marketing intro. Scored picking ranks sentences by shared query stems, and the short-fragment penalty demotes header fragments. The retrieval signal did not drive this: hybrid-plain (hybrid retrieval, naive answering) scores only 0.565 on answer-only, barely above the baseline.
- Abstention rose because the gates exist: the baseline answers everything (abstention 0.0). The score threshold, doc-name consistency, shared-stem and specific-term gates refuse questions the corpus cannot answer, and each refusal records its reason.
- Recall did not change, because it was already saturated: recall@5 is 1.0 for every configuration, including BM25-only. On a 31-chunk corpus the right document is always in the top 5 by keyword alone. Dense retrieval buys no measured recall here at ~50x the latency — that is a finding about this corpus size, not a failure of dense retrieval in general.
- Faithfulness stayed at 1.0 because both pipelines are extractive: every answer sentence is copied verbatim from a cited chunk. Faithfulness measures copying, not correctness — it cannot catch an answer that quotes the right words for the wrong question.
- Eligibility and data quality are new capabilities, not improvements on the baseline: the baseline had no profile, no rules and no record of the corpus's problems. Measured now: eligibility accuracy 1.0 (10/10 golden questions); data-quality checker finds 36 findings (8 resolved, 28 open).

## Failure table — in-corpus questions

| ID | Question | Baseline | Improved | Why |
|---|---|---|---|---|
| q01 | What does the Chevening Scholarship fund? | ✓ | ✓ | — |
| q02 | Is the Chevening Scholarship fully funded? | ✓ | ✓ | — |
| q03 | How long are the master's studies supported by the SI Scholarship for Global Professionals? | ✗ | ✗ | paraphrase gap — most of the expected answer's content words (one, two, year) do not appear in the question, so lexical sentence picking cannot find the sentence; needs dense sentence scoring or a generative answer (Phase 2) |
| q04 | How many countries are eligible for the SI Scholarship for Global Professionals? | ✗ | ✓ | wrong sentence from the right chunk — first-sentence picking returned an intro or unrelated sentence |
| q05 | How many countries does the Fulbright Foreign Student Program operate in? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q06 | How many foreign students receive Fulbright scholarships each year? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q07 | How many scholarship options does the DAAD database list for selection? | ✗ | ✓ | header/nav text picked first — the first sentences of the top chunk are page furniture, not content |
| q08 | Who are DAAD Programmes for Graduates aimed at? | ✓ | ✓ | — |
| q09 | How many institutions from how many countries are involved in an Erasmus Mundus Joint Master? | ✓ | ✓ | — |
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

## Failure table — out-of-corpus questions (abstention)

| ID | Question | Baseline | Improved | Why |
|---|---|---|---|---|
| o01 | What is the deadline for the Chevening Scholarship 2027? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o02 | What is the monthly stipend for the MEXT scholarship in yen? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o03 | What is the acceptance rate for the Erasmus Mundus programme? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |
| o04 | Which university in Canada is ranked number one for Vanier scholars? | ✗ answered | ✗ answered | lexical gates pass: the answer sentences share stems with the question, but no sentence actually names what the question asks for — needs a verification step (Phase 2) |
| o05 | What is the GPA requirement for the DAAD scholarship? | ✗ answered | ✓ abstained | abstained (missing_specific_term) |

## What still fails, and why

- q03, q10, q17 (paraphrase gap): the expected answer's key terms do not appear in the question, so lexical picking cannot rank the right sentence first. Fix planned for Phase 2: dense sentence scoring, or a generative answer with a verification step.
- o04 (Vanier ranked university): the system answers with Vanier sentences that never name a ranked university. The lexical gates check word overlap, not whether the sentences actually answer the question. Fix planned for Phase 2: an answer-verification step that checks the picked sentences contain the question's object.
- The reranker adds no measured value on this corpus (answer-only 0.87 with and without it). It is kept because the mechanism is sound and the corpus is small; on a larger corpus it must be re-measured before it is claimed to help.
- Dense retrieval adds no measured recall and slightly hurts answer-only relevance (dense-only 0.826 vs BM25-only 0.87) at ~50x the latency. On this corpus BM25-only is the better engineering choice; dense is kept in the hybrid because it is expected to matter as the corpus grows, and the ablation will tell us when.
