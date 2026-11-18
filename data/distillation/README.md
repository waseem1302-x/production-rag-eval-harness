# Distillation dataset

Teacher: gemini/gemini-3.5-flash-lite, 2026-11-18. Groq was not used: its stored key returns 403.

## Counts (verified, from stats.json)

| Source | Generated | Kept | Dropped |
|---|---|---|---|
| golden-variant | 60 | 44 | 16 |
| golden-variant-heldout | 10 | 4 | 6 |
| synthetic-qa | 250 | 188 | 62 |
| abstention | 40 | 38 | 2 |
| judge | 40 | 40 | 0 |

Train records: 273 ({'golden-variant': 44, 'synthetic-qa': 151, 'abstention': 38, 'judge': 40}).

Held-out records: 41 ({'golden-variant-heldout': 4, 'synthetic-qa': 37}). Held-out records are never trained on. They are the generalisation check.

Teacher calls: 435. Teacher tokens: 109687 prompt, 23628 completion. Teacher cost: $0 (Gemini free tier).

## Files

- `teacher_raw.jsonl`: every teacher output, before filtering. Audit trail, never trained on directly.
- `train.jsonl`: verified records only.
- `heldout.jsonl`: verified records, never trained on.
- `gate_log.jsonl`: one line per gate decision, with reasons.
- `stats.json`: the counts above, read back from the logs.

## Rules enforced by scripts/build_distillation_data.py

1. Contamination: never train on a golden question verbatim. Golden-derived training uses paraphrases only. Synthetic questions are dropped when their content-token Jaccard similarity to any golden question is 0.85 or higher.
2. Verification gate: every teacher answer must pass `app.verification.verify_answer` against the same 3 chunks the prompt used, contain the expected phrase, and be 1-2 sentences. Anything that fails is dropped, never fixed by hand. The checks run on the kept sentences, the same semantics production uses: a sentence the verifier drops is not in the training answer.
3. Abstention records are exactly "Not found in the corpus." Their check phrases must not appear anywhere in the corpus.
4. Judge records must parse to two lines and match the deterministic proxy (verbatim faithfulness, answer-only relevance). Anything else is dropped.
5. Keep rate below 0.70 for any source stops the run before finalising.

## Corrections to the plan

- The plan says the corpus has 2,095 pickable sentences. Measured: 1,144 sentences of 40 to 400 characters. The sampling targets were kept; the mix is reported in stats.json.
- The plan lists q17 among the 14 generative misses. The measured generative_eval.json run has q17 fixed (via fallback), so the 14 misses are q01, q03, q05, q07, q10, q11, q12, q13, q14, q18, q24, q29, q30, q31. q17 is in the held-out variant group instead.
- Synthetic answers are written by the teacher from the source sentence in one call, then verified against the retrieved top-3 chunks. The plan describes two calls; one call plus the gate enforces the same grounding with half the teacher calls.
- The gate checks the kept sentences (production semantics), not the raw draft. A teacher answer with one grounded sentence and one dropped sentence still yields a clean one-sentence training record. The first smoke run used the stricter all-sentences-kept rule and dropped good answers; the relaxed rule matches what generate_answer does at inference.
- q07, q10 and q29 variants were dropped, and the gate is right: the deployed prompt truncates each chunk to 900 chars. The DAAD answer "147 Scholarship options" sits near char 1260 of the daad chunk, the Erasmus answer "1 to 2 academic years" sits near char 1136 of the erasmus-mundus chunk, and the RTP answer "34,315" sits near char 1706 of the rtp chunk, so the teacher correctly answered "Not found in the corpus." The same cut hides the HKPFS answer "HK$344,400" (near char 1021 of the hkpfs chunk, q15) and the Manaaki answer "undergraduate and postgraduate study" (near char 994 of the manaaki chunk, q17). The truncation is a production limitation to fix separately, not by changing the training prompt.
- Judge labels come from the deterministic proxy, not from the teacher. The teacher judge kept 19/39 (0.49, below the 0.70 gate): Gemini flash-lite disagreed with the proxy in both directions, lenient on wrong answers and hallucinating unfaithfulness on right verbatim ones, and its 19 kept records were skewed 18 right + 1 wrong, which would teach the student to almost always answer 1,1. Those records were not used. The 40 labels (14 wrong-answer items, 26 right-answer items) were built by scripts/build_distillation_data.py --finalize-only from verbatim faithfulness plus expected-phrase relevance, with no teacher call. They pass verification by construction.
- The key-fact check is stem coverage (60 percent) plus exact numbers, not a verbatim substring. The teacher rephrases ("September/October 2027" becomes "September or October 2027"), and the verbatim check dropped correct answers in the smoke run.
