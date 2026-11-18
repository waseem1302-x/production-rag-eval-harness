# Distillation Plan: Teach the Local 0.5B Student with a Stronger Teacher

Status: PLAN ONLY. No training has run. No API has been called. No code has changed.

Date: 18 Nov 2026. Repo: production-rag-eval-harness, main HEAD e4ad3d3, 50 commits, pushed.

## Goal

Make the generative answers better without changing the student at inference time.

The student stays Qwen/Qwen2.5-0.5B-Instruct. It stays local, CPU-only, and $0 per query. The extractive mode does not change at all. It already scores 1.0 on every metric.

The problem to fix: generative mode scores 0.77 overall and 18/32 on factual questions. The same 0.5B model is also the judge, and it scores 1/1 on all 14 factual generative misses. The generator paraphrases past the expected phrase, and the judge cannot tell.

The method: a stronger teacher model writes good answers for our corpus. The student trains on those answers with LoRA. The student learns the behaviour, not the teacher's knowledge. The knowledge stays in the retrieved chunks.

## Baseline numbers (measured 18 Nov 2026, 68-question golden set)

These are the numbers the distilled adapter must beat. Source: `benchmarks/results/generative_eval.json` and `EVAL_RESULTS.md`.

| Metric | Extractive (must not move) | Generative (must improve) |
|---|---|---|
| Answer-only relevance, overall | 1.0 (61/61) | 0.77 (47/61) |
| Answer-only relevance, factual only | 1.0 (32/32) | 0.5625 (18/32) |
| Faithfulness (verbatim proxy) | 1.0 | 0.793 |
| Abstention accuracy | 1.0 (7/7) | 1.0 (7/7) |
| Eligibility accuracy | 1.0 (29/29) | 1.0 (29/29) |
| Judge faithfulness / relevance | n/a | 0.969 / 0.969 (n=32, 0 unparsed) |
| Mean generation latency | n/a | 6.59 s |
| Mean judge latency | n/a | 5.11 s |
| Cost per query | $0.00 | $0.00 |

Known generative misses (14 factual): q01, q03, q05, q07, q10, q11, q12, q13, q14, q17, q18, q24, q29, q30, q31. The judge scores 1/1 on all 14. It also scores 0/0 on q21, the correct Vanier closure answer. The judge is wrong in both directions.

The 68 golden questions are q01 to q32 (factual), o01 to o07 (out of corpus), e01 to e29 (eligibility). The corpus is 19 documents, 101 chunks, 190,525 characters, 2,095 pickable sentences.

## 1. Teacher options

UPDATE 18 Nov 2026, after the plan was written: the Groq option is BLOCKED. The Groq key the user gave on 15 Nov 2026 is stored in ~/.config/free-llm-keys.env, but a re-check on 18 Nov returned 403 Forbidden on the models list, so the key no longer works. Do not retry it. The Gemini key from the same file was verified working on 18 Nov: the models list returned 50 models and gemini-3.5-flash-lite answered a test prompt. Notes from that check: gemini-2.5-flash and gemini-2.5-flash-lite return 404 (retired, use 3.5-flash-lite); gemini-3.5-flash and gemini-3.8-flash returned 503 high demand at that moment, so retry them first and fall back to 3.5-flash-lite if they stay unavailable.

No API key exists yet for any teacher. The user must provide one, or we use the no-key option.

### Option A: Groq free plan (recommended teacher)

Cost: $0. No card is needed. The free plan is a real recurring tier, not a one-time credit.

Limits (published): 30 requests per minute, 1,000 requests per day, 8,000 tokens per minute, 200,000 tokens per day, per model. Teacher model: `openai/gpt-oss-120b`. Confirm the exact model id in the Groq console at execution time. Free tier limits change, so re-check before the run.

What it needs: a Groq API key, created free at console.groq.com. The user creates it and gives it through the secure card. Never in chat.

Why it is the recommendation: the limits are published, so the run can be planned. 200,000 tokens per day covers about 200 teacher calls per day at our prompt size. The full teacher run needs about 350 calls, so it fits in two days of quota. Groq is also very fast, so the calls themselves take minutes. The teacher is a 120B open model, strong enough to follow a "use only the context" instruction.

Risk: the 200,000 token daily cap is the binding limit. If the run grows past it, it pauses until the next day. The generation script must save every teacher output at once, so a pause loses no work.

### Option B: Gemini API free tier

Cost: $0. No card is needed. Free models are Gemini 3.8 Flash and Flash-Lite. Confirm the current model name in Google AI Studio at execution time. Gemini 2.5 Flash is deprecated and routes to 3.5 Flash-Lite from 1 Dec 2026.

Limits: Google no longer publishes per-model free tier numbers. Limits are shown in the AI Studio console only, per project. Treat the quota as unknown until the console is checked.

What it needs: a Gemini API key from Google AI Studio. The user creates it and gives it through the secure card. Never in chat.

Why it is second: the teacher is likely the strongest of the three at instruction following, and the user already has a Google account. But the unknown quota makes planning harder. Also note the term: free tier content may be used to improve Google products. Our corpus is public scholarship pages with no personal data, so this is low risk, but the user should know it.

Use it as the fallback if Groq quota blocks the run, or as a second teacher to cross-check a sample of Groq answers.

### Option C: Bigger open model on Colab T4 (no-key fallback)

Cost: $0. Colab is signed in as waseemmushtaq045@gmail.com. A free T4 GPU is available. No API key is needed.

Limits: T4 has 15 to 16 GB of VRAM. A 7B model in bf16 needs about 15.4 GB, so it does not fit. The teacher must be Qwen/Qwen2.5-7B-Instruct loaded in 4-bit (bitsandbytes, about 4 to 5 GB), or Qwen/Qwen2.5-3B-Instruct in bf16 (about 6.2 GB). A 14B teacher does not fit even in 4-bit on a T4. Colab sessions last at most 12 hours, with an idle timeout near 90 minutes. Files survive only in Google Drive.

What it needs: nothing from the user. The notebook downloads the teacher weights from Hugging Face.

Why it is third: the teacher is weaker than the API options, and same-family transfer is the only advantage (same Qwen tokenizer and style as the student). It is also slower, about 1 to 2 seconds per teacher answer on a T4, and the session can die mid-run. But it needs no key and no quota, so it is the fallback if the user does not want to create an API key.

### Not options

GitHub Models was fully retired on 30 Jul 2026. OpenAI and Anthropic have no free tier. Do not plan around them.

### Recommendation

Use Gemini as the teacher (changed 18 Nov 2026: Groq is blocked, see the update above). Try gemini-3.5-flash or gemini-3.8-flash first, fall back to gemini-3.5-flash-lite if they return 503. Use the Colab 7B 4-bit teacher only if the Gemini key stops working. Record the teacher name, model id, and date in every training file.

## 2. Training data design

### The contamination rule (read first)

The 68 golden questions are the eval. They must stay a clean eval.

Never train on a golden question verbatim. Train on variants of golden questions, and on new synthetic questions from the corpus. Report the full 68-question eval after training, and report the held-out subset separately. If training on golden questions verbatim, the before/after numbers would be meaningless.

### Sources and target counts

Total target: about 340 examples. Counts are targets, not promises. The verification gate in section 2.3 decides the final count.

| Source | Target | What it teaches |
|---|---|---|
| Golden-derived variants | 60 | The exact failure shapes, on fresh wording |
| Synthetic QA from corpus sentences | 200 | Grounded answering across all 19 documents |
| Abstention examples | 40 | Saying "Not found in the corpus" |
| Judge examples | 40 | Grading honestly, including saying 0 |
| Total | 340 | |

#### Golden-derived variants (60)

For each of the 14 known generative misses, the teacher writes 3 paraphrases of the question, each answered from the same evidence chunks. That is 42 examples. For 9 more factual golden questions that the generator already passes, the teacher writes 2 paraphrases each. That is 18 examples. Total 60.

The paraphrase must change the wording, not the ask. The expected answer phrase must stay the same. The original golden question is never in training data.

#### Synthetic QA from corpus sentences (200)

Sample pickable sentences from the 101 chunks. The corpus has 2,095 pickable sentences. Sample to cover all 19 documents, capped at 15 examples per document so Rhodes (29 chunks) does not dominate. For each sampled sentence, the teacher writes one question that the sentence answers, then answers it from the top 3 chunks for that question.

Target mix: about 60 percent fact questions (amounts, dates, durations, counts), 25 percent eligibility-shaped questions (who can apply, which level, which countries), 15 percent definition questions (what the programme is, what it funds). Amounts and dates matter most, because the current misses are mostly dropped numbers and phrases.

#### Abstention examples (40)

The 7 golden out-of-corpus questions stay in eval only. Training abstention uses fresh ones: 20 questions about scholarships not in the corpus (Australia Awards, Chevening is in, so use others such as MEXT is in, so use Gates is in; use genuinely absent ones like Australia Awards, DAAD is in, use Fulbright is in; the script must check the named scholarship against the 19 doc slugs and keep only absent ones), and 20 questions about facts no page states (acceptance rates, GPA cutoffs, interview questions). Target answer is exactly "Not found in the corpus." Nothing else.

#### Judge examples (40)

The judge is the same 0.5B model, so judge training shares the adapter. Keep this slice small, at most 15 percent of the data, so it does not dilute generation.

14 examples: the known generative misses, with the teacher judging the wrong student answer. Correct target is faithfulness 0 or relevance 0, with the exact two-line format from `_JUDGE_PROMPT`. 26 examples: correct answers (extractive answers to golden questions), with the teacher judging them. Correct target is faithfulness 1 and relevance 1.

Risk, stated plainly: judge-format data may hurt generation, or generation data may hurt judging. Measure both separately in section 4. If generation regresses, drop the judge slice and retrain without it.

### Record format

Every record uses the exact deployed prompt format, so the adapter learns the deployed behaviour. The deployed prompt is built by `app/generative.py:build_generation_prompt`. It is one user message: the instruction, then up to 3 chunks as `[Source chunk_id] text` (900 chars max per chunk), then `Question: ...` and `Answer:`.

Training record (JSONL, one object per line):

```json
{
  "messages": [
    {"role": "user", "content": "<exact build_generation_prompt output>"},
    {"role": "assistant", "content": "<verified teacher answer>"}
  ],
  "source": "golden-variant-q10",
  "question_id": "q10-v2",
  "evidence_chunks": ["erasmus-mundus-0"],
  "teacher": "groq/openai/gpt-oss-120b",
  "teacher_date": "2026-11-18",
  "verified": true,
  "verification": {"kept": ["..."], "dropped": []}
}
```

Train on the completion only. Mask the prompt tokens. The assistant content is 1 to 2 sentences, verbatim grounded, or exactly "Not found in the corpus."

Files:

- `data/distillation/teacher_raw.jsonl`: every teacher output, before filtering. Audit trail, never trained on directly.
- `data/distillation/train.jsonl`: verified records only.
- `data/distillation/heldout.jsonl`: 50 synthetic QA records, never trained on. The generalisation check.
- `data/distillation/README.md`: counts, teacher, date, filter rules, and the contamination rule.

### Verification gate (before any record enters training data)

Every teacher answer must pass all checks below. Anything that fails is dropped, never fixed by hand and never trained on blind.

1. It passes `app.verification.verify_answer` against the same 3 chunks the prompt used. This is the same deterministic check production uses.
2. It contains the expected answer phrase (golden variants), or the source sentence's key fact (synthetic QA, checked by running the same question through the extractive answerer and requiring the same expected phrase).
3. It is 1 to 2 sentences. Longer answers are dropped, not trimmed.
4. It adds no number that is absent from the chunks. The production number rule already enforces this; the gate re-checks it.
5. Abstention records must be exactly "Not found in the corpus." Anything else is dropped.
6. Judge records must parse to exactly two lines starting with `faithfulness:` and `relevance:`, with the correct 0 or 1 values. Anything else is dropped.

The gate script logs kept and dropped counts per source. If the keep rate for any source is below 70 percent, stop and inspect the teacher prompts before generating more. A low keep rate means the teacher is not following the instruction, not that the filter is too strict.

## 3. LoRA training config (Colab T4)

Base model: Qwen/Qwen2.5-0.5B-Instruct, bf16. Same revision as production (7ae55760). The student at inference stays exactly this model plus the adapter.

| Setting | Value | Why |
|---|---|---|
| Method | LoRA, not full fine-tuning | The adapter is small and swappable. The base weights stay intact. Full fine-tuning risks catastrophic forgetting on 340 examples. |
| Rank (r) | 16 | Small enough to fit the data size. Larger ranks memorise a small dataset. |
| Alpha | 32 | Alpha is 2 times rank, the standard scaling. Effective scaling is alpha divided by rank, which is 2. |
| Dropout | 0.05 | Light regularisation against overfitting on 340 examples. |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All linear layers. Attention and MLP both need to learn the grounded style. |
| Trainable params | about 9M (about 1.8 percent of 0.5B) | Small enough to train fast, large enough to learn a narrow style. |
| Learning rate | 2e-4, cosine schedule, 5 percent warmup | Standard LoRA rate for a small model. Higher rates destroy the base behaviour on little data. |
| Epochs | 3 maximum, with eval after each epoch | Small data overfits fast. Keep the best epoch, not the last. |
| Batch size | 4 per device, grad accum 4, effective 16 | Effective 16 is stable for 340 examples. Larger batches waste the small dataset. |
| Max sequence length | 1024 tokens | Prompts are about 700 to 900 tokens (3 chunks at 900 chars plus instruction). Measure the p95 prompt length before training. If p95 exceeds 1024, raise to 1280. Never truncate the context, because that changes the prompt from deployment. |
| Optimiser | adamw_torch_fused | Standard, fast on T4. |
| Precision | bf16 | Matches the deployed model. T4 handles bf16 natively. |
| Gradient checkpointing | On | Saves VRAM for longer prompts. Training is slower but fits. |
| Save | One checkpoint per epoch, to Google Drive | Colab can die at any time. Every epoch must survive on Drive. |

### Overfitting guards

Overfitting is the main risk. 340 examples against 9M trainable params will memorise fast.

1. Hold out 50 synthetic QA records before training. Never train on them. They measure generalisation.
2. Hold out 10 golden-derived variants as well, so the variant slice has its own check.
3. Evaluate after every epoch on the held-out set and on the 68-question golden set (generative mode only, fast subset if time is short).
4. Early stopping: stop when the held-out score falls for two epochs in a row, or when the golden factual score falls. Keep the best epoch checkpoint, not the last.
5. If epoch 1 is already worse than the base model, stop. Do not train longer hoping it recovers.
6. Compare the adapter against the base on the same held-out questions, not on training questions. Training accuracy proves nothing.

### What training needs installed (Colab only)

`peft`, `trl`, `datasets`, `accelerate`. These go in a separate `requirements-train.txt` in the repo, not in `requirements.txt`. The app must keep running without them. `bitsandbytes` is needed only if the Colab teacher option is used (4-bit 7B). It also stays out of `requirements.txt`.

Training time estimate on a T4: about 30 to 60 minutes for 3 epochs on 340 examples. This is a rough estimate. Measure the first epoch and extrapolate.

## 4. Eval protocol

Run the same script as the baseline: `python -m scripts.run_generative_eval`. It writes `benchmarks/results/generative_eval.json` and `.md`. Run it on this VM, not on Colab, so latency is comparable to the baseline.

### Before/after table to fill in

| Metric | Before (18 Nov 2026) | After (adapter) | Change |
|---|---|---|---|
| Generative overall answer-only | 0.77 (47/61) | ? | ? |
| Generative factual-only | 0.5625 (18/32) | ? | ? |
| Generative faithfulness (verbatim proxy) | 0.793 | ? | ? |
| Judge faithfulness / relevance | 0.969 / 0.969 (n=32) | ? | ? |
| Judge agreement with proxy | measure at execution | ? | ? |
| Extractive answer-only | 1.0 (61/61) | must stay 1.0 | none allowed |
| Recall@3 / recall@5 | 1.0 / 1.0 | must stay 1.0 | none allowed |
| Abstention accuracy | 1.0 (7/7) | must stay 1.0 | none allowed |
| Eligibility accuracy | 1.0 (29/29) | must stay 1.0 | none allowed |
| Mean generation latency | 6.59 s | ? | ? |
| Mean judge latency | 5.11 s | ? | ? |
| Generative modes (generative / fallback / eligibility) | 23 / 9 / 29 | ? | ? |
| Held-out synthetic QA accuracy | n/a (new) | ? | ? |
| Training cost | n/a | $0 (Colab free) | ? |
| Teacher API cost | n/a | $0 (free tier) | record tokens used |

### Judge agreement (new metric)

The current judge scores are not trustworthy. Judge agreement measures whether the judge agrees with the deterministic proxy.

For each judged question: judge faithfulness should equal the verbatim faithfulness proxy (1 if every answer sentence is verbatim in a chunk, else 0). Judge relevance should equal answer-only relevance (1 if the expected phrase is in the answer, else 0). Agreement is the fraction of questions where both match.

Baseline to measure at execution: the current judge scores 1/1 on all 14 factual misses and 0/0 on q21, so its agreement is poor. Record the exact baseline before claiming any improvement.

### Keep rule (ship the adapter only if all of these hold)

1. Generative factual-only improves. No minimum is set here. Any real improvement counts, but it must be visible on the held-out subset too, not only on trained variants.
2. Zero regressions on every other metric: extractive 1.0, recall 1.0, abstention 1.0, eligibility 1.0, faithfulness 1.0.
3. Judge agreement does not fall. If the judge slice was trained, agreement should rise. If it falls, drop the judge slice and retrain.
4. Held-out synthetic QA accuracy is at least as good as the base model on the same 50 questions. Run the base model on the held-out set first, before training, so the comparison is real.
5. Mean generation latency does not more than double. The adapter adds a merge step at load, not per token, so latency should stay near 6.59 s.

If any rule fails, do not ship the adapter. Record the negative result in `EVAL_RESULTS.md` with the numbers. A measured failure is still a result. The base model stays the default.

### Per-question check

List every question that changed, in both directions. A single regression on a previously correct question blocks shipping, even if the headline number rose. This is the same rule the repo already uses for extractive changes.

## 5. Repo integration

### Where the adapter lives

`adapters/qwen2.5-0.5b-distilled-v1/` in the repo, with:

- `adapter_model.safetensors` (about 18 MB at rank 16, bf16)
- `adapter_config.json` (from PEFT, unedited)
- `training.json` (teacher, model id, date, data counts, LoRA config, epoch kept, held-out score)

### Git, LFS, or Hugging Face Hub (decision)

Commit the adapter to plain git. Reason: about 18 MB is well under GitHub's 100 MB file block and 50 MB warning threshold. It is versioned with the code that loads it.

Do not commit merged full weights (about 988 MB). Never. If a future adapter exceeds 50 MB, move it to Git LFS or Hugging Face Hub and record the decision in this plan. If the adapter is ever hosted on Hugging Face Hub, the repo loads it from there and the local folder holds only the config and a pointer.

### How the app loads the adapter

Add an optional adapter path to `app/generative.py`. A config value `GENERATIVE_ADAPTER_PATH` in `app/config.py`, default `None`. When set, `load_model()` loads the base model, then applies `PeftModel.from_pretrained` with the adapter. When unset, behaviour is exactly the current base model. When the adapter folder is missing or corrupt, log the error and fall back to the base model. Never crash the API because an adapter is missing. This matches the existing embedder fallback pattern.

`POST /ask` with `generate=true` reports the adapter in the `generation` object: `{"model": "Qwen/Qwen2.5-0.5B-Instruct", "adapter": "adapters/qwen2.5-0.5b-distilled-v1"}`. The eval script gains an `--adapter` flag so the same run can measure base and adapter.

### Tests to add

1. Adapter missing: `GENERATIVE_ADAPTER_PATH` points nowhere, the model still loads the base. Fail-safe, like the embedder fallback.
2. Adapter present: the loaded model reports the adapter path in its metadata, and one fixed prompt still completes.
3. Eval parity: the adapter eval run writes the same JSON shape as the base run, so the before/after table can be built from files, not hand-typed numbers.

### Docs to update

- `README.md`: generative section, adapter name and size, new before/after numbers, and the "What is not done yet" bullet about the weak generator.
- `EVAL_RESULTS.md`: a new distillation section with the before/after table, the held-out numbers, and the keep rule outcome.
- `data/distillation/README.md`: data counts, teacher, filter rules.
- This plan: mark each step done with its date when executed.

## 6. Risks and open decisions

| Risk or decision | What happens | Who decides |
|---|---|---|
| No teacher API key exists | Fall back to Option C, the Colab 7B 4-bit teacher. Weaker teacher, slower run, but no key needed. | User: approve Groq or Gemini key, or choose the no-key fallback. |
| Groq daily token cap hit mid-run | Pause until the next day. Resume from `teacher_raw.jsonl`, which already holds every completed call. No work is lost. | Executor: pause and resume, no user action needed. |
| Gemini free tier quota unknown | Check the AI Studio console before starting. If the quota is too small for 350 calls, switch to Groq. | Executor: check console, report the number to the user. |
| Colab session dies | Checkpoints save to Drive after every epoch. Teacher outputs save to Drive after every call. Resume from the last checkpoint. | Executor: resume, no user action needed. |
| Overfitting on 340 examples | Held-out eval after each epoch, early stopping, max 3 epochs. If epoch 1 is worse than base, stop. | Executor: follow the keep rule. |
| Golden set contamination | Variants only, never verbatim golden questions. Held-out subset reported separately. | Executor: enforce the rule, no exception. |
| Judge slice hurts generation | Measure generation and judging separately. Drop the judge slice and retrain if generation regresses. | Executor: follow the keep rule. |
| Adapter does not help | Record the negative result in EVAL_RESULTS.md. Keep the base model. The verification and fallback already cap the harm, so no user-visible answer gets worse. | User: decide whether to try a different teacher or stop. |
| 0.5B is too small to learn the behaviour | Possible. The honest outcome is a measured small gain or no gain. The repo records it either way. A bigger student is out of scope for this phase. | User: decide whether a bigger student is ever worth it. |
| Teacher terms | Gemini free tier content may be used to improve Google products. Groq teacher outputs become our training data, which is standard API use, but re-check the terms at execution time. The corpus is public, with no personal data. | User: confirm the terms are acceptable. |
| Cost if a free tier ends | Stop at once. No payment method is added to any account. The run is designed to cost $0 end to end. | Executor: stop, report, do not add a card. |

## 7. Step order with rough time estimates

Times are active work, not calendar time. The teacher quota can spread step 3 over two days.

| Step | What | Time |
|---|---|---|
| 1 | User decisions: teacher choice and API key, or no-key fallback. See the checklist below. | 15 to 30 min (user) |
| 2 | Build the teacher generation script. It reuses the repo retriever and `build_generation_prompt` exactly, so teacher context matches deployment. Verify one example end to end, from retrieval to verified record. | 1 to 2 h |
| 3 | Generate teacher outputs. About 350 calls. Groq: minutes of calls, spread over 2 days of quota. Gemini: faster if the quota allows. Colab teacher: 1 to 2 h of GPU time. Save every raw output at once. | 1 to 2 h active |
| 4 | Run the verification gate. Build `train.jsonl` and `heldout.jsonl`. Log kept and dropped counts per source. Stop if any source keeps below 70 percent. | 1 to 2 h |
| 5 | Build the Colab training notebook. First run: 1 epoch, then eval, to prove the pipeline works before the full run. | 1 to 2 h setup, 30 to 60 min training |
| 6 | Full training run: 3 epochs with eval after each, checkpoints to Drive. | 30 to 60 min GPU time |
| 7 | Eval the adapter on this VM: base model on held-out set first (for the comparison), then adapter on the 68-question golden set and the held-out set. About 13 min per generative eval run. | 1 h |
| 8 | Decide by the keep rule. If it ships: integrate the adapter, add tests, update README and EVAL_RESULTS.md, commit on a feat branch, push after approval. If it fails: record the negative result, keep the base model. | 2 to 3 h |

Total active work: about 1 to 2 days, spread over 2 to 3 calendar days if the Groq quota paces step 3.

## User decisions needed before execution

- [ ] Teacher choice: Groq free plan (recommended), Gemini free tier, or the no-key Colab 7B teacher.
- [ ] If Groq or Gemini: create the API key and give it through the secure card. Never paste it in chat.
- [ ] Confirm the free tier terms are acceptable, including Gemini's term that free tier content may be used to improve Google products.
- [ ] Confirm the contamination rule: golden questions stay out of training data, only variants are used.
- [ ] Confirm the keep rule: ship the adapter only if generative improves with zero regressions on every other metric.
- [ ] Confirm the judge slice is in scope (40 examples, measured separately, dropped if it hurts generation).
