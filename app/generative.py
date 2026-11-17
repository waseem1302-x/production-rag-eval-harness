"""Generative answers with a verification step (Phase 2, Step 1).

Local model only, $0: Qwen/Qwen2.5-0.5B-Instruct (0.5B params,
~0.99 GB bf16 weights), run on CPU with transformers + torch.
No paid API is ever called. If the model cannot load, the caller
falls back to the extractive answer and says so — a missing model
is never presented as a generated answer.

Pipeline for one question (see `generate_answer`):
1. Retrieve the top chunks (the caller does this; same hybrid
   retrieval as Phase 1) and build the extractive answer first.
   If the extractive answer abstains, the generative path abstains
   too — the generator must not override the abstention gates.
2. Prompt the local model with the top-3 chunks as context and
   an instruction to answer in 1-2 sentences using ONLY the
   context, or reply exactly "Not found in the corpus."
3. Verify every generated sentence against the chunks with
   `app.verification.verify_answer` (deterministic, lexical).
   Unsupported sentences are dropped.
4. Verify the kept sentences actually answer the question with
   `app.answer.verify_answer_relevance` (Phase 2, Step 2: dense
   floor + ranking-focus check — the same check the extractive
   answerer runs). If they do not, fall back to the extractive
   answer, which already passed that check.
5. If at least one sentence survives, that is the answer
   (mode "generative"). If none survives, fall back to the
   extractive answer (mode "generative_fallback_extractive").

The same model is also the LLM-as-judge (`judge_answer`): one
short completion per question returns a faithfulness bit and a
relevance bit, parsed from the text. The judge is an *eval*
instrument, not the guard — the guard is the deterministic
verification in step 3, which runs even when the judge does not.

Testability: all model text comes through `complete()`, which
honours a module-level override set by `set_completion_override`
(a callable `prompt -> str`). Unit tests inject a fake and never
load the 0.5B model; the real model is loaded lazily, once, on
first real use.
"""
import re
import time

from .answer import answer as build_extractive_answer
from .textnorm import segment_sentences
from .verification import verify_answer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_PARAMS = "0.5B"
# Exact on-disk size of model.safetensors, measured on this VM:
# 988,097,824 bytes (988.1 MB / 941.9 MiB), bf16, downloaded
# 18 Nov 2026 from huggingface.co (Qwen/Qwen2.5-0.5B-Instruct,
# revision 7ae557604adf67be50417f59c2c2f167def9a775).
MODEL_FILE_BYTES = 988_097_824
MODEL_FILE_MB = round(MODEL_FILE_BYTES / 1_000_000, 1)

MAX_NEW_TOKENS_ANSWER = 120
MAX_NEW_TOKENS_JUDGE = 16
MAX_CONTEXT_CHARS_PER_CHUNK = 900
MAX_CONTEXT_CHUNKS = 3

_MODEL = None
_TOKENIZER = None
_LOAD_ERROR = None
_COMPLETION_OVERRIDE = None


def set_completion_override(fn) -> None:
    """Inject a fake text-completion function (tests only).

    `fn` takes a prompt string and returns the model's raw
    completion text. Pass None to restore the real model.
    """
    global _COMPLETION_OVERRIDE
    _COMPLETION_OVERRIDE = fn


def model_load_error() -> str | None:
    return _LOAD_ERROR


def load_model():
    """Load (and cache) the tokenizer + model, or record the error.

    bf16 halves the weights vs float32 (~0.99 GB vs ~1.98 GB) and
    is what the model card ships; on this CPU-only VM it is the
    difference between fitting in RAM and not. If bf16 load fails,
    fall back to float32 once before giving up.
    """
    global _MODEL, _TOKENIZER, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    if _LOAD_ERROR is not None:
        return None, None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Measured on this VM (1.5 shared CPUs): 2 threads generate
        # in ~11 s vs ~19.5 s with the default 4 threads (thread
        # contention under the CPU quota); model load is slower but
        # load happens once per process.
        torch.set_num_threads(2)
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        try:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16)
        except Exception:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float32)
        _MODEL.eval()
        return _MODEL, _TOKENIZER
    except Exception as exc:  # noqa: BLE001 - model load can fail many ways
        _LOAD_ERROR = str(exc)
        _MODEL = None
        _TOKENIZER = None
        return None, None


def complete(prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """Run one text completion; returns (text, latency_seconds).

    Uses the test override when set, otherwise the real local
    model (greedy decoding, so runs are deterministic).
    Raises RuntimeError when the model is unavailable.
    """
    start = time.monotonic()
    if _COMPLETION_OVERRIDE is not None:
        return _COMPLETION_OVERRIDE(prompt), time.monotonic() - start
    model, tokenizer = load_model()
    if model is None:
        raise RuntimeError(f"Generative model unavailable: {_LOAD_ERROR}")
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated = out[0][inputs["input_ids"].shape[1]:]
    latency = time.monotonic() - start
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), latency


def build_generation_prompt(question: str, chunks: list[dict]) -> str:
    """Context + instruction prompt for the answer generator."""
    context = "\n\n".join(
        f"[Source {c['chunk_id']}] {c['text'][:MAX_CONTEXT_CHARS_PER_CHUNK]}"
        for c in chunks[:MAX_CONTEXT_CHUNKS])
    return (
        "Answer the question using ONLY the context below. "
        "Reply in 1-2 short sentences. Do not add any fact that is "
        "not stated in the context. If the context does not answer "
        "the question, reply exactly: Not found in the corpus.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")


def _citations(chunks: list[dict]) -> list[dict]:
    return [{"doc": c["doc"], "chunk_id": c["chunk_id"],
             "score": c["score"], "text": c["text"]}
            for c in chunks[:MAX_CONTEXT_CHUNKS]]


def generate_answer(question: str, retrieved: list[dict],
                    threshold: float | None = None) -> dict:
    """Generative answer with verification, or a stated fallback.

    Returns an answer dict in the same shape as `app.answer.answer`
    plus `answer_mode`, `generation` metadata and, when generative
    text survived verification, `verification` (kept/dropped).
    `answer_mode` is one of: "generative",
    "generative_fallback_extractive" (nothing verified, or the model
    was unavailable), "abstained" (the extractive gates refused, so
    the generator never ran).
    """
    from .config import ABSTAIN_THRESHOLD

    if threshold is None:
        threshold = ABSTAIN_THRESHOLD
    extractive = build_extractive_answer(question, retrieved, threshold=threshold)
    if not extractive["found"]:
        result = dict(extractive)
        result["answer_mode"] = "abstained"
        result["generation"] = {"model": MODEL_ID,
                                "reason": "extractive_abstained",
                                "abstention_reason": extractive.get("abstention_reason")}
        return result

    chunks = retrieved[:MAX_CONTEXT_CHUNKS]
    # Same doc-name consistency as the extractive answerer: when
    # the question names a scholarship, only that scholarship's
    # chunks may ground the answer. Without this, the model blends
    # chunks from two scholarships — measured on q17, where the
    # unfiltered top-3 (Manaaki + 2x SI) produced a Manaaki answer
    # listing the SI Scholarship's study fields, which verification
    # then passed because the SI chunk was in the context.
    from .answer import _named_docs

    named = _named_docs(question)
    if named:
        named_chunks = [c for c in retrieved if c["doc"] in named][:MAX_CONTEXT_CHUNKS]
        if named_chunks:
            chunks = named_chunks
    try:
        prompt = build_generation_prompt(question, chunks)
        draft, latency = complete(prompt, MAX_NEW_TOKENS_ANSWER)
    except RuntimeError as exc:
        result = dict(extractive)
        result["answer_mode"] = "generative_fallback_extractive"
        result["generation"] = {"model": MODEL_ID, "error": str(exc),
                                "reason": "model_unavailable"}
        return result

    if not draft or draft.strip().lower().startswith("not found in the corpus"):
        result = dict(extractive)
        result["answer_mode"] = "generative_fallback_extractive"
        result["generation"] = {"model": MODEL_ID, "latency_s": round(latency, 2),
                                "reason": "model_abstained",
                                "draft": draft}
        return result

    check = verify_answer(draft, chunks, question=question)
    if not check["kept"]:
        result = dict(extractive)
        result["answer_mode"] = "generative_fallback_extractive"
        result["generation"] = {"model": MODEL_ID, "latency_s": round(latency, 2),
                                "reason": "verification_failed_all_dropped",
                                "draft": draft,
                                "dropped": check["dropped"]}
        return result

    # Phase 2, Step 2: the kept sentences are grounded, but
    # grounding is not relevance (the o04 gap: Vanier sentences
    # that never name a ranked university are all "supported").
    # Run the same answer-verification the extractive answerer
    # runs; on failure fall back to the extractive answer, which
    # already passed it.
    from .answer import verify_answer_relevance

    relevance = verify_answer_relevance(question, check["kept"])
    if not relevance["answers"]:
        result = dict(extractive)
        result["answer_mode"] = "generative_fallback_extractive"
        result["generation"] = {"model": MODEL_ID, "latency_s": round(latency, 2),
                                "reason": "relevance_verification_failed",
                                "draft": draft,
                                "relevance_verification": relevance}
        return result

    return {
        "question": question,
        "found": True,
        "answer": check["answer"],
        "score": extractive["score"],
        "citations": _citations(chunks),
        "answer_sentences": check["kept"],
        "answer_mode": "generative",
        "generation": {"model": MODEL_ID, "latency_s": round(latency, 2),
                       "reason": "verified"},
        "verification": {"kept": check["kept"], "dropped": check["dropped"],
                         "n_dropped": len(check["dropped"])},
        "relevance_verification": relevance,
    }


_JUDGE_PROMPT = (
    "You are grading a scholarship answer against its source text.\n"
    "Question: {question}\n\nSource text:\n{context}\n\n"
    "Answer to grade:\n{answer}\n\n"
    "Reply with exactly two lines and nothing else. The first line "
    "MUST start with the word faithfulness, the second line MUST "
    "start with the word relevance:\n"
    "faithfulness: 1 if every claim in the answer is stated in the "
    "source text, else 0\n"
    "relevance: 1 if the answer actually answers the question, else 0\n")


def judge_answer(question: str, answer_text: str, chunks: list[dict],
                 judge_fn=None) -> dict:
    """LLM-as-judge: one short completion, two 0/1 scores.

    Returns {"faithfulness": 0|1|None, "relevance": 0|1|None,
    "raw": str, "parsed": bool, "latency_s": float}. Unparseable
    output scores None on both — it is excluded from the eval
    means, never counted as 0 or 1. `judge_fn`, when given, is a
    `prompt -> str` callable used instead of the model (tests).
    """
    context = "\n\n".join(c["text"][:MAX_CONTEXT_CHARS_PER_CHUNK]
                          for c in chunks[:MAX_CONTEXT_CHUNKS])
    prompt = _JUDGE_PROMPT.format(question=question, context=context,
                                  answer=answer_text)
    if judge_fn is not None:
        start = time.monotonic()
        raw = judge_fn(prompt)
        latency = time.monotonic() - start
    else:
        raw, latency = complete(prompt, MAX_NEW_TOKENS_JUDGE)
    # Label-aware parsing: the 0.5B judge sometimes answers the
    # lines in the opposite order ("Relevance: 1 / Faithfulness: 0"),
    # so positional parsing would swap the two scores. Match each
    # label first; only fall back to "first two digits in order"
    # when a line has no label at all.
    text = raw or ""
    faith_m = re.search(r"faithfulness\s*:\s*([01])", text, re.IGNORECASE)
    rel_m = re.search(r"relevance\s*:\s*([01])", text, re.IGNORECASE)
    if faith_m is not None and rel_m is not None:
        parsed = True
        faithfulness, relevance = int(faith_m.group(1)), int(rel_m.group(1))
    else:
        digits = re.findall(r"(?<![0-9])[01](?![0-9])", text)
        parsed = len(digits) >= 2
        faithfulness = int(digits[0]) if parsed else None
        relevance = int(digits[1]) if parsed else None
    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "raw": raw,
        "parsed": parsed,
        "latency_s": round(latency, 3),
    }
