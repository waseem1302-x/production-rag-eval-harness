"""Build distillation training data with a Gemini teacher.

Step 1 of the distillation plan (docs/DISTILLATION_PLAN.md): generate
teacher outputs, verify every one through the production gate, and
write the verified train / held-out datasets. No training runs here.
No model is loaded here either: retrieval and verification are
deterministic and CPU-only.

Teacher: Gemini, first working model in this order (probe call):
gemini-3.5-flash, gemini-3.8-flash, gemini-3.5-flash-lite. The 3.5
and 3.8 flash models returned 503 high demand on 18 Nov 2026;
3.5-flash-lite answered. One model is used for the whole run and
its id is recorded in every file. Groq is NOT used: its stored key
returns 403. Do not retry it.

The API key is read from ~/.config/free-llm-keys.env inside this
script only. It is never printed, never logged, and never written
to any file. Teacher calls go to generativelanguage.googleapis.com.

Contamination rule (plan section 2): NEVER train on a golden
question verbatim. Golden-derived training uses paraphrases only.
Synthetic questions are checked against all 68 golden questions
and dropped when too similar. Held-out records are never trained on.

Verification gate (plan section 2.3): every teacher answer must
pass `app.verification.verify_answer` against the same 3 chunks the
prompt used, contain the expected phrase, and be 1-2 sentences.
Anything that fails is dropped, never fixed by hand. If any source
keeps below 70 percent, the run stops before finalising.

Resume: every teacher call is appended to
data/distillation/teacher_raw.jsonl at once. A re-run skips item
ids already in that file, so a quota pause loses no work.

Usage (from the repo root, with the repo venv):
  .venv/bin/python -m scripts.build_distillation_data --smoke
  .venv/bin/python -m scripts.build_distillation_data --sources variants synthetic
  .venv/bin/python -m scripts.build_distillation_data
The full run makes about 390 teacher calls. Run it detached
(nohup ... &) and read the log; it prints one line per item.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from app.answer import _named_docs, _sentences
from app.answer import answer as build_extractive_answer
from app.config import CHUNKS_FILE
from app.eval import load_golden
from app.generative import _JUDGE_PROMPT, build_generation_prompt
from app.retrieval import Retriever, _query_terms
from app.textnorm import norm_text, segment_sentences
from app.verification import _numbers, verify_answer

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "distillation"
RAW_FILE = OUT_DIR / "teacher_raw.jsonl"
GATE_FILE = OUT_DIR / "gate_log.jsonl"
TRAIN_FILE = OUT_DIR / "train.jsonl"
HELDOUT_FILE = OUT_DIR / "heldout.jsonl"
STATS_FILE = OUT_DIR / "stats.json"
KEY_FILE = Path.home() / ".config" / "free-llm-keys.env"

TEACHER_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash",
                  "gemini-3.5-flash-lite"]
TEACHER_DATE = "2026-11-18"
MAX_ATTEMPTS = 2
SLEEP_BETWEEN_CALLS = 8.0
RANDOM_SEED = 20261118

# Golden-derived variants: 14 generative misses x 3 paraphrases,
# 9 passing factual questions x 2, plus 5 more passing questions x 2
# held out (never trained on). The miss list is the measured
# generative_eval.json run, not the plan's stale list (q17 is fixed
# there, so it is in the held-out group, not the misses).
MISS_IDS = ["q01", "q03", "q05", "q07", "q10", "q11", "q12", "q13",
            "q14", "q18", "q24", "q29", "q30", "q31"]
PASS_IDS = ["q06", "q08", "q09", "q19", "q22", "q23", "q25", "q26", "q28"]
HELDOUT_VARIANT_IDS = ["q02", "q04", "q15", "q16", "q17"]

# Abstention questions. Each carries the phrases that must not
# appear anywhere in the corpus; a question is dropped if any
# phrase is found. Absent scholarships are checked against the 19
# doc slugs as well. Target answer is always exactly
# "Not found in the corpus."
ABSENT_SCHOLARSHIPS = [
    ("What is the deadline for the Australia Awards Scholarship?",
     ["australia awards"]),
    ("How much does the Clarendon Fund Scholarship pay per month?",
     ["clarendon"]),
    ("Who can apply for the Lester B. Pearson Scholarship?",
     ["lester b. pearson", "pearson scholarship"]),
    ("What does the Pierre Elliott Trudeau Scholarship cover?",
     ["trudeau"]),
    ("What is the deadline for the Türkiye Burslari Scholarship?",
     ["turkiye burslari", "türkiye burslari"]),
    ("How much is the KAIST International Student Scholarship worth?",
     ["kaist"]),
    ("Who can apply for the NUS Research Scholarship?",
     ["nus research scholarship"]),
    ("What is the monthly stipend of the ETH Excellence Scholarship?",
     ["eth excellence"]),
    ("What degrees does the SINGA Scholarship fund?",
     ["singa"]),
    ("Who can apply for the ANSO Scholarship?",
     ["anso scholarship"]),
    ("What is the deadline for the CSC Chinese Government Scholarship?",
     ["chinese government scholarship"]),
    ("How much does the Brunei Darussalam Scholarship pay?",
     ["brunei"]),
    ("Who can apply for the Oxford-Weidenfeld Scholarship?",
     ["weidenfeld"]),
    ("What does the Rotary Global Grant fund?",
     ["rotary global grant"]),
    ("What is the deadline for the Aga Khan Foundation Scholarship?",
     ["aga khan"]),
    ("Who can apply for the Inlaks Shivdasani Scholarship?",
     ["inlaks"]),
    ("What is the deadline for the JN Tata Endowment loan scholarship?",
     ["jn tata"]),
    ("Who can apply for the Heinrich Böll Scholarship?",
     ["heinrich böll", "heinrich boll"]),
    ("What does the Konrad Adenauer Scholarship cover?",
     ["konrad adenauer"]),
    ("How much is the Friedrich Ebert Scholarship worth?",
     ["friedrich ebert"]),
]
UNSTATED_FACTS = [
    ("What is the acceptance rate for the Chevening Scholarship?",
     ["acceptance rate"]),
    ("What GPA cutoff does the Gates Cambridge Scholarship require?",
     ["gpa"]),
    ("What questions are asked in the Rhodes Scholarship interview?",
     ["interview questions"]),
    ("How many applicants were rejected by the DAAD last year?",
     ["rejected"]),
    ("What is the average GRE score of Schwarzman Scholars?",
     ["gre score"]),
    ("What is the acceptance rate for the Fulbright Foreign Student Program?",
     ["acceptance rate"]),
    ("What class rank does the Knight-Hennessy Scholarship require?",
     ["class rank"]),
    ("What GPA does the MEXT Scholarship require?",
     ["gpa"]),
    ("What questions are asked in the Eiffel Scholarship interview?",
     ["interview questions"]),
    ("What is the success rate for Vanier Scholarship applicants?",
     ["success rate"]),
    ("How many applicants apply for the Manaaki New Zealand Scholarship each year?",
     ["number of applicants"]),
    ("What questions are asked in the Commonwealth Shared Scholarship interview?",
     ["interview questions"]),
    ("What is the acceptance rate for the SI Scholarship for Global Professionals?",
     ["acceptance rate"]),
    ("What GPA cutoff does the Marshall Scholarship require?",
     ["gpa"]),
    ("What is the rejection rate for Erasmus Mundus Joint Masters?",
     ["rejection rate"]),
    ("What is the acceptance rate for the Hong Kong PhD Fellowship Scheme?",
     ["acceptance rate"]),
    ("What questions are asked in the RTP Scholarship interview?",
     ["interview questions"]),
    ("What is the acceptance rate for the Pioneering Women in STEM Scholarship?",
     ["acceptance rate"]),
    ("How many recommendation letters does the Chevening Scholarship require?",
     ["recommendation letters"]),
    ("How many applicants were rejected by the Gates Cambridge Scholarship last year?",
     ["rejected"]),
]
DOC_KEYWORDS = ["chevening", "rhodes", "gates", "daad", "fulbright",
                "mext", "eiffel", "vanier", "manaaki", "erasmus",
                "schwarzman", "marshall", "knight", "rtp", "hkpfs",
                "commonwealth", "sweden", "pioneering"]

JUDGE_RIGHT_IDS = [f"q{i:02d}" for i in range(1, 27)]


def load_key() -> str:
    """Read the Gemini key from the env file. Never print it."""
    for line in KEY_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"").strip("'")
    raise SystemExit("GEMINI_API_KEY not found in ~/.config/free-llm-keys.env")


class Teacher:
    """One Gemini model, chosen by probe, used for the whole run."""

    def __init__(self, key: str):
        self.key = key
        self.model = None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.raw_path = RAW_FILE

    def probe(self) -> str:
        for model in TEACHER_MODELS:
            try:
                text, _ = self._call_once(
                    model, "Reply with exactly: OK", max_tokens=16)
                if text.strip() == "OK":
                    self.model = model
                    print(f"teacher: {model}", flush=True)
                    return model
            except urllib.error.HTTPError as exc:
                print(f"teacher probe {model}: HTTP {exc.code}", flush=True)
            except Exception as exc:  # noqa: BLE001 - probe must not crash
                print(f"teacher probe {model}: {type(exc).__name__}",
                      flush=True)
        raise SystemExit("no Gemini teacher model answered the probe")

    def _call_once(self, model: str, prompt: str,
                   max_tokens: int) -> tuple[str, dict]:
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2,
                                 "maxOutputTokens": max_tokens},
        }).encode()
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent",
            data=body,
            headers={"x-goog-api-key": self.key,
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        candidates = data.get("candidates") or []
        if not candidates or "content" not in candidates[0]:
            raise ValueError("no candidates in Gemini response")
        parts = candidates[0]["content"].get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            raise ValueError("empty text in Gemini response")
        usage = data.get("usageMetadata", {})
        return text, usage

    def call(self, item_id: str, kind: str, prompt: str,
             max_tokens: int = 512) -> str | None:
        """One teacher call with retries. Returns the raw text, or
        None when the model stays unavailable. Every completed call
        is appended to teacher_raw.jsonl at once.

        On a 429 rate limit, the run does NOT stop: it waits out
        the retry delay the API returns (default 30 s, capped at
        120 s) and tries the same call again, up to 5 waits. The
        free tier recovered within minutes in the first full run,
        so the limit is a short window, not a daily cap. Only after
        5 waits does the run stop, and a re-run resumes from
        teacher_raw.jsonl, so nothing is lost."""
        delay = 2.0
        last_error = None
        waits = 0
        for attempt in range(1, 6):
            try:
                text, usage = self._call_once(self.model, prompt, max_tokens)
                self.calls += 1
                self.prompt_tokens += usage.get("promptTokenCount", 0)
                self.completion_tokens += usage.get("candidatesTokenCount", 0)
                with self.raw_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "item_id": item_id, "kind": kind,
                        "teacher": f"gemini/{self.model}",
                        "teacher_date": TEACHER_DATE,
                        "prompt": prompt, "raw": text,
                        "usage": usage,
                    }) + "\n")
                time.sleep(SLEEP_BETWEEN_CALLS)
                return text
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code == 429:
                    waits += 1
                    if waits > 5:
                        print(f"  {item_id}: quota 429 after 5 waits, "
                              "stopping run", flush=True)
                        raise SystemExit(
                            "Gemini quota still hit after 5 waits. "
                            "Re-run later; teacher_raw.jsonl already "
                            "holds every completed call, so nothing "
                            "is lost.")
                    wait_s = self._retry_delay(exc)
                    print(f"  {item_id}: quota 429, waiting {wait_s}s "
                          f"(wait {waits}/5)", flush=True)
                    time.sleep(wait_s)
                    continue
                print(f"  {item_id}: HTTP {exc.code} attempt {attempt}/5",
                      flush=True)
                time.sleep(delay)
                delay *= 2
            except Exception as exc:  # noqa: BLE001 - retry, then give up
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"  {item_id}: {last_error} attempt {attempt}/5",
                      flush=True)
                time.sleep(delay)
                delay *= 2
        print(f"  {item_id}: FAILED after 5 attempts ({last_error})",
              flush=True)
        return None

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError) -> float:
        """Seconds to wait after a 429, from the API's RetryInfo
        detail when present, else the Retry-After header, else 30.
        Capped at 120."""
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            for detail in body.get("error", {}).get("details", []):
                retry = detail.get("retryDelay", "")
                if retry.endswith("s"):
                    return min(120.0, float(retry[:-1]))
        except (ValueError, TypeError, AttributeError):
            pass
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                return min(120.0, float(header))
            except ValueError:
                pass
        return 30.0


# ---- pure gate helpers (unit-tested in tests/test_distillation_gate.py) ----

def count_sentences(text: str) -> int:
    return len([s for s in segment_sentences(text) if s.strip()])


def contains_phrase(text: str, phrase: str) -> bool:
    """Case-insensitive substring check on normalised text, the
    same comparison the eval harness makes (it lowercases both
    sides before checking)."""
    return norm_text(phrase.lower()) in norm_text(text.lower())


def no_new_numbers(answer: str, chunks: list[dict]) -> bool:
    sent_numbers = _numbers(answer)
    if not sent_numbers:
        return True
    all_chunk_numbers: set[str] = set()
    for chunk in chunks:
        all_chunk_numbers |= _numbers(chunk["text"])
    return not (sent_numbers - all_chunk_numbers)


def parse_judge_lines(text: str) -> tuple[int, int] | None:
    """Parse exactly two lines, faithfulness then relevance, 0 or 1
    each. Same label-aware parsing as app.generative.judge_answer."""
    faith_m = re.search(r"faithfulness\s*:\s*([01])", text, re.IGNORECASE)
    rel_m = re.search(r"relevance\s*:\s*([01])", text, re.IGNORECASE)
    if faith_m is not None and rel_m is not None:
        return int(faith_m.group(1)), int(rel_m.group(1))
    return None


def key_fact_covered(answer: str, key_fact: str) -> bool:
    """Does the answer carry the key fact? Not a verbatim check:
    the teacher naturally rephrases ("September/October 2027"
    becomes "September or October 2027"), so the check is content
    stems plus exact numbers. At least 60 percent of the key
    fact's content stems must appear in the answer, and every
    number in the key fact must appear in the answer. The bar is
    0.6, not 0.7: synonyms the stemmer cannot bridge ("at least"
    vs "minimum", "long" vs "duration", "consist" vs "comprise")
    cost real coverage on correct answers, measured in the smoke
    run."""
    from app.verification import _content_stems

    fact_stems = _content_stems(key_fact)
    if not fact_stems:
        return False
    answer_stems = _content_stems(answer)
    coverage = len(fact_stems & answer_stems) / len(fact_stems)
    fact_numbers = _numbers(key_fact)
    if fact_numbers and not fact_numbers <= _numbers(answer):
        return False
    return coverage >= 0.6


def jaccard_similarity(a: str, b: str) -> float:
    """Content-token Jaccard between two questions. 1.0 is identical."""
    sa, sb = _query_terms(a), _query_terms(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_verbatim_golden(question: str, golden: list[dict]) -> bool:
    nq = norm_text(question)
    return any(norm_text(g["question"]) == nq for g in golden)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def faithfulness_proxy(answer_text: str, citations: list[dict]) -> int:
    """1 when every answer sentence is verbatim in the citations,
    else 0. Same proxy as app/eval.py."""
    sents = _sentences(answer_text)
    if not sents:
        return 0
    cited = norm_text(" ".join(c["text"] for c in citations))
    return 1 if all(norm_text(s) in cited for s in sents) else 0


class Builder:
    """Builds every item, gates it, and writes the datasets."""

    def __init__(self, teacher: Teacher, retriever: Retriever,
                 chunks: list[dict], golden: list[dict],
                 gen_eval: dict, smoke: bool = False):
        self.teacher = teacher
        self.retriever = retriever
        self.chunks = chunks
        self.golden = {g["id"]: g for g in golden}
        self.golden_list = golden
        self.gen_eval = gen_eval
        self.smoke = smoke
        self.done_ids = self._load_done_ids()
        self.records: list[dict] = []
        self.heldout: list[dict] = []
        self.stats: Counter = Counter()
        self.drop_reasons: Counter = Counter()
        self.all_chunk_text = norm_text(" ".join(c["text"] for c in chunks))

    def _load_done_ids(self) -> set[str]:
        if not RAW_FILE.exists():
            return set()
        return {json.loads(line)["item_id"] for line in
                RAW_FILE.read_text().splitlines() if line.strip()}

    def log_gate(self, item_id: str, source: str, kept: bool,
                 reasons: list[str]) -> None:
        with GATE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"item_id": item_id, "source": source,
                                 "kept": kept, "reasons": reasons}) + "\n")
        self.stats[f"{source}:generated"] += 1
        if kept:
            self.stats[f"{source}:kept"] += 1
        else:
            self.stats[f"{source}:dropped"] += 1
            for reason in reasons:
                self.drop_reasons[f"{source}:{reason}"] += 1

    def top_chunks(self, question: str) -> list[dict]:
        """Top-3 chunks exactly as app.generative.generate_answer
        picks them: hybrid top-5, then the named-doc filter."""
        retrieved = self.retriever.search(question, top_k=5)
        named = _named_docs(question)
        if named:
            named_chunks = [c for c in retrieved if c["doc"] in named][:3]
            if named_chunks:
                return named_chunks
        return retrieved[:3]

    def make_record(self, question_id: str, source: str, prompt: str,
                    answer_text: str, chunks: list[dict],
                    expected_phrase: str | None,
                    heldout: bool = False,
                    extra: dict | None = None) -> dict:
        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_text},
            ],
            "source": source,
            "question_id": question_id,
            "evidence_chunks": [c["chunk_id"] for c in chunks],
            "teacher": f"gemini/{self.teacher.model}",
            "teacher_date": TEACHER_DATE,
            "verified": True,
            "expected_phrase": expected_phrase,
            "verification": {"kept": [s for s in
                                      segment_sentences(answer_text)
                                      if s.strip()],
                             "dropped": []},
        }
        if extra:
            record.update(extra)
        (self.heldout if heldout else self.records).append(record)
        return record

    # ---- golden-derived variants ----

    def paraphrases(self, qid: str, n: int) -> list[str]:
        """Ask the teacher for n paraphrases of one golden question."""
        item_id = f"para-{qid}"
        if item_id in self.done_ids:
            # A re-run reuses the saved raw output, but a truncated
            # first attempt parses to nothing, so try every saved
            # line and fall through to a fresh call when none parses.
            if RAW_FILE.exists():
                for line in RAW_FILE.read_text().splitlines():
                    row = json.loads(line)
                    if row["item_id"] == item_id:
                        parsed = self._parse_paraphrases(row["raw"], qid, n)
                        if parsed:
                            return parsed
                return self._fresh_paraphrases(item_id, qid, n)
            return []
        return self._fresh_paraphrases(item_id, qid, n)

    def _fresh_paraphrases(self, item_id: str, qid: str, n: int) -> list[str]:
        question = self.golden[qid]["question"]
        expected = self.golden[qid]["expected_answer"]
        prompt = (
            f"Rewrite the following scholarship question in {n} "
            "different ways. Each paraphrase must ask exactly the "
            "same thing, in different words, and must still be "
            "answered by this exact phrase: "
            f"'{expected}'. Do not answer it. Do not change what is "
            "being asked. Reply with a JSON array "
            f"of {n} strings, nothing else.\n\nQuestion: {question}")
        for _ in range(MAX_ATTEMPTS):
            raw = self.teacher.call(item_id, "paraphrase", prompt,
                                    max_tokens=512)
            if raw is None:
                return []
            parsed = self._parse_paraphrases(raw, qid, n)
            if parsed:
                return parsed
            print(f"  para-{qid}: unparseable paraphrase list, retrying",
                  flush=True)
        print(f"  para-{qid}: unparseable paraphrase list", flush=True)
        return []

    def _parse_paraphrases(self, raw: str, qid: str, n: int) -> list[str]:
        original = self.golden[qid]["question"]
        try:
            items = json.loads(strip_code_fences(raw))
        except (json.JSONDecodeError, ValueError):
            print(f"  para-{qid}: unparseable paraphrase list", flush=True)
            return []
        out: list[str] = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item or norm_text(item) == norm_text(original):
                continue
            if jaccard_similarity(item, original) >= 0.85:
                continue
            if any(jaccard_similarity(item, kept) >= 0.9 for kept in out):
                continue
            out.append(item)
        return out[:n]

    def _last_raw(self, item_id: str) -> dict | None:
        """Last teacher output for an item (a retry appends a new
        line, so the last line is the newest attempt)."""
        found = None
        if RAW_FILE.exists():
            for line in RAW_FILE.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["item_id"] == item_id:
                    found = row
        return found

    def _gate_entries(self, item_id: str) -> list[dict]:
        if not GATE_FILE.exists():
            return []
        return [json.loads(line) for line in
                GATE_FILE.read_text().splitlines()
                if line.strip() and json.loads(line)["item_id"] == item_id]

    def _attempts_done(self, item_id: str) -> int:
        return len(self._gate_entries(item_id))

    def _already_kept(self, item_id: str) -> bool:
        entries = self._gate_entries(item_id)
        return bool(entries) and entries[-1]["kept"]

    def variant_answers(self, qid: str, paraphrase_list: list[str],
                        heldout: bool = False) -> None:
        expected = self.golden[qid]["expected_answer"]
        for i, paraphrase in enumerate(paraphrase_list, start=1):
            item_id = f"{qid}-v{i}" + ("-h" if heldout else "")
            if self._already_kept(item_id):
                self._regenerate_variant(item_id, qid, paraphrase,
                                         expected, heldout)
                continue
            chunks = self.top_chunks(paraphrase)
            prompt = build_generation_prompt(paraphrase, chunks)
            source = "golden-variant-heldout" if heldout else "golden-variant"
            self._answer_with_gate(item_id, source, prompt, chunks,
                                   paraphrase, expected, heldout,
                                   question_id=item_id)

    def _regenerate_variant(self, item_id: str, qid: str, paraphrase: str,
                            expected: str, heldout: bool) -> None:
        """A re-run re-gates the saved raw output, so resume is
        exact and costs no new teacher call."""
        row = self._last_raw(item_id)
        if row is None:
            return
        chunks = self.top_chunks(paraphrase)
        prompt = build_generation_prompt(paraphrase, chunks)
        source = ("golden-variant-heldout" if heldout
                  else "golden-variant")
        self._gate_answer(item_id, source, prompt, chunks,
                          row["raw"], expected, heldout,
                          question_id=item_id)

    def _answer_with_gate(self, item_id: str, source: str, prompt: str,
                          chunks: list[dict], question: str,
                          expected: str, heldout: bool,
                          question_id: str) -> None:
        while self._attempts_done(item_id) < MAX_ATTEMPTS:
            raw = self.teacher.call(item_id, "answer", prompt,
                                    max_tokens=256)
            if raw is None:
                return
            if self._gate_answer(item_id, source, prompt, chunks, raw,
                                 expected, heldout,
                                 question_id=question_id):
                return
        print(f"  {item_id}: dropped after {MAX_ATTEMPTS} attempts",
              flush=True)

    def _gate_answer(self, item_id: str, source: str, prompt: str,
                     chunks: list[dict], raw: str, expected: str,
                     heldout: bool, question_id: str) -> bool:
        """Production semantics: verify_answer keeps the grounded
        sentences and drops the rest, and the kept sentences are
        the answer. The gate therefore checks the KEPT answer, not
        the raw draft: it must contain the expected phrase, be 1-2
        sentences, and add no new number. An answer with nothing
        kept is dropped."""
        answer_text = strip_code_fences(raw).strip()
        check = verify_answer(answer_text, chunks)
        if not check["kept"]:
            self.log_gate(item_id, source, False,
                          ["verification_dropped"])
            return False
        kept_answer = check["answer"]
        reasons: list[str] = []
        if not (1 <= count_sentences(kept_answer) <= 2):
            reasons.append("sentence_count")
        if expected and not contains_phrase(kept_answer, expected):
            reasons.append("missing_expected_phrase")
        if not no_new_numbers(kept_answer, chunks):
            reasons.append("new_number")
        if reasons:
            self.log_gate(item_id, source, False, reasons)
            return False
        self.log_gate(item_id, source, True, [])
        self.make_record(question_id, source, prompt, kept_answer,
                         chunks, expected, heldout=heldout)
        return True

    def run_variants(self) -> None:
        groups = [(MISS_IDS, 3, False), (PASS_IDS, 2, False),
                  (HELDOUT_VARIANT_IDS, 2, True)]
        for ids, n, heldout in groups:
            for qid in ids:
                if self.smoke and qid != MISS_IDS[0]:
                    continue
                paras = self.paraphrases(qid, n)
                if not paras:
                    print(f"  {qid}: no paraphrases", flush=True)
                    continue
                self.variant_answers(qid, paras, heldout=heldout)
        self._check_keep_rate("golden-variant")

    # ---- synthetic QA from corpus sentences ----

    def sample_sentences(self, total: int = 250) -> list[dict]:
        """Stratified sample of pickable sentences: 60 percent fact
        (has a digit), 25 percent eligibility-shaped, 15 percent
        definition. Capped at 15 sentences per document so Rhodes
        does not dominate. Deterministic seed."""
        elig_words = ["eligib", "citizen", "nationality", "degree",
                      "level", "apply", "applicant", "countr"]
        by_class: dict[str, list[dict]] = {"fact": [], "elig": [],
                                           "def": []}
        seen: set[str] = set()
        junk = ("date modified", "last updated", "all rights reserved",
                "copyright", "pursue your dream", "your dream",
                "apply now", "amazing opportunity", "life-changing",
                "do not miss")
        for chunk in self.chunks:
            for sent in segment_sentences(chunk["text"]):
                sent = sent.strip()
                if not (40 <= len(sent) <= 400) or sent in seen:
                    continue
                if any(j in sent.lower() for j in junk):
                    continue
                seen.add(sent)
                low = sent.lower()
                if re.search(r"\d", sent):
                    cls = "fact"
                elif any(w in low for w in elig_words):
                    cls = "elig"
                else:
                    cls = "def"
                by_class[cls].append({"sentence": sent,
                                      "doc": chunk["doc"],
                                      "chunk_id": chunk["chunk_id"]})
        rng = random.Random(RANDOM_SEED)
        for cls in by_class:
            rng.shuffle(by_class[cls])
        targets = {"fact": 150, "elig": 62, "def": 38}
        picked: list[dict] = []
        per_doc: Counter = Counter()
        for cls, target in targets.items():
            for item in by_class[cls]:
                if len(picked) >= total:
                    break
                if len([p for p in picked if p["cls"] == cls]) >= target:
                    break
                if per_doc[item["doc"]] >= 15:
                    continue
                item["cls"] = cls
                picked.append(item)
                per_doc[item["doc"]] += 1
        # Fill any shortfall from the other classes, same cap.
        if len(picked) < total:
            rest: list[dict] = []
            for cls in ("def", "elig", "fact"):
                rest.extend(by_class[cls])
            rng.shuffle(rest)
            picked_ids = {(p["sentence"]) for p in picked}
            for item in rest:
                if len(picked) >= total:
                    break
                if item["sentence"] in picked_ids:
                    continue
                if per_doc[item["doc"]] >= 15:
                    continue
                item.setdefault("cls", "def")
                picked.append(item)
                per_doc[item["doc"]] += 1
        print(f"sampled sentences: {len(picked)} "
              f"(fact={sum(1 for p in picked if p['cls'] == 'fact')}, "
              f"elig={sum(1 for p in picked if p['cls'] == 'elig')}, "
              f"def={sum(1 for p in picked if p['cls'] == 'def')})",
              flush=True)
        return picked

    def run_synthetic(self) -> None:
        sampled = self.sample_sentences(250 if not self.smoke else 6)
        for i, item in enumerate(sampled):
            item_id = f"syn-{item['doc']}-{i:03d}"
            # Every 5th sampled item is held out (never trained on).
            # The sample order is shuffled within each class, so the
            # held-out set keeps the class and doc mix.
            heldout = (i % 5 == 0)
            if self.smoke and i >= 6:
                break
            if item_id in self.done_ids:
                self._regenerate_synthetic(item_id, item, heldout)
                continue
            prompt = (
                "You are given one sentence from a scholarship "
                "information page.\n\n"
                f"Sentence: {item['sentence']}\nSource: {item['doc']}\n\n"
                "Write:\n"
                '1. "question": one question that this sentence '
                "answers. The question must be answerable from the "
                "sentence alone.\n"
                '2. "key_fact": a short phrase copied exactly from '
                "the sentence that answers the question.\n"
                '3. "answer": a 1-2 sentence answer using ONLY the '
                "sentence. Do not add any fact not stated in the "
                "sentence.\n\n"
                "Reply with a JSON object only: "
                '{"question": ..., "key_fact": ..., "answer": ...}')
            raw = self.teacher.call(item_id, "synthetic", prompt,
                                    max_tokens=512)
            if raw is None:
                continue
            self._gate_synthetic(item_id, item, raw, heldout)
        self._check_keep_rate("synthetic-qa")

    def _regenerate_synthetic(self, item_id: str, item: dict,
                              heldout: bool) -> None:
        row = self._last_raw(item_id)
        if row is not None:
            self._gate_synthetic(item_id, item, row["raw"], heldout)

    def _gate_synthetic(self, item_id: str, item: dict, raw: str,
                        heldout: bool) -> None:
        try:
            data = json.loads(strip_code_fences(raw))
        except (json.JSONDecodeError, ValueError):
            self.log_gate(item_id, "synthetic-qa", False, ["unparseable"])
            return
        question = str(data.get("question", "")).strip()
        key_fact = str(data.get("key_fact", "")).strip()
        answer_text = str(data.get("answer", "")).strip()
        reasons: list[str] = []
        chunks: list[dict] = []
        if not question or not key_fact or not answer_text:
            reasons.append("missing_field")
        else:
            chunks = self.top_chunks(question)
            if is_verbatim_golden(question, self.golden_list):
                reasons.append("golden_verbatim")
            elif max((jaccard_similarity(question, g["question"])
                      for g in self.golden_list), default=0.0) >= 0.85:
                reasons.append("too_close_to_golden")
            elif not contains_phrase(item["sentence"], key_fact):
                reasons.append("key_fact_not_in_sentence")
            elif not key_fact_covered(answer_text, key_fact):
                reasons.append("key_fact_not_in_answer")
            elif not (1 <= count_sentences(answer_text) <= 2):
                reasons.append("sentence_count")
            elif not no_new_numbers(answer_text, chunks):
                reasons.append("new_number")
        if reasons:
            self.log_gate(item_id, "synthetic-qa", False, reasons)
            return
        check = verify_answer(answer_text, chunks, question=question)
        if not check["kept"]:
            self.log_gate(item_id, "synthetic-qa", False,
                          ["verification_dropped"])
            return
        kept_answer = check["answer"]
        if not (1 <= count_sentences(kept_answer) <= 2):
            self.log_gate(item_id, "synthetic-qa", False,
                          ["sentence_count"])
            return
        if not key_fact_covered(kept_answer, key_fact):
            self.log_gate(item_id, "synthetic-qa", False,
                          ["key_fact_not_in_answer"])
            return
        self.log_gate(item_id, "synthetic-qa", True, [])
        prompt = build_generation_prompt(question, chunks)
        self.make_record(item_id, "synthetic-qa", prompt,
                         kept_answer, chunks, key_fact,
                         heldout=heldout)

    # ---- abstention ----

    def run_abstention(self) -> None:
        items = ([(q, p, "absent") for q, p in ABSENT_SCHOLARSHIPS]
                 + [(q, p, "unstated") for q, p in UNSTATED_FACTS])
        for i, (question, phrases, kind) in enumerate(items):
            item_id = f"abs-{i:02d}"
            if self.smoke and i >= 4:
                break
            reasons: list[str] = []
            for phrase in phrases:
                if re.search(r"\b" + re.escape(norm_text(phrase)) + r"\b",
                             self.all_chunk_text):
                    reasons.append("phrase_in_corpus")
                    break
            for keyword in DOC_KEYWORDS:
                if re.search(r"\b" + re.escape(keyword) + r"\b",
                             norm_text(question)):
                    reasons.append("names_in_corpus_doc")
                    break
            if reasons:
                self.log_gate(item_id, "abstention", False, reasons)
                continue
            chunks = self.top_chunks(question)
            prompt = build_generation_prompt(question, chunks)
            extractive = build_extractive_answer(question,
                                                 self.retriever.search(
                                                     question, top_k=5))
            self.log_gate(item_id, "abstention", True, [])
            self.make_record(
                item_id, "abstention", prompt, "Not found in the corpus.",
                chunks, "Not found in the corpus.")
            print(f"  {item_id} ({kind}): kept "
                  f"(extractive found={extractive['found']})", flush=True)
        self._check_keep_rate("abstention")

    # ---- judge ----

    def run_judge(self) -> None:
        gen_rows = {r["id"]: r for r in
                    self.gen_eval["generative"]["per_question"]}
        ext_rows = {r["id"]: r for r in
                    self.gen_eval["extractive"]["per_question"]}
        for qid in MISS_IDS:
            if self.smoke and qid != MISS_IDS[0]:
                continue
            row = gen_rows[qid]
            self._judge_item(f"judge-w-{qid}", "judge",
                             row["question"], row["answer"],
                             self._judge_citations(row),
                             self.golden[qid]["expected_answer"])
        for qid in JUDGE_RIGHT_IDS:
            if self.smoke:
                continue
            row = ext_rows[qid]
            retrieved = self.retriever.search(row["question"], top_k=5)
            result = build_extractive_answer(row["question"], retrieved)
            self._judge_item(f"judge-r-{qid}", "judge",
                             row["question"], result["answer"],
                             result["citations"],
                             self.golden[qid]["expected_answer"])
        self._check_keep_rate("judge")

    def _judge_citations(self, row: dict) -> list[dict]:
        retrieved = self.retriever.search(row["question"], top_k=5)
        if row.get("answer_mode") == "generative":
            named = _named_docs(row["question"])
            if named:
                named_chunks = [c for c in retrieved
                                if c["doc"] in named][:3]
                if named_chunks:
                    return named_chunks
            return retrieved[:3]
        result = build_extractive_answer(row["question"], retrieved)
        return result["citations"]

    def _judge_item(self, item_id: str, source: str, question: str,
                    answer_text: str, chunks: list[dict],
                    expected: str) -> None:
        context = "\n\n".join(c["text"][:900] for c in chunks[:3])
        prompt = _JUDGE_PROMPT.format(question=question, context=context,
                                      answer=answer_text)
        target_faith = faithfulness_proxy(answer_text, chunks)
        target_rel = 1 if contains_phrase(answer_text, expected) else 0
        target = (target_faith, target_rel)
        if item_id in self.done_ids:
            row = self._last_raw(item_id)
            if row is not None and self._already_kept(item_id):
                self._gate_judge(item_id, source, prompt, chunks,
                                 row["raw"], target)
                return
        while self._attempts_done(item_id) < MAX_ATTEMPTS:
            raw = self.teacher.call(item_id, "judge", prompt,
                                    max_tokens=64)
            if raw is None:
                return
            if self._gate_judge(item_id, source, prompt, chunks, raw,
                                target):
                return

    def _gate_judge(self, item_id: str, source: str, prompt: str,
                    chunks: list[dict], raw: str,
                    target: tuple[int, int]) -> bool:
        parsed = parse_judge_lines(raw)
        if parsed is None:
            self.log_gate(item_id, source, False, ["unparseable"])
            return False
        if parsed != target:
            self.log_gate(item_id, source, False,
                          ["disagrees_with_proxy"])
            return False
        self.log_gate(item_id, source, True, [])
        answer_text = (f"faithfulness: {target[0]}\n"
                       f"relevance: {target[1]}")
        self.make_record(item_id, source, prompt, answer_text, chunks,
                         None,
                         extra={"expected_judge":
                                {"faithfulness": target[0],
                                 "relevance": target[1]}})
        return True

    # ---- keep rate and finalize ----

    def _source_items(self, source: str) -> tuple[set[str], set[str]]:
        """Distinct item ids per source from the gate log (cumulative
        across resumed runs), and the subset whose last entry kept."""
        items: dict[str, bool] = {}
        if GATE_FILE.exists():
            for line in GATE_FILE.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["source"] == source:
                    items[row["item_id"]] = row["kept"]
        kept = {i for i, k in items.items() if k}
        return set(items), kept

    def _check_keep_rate(self, source: str) -> None:
        items, kept = self._source_items(source)
        rate = len(kept) / len(items) if items else 0.0
        print(f"keep rate {source}: {len(kept)}/{len(items)} = {rate:.2f}",
              flush=True)
        # The 0.70 rule is for a full source. On a smoke sample of
        # fewer than 10 items one drop swings the rate by 10 points
        # or more, so report it but do not stop the run.
        if len(items) < 10:
            print(f"  (fewer than 10 items: keep-rate stop skipped)",
                  flush=True)
            return
        if items and rate < 0.70:
            raise SystemExit(
                f"keep rate for {source} is {rate:.2f}, below 0.70. "
                "Stopping before finalising. Inspect the teacher "
                "prompts and gate log first.")

    def finalize(self) -> dict:
        """Write train / held-out files from this run's records and
        return cumulative stats read back from the log files.

        Merged with any records already on disk, keyed by
        question_id, so a resumed run never wipes an earlier run's
        verified records."""
        merged_train: dict[str, dict] = {}
        merged_held: dict[str, dict] = {}
        for path, merged in ((TRAIN_FILE, merged_train),
                             (HELDOUT_FILE, merged_held)):
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        merged[row["question_id"]] = row
        for record in self.records:
            merged_held.pop(record["question_id"], None)
            merged_train[record["question_id"]] = record
        for record in self.heldout:
            merged_train.pop(record["question_id"], None)
            merged_held[record["question_id"]] = record
        train = list(merged_train.values())
        heldout = list(merged_held.values())
        TRAIN_FILE.write_text("".join(json.dumps(r) + "\n" for r in train))
        HELDOUT_FILE.write_text("".join(json.dumps(r) + "\n"
                                        for r in heldout))
        calls = prompt_tokens = completion_tokens = 0
        if RAW_FILE.exists():
            for line in RAW_FILE.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                calls += 1
                usage = row.get("usage", {})
                prompt_tokens += usage.get("promptTokenCount", 0)
                completion_tokens += usage.get("candidatesTokenCount", 0)
        drop_reasons: Counter = Counter()
        if GATE_FILE.exists():
            for line in GATE_FILE.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row["kept"]:
                    for reason in row["reasons"]:
                        drop_reasons[f"{row['source']}:{reason}"] += 1
        per_source = {}
        for src in ("golden-variant", "golden-variant-heldout",
                    "synthetic-qa", "abstention", "judge"):
            items, kept = self._source_items(src)
            per_source[src] = {"generated": len(items),
                               "kept": len(kept),
                               "dropped": len(items) - len(kept)}
        by_source = Counter(r["source"] for r in train)
        by_source_held = Counter(r["source"] for r in heldout)
        stats = {
            "teacher": f"gemini/{self.teacher.model}",
            "teacher_date": TEACHER_DATE,
            "teacher_calls": calls,
            "teacher_prompt_tokens": prompt_tokens,
            "teacher_completion_tokens": completion_tokens,
            "generated_kept_dropped": per_source,
            "drop_reasons": dict(drop_reasons),
            "train_records": len(train),
            "train_by_source": dict(by_source),
            "heldout_records": len(heldout),
            "heldout_by_source": dict(by_source_held),
        }
        STATS_FILE.write_text(json.dumps(stats, indent=2))
        return stats


def set_out_dir(path: Path) -> None:
    """Point all output files at another directory (smoke runs)."""
    global OUT_DIR, RAW_FILE, GATE_FILE, TRAIN_FILE, HELDOUT_FILE, STATS_FILE
    OUT_DIR = path
    RAW_FILE = OUT_DIR / "teacher_raw.jsonl"
    GATE_FILE = OUT_DIR / "gate_log.jsonl"
    TRAIN_FILE = OUT_DIR / "train.jsonl"
    HELDOUT_FILE = OUT_DIR / "heldout.jsonl"
    STATS_FILE = OUT_DIR / "stats.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_dataset_readme(stats: dict) -> None:
    """Dataset card with verified numbers only, from stats.json."""
    gkd = stats["generated_kept_dropped"]
    lines = [
        "# Distillation dataset",
        "",
        f"Teacher: {stats['teacher']}, {stats['teacher_date']}. "
        "Groq was not used: its stored key returns 403.",
        "",
        "## Counts (verified, from stats.json)",
        "",
        "| Source | Generated | Kept | Dropped |",
        "|---|---|---|---|",
    ]
    for src in ("golden-variant", "golden-variant-heldout",
                "synthetic-qa", "abstention", "judge"):
        row = gkd[src]
        lines.append(f"| {src} | {row['generated']} | {row['kept']} "
                     f"| {row['dropped']} |")
    lines += [
        "",
        f"Train records: {stats['train_records']} "
        f"({stats['train_by_source']}).",
        "",
        f"Held-out records: {stats['heldout_records']} "
        f"({stats['heldout_by_source']}). Held-out records are never "
        "trained on. They are the generalisation check.",
        "",
        f"Teacher calls: {stats['teacher_calls']}. Teacher tokens: "
        f"{stats['teacher_prompt_tokens']} prompt, "
        f"{stats['teacher_completion_tokens']} completion. "
        "Teacher cost: $0 (Gemini free tier).",
        "",
        "## Files",
        "",
        "- `teacher_raw.jsonl`: every teacher output, before filtering. "
        "Audit trail, never trained on directly.",
        "- `train.jsonl`: verified records only.",
        "- `heldout.jsonl`: verified records, never trained on.",
        "- `gate_log.jsonl`: one line per gate decision, with reasons.",
        "- `stats.json`: the counts above, read back from the logs.",
        "",
        "## Rules enforced by scripts/build_distillation_data.py",
        "",
        "1. Contamination: never train on a golden question verbatim. "
        "Golden-derived training uses paraphrases only. Synthetic "
        "questions are dropped when their content-token Jaccard "
        "similarity to any golden question is 0.85 or higher.",
        "2. Verification gate: every teacher answer must pass "
        "`app.verification.verify_answer` against the same 3 chunks "
        "the prompt used, contain the expected phrase, and be 1-2 "
        "sentences. Anything that fails is dropped, never fixed by hand. "
        "The checks run on the kept sentences, the same semantics "
        "production uses: a sentence the verifier drops is not in "
        "the training answer.",
        "3. Abstention records are exactly \"Not found in the corpus.\" "
        "Their check phrases must not appear anywhere in the corpus.",
        "4. Judge records must parse to two lines and match the "
        "deterministic proxy (verbatim faithfulness, answer-only "
        "relevance). Anything else is dropped.",
        "5. Keep rate below 0.70 for any source stops the run before "
        "finalising.",
        "",
        "## Corrections to the plan",
        "",
        "- The plan says the corpus has 2,095 pickable sentences. "
        "Measured: 1,144 sentences of 40 to 400 characters. The "
        "sampling targets were kept; the mix is reported in stats.json.",
        "- The plan lists q17 among the 14 generative misses. The "
        "measured generative_eval.json run has q17 fixed (via "
        "fallback), so the 14 misses are q01, q03, q05, q07, q10, "
        "q11, q12, q13, q14, q18, q24, q29, q30, q31. q17 is in the "
        "held-out variant group instead.",
        "- Synthetic answers are written by the teacher from the "
        "source sentence in one call, then verified against the "
        "retrieved top-3 chunks. The plan describes two calls; one "
        "call plus the gate enforces the same grounding with half "
        "the teacher calls.",
        "- The gate checks the kept sentences (production "
        "semantics), not the raw draft. A teacher answer with one "
        "grounded sentence and one dropped sentence still yields a "
        "clean one-sentence training record. The first smoke run "
        "used the stricter all-sentences-kept rule and dropped good "
        "answers; the relaxed rule matches what generate_answer "
        "does at inference.",
        "- q07, q10 and q29 variants were dropped, and the gate "
        "is right: the deployed prompt truncates each chunk to "
        "900 chars. The DAAD answer \"147 Scholarship options\" "
        "sits near char 1100 of the daad chunk, the Erasmus "
        "answer \"1 to 2 academic years\" sits near char 1136 of "
        "the erasmus-mundus chunk, and the RTP answer \"34,315\" "
        "sits near char 1706 of the rtp chunk, so the teacher "
        "correctly answered \"Not found in the corpus.\" The "
        "truncation is a production limitation to fix "
        "separately, not by changing the training prompt.",
        "- The key-fact check is stem coverage (60 percent) plus "
        "exact numbers, not a verbatim substring. The teacher "
        "rephrases (\"September/October 2027\" becomes \"September "
        "or October 2027\"), and the verbatim check dropped correct "
        "answers in the smoke run.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (1 miss, 6 synthetic, 4 "
                        "abstention) into /tmp/distill-smoke")
    parser.add_argument("--sources", nargs="+",
                        choices=["variants", "synthetic", "abstention",
                                 "judge"],
                        default=["variants", "synthetic", "abstention",
                                 "judge"])
    args = parser.parse_args()
    if args.smoke:
        set_out_dir(Path("/tmp/distill-smoke"))
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    teacher = Teacher(load_key())
    teacher.probe()

    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text()
              .splitlines() if line.strip()]
    print(f"chunks: {len(chunks)}", flush=True)
    retriever = Retriever(chunks)
    print(f"embedder: {retriever.embedder}", flush=True)
    golden = load_golden()
    gen_eval = json.loads(
        (REPO / "benchmarks" / "results" / "generative_eval.json")
        .read_text())
    builder = Builder(teacher, retriever, chunks, golden, gen_eval,
                      smoke=args.smoke)

    if "variants" in args.sources:
        print("== golden variants ==", flush=True)
        builder.run_variants()
    if "synthetic" in args.sources:
        print("== synthetic QA ==", flush=True)
        builder.run_synthetic()
    if "abstention" in args.sources:
        print("== abstention ==", flush=True)
        builder.run_abstention()
    if "judge" in args.sources:
        print("== judge ==", flush=True)
        builder.run_judge()

    stats = builder.finalize()
    if not args.smoke:
        write_dataset_readme(stats)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"train: {stats['train_records']}  "
          f"heldout: {stats['heldout_records']}", flush=True)


if __name__ == "__main__":
    main()
