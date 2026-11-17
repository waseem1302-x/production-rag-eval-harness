"""Shared paths and constants for the production RAG eval harness."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_FILE = DATA_DIR / "sources.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
INDEX_FILE = PROCESSED_DIR / "faiss.index"
# Programme closures the eligibility module reads (Phase 2, Step 4):
# one entry per programme the corpus itself says is closed, with
# verbatim evidence. See data/closures.json.
CLOSURES_FILE = DATA_DIR / "closures.json"
GOLDEN_FILE = ROOT / "eval" / "golden.json"

# Chunking: ~512 tokens, ~1 token = ~4 chars of English text.
# Phase 2, Step 5 (chunk-size ablation): the token size is the
# parameter; chars are derived from it so 256 / 512 / 1,024 tokens
# map to 1,024 / 2,048 / 4,096 chars. Overlap is held CONSTANT at
# 200 chars for every size, so chunk size is the only variable in
# the ablation (a proportional overlap would vary two things at once).
CHARS_PER_TOKEN = 4
CHUNK_TOKENS = 512
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = 200


def chunk_chars_for_tokens(tokens: int) -> int:
    """Chars budget for a chunk-size in tokens (1 token ~ 4 chars)."""
    return tokens * CHARS_PER_TOKEN

# Retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
HYBRID_DENSE_WEIGHT = 0.5
HYBRID_BM25_WEIGHT = 0.5
DEFAULT_TOP_K = 5
RERANK_CANDIDATES = 10
# Below this best hybrid score, abstain. Tuned after the first eval run;
# see EVAL_RESULTS.md for the value in use and why.
ABSTAIN_THRESHOLD = 0.15

# Cross-encoder reranker (Phase 2, Step 3): ms-marco MiniLM-L6,
# 22.7M params, ~91 MB on disk, CPU-only, $0. See app/cross_encoder.py.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Dense sentence scoring (Phase 2, Step 2): weight of the cosine
# similarity term added to the lexical sentence score in
# app.answer._pick_sentences. Lexical scores run ~0-10; MiniLM
# cosine sims run ~0.2-0.9. Measured on the golden set: weights
# 0, 1 and 2 change no golden answer at all; weights 3, 3.5 and 4
# fix q03 but lose q16 (its correct sentence scores only 0.394
# dense similarity, below the boilerplate that replaces it), net
# answer-only 0.909 either way. 2.0 is therefore the largest
# weight with zero regressions — the mechanism is in the path and
# measured, not claimed to earn a gain on this corpus.
DENSE_SENTENCE_WEIGHT = 2.0
# Answer verification (Phase 2, Step 2): the best picked sentence
# must reach at least this cosine similarity to the question, or the
# answer is refused as "low_answer_similarity". Calibrated below
# the lowest answerable question in the golden set (q19, max sim
# 0.562) with margin; it is a floor for unrelated questions, not
# the o04 catcher — o04's picked sentences score 0.667, above
# several answerable questions, so dense similarity alone cannot
# separate it (see app/answer.py).
DENSE_VERIFY_MIN_SIM = 0.45

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) production-rag-eval-harness/1.0 "
    "(research project; fetches public official scholarship pages)"
)
