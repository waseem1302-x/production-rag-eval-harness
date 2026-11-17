"""Shared paths and constants for the production RAG eval harness."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SOURCES_FILE = DATA_DIR / "sources.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
INDEX_FILE = PROCESSED_DIR / "faiss.index"
GOLDEN_FILE = ROOT / "eval" / "golden.json"

# Chunking: ~512 tokens, ~1 token = ~4 chars of English text.
CHUNK_CHARS = 2048
CHUNK_OVERLAP_CHARS = 200

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

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) production-rag-eval-harness/1.0 "
    "(research project; fetches public official scholarship pages)"
)
