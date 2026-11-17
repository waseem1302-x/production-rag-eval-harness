"""Embeddings: sentence-transformers, with an honest BM25-only fallback.

Primary: all-MiniLM-L6-v2 (384-dim, ~90 MB, runs on CPU).
If the model cannot be loaded (no network, no cache), the embedder
reports available=False and the retriever uses BM25 only. The API
response always names the embedder actually used, so a BM25-only
run is never presented as a dense run.
"""
import numpy as np

from .config import EMBEDDING_DIM, EMBEDDING_MODEL

_MODEL = None
_LOAD_ERROR = None


def load_model():
    """Load (and cache) the sentence-transformers model, or record the error."""
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _LOAD_ERROR is not None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
        return _MODEL
    except Exception as exc:  # noqa: BLE001 - model download can fail many ways
        _LOAD_ERROR = str(exc)
        return None


def embedder_name() -> str:
    return EMBEDDING_MODEL if load_model() is not None else "bm25-only"


def load_error() -> str | None:
    load_model()
    return _LOAD_ERROR


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts, L2-normalised (so FAISS inner product = cosine)."""
    model = load_model()
    if model is None:
        raise RuntimeError(f"Embedding model unavailable: {_LOAD_ERROR}")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embedding_dim() -> int:
    return EMBEDDING_DIM
