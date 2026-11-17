"""Cross-encoder reranker: sentence-transformers CrossEncoder, local CPU, $0.

Primary: cross-encoder/ms-marco-MiniLM-L-6-v2 (MiniLM-L6, 22.7M
params, model.safetensors 90,870,598 bytes, max_seq_length 512).
It scores (question, chunk) pairs jointly, so unlike the bi-encoder
it sees the question and the chunk together — in principle it can
tell a chunk that merely repeats the query terms apart from one
that actually answers the question.

Measured honestly on this VM (1.5 shared CPUs, no GPU): scoring
10 full chunks takes ~2.5-3.4 s per question (long sequences,
truncated to 512 tokens), vs ~0 ms for the lexical reranker. Short
pairs are ~0.06 s each. The model loads in ~10 s warm (~57 s cold
including the ~91 MB download). If the model cannot be loaded (no
network, no cache), `score_pairs` returns None and the retriever
falls back to the lexical reranker — a missing model degrades the
reranker, never crashes a search, and the trace says which reranker
actually ran.
"""
import math

from .config import CROSS_ENCODER_MODEL

_MODEL = None
_LOAD_ERROR = None


def load_model():
    """Load (and cache) the cross-encoder model, or record the error."""
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _LOAD_ERROR is not None:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(CROSS_ENCODER_MODEL)
        return _MODEL
    except Exception as exc:  # noqa: BLE001 - model download can fail many ways
        _LOAD_ERROR = str(exc)
        return None


def reranker_name() -> str:
    return CROSS_ENCODER_MODEL if load_model() is not None else "lexical-only"


def load_error() -> str | None:
    load_model()
    return _LOAD_ERROR


def score_pairs(question: str, texts: list[str]) -> list[float] | None:
    """Raw cross-encoder logits for (question, text) pairs, in order.

    Logits are unbounded (typically about -11 to +8 on this corpus:
    a relevant pair scores ~+5 to +7, an unrelated one ~-7 to -11).
    The retriever maps them through a sigmoid to a 0-1 rerank
    score, so the abstention threshold keeps its meaning. Returns
    None when the model is unavailable.
    """
    model = load_model()
    if model is None:
        return None
    scores = model.predict([(question, text) for text in texts],
                           show_progress_bar=False)
    return [round(float(s), 4) for s in scores]


def sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))
