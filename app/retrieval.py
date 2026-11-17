"""Hybrid retrieval: FAISS dense + BM25, then a lexical rerank.

Scores are each normalised to 0-1 (min-max over the candidate set)
and combined 0.5 dense + 0.5 BM25. The reranker is transparent:
it re-sorts the top candidates by combined score plus a bonus for
query-term coverage in the chunk. It is NOT a cross-encoder;
Phase 2 adds one and measures whether it helps.

When dense embeddings are unavailable, retrieval is BM25-only and
`embedder` in the result says so.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from . import embeddings
from .config import (
    DEFAULT_TOP_K,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    RERANK_CANDIDATES,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _query_terms(question: str) -> set[str]:
    stop = {"what", "which", "where", "when", "who", "how", "does", "do",
            "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
            "and", "or", "by", "with", "from", "it", "its", "this", "that"}
    return {t for t in tokenize(question) if t not in stop and len(t) > 2}


class Retriever:
    """Built once from chunks; `search` answers one question.

    `mode` selects the retrieval signal, and exists so the ablation
    can measure each signal's contribution instead of assuming it:
    - "bm25": keyword scores only. The embedding model is never
      loaded, so BM25 latency is honest and the configuration runs
      without the model.
    - "dense": embedding cosine only (FAISS IndexFlatIP).
    - "hybrid": 0.5 * dense + 0.5 * BM25 after min-max normalisation
      (the Phase 1 default).
    In "bm25" mode dense_score is 0.0 and in "dense" mode bm25_score
    is 0.0 — "not used in this mode", not "scored zero"."""

    def __init__(self, chunks: list[dict], mode: str = "hybrid"):
        if mode not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"unknown retrieval mode: {mode!r} "
                             "(expected 'bm25', 'dense' or 'hybrid')")
        self.mode = mode
        self.chunks = chunks
        self.corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if chunks else None
        self.vectors = None
        self.index = None
        self.embedder = embeddings.embedder_name() if mode != "bm25" else "bm25-only"
        if self.embedder != "bm25-only" and chunks:
            self.vectors = embeddings.embed(chunks_text(chunks))
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.vectors.shape[1])
                self.index.add(self.vectors)
            except Exception:
                self.index = None  # numpy cosine fallback in search()

    def _dense_scores(self, query: str, n: int) -> dict[int, float]:
        if self.vectors is None:
            return {}
        q = embeddings.embed([query])[0]
        if self.index is not None:
            scores, idxs = self.index.search(q.reshape(1, -1), min(n, len(self.chunks)))
            return {int(i): float(s) for i, s in zip(idxs[0], scores[0]) if i >= 0}
        sims = self.vectors @ q
        top = np.argsort(-sims)[:n]
        return {int(i): float(sims[i]) for i in top}

    def _rerank(self, candidates: list[dict], query_terms: set[str]) -> list[dict]:
        for cand in candidates:
            chunk_terms = set(tokenize(cand["text"]))
            coverage = len(query_terms & chunk_terms) / max(1, len(query_terms))
            cand["rerank_score"] = cand["score"] + 0.1 * coverage
            cand["term_coverage"] = round(coverage, 3)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    def search(self, question: str, top_k: int = DEFAULT_TOP_K,
               rerank: bool = True) -> list[dict]:
        """Search the index. `rerank=False` returns candidates in
        combined-score order; the benchmark uses it for the naive
        baseline so the reranker's contribution is measured, not
        assumed."""
        if not self.chunks:
            return []
        n_cand = min(RERANK_CANDIDATES, len(self.chunks))
        # Only the signals this mode uses are computed: in "bm25"
        # mode the query is never embedded, in "dense" mode BM25
        # scores are never computed. Computing an unused signal
        # would hide the mode's true latency.
        dense = self._dense_scores(question, n_cand) if self.mode != "bm25" else {}
        bm25_scores = (self.bm25.get_scores(tokenize(question)) if self.bm25 is not None and self.mode != "dense"
                       else np.zeros(len(self.chunks)))
        top_bm25 = np.argsort(-bm25_scores)[:n_cand] if self.mode != "dense" else []
        cand_ids = sorted(set(dense) | {int(i) for i in top_bm25})

        dense_vals = [dense.get(i, 0.0) for i in cand_ids]
        bm25_vals = [float(bm25_scores[i]) for i in cand_ids]
        dense_norm = _minmax(dense_vals)
        bm25_norm = _minmax(bm25_vals)

        candidates = []
        for pos, cid in enumerate(cand_ids):
            if self.mode == "bm25":
                score = bm25_norm[pos]
            elif self.mode == "dense":
                score = dense_norm[pos]
            else:
                score = (HYBRID_DENSE_WEIGHT * dense_norm[pos]
                         + HYBRID_BM25_WEIGHT * bm25_norm[pos])
            candidates.append({
                "chunk_id": self.chunks[cid]["chunk_id"],
                "doc": self.chunks[cid]["doc"],
                "text": self.chunks[cid]["text"],
                "score": round(float(score), 4),
                "dense_score": round(float(dense_norm[pos]), 4),
                "bm25_score": round(float(bm25_norm[pos]), 4),
            })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        for cand in candidates:
            cand["rerank_score"] = cand["score"]
        if not rerank:
            return candidates[:top_k]
        return self._rerank(candidates[:n_cand], _query_terms(question))[:top_k]


def chunks_text(chunks: list[dict]) -> list[str]:
    return [c["text"] for c in chunks]
