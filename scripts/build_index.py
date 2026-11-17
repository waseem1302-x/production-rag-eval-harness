"""Chunk + embed + build the FAISS index.

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --chunk-tokens 256 --out-dir benchmarks/results/chunk_ablation/chunk-256

Defaults (no flags) reproduce the shipped behaviour exactly:
512-token chunks, overlap 200 chars, outputs to data/processed/
(chunks.jsonl + faiss.index). `--chunk-tokens` is the Phase 2,
Step 5 ablation parameter: 256 / 512 / 1,024 tokens from the same
.txt corpus, overlap held constant at 200 chars so chunk size is
the only variable. `--out-dir` keeps an ablation index out of
data/processed/ so the working 512-token index is never overwritten.
"""
import argparse
import json

from app.chunking import chunk_document
from app.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_TOKENS,
    PROCESSED_DIR,
    chunk_chars_for_tokens,
)
from app.embeddings import embed, embedder_name
from app.ingest import load_documents

CHUNK_TOKEN_CHOICES = (256, 512, 1024)


def build_index(chunk_tokens: int = CHUNK_TOKENS, out_dir=None) -> dict:
    """Build chunks + FAISS index for one chunk size.

    Returns stats: chunk_tokens, chunk_chars, overlap_chars,
    n_documents, n_chunks, chunks_file, index_file, embedder.
    Raises ValueError for an unsupported chunk size.
    """
    if chunk_tokens not in CHUNK_TOKEN_CHOICES:
        raise ValueError(
            f"unsupported chunk size: {chunk_tokens!r} "
            f"(expected one of {list(CHUNK_TOKEN_CHOICES)})")
    out_dir = PROCESSED_DIR if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_chars = chunk_chars_for_tokens(chunk_tokens)
    chunks_file = out_dir / "chunks.jsonl"
    index_file = out_dir / "faiss.index"

    docs = load_documents()
    print(f"documents: {len(docs)}")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc["doc"], doc["text"],
                                chunk_chars=chunk_chars,
                                overlap_chars=CHUNK_OVERLAP_CHARS)
        print(f"  {doc['doc']}: {len(doc['text'])} chars -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    with chunks_file.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"total chunks: {len(all_chunks)} (saved to {chunks_file})")
    print(f"embedder: {embedder_name()}")
    n_vectors = 0
    if embedder_name() != "bm25-only":
        vectors = embed([c["text"] for c in all_chunks])
        import faiss
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_file))
        n_vectors = index.ntotal
        print(f"FAISS index: {n_vectors} vectors, dim {vectors.shape[1]} (saved to {index_file})")
    else:
        print("WARNING: embedding model unavailable; index not built, retrieval will be BM25-only.")
    return {
        "chunk_tokens": chunk_tokens,
        "chunk_chars": chunk_chars,
        "overlap_chars": CHUNK_OVERLAP_CHARS,
        "n_documents": len(docs),
        "n_chunks": len(all_chunks),
        "n_vectors": n_vectors,
        "chunks_file": str(chunks_file),
        "index_file": str(index_file),
        "embedder": embedder_name(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk + embed + build the FAISS index.")
    parser.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS,
                        choices=list(CHUNK_TOKEN_CHOICES),
                        help="chunk size in tokens (default: 512)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: data/processed/)")
    args = parser.parse_args()
    from pathlib import Path
    build_index(chunk_tokens=args.chunk_tokens,
                out_dir=Path(args.out_dir) if args.out_dir else None)
