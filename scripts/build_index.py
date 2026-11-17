"""Chunk + embed + build the FAISS index. Usage: python -m scripts.build_index"""
import json

from app.chunking import chunk_document
from app.config import CHUNKS_FILE, INDEX_FILE, PROCESSED_DIR
from app.embeddings import embed, embedder_name
from app.ingest import load_documents

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents()
    print(f"documents: {len(docs)}")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc["doc"], doc["text"])
        print(f"  {doc['doc']}: {len(doc['text'])} chars -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"total chunks: {len(all_chunks)} (saved to {CHUNKS_FILE})")
    print(f"embedder: {embedder_name()}")
    if embedder_name() != "bm25-only":
        vectors = embed([c["text"] for c in all_chunks])
        import faiss
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(INDEX_FILE))
        print(f"FAISS index: {index.ntotal} vectors, dim {vectors.shape[1]} (saved to {INDEX_FILE})")
    else:
        print("WARNING: embedding model unavailable; index not built, retrieval will be BM25-only.")
