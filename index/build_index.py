"""
index/build_index.py
Builds two indices from all_chunks.jsonl:
  1. ChromaDB (dense, BAAI/bge-small-en-v1.5 embeddings)
  2. BM25 pickle (rank_bm25)
Run once; both indices are persistent.
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (CHUNKS_DIR, CHROMA_DIR, BM25_PATH,
                    EMBED_MODEL, EMBED_DEVICE, CHROMA_COLLECTION)

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Run: pip install rank-bm25")


BATCH_SIZE = 256   # ChromaDB upsert batch size


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenisation for BM25."""
    import re
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def build_chroma(chunks: list[dict], embed_model: SentenceTransformer):
    print("Building ChromaDB index …")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete existing collection to rebuild cleanly
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    texts    = [c["text"] for c in chunks]
    ids      = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "arxiv_id":  c["arxiv_id"],
        "title":     c["title"][:200],
        "published": c["published"],
        "chunk_idx": c["chunk_idx"],
    } for c in chunks]

    print(f"  Embedding {len(texts)} chunks (batch={BATCH_SIZE}) …")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids   = ids[i:i+BATCH_SIZE]
        batch_meta  = metadatas[i:i+BATCH_SIZE]

        embeddings = embed_model.encode(
            batch_texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )
        if (i // BATCH_SIZE + 1) % 5 == 0:
            print(f"    {i+len(batch_texts)}/{len(texts)} chunks indexed")

    print(f"  ChromaDB collection '{CHROMA_COLLECTION}' built with {len(texts)} chunks.")
    return collection


def build_bm25(chunks: list[dict]):
    print("Building BM25 index …")
    corpus_tokens = [tokenize_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    index_data = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "corpus_tokens": corpus_tokens,
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(index_data, f)
    print(f"  BM25 index saved to {BM25_PATH}")
    return bm25


def main():
    chunks_path = CHUNKS_DIR / "all_chunks.jsonl"
    if not chunks_path.exists():
        print("Run scraper/chunk_text.py first.")
        return

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Load embedding model once
    print(f"Loading embedding model: {EMBED_MODEL} …")
    embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    # Build both indices
    build_chroma(chunks, embed_model)
    build_bm25(chunks)

    print("\n✓ Both indices built successfully.")


if __name__ == "__main__":
    main()