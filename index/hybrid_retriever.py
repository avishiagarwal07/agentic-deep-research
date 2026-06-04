"""
index/hybrid_retriever.py
Combines dense (ChromaDB) + lexical (BM25) via Reciprocal Rank Fusion,
then optionally reranks top candidates with a cross-encoder.
"""

import sys
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBED_MODEL, EMBED_DEVICE, CHROMA_DIR, CHROMA_COLLECTION,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_RRF, TOP_K_RERANK,
    RRF_K, RERANKER_MODEL, CHUNKS_DIR
)
from index.bm25_index import search as bm25_search

import json

# ── lazy singletons ────────────────────────────────────────────────────────

_embed_model  = None
_chroma_col   = None
_reranker     = None
_chunk_lookup = None   # chunk_id → text


def _get_embed():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _embed_model


def _get_chroma():
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _chroma_col = client.get_collection(CHROMA_COLLECTION)
    return _chroma_col


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def _get_chunk_lookup() -> dict[str, dict]:
    """Load all chunks into a dict for fast lookup by chunk_id."""
    global _chunk_lookup
    if _chunk_lookup is None:
        path = CHUNKS_DIR / "all_chunks.jsonl"
        _chunk_lookup = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    _chunk_lookup[c["chunk_id"]] = c
    return _chunk_lookup


# ── Dense retrieval ────────────────────────────────────────────────────────

def dense_search(query: str, top_k: int = TOP_K_DENSE) -> list[tuple[str, float]]:
    """Return (chunk_id, cosine_score) pairs."""
    model = _get_embed()
    col   = _get_chroma()

    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    results = col.query(
        query_embeddings=q_emb,
        n_results=min(top_k, col.count()),
        include=["distances", "metadatas"]
    )
    ids       = results["ids"][0]
    distances = results["distances"][0]   # cosine distance [0,2]; lower=better
    scores    = [1.0 - d / 2.0 for d in distances]   # convert to similarity
    return list(zip(ids, scores))


# ── RRF fusion ─────────────────────────────────────────────────────────────

def rrf_fuse(
    dense_results: list[tuple[str, float]],
    bm25_results:  list[tuple[str, float]],
    k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists → merged ranked list."""
    scores: dict[str, float] = {}

    for rank, (cid, _) in enumerate(dense_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, (cid, _) in enumerate(bm25_results):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ── Cross-encoder reranking ────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[tuple[str, float]],
    top_k: int = TOP_K_RERANK
) -> list[tuple[str, float]]:
    """Rerank candidates using cross-encoder; returns top_k."""
    lookup = _get_chunk_lookup()
    reranker = _get_reranker()

    pairs  = [(query, lookup[cid]["text"]) for cid, _ in candidates if cid in lookup]
    cids   = [cid for cid, _ in candidates if cid in lookup]

    if not pairs:
        return candidates[:top_k]

    ce_scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(cids, ce_scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── Public API ─────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    use_hybrid:  bool = True,
    use_reranker: bool = True,
    top_k: int = TOP_K_RERANK,
    exclude_ids: set | None = None,
) -> list[dict]:
    """
    Main retrieval entry point.
    Returns list of chunk dicts (with 'text', 'arxiv_id', 'title', etc.)
    """
    lookup = _get_chunk_lookup()

    if use_hybrid:
        dense  = dense_search(query, top_k=TOP_K_DENSE)
        bm25   = bm25_search(query,  top_k=TOP_K_BM25)
        merged = rrf_fuse(dense, bm25)[:TOP_K_RRF]
    else:
        # Dense-only
        merged = dense_search(query, top_k=TOP_K_RRF)

    if exclude_ids:
        merged = [(cid, s) for cid, s in merged
                  if lookup.get(cid, {}).get("arxiv_id") not in exclude_ids]

    if use_reranker and len(merged) > top_k:
        merged = rerank(query, merged, top_k=top_k)
    else:
        merged = merged[:top_k]

    seen_papers = set()
    results = []

    for cid, score in merged:
        if cid not in lookup:
            continue

        chunk = dict(lookup[cid])

        arxiv_id = chunk["arxiv_id"]

        if arxiv_id in seen_papers:
            continue

        seen_papers.add(arxiv_id)

        chunk["retrieval_score"] = score
        results.append(chunk)

        if len(results) >= top_k:
            break

    return results