"""
index/bm25_index.py
Thin wrapper around the persisted BM25 index.
"""

import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BM25_PATH, TOP_K_BM25

_cache = {}   # module-level cache so we load once per process


def _load():
    if "index" not in _cache:
        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {BM25_PATH}. Run index/build_index.py first."
            )
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        _cache["index"]  = data["bm25"]
        _cache["ids"]    = data["chunk_ids"]
    return _cache["index"], _cache["ids"]


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def search(query: str, top_k: int = TOP_K_BM25) -> list[tuple[str, float]]:
    """Return list of (chunk_id, score) sorted descending."""
    bm25, ids = _load()
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(ids[i], float(s)) for i, s in ranked if s > 0]