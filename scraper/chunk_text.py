"""
scraper/chunk_text.py
Chunks parsed paper text using RecursiveCharacterTextSplitter.
Each chunk carries metadata: arxiv_id, title, authors, chunk_id, published.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PARSED_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("Run: pip install langchain-text-splitters")

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def token_len(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    # Fallback: 1 token ≈ 4 chars
    def token_len(text: str) -> int:
        return len(text) // 4


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_paper(record: dict, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Return list of chunk dicts for one paper."""
    text = record.get("text", "")
    if not text.strip():
        return []

    # Prepend title + abstract to give each chunk context anchor
    header = f"Title: {record['title']}\nAbstract: {record.get('abstract','')}\n\n"
    full_text = header + text

    raw_chunks = splitter.split_text(full_text)

    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({
            "chunk_id":  f"{record['arxiv_id']}::chunk_{idx:04d}",
            "arxiv_id":  record["arxiv_id"],
            "title":     record["title"],
            "authors":   record.get("authors", []),
            "published": record.get("published", ""),
            "chunk_idx": idx,
            "text":      chunk,
            "token_len": token_len(chunk),
        })
    return chunks


def chunk_all():
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    # exclude metadata.json
    parsed_files = [f for f in parsed_files if f.name != "metadata.json"]
    print(f"Chunking {len(parsed_files)} parsed papers …")

    splitter = build_splitter()
    total_chunks = 0
    all_chunks_path = CHUNKS_DIR / "all_chunks.jsonl"

    with open(all_chunks_path, "w", encoding="utf-8") as out_f:
        for i, path in enumerate(parsed_files):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            chunks = chunk_paper(record, splitter)
            for chunk in chunks:
                out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(parsed_files)}  chunks so far: {total_chunks}")

    print(f"\nDone. Total chunks: {total_chunks}")
    print(f"Saved to: {all_chunks_path}")
    return all_chunks_path


if __name__ == "__main__":
    chunk_all()