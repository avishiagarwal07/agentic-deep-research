"""
fetch_broad_corpus.py
Fetches a broad corpus of 400-700 LLM agent papers from arXiv.
Run AFTER quick_fetch_targeted.py so targeted papers are already downloaded.
Adds to existing metadata without duplicating.
"""

import arxiv
import json
import time
import re
import urllib.request
from pathlib import Path

PARSED_DIR = Path("data/parsed")
PAPERS_DIR = Path("data/papers")
PARSED_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 400  # total papers to reach

BROAD_QUERIES = [
    ("all:agent", 50),
    ("all:\"language model agent\"", 50),
    ("all:\"LLM agent\"", 50),
    ("all:\"multi-agent\"", 50),
    ("all:\"tool use\"", 50),
    ("all:\"agentic RAG\"", 50),
    ("all:\"computer use\"", 50),
    ("all:SWE-Agent", 50),
    ("all:SWE-bench", 50),
    ("all:OpenHands", 50),
    ("all:AppWorld", 50),
    ("all:WebArena", 50),
    ("all:OSWorld", 50),
    ("all:AgentBench", 50),
    ("all:tau-bench", 50),
    ("all:UI-TARS", 50),

    ("all:\"computer use agent\"", 50),
    ("all:\"web agent\"", 50),
    ("all:\"coding agent\"", 50),
    ("all:\"software engineering agent\"", 50),

    ("all:AutoGen", 50),
    ("all:MetaGPT", 50),
    ("all:CAMEL", 50),

]


def fetch_query(query: str, client: arxiv.Client, max_results: int) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    try:
        for r in client.results(search):
            published = r.published.strftime("%Y-%m-%d")
            if published < "2024-01-01":
                continue
            aid = r.entry_id.split("/abs/")[-1]
            aid = re.sub(r'v\d+$', '', aid)
            results.append({
                "arxiv_id":  aid,
                "title":     r.title,
                "abstract":  r.summary.replace("\n", " "),
                "authors":   [a.name for a in r.authors],
                "published": published,
                "pdf_url":   f"https://arxiv.org/pdf/{aid}.pdf",
            })
    except Exception as e:
        print(f"    [WARN] {e}")
    return results


def download_pdf(paper: dict) -> bool:
    safe_id = paper["arxiv_id"].replace("/", "_")
    dest = PAPERS_DIR / f"{safe_id}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    try:
        req = urllib.request.Request(
            paper["pdf_url"],
            headers={"User-Agent": "research-bot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) > 5_000:
            dest.write_bytes(data)
            return True
    except Exception:
        pass
    return False


def main():
    meta_path = PARSED_DIR / "metadata.json"

    # Load existing
    existing = []
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
    seen = {p["arxiv_id"] for p in existing}
    papers = list(existing)
    print(f"Starting with {len(papers)} existing papers. Target: {TARGET}")

    client = arxiv.Client(page_size=50, delay_seconds=10, num_retries=8)

    # Fetch broad queries
    for i, (query, max_r) in enumerate(BROAD_QUERIES):
        if len(papers) >= TARGET:
            print(f"Reached target of {TARGET} papers.")
            break
        print(f"\n[{i+1}/{len(BROAD_QUERIES)}] {query[:70]}")
        results = fetch_query(query, client, max_results=max_r)
        added = 0
        for r in results:
            if r["arxiv_id"] not in seen:
                seen.add(r["arxiv_id"])
                papers.append(r)
                added += 1
        print(f"  +{added} new papers  (total={len(papers)})")

        # Save after each query (resumable if interrupted)
        meta_path.write_text(json.dumps(papers, indent=2))
        time.sleep(5)

    print(f"\nMetadata done: {len(papers)} papers saved.")

    # Download PDFs for new papers only
    need_download = [
        p for p in papers
        if not (PAPERS_DIR / f"{p['arxiv_id'].replace('/','_')}.pdf").exists()
    ]
    print(f"\nDownloading {len(need_download)} new PDFs…")
    ok = fail = 0
    for i, p in enumerate(need_download):
        if download_pdf(p):
            ok += 1
        else:
            fail += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(need_download)}  ok={ok} fail={fail}")
        time.sleep(1.5)

    print(f"\nAll done!  PDFs: ok={ok} fail={fail}")
    print(f"Total corpus: {len(papers)} papers")
    print(f"\nNext steps:")
    print("  python scraper/parse_pdfs.py")
    print("  python scraper/chunk_text.py")
    print("  python run_pipeline.py --phase index")


if __name__ == "__main__":
    main()