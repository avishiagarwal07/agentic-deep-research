"""
scraper/fetch_papers.py — uses official arxiv Python library
"""
import arxiv
import json
import time
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAPERS_DIR, PARSED_DIR, MAX_PAPERS, ARXIV_START, ARXIV_END

PARSED_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

QUERY = (
    "(ti:\"LLM agent\" OR ti:\"language model agent\" OR ti:\"agentic\" "
    "OR ti:\"tool use\" OR ti:\"multi-agent\" OR abs:\"agentic RAG\" "
    "OR abs:\"tool-augmented language\") AND (cat:cs.CL OR cat:cs.AI OR cat:cs.LG)"
)

def main():
    meta_path = PARSED_DIR / "metadata.json"

    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if len(existing) > 10:
            print(f"Cached metadata: {len(existing)} papers")
            papers = existing
        else:
            papers = []
    else:
        papers = []

    if not papers:
        print(f"Searching arXiv (target={MAX_PAPERS})…")
        client = arxiv.Client(
            page_size=100,
            delay_seconds=5,      # 5s between pages — very polite
            num_retries=10,
        )
        search = arxiv.Search(
            query=QUERY,
            max_results=MAX_PAPERS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        seen = set()
        for i, result in enumerate(client.results(search)):
            published = result.published.strftime("%Y-%m-%d")
            if published < ARXIV_START or published > ARXIV_END:
                continue
            arxiv_id = result.entry_id.split("/abs/")[-1]
            arxiv_id = arxiv_id.rsplit("v", 1)[0] if arxiv_id[-2] == "v" else arxiv_id
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            papers.append({
                "arxiv_id":  arxiv_id,
                "title":     result.title,
                "abstract":  result.summary.replace("\n", " "),
                "authors":   [a.name for a in result.authors],
                "published": published,
                "pdf_url":   f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            })
            if (i+1) % 50 == 0:
                print(f"  {len(papers)} papers collected…")

        meta_path.write_text(json.dumps(papers, indent=2))
        print(f"Saved {len(papers)} papers to {meta_path}")

    # Download PDFs
    print(f"\nDownloading {len(papers)} PDFs…")
    ok = fail = skip = 0
    for i, p in enumerate(papers):
        safe_id = p["arxiv_id"].replace("/", "_")
        dest = PAPERS_DIR / f"{safe_id}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            skip += 1
            continue
        try:
            req = urllib.request.Request(
                p["pdf_url"],
                headers={"User-Agent": "research-bot/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) > 5000:
                dest.write_bytes(data)
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(papers)} ok={ok} fail={fail} skip={skip}")
        time.sleep(2)

    print(f"Done. ok={ok} fail={fail} skip={skip}")

if __name__ == "__main__":
    main()