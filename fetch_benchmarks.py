import arxiv
import json
import re
import time
import urllib.request
from pathlib import Path

PARSED_DIR = Path("data/parsed")
PAPERS_DIR = Path("data/papers")

meta_path = PARSED_DIR / "metadata.json"

BENCHMARK_QUERIES = [
    "all:AppWorld",
    "all:SWE-Agent",
    "all:AgentBench",
    "all:OSWorld",
    "all:WebArena",
    "all:WebShop",
    "all:OpenHands",
    "all:tau-bench",
    "all:UI-TARS",
    "all:BrowserArena",
]

client = arxiv.Client(
    page_size=25,
    delay_seconds=5,
    num_retries=8,
)


def download_pdf(pdf_url, arxiv_id):
    dest = PAPERS_DIR / f"{arxiv_id}.pdf"

    if dest.exists():
        return True

    try:
        req = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "research-bot/1.0"}
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        if len(data) > 5000:
            dest.write_bytes(data)
            return True

    except Exception as e:
        print("Download failed:", e)

    return False


if meta_path.exists():
    papers = json.loads(meta_path.read_text())
else:
    papers = []

seen = {p["arxiv_id"] for p in papers}

added = 0

for query in BENCHMARK_QUERIES:

    print(f"\nSearching: {query}")

    search = arxiv.Search(
        query=query,
        max_results=20,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    try:
        for r in client.results(search):

            aid = r.entry_id.split("/abs/")[-1]
            aid = re.sub(r"v\d+$", "", aid)

            if aid in seen:
                continue

            paper = {
                "arxiv_id": aid,
                "title": r.title,
                "abstract": r.summary.replace("\n", " "),
                "authors": [a.name for a in r.authors],
                "published": r.published.strftime("%Y-%m-%d"),
                "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
            }

            papers.append(paper)
            seen.add(aid)

            if download_pdf(paper["pdf_url"], aid):
                added += 1
                print("  +", paper["title"])

        time.sleep(3)

    except Exception as e:
        print("ERROR:", e)

meta_path.write_text(json.dumps(papers, indent=2))

print("\nDone.")
print("New papers added:", added)
print("Total corpus size:", len(papers))