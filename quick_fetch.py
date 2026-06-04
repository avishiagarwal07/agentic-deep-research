# save as: quick_fetch.py in your project root
import arxiv, json, time, urllib.request
from pathlib import Path

PARSED_DIR = Path("data/parsed")
PAPERS_DIR = Path("data/papers")
PARSED_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

client = arxiv.Client(page_size=25, delay_seconds=8, num_retries=5)
search = arxiv.Search(
    query='ti:agent AND (cat:cs.CL OR cat:cs.AI)',
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

papers, seen = [], set()
for r in client.results(search):
    aid = r.entry_id.split("/abs/")[-1].split("v")[0]
    if aid in seen: continue
    seen.add(aid)
    papers.append({
        "arxiv_id": aid, "title": r.title,
        "abstract": r.summary.replace("\n"," "),
        "authors": [a.name for a in r.authors],
        "published": r.published.strftime("%Y-%m-%d"),
        "pdf_url": f"https://arxiv.org/pdf/{aid}.pdf",
    })
    print(f"  [{len(papers)}] {aid} — {r.title[:60]}")

Path("data/parsed/metadata.json").write_text(json.dumps(papers, indent=2))
print(f"\nSaved {len(papers)} papers.")

# Download PDFs
print("Downloading PDFs...")
ok=0
for i,p in enumerate(papers):
    dest = PAPERS_DIR / f"{p['arxiv_id'].replace('/','_')}.pdf"
    if dest.exists() and dest.stat().st_size > 10000:
        ok+=1; continue
    try:
        req = urllib.request.Request(p["pdf_url"], headers={"User-Agent":"bot/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) > 5000:
            dest.write_bytes(data); ok+=1
    except: pass
    time.sleep(1.5)
    if (i+1)%10==0: print(f"  {i+1}/50 downloaded={ok}")

print(f"Done. {ok} PDFs ready.")