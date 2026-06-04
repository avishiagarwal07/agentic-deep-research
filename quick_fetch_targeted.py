"""
quick_fetch_targeted.py
Fetches the SPECIFIC papers referenced in the 30 eval questions,
plus a broader corpus batch. Run this instead of quick_fetch.py.
"""

import arxiv
import json
import time
import urllib.request
from pathlib import Path

PARSED_DIR = Path("data/parsed")
PAPERS_DIR = Path("data/papers")
PARSED_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# ── Specific papers the questions ask about (fetch by title search) ────────
REQUIRED_QUERIES = [
    "Mem0 memory architecture agents",
    "tau-bench reliability metric tool-using agents",
    "OSWorld benchmark computer use tasks",
    "SWE-agent ACI agent computer interface",
    "Agent Interoperability Protocols survey MCP A2A",
    "Agentic RAG survey patterns",
    "AppWorld benchmark controllable apps tasks",
    "UI-TARS computer use benchmark GUI agent",
    "OpenHands event-driven architecture agents",
    "OS-MAP taxonomy computer using agent capabilities",
    "A-MEM agentic memory LLM",
    "UI-TARS-2 GUI agent training pipeline",
    "Multi-Turn Multi-Agent Orchestration",
    "Multi-Agent Collaboration Evolving Orchestration",
    "Can LLM Agents Really Debate multi-agent",
    "Multi-Agent Collaboration Mechanisms survey",
    "Deep Research Agents survey autonomous",
    "Deep Research Survey Autonomous Research Agents",
    "Open Reproducible Deep Research",
    "From Web Search Towards Agentic Deep Research",
    "reflection self-critique loop LLM agents",
    "long-term memory LLM agents vector graph hierarchical",
    "multi-agent systems single LLM comparison empirical",
    "GUI agent evaluation benchmark 2025",
    "deep research agent planner retriever synthesizer",
    "MCP ACP ANP protocol agent interoperability",
    "SWE-agent OpenHands code agent ACI interface",
    "hybrid retrieval lexical dense reranking RAG",
    "agentic AI evaluation open problems benchmarks",
    "agent memory tool use reflection failure modes",
]

# ── Broader corpus queries ─────────────────────────────────────────────────
BROAD_QUERIES = [
    "LLM agent memory retrieval 2024",
    "agentic RAG retrieval augmented generation agent",
    "multi-agent collaboration LLM benchmark",
    "tool use language model agent evaluation",
    "computer use agent GUI web automation",
    "deep research agent autonomous LLM",
    "agent planning reflection self-critique loop",
    "code agent software engineering LLM",
]


def fetch_query(query: str, client: arxiv.Client, max_results: int = 5) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    try:
        for r in client.results(search):
            published = r.published.strftime("%Y-%m-%d")
            if published < "2024-01-01":
                continue
            aid = r.entry_id.split("/abs/")[-1]
            # strip version suffix
            import re
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
        print(f"    [WARN] query failed: {e}")
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

    # Load existing papers (from quick_fetch.py if run before)
    existing = []
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
    seen = {p["arxiv_id"] for p in existing}
    papers = list(existing)
    print(f"Starting with {len(papers)} existing papers.")

    client = arxiv.Client(page_size=10, delay_seconds=4, num_retries=5)

    # ── Phase 1: Required papers (top-5 per query, relevance-sorted) ──────
    print(f"\nFetching {len(REQUIRED_QUERIES)} required-paper queries…")
    for i, q in enumerate(REQUIRED_QUERIES):
        print(f"  [{i+1}/{len(REQUIRED_QUERIES)}] {q[:60]}")
        results = fetch_query(q, client, max_results=5)
        added = 0
        for r in results:
            if r["arxiv_id"] not in seen:
                seen.add(r["arxiv_id"])
                papers.append(r)
                added += 1
        print(f"    → +{added} new  (total={len(papers)})")
        time.sleep(4)

    # ── Phase 2: Broader corpus ────────────────────────────────────────────
    print(f"\nFetching {len(BROAD_QUERIES)} broad corpus queries…")
    for i, q in enumerate(BROAD_QUERIES):
        print(f"  [{i+1}/{len(BROAD_QUERIES)}] {q[:60]}")
        results = fetch_query(q, client, max_results=15)
        added = 0
        for r in results:
            if r["arxiv_id"] not in seen:
                seen.add(r["arxiv_id"])
                papers.append(r)
                added += 1
        print(f"    → +{added} new  (total={len(papers)})")
        time.sleep(4)

    # Save metadata
    meta_path.write_text(json.dumps(papers, indent=2))
    print(f"\nSaved {len(papers)} papers to metadata.json")

    # ── Phase 3: Download PDFs ─────────────────────────────────────────────
    print(f"\nDownloading PDFs for {len(papers)} papers…")
    ok = fail = skip = 0
    for i, p in enumerate(papers):
        safe_id = p["arxiv_id"].replace("/", "_")
        dest = PAPERS_DIR / f"{safe_id}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            skip += 1
            continue
        if download_pdf(p):
            ok += 1
        else:
            fail += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(papers)}  ok={ok} fail={fail} skip={skip}")
        time.sleep(1.5)

    print(f"\nAll done. PDFs: ok={ok} fail={fail} skip={skip}")
    print(f"Total corpus size: {len(papers)} papers")


if __name__ == "__main__":
    main()