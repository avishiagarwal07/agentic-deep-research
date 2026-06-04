"""
scraper/parse_pdfs.py
Extracts clean text from downloaded PDFs using PyMuPDF (fitz).
Saves one JSON per paper with text + metadata.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAPERS_DIR, PARSED_DIR

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Run: pip install pymupdf")


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF, cleaning arXiv boilerplate."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return ""

    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    doc.close()

    full_text = "\n".join(pages)

    # Basic cleaning
    lines = []
    for line in full_text.split("\n"):
        line = line.strip()
        # Skip very short lines (page numbers, headers)
        if len(line) < 3:
            continue
        # Skip lines that are purely numeric (page numbers)
        if line.isdigit():
            continue
        lines.append(line)

    return "\n".join(lines)


def parse_all():
    meta_path = PARSED_DIR / "metadata.json"
    if not meta_path.exists():
        print("Run fetch_papers.py first.")
        return

    papers = json.loads(meta_path.read_text())
    print(f"Parsing {len(papers)} papers …")

    parsed = []
    missing, ok, failed = 0, 0, 0

    for i, paper in enumerate(papers):
        safe_id = paper["arxiv_id"].replace("/", "_")
        pdf_path = PAPERS_DIR / f"{safe_id}.pdf"
        out_path = PARSED_DIR / f"{safe_id}.json"

        if out_path.exists():
            parsed.append(out_path)
            ok += 1
            continue

        if not pdf_path.exists():
            missing += 1
            continue

        text = extract_text(pdf_path)
        if not text.strip():
            failed += 1
            continue

        record = {**paper, "text": text, "char_count": len(text)}
        out_path.write_text(json.dumps(record, ensure_ascii=False))
        parsed.append(out_path)
        ok += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(papers)}  ok={ok} missing={missing} failed={failed}")

    print(f"\nDone. Parsed={ok}  Missing PDF={missing}  Extraction failed={failed}")
    print(f"Parsed JSONs in: {PARSED_DIR}")


if __name__ == "__main__":
    parse_all()