"""
config.py — Central configuration for the Agentic Deep Research system.
All paths, model names, and hyperparameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
PAPERS_DIR    = DATA_DIR / "papers"          # raw PDFs
PARSED_DIR    = DATA_DIR / "parsed"          # per-paper JSON
CHUNKS_DIR    = DATA_DIR / "chunks"          # chunked JSON
CHROMA_DIR    = DATA_DIR / "chroma_db"       # ChromaDB persistent store
BM25_PATH     = DATA_DIR / "bm25_index.pkl"
PREDICTIONS_DIR = ROOT / "predictions"
EVAL_DIR      = ROOT / "eval"
QUESTIONS_PATH  = EVAL_DIR / "questions.jsonl"

for d in [DATA_DIR, PAPERS_DIR, PARSED_DIR, CHUNKS_DIR,
          CHROMA_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── arXiv scraping ─────────────────────────────────────────────────────────
ARXIV_QUERY   = (
    "(ti:\"LLM agent\" OR ti:\"language model agent\" OR ti:\"agentic RAG\" "
    "OR ti:\"tool use\" OR ti:\"agent memory\" OR ti:\"computer use agent\" "
    "OR abs:\"agentic\" OR abs:\"multi-agent\" OR abs:\"tool-augmented\")"
)
ARXIV_CATS    = ["cs.CL", "cs.AI", "cs.LG"]
ARXIV_START   = "2024-01-01"
ARXIV_END     = "2026-04-30"
MAX_PAPERS    = 700
FETCH_DELAY   = 3          # seconds between arXiv API calls (be polite)

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400        # tokens
CHUNK_OVERLAP = 80

# ── Embeddings ────────────────────────────────────────────────────────────
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE  = "cpu"      # change to "cuda" if available
CHROMA_COLLECTION = "agent_papers"

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K_DENSE   = 50         # dense retrieval candidates
TOP_K_BM25    = 50         # BM25 candidates
TOP_K_RRF     = 40         # after RRF fusion
TOP_K_RERANK  = 8          # after cross-encoder reranking
RRF_K         = 60         # RRF constant

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM (Groq) ────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
PLAN_MODEL = "gemma-4-31b-it"
SYNTH_MODEL = "gemma-4-31b-it"
FAST_MODEL = "gemini-2.5-flash"

MAX_TOKENS_PLAN = 1024
MAX_TOKENS_SYNTH = 2048
MAX_TOKENS_FAST = 512

TEMPERATURE = 0.1

# ── Agent loop ────────────────────────────────────────────────────────────
MAX_ITERATIONS = 2         # max reflect-retrieve cycles
MIN_EVIDENCE_CHUNKS = 6    # reflector: stop if we have at least this many

# ── Evaluation ────────────────────────────────────────────────────────────
JUDGE_MODEL   = PLAN_MODEL  # LLM-as-judge uses same big model
ACCURACY_THRESHOLD = 0.5    # judge score treated as correct if >= this

# ── Ablation configs ──────────────────────────────────────────────────────
# Each key matches a predictions/<config>.jsonl filename
CONFIGS = {
    "full_agent":         dict(use_planner=True,  use_reranker=True,  use_reflector=True,  use_hybrid=True,  use_verifier=True),
    "baseline":           dict(use_planner=False, use_reranker=False, use_reflector=False, use_hybrid=False, use_verifier=False),
    "no_planner":         dict(use_planner=False, use_reranker=True,  use_reflector=True,  use_hybrid=True,  use_verifier=True),
    "no_reranker":        dict(use_planner=True,  use_reranker=False, use_reflector=True,  use_hybrid=True,  use_verifier=True),
    "no_reflector":       dict(use_planner=True,  use_reranker=True,  use_reflector=False, use_hybrid=True,  use_verifier=True),
    "no_hybrid":          dict(use_planner=True,  use_reranker=True,  use_reflector=True,  use_hybrid=False, use_verifier=True),
    "no_verifier":        dict(use_planner=True,  use_reranker=True,  use_reflector=True,  use_hybrid=True,  use_verifier=False),
}