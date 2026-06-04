"""
run_pipeline.py
Single-command entry point.

Usage:
  python run_pipeline.py --phase all            # full run
  python run_pipeline.py --phase scrape         # fetch + parse + chunk
  python run_pipeline.py --phase index          # build indices
  python run_pipeline.py --phase run            # run all agent configs
  python run_pipeline.py --phase eval           # evaluate all predictions
  python run_pipeline.py --config full_agent    # run + eval one config
  python run_pipeline.py --phase run --config baseline   # one ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path

from config import CONFIGS, PREDICTIONS_DIR, EVAL_DIR, QUESTIONS_PATH


def phase_scrape():
    print("\n═══ PHASE 1: SCRAPE ═══")
    from scraper.fetch_papers import main as fetch_main
    from scraper.parse_pdfs import parse_all
    from scraper.chunk_text import chunk_all

    fetch_main()
    parse_all()
    chunk_all()


def phase_index():
    print("\n═══ PHASE 2: INDEX ═══")
    from index.build_index import main as build_main
    build_main()


def phase_run(config_names: list[str] | None = None):
    print("\n═══ PHASE 3: RUN AGENT ═══")
    from agent.run_agent import run_agent, state_to_dict

    if not QUESTIONS_PATH.exists():
        print(f"[ERROR] {QUESTIONS_PATH} not found.")
        sys.exit(1)

    questions = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions.")

    configs_to_run = config_names or list(CONFIGS.keys())

    for cfg_name in configs_to_run:
        cfg = CONFIGS[cfg_name]
        out_path = PREDICTIONS_DIR / f"{cfg_name}.jsonl"

        # Skip if already done
        if out_path.exists():
            existing = sum(1 for _ in open(out_path))
            if existing >= len(questions):
                print(f"  [SKIP] {cfg_name} already complete ({existing} predictions)")
                continue

        print(f"\n  Running config: {cfg_name} {cfg}")
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, q in enumerate(questions):
                print(f"    Q{i+1}/{len(questions)}: {q['question'][:70]} …")
                try:
                    state = run_agent(
                        question=q["question"],
                        question_id=q["id"],
                        **cfg,
                    )
                    record = state_to_dict(state)
                    # Attach evidence chunks for faithfulness scoring
                    record["trace"]["evidence_chunks"] = [
                        {"arxiv_id": c["arxiv_id"], "text": c["text"][:300]}
                        for c in state.evidence[:10]
                    ]
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print(f"      ✓ lat={state.latency_seconds}s "
                          f"calls={state.tool_call_count} "
                          f"cit={len(state.citations)}")
                except Exception as e:
                    print(f"      ✗ ERROR: {e}")
                    out_f.write(json.dumps({
                        "id": q["id"], "question": q["question"],
                        "answer": f"ERROR: {e}", "citations": [], "trace": {}
                    }) + "\n")
                    out_f.flush()
                time.sleep(0.5)   # brief pause between questions

        print(f"  Saved: {out_path}")


def phase_eval(config_names: list[str] | None = None):
    print("\n═══ PHASE 4: EVALUATE ═══")
    import subprocess
    configs = config_names or list(CONFIGS.keys())
    for cfg in configs:
        subprocess.run(
            [sys.executable, "eval/run_eval.py", "--config", cfg],
            check=False
        )


def main():
    parser = argparse.ArgumentParser(description="Agentic Deep Research Pipeline")
    parser.add_argument("--phase", default="all",
                        choices=["all", "scrape", "index", "run", "eval"],
                        help="Which phase to run")
    parser.add_argument("--config", default=None,
                        help="Specific config(s) to run (comma-separated), default=all")
    args = parser.parse_args()

    config_names = args.config.split(",") if args.config else None

    t0 = time.time()
    if args.phase == "scrape" or args.phase == "all":
        phase_scrape()
    if args.phase == "index" or args.phase == "all":
        phase_index()
    if args.phase == "run" or args.phase == "all":
        phase_run(config_names)
    if args.phase == "eval" or args.phase == "all":
        phase_eval(config_names)

    print(f"\n✓ Pipeline complete in {round(time.time()-t0, 1)}s")


if __name__ == "__main__":
    main()