"""
eval/run_eval.py
Evaluates a predictions/<config>.jsonl file against eval/questions.jsonl.
Writes eval/results/<config>_scores.json and prints a summary.

Usage:
  python eval/run_eval.py --config full_agent
  python eval/run_eval.py --config all       # run all configs
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PREDICTIONS_DIR, EVAL_DIR, CONFIGS
from eval.judge import judge_accuracy, judge_faithfulness
from eval.metrics import citation_metrics, aggregate_scores, print_summary_table

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_questions(path: Path) -> dict[str, dict]:
    """Returns dict: id → question record."""
    questions = {}
    for q in load_jsonl(path):
        questions[q["id"]] = q
    return questions


def evaluate_config(config_name: str, questions: dict) -> list[dict]:
    pred_path = PREDICTIONS_DIR / f"{config_name}.jsonl"
    if not pred_path.exists():
        print(f"  [SKIP] {pred_path} not found.")
        return []

    predictions = load_jsonl(pred_path)
    print(f"\nEvaluating '{config_name}' ({len(predictions)} predictions) …")

    per_question = []
    for pred in predictions:
        qid = pred.get("id", "")
        q   = questions.get(qid, {})

        answer      = pred.get("answer", "")
        citations   = pred.get("citations", [])

        # Ground truth (may be absent if held out)
        reference   = q.get("answer", "")
        must_cite   = q.get("must_cite", [])
        evidence    = pred.get("trace", {}).get("evidence_chunks", [])

        # Scores
        acc_score, acc_reason = judge_accuracy(answer, reference) if reference else (0.0, "no reference")
        faith_score, faith_reason = judge_faithfulness(answer, evidence) if evidence else (0.5, "no evidence in trace")

        cit = citation_metrics(citations, must_cite) if must_cite else {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        trace = pred.get("trace", {})
        row = {
            "id":                  qid,
            "accuracy":            acc_score,
            "accuracy_reason":     acc_reason,
            "faithfulness":        faith_score,
            "faithfulness_reason": faith_reason,
            "citation_precision":  cit["precision"],
            "citation_recall":     cit["recall"],
            "citation_f1":         cit["f1"],
            "latency_seconds":     trace.get("latency_seconds", 0.0),
            "tool_calls":          trace.get("tool_calls", 0),
            "n_citations":         len(citations),
        }
        per_question.append(row)
        print(f"  [{qid}] acc={acc_score:.2f} faith={faith_score:.2f} "
              f"cit-R={cit['recall']:.2f}")

    # Save per-question scores
    out_path = RESULTS_DIR / f"{config_name}_scores.json"
    out_path.write_text(json.dumps(per_question, indent=2))

    # Aggregate
    summary = aggregate_scores(per_question)
    summary_path = RESULTS_DIR / f"{config_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  → Saved to {out_path}")

    return per_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_agent",
                        help="Config name or 'all'")
    args = parser.parse_args()

    questions_path = EVAL_DIR / "questions.jsonl"
    if not questions_path.exists():
        print(f"[ERROR] {questions_path} not found. Make sure eval/questions.jsonl exists.")
        sys.exit(1)

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions.")

    if args.config == "all":
        config_names = list(CONFIGS.keys())
    else:
        config_names = [args.config]

    all_summaries = {}
    for cfg in config_names:
        per_q = evaluate_config(cfg, questions)
        if per_q:
            all_summaries[cfg] = aggregate_scores(per_q)

    if len(all_summaries) > 1:
        print_summary_table(all_summaries)
    elif all_summaries:
        cfg = list(all_summaries.keys())[0]
        print(f"\nSummary for '{cfg}':")
        for k, v in all_summaries[cfg].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()