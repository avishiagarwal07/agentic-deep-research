"""
eval/metrics.py
Citation precision/recall (exact set overlap) and
aggregation of judge scores across a prediction file.
"""

import re
from typing import Any


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Strip version suffix (v1, v2 …) for matching."""
    return re.sub(r"v\d+$", "", arxiv_id.strip()).strip()


def citation_metrics(
    predicted: list[str],
    must_cite: list[str],
) -> dict[str, float]:
    """
    Compute citation precision, recall, F1 against a must-cite set.
    predicted  : arXiv IDs in the system answer
    must_cite  : ground-truth arXiv IDs that should be cited
    """
    pred_set = {normalize_arxiv_id(c) for c in predicted}
    true_set = {normalize_arxiv_id(c) for c in must_cite}

    if not true_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(true_set)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


def aggregate_scores(per_question: list[dict]) -> dict[str, Any]:
    """
    Aggregate per-question metric dicts into a summary table.
    Each element of per_question should have keys:
      accuracy, faithfulness, citation_precision, citation_recall,
      citation_f1, latency_seconds, tool_calls
    """
    if not per_question:
        return {}

    keys = ["accuracy", "faithfulness", "citation_precision",
            "citation_recall", "citation_f1", "latency_seconds", "tool_calls"]

    summary = {}
    for k in keys:
        vals = [q[k] for q in per_question if k in q]
        if vals:
            summary[f"{k}_mean"] = round(sum(vals) / len(vals), 4)
            summary[f"{k}_min"]  = round(min(vals), 4)
            summary[f"{k}_max"]  = round(max(vals), 4)

    summary["n_questions"] = len(per_question)
    return summary


def print_summary_table(config_results: dict[str, dict]):
    """Print a nicely formatted ablation table."""
    configs = list(config_results.keys())
    metrics = ["accuracy_mean", "faithfulness_mean",
               "citation_precision_mean", "citation_recall_mean",
               "latency_seconds_mean", "tool_calls_mean"]
    short = ["Acc", "Faith", "Cit-P", "Cit-R", "Lat(s)", "Calls"]

    # Header
    col_w = 14
    header = f"{'Config':<22}" + "".join(f"{s:>{col_w}}" for s in short)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for cfg in configs:
        row_data = config_results[cfg]
        row = f"{cfg:<22}"
        for m in metrics:
            val = row_data.get(m, float("nan"))
            row += f"{val:>{col_w}.4f}" if isinstance(val, float) else f"{'N/A':>{col_w}}"
        print(row)
    print("=" * len(header))