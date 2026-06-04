"""
eval/judge.py
LLM-as-judge for answer accuracy (vs ground truth) and faithfulness (grounded in evidence).
Uses Groq with the big model.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JUDGE_MODEL, MAX_TOKENS_FAST, TEMPERATURE
from agent.llm import call_llm

# ── Accuracy judge ─────────────────────────────────────────────────────────

ACCURACY_SYSTEM = """You are an objective research answer evaluator. Compare a system answer to a reference answer.

Score the system answer on a scale of 0.0–1.0:
- 1.0: Fully correct, covers all key points of the reference
- 0.7: Mostly correct, minor omissions
- 0.5: Partially correct, covers some but not all key points
- 0.3: Mostly incorrect but has some relevant information
- 0.0: Completely wrong or irrelevant

Respond in this EXACT format:
SCORE: <float 0.0-1.0>
REASON: <one sentence>"""

ACCURACY_PROMPT = """Reference answer: {reference}

System answer: {answer}

Score the system answer:"""


def judge_accuracy(answer: str, reference: str) -> tuple[float, str]:
    """Returns (score 0-1, reason)."""
    if not answer.strip() or not reference.strip():
        return 0.0, "empty answer or reference"

    prompt = ACCURACY_PROMPT.format(
        reference=reference[:1000],
        answer=answer[:1500],
    )
    response = call_llm(
        system=ACCURACY_SYSTEM,
        user=prompt,
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS_FAST,
        temperature=0.0,
    )

    score = 0.0
    reason = ""
    for line in response.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score, reason


# ── Faithfulness judge ─────────────────────────────────────────────────────

FAITH_SYSTEM = """You are evaluating whether an answer is faithful to its source evidence (no hallucinations).

Score faithfulness 0.0–1.0:
- 1.0: Every claim is directly supported by the evidence
- 0.7: Most claims supported, minor extrapolation
- 0.5: Some claims unsupported or speculative
- 0.3: Many unsupported claims
- 0.0: Answer is mostly hallucinated or contradicts evidence

Respond in this EXACT format:
SCORE: <float 0.0-1.0>
REASON: <one sentence>"""

FAITH_PROMPT = """Evidence (retrieved passages):
{evidence}

System answer:
{answer}

Score faithfulness (is the answer grounded in the evidence?):"""


def judge_faithfulness(answer: str, evidence_chunks: list[dict]) -> tuple[float, str]:
    """Returns (score 0-1, reason)."""
    if not answer.strip() or not evidence_chunks:
        return 0.0, "empty answer or no evidence"

    # Summarize evidence
    evidence_text = "\n---\n".join(
        f"[{c['arxiv_id']}] {c['text'][:300]}"
        for c in evidence_chunks[:10]
    )

    prompt = FAITH_PROMPT.format(
        evidence=evidence_text[:3000],
        answer=answer[:1500],
    )
    response = call_llm(
        system=FAITH_SYSTEM,
        user=prompt,
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS_FAST,
        temperature=0.0,
    )

    score = 0.0
    reason = ""
    for line in response.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score, reason