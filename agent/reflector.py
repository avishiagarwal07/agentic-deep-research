"""
agent/reflector.py
Decides whether the current evidence is sufficient to answer the question,
or whether to search again with refined queries.
Uses the fast model (Llama 3.1 8B) to keep latency low.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAST_MODEL, MAX_TOKENS_FAST, TEMPERATURE, MAX_ITERATIONS, MIN_EVIDENCE_CHUNKS
from agent.state import AgentState
from agent.llm import call_llm

REFLECTOR_SYSTEM = """You are a critical research assistant evaluating whether retrieved evidence is sufficient to answer a question.

Respond in this EXACT format:
SUFFICIENT: yes|no
MISSING: <one sentence describing what is missing, or "nothing" if sufficient>
FOLLOW_UP: <one refined search query to find missing info, or "none" if sufficient>"""

REFLECTOR_PROMPT = """Question: {question}

Evidence collected so far ({n_chunks} chunks from {n_papers} papers):
{evidence_summary}

Is this evidence sufficient to write a complete, well-cited answer?"""


def _summarize_evidence(state: AgentState, max_chars: int = 3000) -> str:
    """Brief summary of evidence for the reflector prompt."""
    lines = []
    for i, c in enumerate(state.evidence[:15]):
        snippet = c["text"][:200].replace("\n", " ")
        lines.append(f"  [{i+1}] {c['arxiv_id']} — {c['title'][:60]}: {snippet}")
    summary = "\n".join(lines)
    return summary[:max_chars]


def reflect(
    state: AgentState,
    use_reflector: bool = True,
) -> tuple[AgentState, list[str]]:
    """
    Returns (updated_state, follow_up_queries).
    follow_up_queries is [] if evidence is sufficient or reflector is disabled.
    """
    state.iteration += 1

    # Hard stops: max iterations or enough evidence
    if not use_reflector:
        state.sufficient = True
        state.log("reflect", mode="disabled", sufficient=True)
        return state, []

    if state.iteration >= MAX_ITERATIONS:
        state.sufficient = True
        state.log("reflect", mode="max_iter", sufficient=True)
        return state, []

    if len(state.evidence) >= MIN_EVIDENCE_CHUNKS and state.iteration >= 2:
        state.sufficient = True
        state.log("reflect", mode="enough_evidence", sufficient=True,
                  n_evidence=len(state.evidence))
        return state, []

    # LLM reflection
    evidence_summary = _summarize_evidence(state)
    prompt = REFLECTOR_PROMPT.format(
        question=state.question,
        n_chunks=len(state.evidence),
        n_papers=len(set(c["arxiv_id"] for c in state.evidence)),
        evidence_summary=evidence_summary,
    )
    response = call_llm(
        system=REFLECTOR_SYSTEM,
        user=prompt,
        model=FAST_MODEL,
        max_tokens=MAX_TOKENS_FAST,
        temperature=TEMPERATURE,
    )
    state.tool_call_count += 1

    # Parse response
    sufficient = False
    follow_up = []
    missing = ""

    for line in response.strip().split("\n"):
        if line.startswith("SUFFICIENT:"):
            val = line.split(":", 1)[1].strip().lower()
            sufficient = val.startswith("y")
        elif line.startswith("MISSING:"):
            missing = line.split(":", 1)[1].strip()
        elif line.startswith("FOLLOW_UP:"):
            val = line.split(":", 1)[1].strip()
            if val.lower() not in ("none", "nothing", ""):
                follow_up = [val]

    state.sufficient = sufficient
    state.reflection_notes = missing

    state.log(
        "reflect",
        iteration=state.iteration,
        sufficient=sufficient,
        missing=missing,
        follow_up=follow_up,
        raw=response[:200],
    )

    return state, ([] if sufficient else follow_up)