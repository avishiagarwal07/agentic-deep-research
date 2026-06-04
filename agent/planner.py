"""
agent/planner.py
Decomposes the user's research question into 2-5 focused sub-questions
that can each be answered by retrieving evidence chunks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLAN_MODEL, MAX_TOKENS_PLAN, TEMPERATURE
from agent.state import AgentState
from agent.llm import call_llm

PLANNER_SYSTEM = """You are a research planning assistant. Your job is to break down a complex research question about LLM agents and AI papers into 2-5 specific sub-questions that will each retrieve different relevant evidence.

Rules:
- Output ONLY a numbered list of sub-questions, nothing else.
- Each sub-question should target a distinct aspect of the main question.
- Sub-questions should be specific enough to retrieve focused evidence.
- Do not repeat sub-questions or make them too broad.
- For simple factoid questions, 2 sub-questions are enough.
- For comparative questions, create one sub-question per item being compared.
- For survey questions, create sub-questions covering different aspects."""

PLANNER_PROMPT = """Main research question: {question}

Break this into 2-5 specific sub-questions for evidence retrieval:"""


def plan(state: AgentState, use_planner: bool = True) -> AgentState:
    """
    If use_planner=True: decompose into sub-questions via LLM.
    If use_planner=False (baseline): use question as-is.
    """
    if not use_planner:
        state.sub_questions = [state.question]
        state.log("plan", mode="passthrough", sub_questions=state.sub_questions)
        return state

    prompt = PLANNER_PROMPT.format(question=state.question)
    response = call_llm(
        system=PLANNER_SYSTEM,
        user=prompt,
        model=PLAN_MODEL,
        max_tokens=MAX_TOKENS_PLAN,
        temperature=TEMPERATURE,
    )
    state.tool_call_count += 1

    # Parse numbered list
    sub_questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Remove leading number+dot/paren: "1. " or "1) "
        import re
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned) > 10:
            sub_questions.append(cleaned)

    # Fallback
    if not sub_questions:
        sub_questions = [state.question]

    state.sub_questions = sub_questions[:5]   # cap at 5
    state.log("plan", sub_questions=state.sub_questions, raw=response[:300])
    return state