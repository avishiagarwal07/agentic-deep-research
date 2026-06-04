"""
agent/run_agent.py — outputs cited_papers per submission format
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.state import AgentState
from agent.planner import plan
from agent.retriever import retrieve_evidence
from agent.reflector import reflect
from agent.synthesizer import synthesize
from agent.verifier import verify


def run_agent(
    question: str,
    question_id: str = "",
    use_planner:   bool = True,
    use_reranker:  bool = True,
    use_reflector: bool = True,
    use_hybrid:    bool = True,
    use_verifier:  bool = True,
) -> AgentState:
    t0 = time.time()
    state = AgentState(question=question, question_id=question_id)
    state.log("start", question=question)

    # 1. Plan
    state = plan(state, use_planner=use_planner)

    # 2. Initial retrieval
    state = retrieve_evidence(state, use_hybrid=use_hybrid, use_reranker=use_reranker)

    # 3. Reflect loop
    while not state.sufficient:
        state, follow_ups = reflect(state, use_reflector=use_reflector)
        if state.sufficient or not follow_ups:
            break
        state = retrieve_evidence(state, use_hybrid=use_hybrid,
                                  use_reranker=use_reranker, queries=follow_ups)

    # 4. Synthesize
    state = synthesize(state)

    # 5. Verify
    state = verify(state, use_verifier=use_verifier)

    state.latency_seconds = round(time.time() - t0, 2)
    state.log("done", latency=state.latency_seconds, tool_calls=state.tool_call_count)
    return state


def state_to_dict(state: AgentState) -> dict:
    """Submission-format dict. cited_papers is the authoritative field."""
    final_answer = state.verified_answer or state.answer
    return {
        "id":           state.question_id,
        "answer":       final_answer,
        "cited_papers": state.citations,   # ← matches grader expectation
        "trace": {
            "sub_questions":    state.sub_questions,
            "iterations":       state.iteration,
            "tool_calls":       state.tool_call_count,
            "latency_seconds":  state.latency_seconds,
            "reflection_notes": state.reflection_notes,
            "evidence_chunks":  [
                {"arxiv_id": c["arxiv_id"], "text": c["text"][:300]}
                for c in state.evidence[:10]
            ],
            "steps": state.trace,
        },
    }