"""
agent/retriever.py
Retrieves evidence chunks for each sub-question and adds them to state.
Supports hybrid+reranker or dense-only depending on ablation flags.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K_RERANK
from agent.state import AgentState
from index.hybrid_retriever import retrieve


def retrieve_evidence(
    state: AgentState,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    queries: list[str] | None = None,
) -> AgentState:
    """
    Retrieve chunks for each query (default: state.sub_questions).
    Merges into state.evidence, deduplicating.
    """
    queries = queries or state.sub_questions or [state.question]

    # Build set of already-seen arxiv_ids for diversity (optional exclude)
    seen_arxiv = set(state.cited_arxiv_ids) if state.iteration > 1 else None

    for query in queries:
        chunks = retrieve(
            query=query,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            top_k=TOP_K_RERANK,
            exclude_ids=None,  # don't exclude; let dedup handle it
        )
        state.add_evidence(chunks)
        state.log(
            "retrieve",
            query=query,
            n_chunks=len(chunks),
            iteration=state.iteration,
        )

    return state