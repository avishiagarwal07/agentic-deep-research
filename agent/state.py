"""
agent/state.py
Dataclass holding the full mutable state of one agent run.
Passed through the planner → retriever → reflector → synthesizer → verifier loop.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    # Input
    question: str = ""
    question_id: str = ""

    # Planning
    sub_questions: list[str] = field(default_factory=list)

    # Retrieved evidence: list of chunk dicts
    evidence: list[dict] = field(default_factory=list)

    # Ids already seen so we don't re-retrieve the same chunks
    seen_chunk_ids: set = field(default_factory=set)

    # Papers cited across all evidence
    cited_arxiv_ids: list[str] = field(default_factory=list)

    # Reflector state
    iteration: int = 0
    sufficient: bool = False
    reflection_notes: str = ""

    # Output
    answer: str = ""
    verified_answer: str = ""
    citations: list[str] = field(default_factory=list)   # final arxiv IDs

    # Observability / trace
    trace: list[dict] = field(default_factory=list)

    # Metrics
    tool_call_count: int = 0
    latency_seconds: float = 0.0

    def log(self, step: str, **kwargs):
        """Append a trace event."""
        self.trace.append({"step": step, **kwargs})

    def add_evidence(self, chunks: list[dict]):
        """Merge new chunks, deduplicating by chunk_id."""
        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            if cid not in self.seen_chunk_ids:
                self.seen_chunk_ids.add(cid)
                self.evidence.append(chunk)
                arxiv_id = chunk.get("arxiv_id", "")
                if arxiv_id and arxiv_id not in self.cited_arxiv_ids:
                    self.cited_arxiv_ids.append(arxiv_id)
        self.tool_call_count += 1

    def evidence_text(self, max_chunks: int = 20) -> str:
        """Format top evidence chunks as a numbered context block."""
        lines = []
        for i, c in enumerate(self.evidence[:max_chunks]):
            lines.append(
                f"[{i+1}] arXiv:{c['arxiv_id']}  \"{c['title']}\"\n{c['text']}"
            )
        return "\n\n---\n\n".join(lines)