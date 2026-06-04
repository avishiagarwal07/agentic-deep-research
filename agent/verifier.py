"""
agent/verifier.py
Citation grounding verifier:
For each inline citation in the answer, checks that the corresponding
evidence chunk actually supports the claim being made.
Removes or flags unsupported citations.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FAST_MODEL, MAX_TOKENS_FAST, TEMPERATURE
from agent.state import AgentState
from agent.llm import call_llm

VERIFY_SYSTEM = """You are a citation fact-checker. Given a claim and a source passage, determine if the passage supports the claim.

Respond with ONLY:
SUPPORTED: yes|no
REASON: <one sentence>"""

VERIFY_PROMPT = """Claim: {claim}

Source passage (arXiv:{arxiv_id}):
{passage}

Does this passage support the claim?"""


def _extract_claims(answer: str) -> list[tuple[str, str]]:
    """
    Extract (sentence_with_citation, arxiv_id) pairs from the answer.
    Looks for sentences ending with [arXiv:XXXX.XXXXX].
    """
    # Split answer into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    claims = []
    for sent in sentences:
        matches = re.findall(r"\[arXiv:([\d\.]+(?:v\d+)?)\]", sent)
        for arxiv_id in matches:
            # Clean the sentence (remove citation bracket for the claim)
            claim = re.sub(r"\s*\[arXiv:[\d\.v]+\]", "", sent).strip()
            if len(claim) > 20:
                claims.append((claim, re.sub(r"v\d+$", "", arxiv_id)))
    return claims


def _find_passage(arxiv_id: str, state: AgentState) -> str | None:
    """Find the best evidence chunk for a given arXiv ID."""
    for chunk in state.evidence:
        if chunk.get("arxiv_id", "").startswith(arxiv_id):
            return chunk["text"][:800]
    return None


def verify(state: AgentState, use_verifier: bool = True) -> AgentState:
    """
    Verify each cited claim. If use_verifier=False, pass through unchanged.
    Updates state.verified_answer with unsupported citations flagged.
    """
    if not use_verifier:
        state.verified_answer = state.answer
        state.log("verify", mode="disabled")
        return state

    claims = _extract_claims(state.answer)
    if not claims:
        state.verified_answer = state.answer
        state.log("verify", n_claims=0)
        return state

    unsupported = set()
    results = []

    # Limit verification to max 10 claims to control latency
    for claim, arxiv_id in claims[:10]:
        passage = _find_passage(arxiv_id, state)
        if passage is None:
            unsupported.add(arxiv_id)
            results.append({"claim": claim[:80], "arxiv_id": arxiv_id,
                            "supported": False, "reason": "passage not in evidence"})
            continue

        prompt = VERIFY_PROMPT.format(
            claim=claim[:300],
            arxiv_id=arxiv_id,
            passage=passage,
        )
        response = call_llm(
            system=VERIFY_SYSTEM,
            user=prompt,
            model=FAST_MODEL,
            max_tokens=MAX_TOKENS_FAST,
            temperature=0.0,
        )
        state.tool_call_count += 1

        supported = False
        reason = ""
        for line in response.strip().split("\n"):
            if line.startswith("SUPPORTED:"):
                supported = line.split(":", 1)[1].strip().lower().startswith("y")
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        if not supported:
            unsupported.add(arxiv_id)
        results.append({
            "claim": claim[:80],
            "arxiv_id": arxiv_id,
            "supported": supported,
            "reason": reason,
        })

    # Remove unsupported citations from the answer
    verified = state.answer
    for arxiv_id in unsupported:
        verified = re.sub(
            rf"\s*\[arXiv:{re.escape(arxiv_id)}(?:v\d+)?\]",
            " [UNSUPPORTED]",
            verified
        )

    # Update citations list: keep only supported ones
    state.citations = [c for c in state.citations if c not in unsupported]
    state.verified_answer = verified

    state.log(
        "verify",
        n_claims=len(results),
        n_unsupported=len(unsupported),
        results=results[:5],  # truncate trace
    )
    return state