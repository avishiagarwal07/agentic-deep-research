"""
agent/synthesizer.py — updated to use cited_papers key per submission format
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYNTH_MODEL, MAX_TOKENS_SYNTH, TEMPERATURE
from agent.state import AgentState
from agent.llm import call_llm

SYNTH_SYSTEM = """You are a research synthesis assistant. Write a precise, evidence-based answer to the research question using ONLY the provided evidence chunks.

Rules:
- Cite sources inline as [arxiv_id] immediately after the claim they support. Example: [2504.19413]
- Use ONLY information from the evidence — no external knowledge or assumptions.
- Be specific: mention method names, numbers, results from the papers.
- For factoid questions: answer in 1-3 sentences, be direct.
- For comparative questions: compare side-by-side, 100-300 words.
- For survey questions: organize by theme, 250-600 words, cite at least 4 papers.
- End with a line: CITED: 2504.19413, 2502.12110, ... (all arXiv IDs you cited)"""

SYNTH_PROMPT = """Question: {question}

Evidence:
{evidence}

Write a well-cited answer based solely on the above evidence."""


def synthesize(state: AgentState) -> AgentState:
    evidence_text = state.evidence_text(max_chunks=20)

    if not evidence_text.strip():
        state.answer = "Insufficient evidence retrieved to answer this question."
        state.citations = []
        return state

    prompt = SYNTH_PROMPT.format(
        question=state.question,
        evidence=evidence_text,
    )

    answer = call_llm(
        system=SYNTH_SYSTEM,
        user=prompt,
        model=SYNTH_MODEL,
        max_tokens=MAX_TOKENS_SYNTH,
        temperature=TEMPERATURE,
    )
    state.tool_call_count += 1

    # Extract CITED: line
    cited = []
    cited_line_match = re.search(r"CITED:\s*(.+)", answer)
    if cited_line_match:
        raw = cited_line_match.group(1)
        cited = [x.strip() for x in re.split(r"[,\s]+", raw) if re.match(r"\d{4}\.\d{4,5}", x.strip())]
        # Remove the CITED line from answer
        answer = answer[:cited_line_match.start()].strip()

    # Also extract inline [arxiv_id] citations
    inline = re.findall(r"\[(\d{4}\.\d{4,5}(?:v\d+)?)\]", answer)
    for c in inline:
        base = re.sub(r"v\d+$", "", c)
        if base not in cited:
            cited.append(base)

    # Normalize all: strip version suffix
    cited = [re.sub(r"v\d+$", "", c) for c in cited]
    # Deduplicate preserving order
    seen = set()
    unique = []
    for c in cited:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    state.answer = answer
    state.citations = unique
    state.log("synthesize", n_evidence=len(state.evidence), n_cited=len(unique))
    return state