# Agentic Deep Research

An agentic research assistant for answering research questions over recent LLM-agent literature.

The system builds a corpus of arXiv papers (2024–2026), retrieves relevant evidence using hybrid search, and generates citation-grounded answers through a multi-stage agent pipeline.

## Features

* Corpus of 568 LLM-agent papers from arXiv
* Hybrid retrieval (BGE embeddings + BM25)
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Query planning via sub-question decomposition
* Reflection-driven retrieval refinement
* Citation verification
* Ablation framework for component analysis

## Architecture

Question
→ Planner
→ Hybrid Retrieval
→ Reranker
→ Reflector
→ Synthesizer
→ Citation Verifier

### Models

| Component   | Model                |
| ----------- | -------------------- |
| Planner     | Gemma 4 31B Instruct |
| Synthesizer | Gemma 4 31B Instruct |
| Judge       | Gemma 4 31B Instruct |
| Reflector   | Gemini 2.5 Flash     |
| Verifier    | Gemini 2.5 Flash     |

## Retrieval Stack

* Embeddings: `BAAI/bge-small-en-v1.5`
* Vector Store: `ChromaDB`
* Lexical Search: `BM25`
* Fusion: `Reciprocal Rank Fusion`
* Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Running

```bash
python run_pipeline.py --phase scrape
python run_pipeline.py --phase index
python run_pipeline.py --phase run
python run_pipeline.py --phase eval
```

Or run everything:

```bash
python run_pipeline.py --phase all
```

## Ablations

The system supports the following configurations:

* `full_agent`
* `baseline`
* `no_planner`
* `no_reranker`
* `no_reflector`
* `no_hybrid`
* `no_verifier`

Each configuration disables exactly one component, making it possible to measure its impact on answer faithfulness, latency, and overall system behavior.

## Key Engineering Challenges

* arXiv API rate limiting during corpus collection
* Missing benchmark papers in the initial corpus
* LLM rate limits during large-scale evaluation
* Migration from Groq to Google AI Studio
* Multi-model allocation to balance quality and throughput

## Repository Structure

```text
agent/          Agent pipeline
scraper/        Paper collection and parsing
index/          Retrieval indices
eval/           Evaluation framework
predictions/    Generated answers
corpus/         Parsed paper corpus
```

## Author
Avishi Agarwal