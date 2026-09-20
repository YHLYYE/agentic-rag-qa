# AgenticRAG-QA

Self-correcting, routing Agentic RAG Q&A over FinanceBench, orchestrated by
LangGraph, with a reproducible three-group evaluation.

## Quick start
1. `pip install -e .[dev]`
2. `python -m src.ingest.download` to fetch FinanceBench
3. Build index, then run `python -m src.eval.run_comparison`

## Architecture
Offline: PDFs -> parse -> chunk -> bge-m3 embed -> Faiss + BM25.
Online (LangGraph): route -> retrieve -> CRAG critique -> generate ->
verify-then-answer, with bounded retry loops.

## Why the pieces are where they are
- Retrieval quality (MRR/HitRate) and generation quality (RAGAS) are measured
  separately so one cannot mask the other.
- Citations are a hard gate: an answer whose citations don't resolve to
  retrieved chunks is rejected or downgraded.
- bge-m3 is kept despite the small corpus because the value is in structured
  chunking + self-correction, not embedding size.
