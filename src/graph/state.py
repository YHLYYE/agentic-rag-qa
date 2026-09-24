from typing import TypedDict


class AgenticRAGState(TypedDict):
    question: str
    intent: str
    route_decision: dict
    retrieved_chunks: list
    retrieval_verdict: str
    candidate_answer: str
    citations: list
    grounding_verdict: str
    final_answer: str
    retry_count: int
    needs_clarify: bool
    clarification: str
