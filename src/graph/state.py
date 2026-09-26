from typing import TypedDict


class AgenticRAGState(TypedDict):
    question: str
    history: list          # 多轮对话历史 [{"question":..., "answer":...}]，只进生成、不进检索
    rewritten: bool        # 本轮 question 是否被指代消解改写过
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
