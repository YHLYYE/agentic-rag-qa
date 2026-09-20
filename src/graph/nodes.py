from graph.state import AgenticRAGState
from rag.citation import parse_citations


def _route_by_rules(question: str) -> str | None:
    q = question.lower()
    if "definition" in q or "define" in q:
        return "keyword"
    return None


def route_node(state: AgenticRAGState) -> dict:
    q = state["question"]
    intent = _route_by_rules(q) or "semantic"
    return {"intent": intent, "route_decision": {"intent": intent, "source": "rule"}}


def verify_node(state: AgenticRAGState) -> dict:
    retrieved_ids = {rc.get("chunk_id") for rc in state["retrieved_chunks"]}
    cits = state.get("citations", [])
    if not cits:
        return {"grounding_verdict": "unsupported"}
    supported = all(c in retrieved_ids for c in cits)
    return {"grounding_verdict": "supported" if supported else "unsupported"}


def retrieve_node(state: AgenticRAGState, retrievers: dict) -> dict:
    from rag.hybrid import merge_and_rerank
    q = state["question"]
    intent = state["intent"]
    if intent == "hybrid":
        lists = [r(q, 8) for r in retrievers.values()]
        merged = merge_and_rerank(lists, top_k=8)
    elif intent == "keyword":
        merged = retrievers["bm25"](q, 8)
    else:
        merged = retrievers["dense"](q, 8)
    out = [{"chunk_id": rc.chunk.chunk_id, "text": rc.chunk.text, "score": rc.score}
           for rc in merged]
    return {"retrieved_chunks": out}


def critique_node(state: AgenticRAGState) -> dict:
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {"retrieval_verdict": "incorrect"}
    return {"retrieval_verdict": "correct"}


def generate_node(state: AgenticRAGState, llm) -> dict:
    chunks = state["retrieved_chunks"]
    ctx = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    prompt = (
        f"Question: {state['question']}\n\nContext:\n{ctx}\n\n"
        "Answer concisely and cite sources inline as {{chunk_id}}."
    )
    answer = llm.complete(prompt)
    cits = parse_citations(answer)
    return {"candidate_answer": answer, "citations": cits}
