from graph.state import AgenticRAGState


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
