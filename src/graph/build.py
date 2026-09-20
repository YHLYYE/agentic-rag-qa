from langgraph.graph import StateGraph, END

from graph.state import AgenticRAGState
from graph.nodes import route_node, retrieve_node, critique_node, generate_node, verify_node


def _route_after_critique(state: AgenticRAGState) -> str:
    if state["retrieval_verdict"] == "correct":
        return "generate"
    if state["retry_count"] >= 2:
        return "generate"
    return "retrieve"


def _route_after_verify(state: AgenticRAGState) -> str:
    if state["grounding_verdict"] == "supported":
        return END
    if state["retry_count"] >= 2:
        return END
    return "retrieve"


def build_graph(retrievers: dict, llm, max_retry: int = 2):
    g = StateGraph(AgenticRAGState)
    g.add_node("route", route_node)
    g.add_node("retrieve", lambda s: retrieve_node(s, retrievers))
    g.add_node("critique", critique_node)
    g.add_node("generate", lambda s: generate_node(s, llm))
    g.add_node("verify", verify_node)

    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "critique")
    g.add_conditional_edges("critique", _route_after_critique,
                            {"generate": "generate", "retrieve": "retrieve"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", _route_after_verify,
                            {END: END, "retrieve": "retrieve"})
    return g.compile()
