from langgraph.graph import StateGraph, END

from graph.state import AgenticRAGState
from graph.nodes import route_node, retrieve_node, critique_node, generate_node, verify_node, clarify_node


def _route_after_route(state: AgenticRAGState) -> str:
    if state.get("needs_clarify"):
        return "clarify"
    return "retrieve"


def _route_after_critique(state: AgenticRAGState, max_retry: int = 2) -> str:
    if state["retrieval_verdict"] == "correct":
        return "generate"
    if state.get("retry_count", 0) >= max_retry:
        return "generate"
    return "retrieve"


def _route_after_verify(state: AgenticRAGState, max_retry: int = 2) -> str:
    if state["grounding_verdict"] == "supported":
        return END
    if state.get("retry_count", 0) >= max_retry:
        return END
    return "retrieve"


def build_graph(retrievers: dict, llm, max_retry: int = 2, checkpointer=None):
    g = StateGraph(AgenticRAGState)
    # 传入 llm 才会走 LLM 意图分类；llm=None 时自动退回关键词规则兜底
    g.add_node("route", lambda s: route_node(s, llm))
    g.add_node("clarify", clarify_node)
    g.add_node("retrieve", lambda s: retrieve_node(s, retrievers))
    g.add_node("critique", critique_node)
    g.add_node("generate", lambda s: generate_node(s, llm))
    g.add_node("verify", lambda s: verify_node(s, max_retry))

    g.set_entry_point("route")
    g.add_conditional_edges("route", _route_after_route,
                            {"clarify": "clarify", "retrieve": "retrieve"})
    g.add_edge("clarify", "route")  # 澄清后回到 route 重新路由
    g.add_edge("retrieve", "critique")
    g.add_conditional_edges("critique", lambda s: _route_after_critique(s, max_retry),
                            {"generate": "generate", "retrieve": "retrieve"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", lambda s: _route_after_verify(s, max_retry),
                            {END: END, "retrieve": "retrieve"})
    return g.compile(checkpointer=checkpointer)
