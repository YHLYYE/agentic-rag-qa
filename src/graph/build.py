from langgraph.graph import StateGraph, END

from graph.state import AgenticRAGState
from graph.nodes import (route_node, retrieve_node, critique_node, generate_node,
                         verify_node, clarify_node, give_up_node, rewrite_node)


def _route_after_route(state: AgenticRAGState) -> str:
    if state.get("needs_clarify"):
        return "clarify"
    return "retrieve"


def _route_after_critique(state: AgenticRAGState, max_retry: int = 2) -> str:
    if state["retrieval_verdict"] == "correct":
        return "generate"
    if state.get("retry_count", 0) >= max_retry:
        # 重试耗尽仍判 incorrect → 明确拒答；ambiguous 才带着引用硬闸门尽力作答
        return "give_up" if state["retrieval_verdict"] == "incorrect" else "generate"
    return "retrieve"


def _route_after_verify(state: AgenticRAGState, max_retry: int = 2) -> str:
    if state["grounding_verdict"] == "supported":
        return END
    if state.get("retry_count", 0) >= max_retry:
        return END
    return "retrieve"


def build_graph(retrievers: dict, llm, max_retry: int = 2, checkpointer=None,
                reranker=None):
    """reranker=None 时退化为「无精排」的原行为；传入则启用统一重排。"""
    g = StateGraph(AgenticRAGState)
    # 多轮对话的指代消解放最前：没有历史时是空操作，不影响单轮行为
    g.add_node("rewrite", lambda s: rewrite_node(s, llm))
    # 传入 llm 才会走 LLM 意图分类；llm=None 时自动退回关键词规则兜底
    g.add_node("route", lambda s: route_node(s, llm))
    g.add_node("clarify", clarify_node)
    g.add_node("retrieve", lambda s: retrieve_node(s, retrievers, reranker=reranker))
    g.add_node("critique", lambda s: critique_node(s, llm))
    g.add_node("give_up", give_up_node)
    g.add_node("generate", lambda s: generate_node(s, llm))
    g.add_node("verify", lambda s: verify_node(s, max_retry))

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "route")
    g.add_conditional_edges("route", _route_after_route,
                            {"clarify": "clarify", "retrieve": "retrieve"})
    g.add_edge("clarify", "route")  # 澄清后回到 route 重新路由
    g.add_edge("retrieve", "critique")
    g.add_conditional_edges("critique", lambda s: _route_after_critique(s, max_retry),
                            {"generate": "generate", "retrieve": "retrieve",
                             "give_up": "give_up"})
    g.add_edge("give_up", END)
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", lambda s: _route_after_verify(s, max_retry),
                            {END: END, "retrieve": "retrieve"})
    return g.compile(checkpointer=checkpointer)
