from graph.state import AgenticRAGState
from rag.citation import parse_citations


# 语义意图 → 检索策略 的映射
_INTENT_MAP = {
    "factoid": "keyword",     # 查事实/数字 → BM25 关键词
    "comparison": "semantic", # 对比 → dense 语义
    "multi-hop": "hybrid",    # 多跳 → dense+BM25 混合
}


def _route_by_rules(question: str) -> str | None:
    q = question.lower()
    if "definition" in q or "define" in q:
        return "keyword"
    return None


def _route_by_llm(question: str, llm) -> str:
    prompt = (
        "把以下问题分类为三类之一，只输出类别名（factoid / comparison / multi-hop）：\n"
        "- factoid：查一个具体事实或数字（单步查询，如「谁发明了电话」）\n"
        "- comparison：对比两个事物（如「A 和 B 哪个更高」）\n"
        "- multi-hop：需要多步推理才能回答（如「A 主演的电影的导演是谁」）\n\n"
        f"问题：{question}\n\n类别："
    )
    resp = llm.complete(prompt).strip().lower()
    for key in ("factoid", "comparison", "multi-hop", "multihop"):
        if key in resp:
            return "multi-hop" if key == "multihop" else key
    return "factoid"  # 兜底


def route_node(state: AgenticRAGState, llm=None) -> dict:
    q = state["question"].strip()
    # 简单歧义检测：去掉空格后 < 5 个字符（中英文通用，如 "who?" / "谁？"）→ 需要澄清
    if len(q.replace(" ", "")) < 5:
        return {
            "intent": "semantic",
            "needs_clarify": True,
            "route_decision": {"semantic_intent": "ambiguous", "strategy": "semantic", "source": "rule"},
        }
    if llm is not None:
        semantic = _route_by_llm(q, llm)
        strategy = _INTENT_MAP.get(semantic, "semantic")
        source = "llm"
    else:
        strategy = _route_by_rules(q) or "semantic"
        semantic = None
        source = "rule"
    return {
        "intent": strategy,
        "needs_clarify": False,
        "route_decision": {"semantic_intent": semantic, "strategy": strategy, "source": source},
    }


def clarify_node(state: AgenticRAGState) -> dict:
    """interrupt() 人工澄清：暂停等用户补充信息，恢复后回到 route 重新路由。"""
    from langgraph.types import interrupt
    clarification = interrupt({"clarification": "请澄清你的问题（补充缺失信息）"})
    return {"question": clarification, "needs_clarify": False, "clarification": clarification}


_DEGRADED_ANSWER = (
    "抱歉，我没有在知识库中找到足以支撑该结论的资料（已重试 {n} 次）。"
    "请补充问题信息，或直接查阅原始文档。"
)


def verify_node(state: AgenticRAGState, max_retry: int = 2) -> dict:
    retrieved_ids = {rc.get("chunk_id") for rc in state["retrieved_chunks"]}
    cits = state.get("citations", [])
    if cits and all(c in retrieved_ids for c in cits):
        return {"grounding_verdict": "supported",
                "final_answer": state.get("candidate_answer", "")}
    if state.get("retry_count", 0) >= max_retry:
        # 重试已耗尽：降级为「查不到」，而不是留空或抛异常
        return {"grounding_verdict": "unsupported",
                "final_answer": _DEGRADED_ANSWER.format(n=max_retry)}
    return {"grounding_verdict": "unsupported"}


def retrieve_node(state: AgenticRAGState, retrievers: dict) -> dict:
    from rag.hybrid import merge_and_rerank
    q = state["question"]
    attempt = state.get("retry_count", 0)
    # 回退重查必须改变检索行为：沿用原策略+同一 query 会拿到完全相同的 chunks（原地打转）
    intent = "hybrid" if attempt > 0 else state["intent"]
    top_k = 8 if attempt == 0 else 12
    if intent == "hybrid":
        lists = [r(q, top_k) for r in retrievers.values()]
        merged = merge_and_rerank(lists, top_k=top_k)
    elif intent == "keyword":
        merged = retrievers["bm25"](q, top_k)
    else:
        merged = retrievers["dense"](q, top_k)
    out = [{"chunk_id": rc.chunk.chunk_id, "text": rc.chunk.text, "score": rc.score}
           for rc in merged]
    return {"retrieved_chunks": out, "retry_count": attempt + 1}


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
