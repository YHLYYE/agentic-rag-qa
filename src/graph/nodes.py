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
    q = state["question"]
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
        "route_decision": {"semantic_intent": semantic, "strategy": strategy, "source": source},
    }


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
